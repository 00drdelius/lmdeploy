from typing import *

import torch
from torch import nn
import torch.nn.functional as F

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, FlashAttention, RMSNorm, RopeType, SiluAndMul,
                                 build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_merged_colwise_linear, build_qkv_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin, next_power_of_2
from .utils.model import DeployModelMixin
from ...utils import get_logger
from ..distributed import get_world_rank
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig, Qwen2_5_VLConfig

logger = get_logger(__name__)


def _apply_mrope_selection(hidden_states: torch.Tensor, mrope_position_ids: torch.Tensor, mrope_section: List[int],
                           position_ids: torch.Tensor, rotary_emb_func: Callable):
    _mrope_position_ids = torch.zeros(3, position_ids.shape[-1], dtype=position_ids.dtype, device=position_ids.device)
    _mrope_position_ids[:, :mrope_position_ids.shape[-1]] = mrope_position_ids
    cos, sin = rotary_emb_func(hidden_states, _mrope_position_ids)
    _cos = torch.zeros(cos.shape[1], cos.shape[-1], dtype=cos.dtype, device=cos.device)
    _sin = torch.zeros_like(_cos)
    mrope_section = mrope_section * 2

    def _apply_split(src, dst):
        start = 0
        for i, m in enumerate(src.split(mrope_section, dim=-1)):
            dst[:, start:start + mrope_section[i]] = m[i % 3]
            start += mrope_section[i]

    _apply_split(cos, _cos)
    _apply_split(sin, _sin)

    return _cos, _sin


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig,
                 dtype:torch.dtype=None,
                 device:torch.device=None,
                 bias: bool = False):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)

        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=bias,
            dtype=dtype, device= device,
            quant_config=quantization_config,
            is_tp=True,
        )
        self.act_fn = SiluAndMul(inplace=True)

        self.down_proj = build_rowwise_linear(config.intermediate_size,
                                              config.hidden_size,
                                              bias=bias,
                                              quant_config=quantization_config,
                                              dtype=dtype,
                                              device=device,
                                              is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
        dtype: torch.dtype = None,
        device: torch.device = None
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
            dtype=dtype,
            device=device
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0, device: torch.device = None) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self,
                 dim: int,
                 context_dim: int,
                 spatial_merge_size: int = 2,
                 dtype: torch.dtype = None,
                 device: torch.device = None) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6, dtype=dtype, device=device)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class Qwen2_5_VLVisionAttention(nn.Module):
    """Vision attention."""

    def __init__(self, config: Qwen2_5_VLVisionConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        dim = config.hidden_size
        num_heads = config.num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        # packed qkv
        self.qkv = build_qkv_proj(
            dim,
            num_q_heads=num_heads,
            num_kv_heads=num_heads,
            head_size=head_dim,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attention = FlashAttention(
            num_heads,
            head_dim,
            causal=False,
        )

        # o_proj
        self.proj = build_rowwise_linear(dim,
                                         dim,
                                         bias=True,
                                         quant_config=quantization_config,
                                         dtype=dtype,
                                         device=device,
                                         is_tp=True)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor]) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        # qkv proj
        qkv_states = self.qkv(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        q, k, v = self.qkv.split_qkv(qkv_states)

        cos, sin = rotary_pos_emb
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = self.attention(
            q,
            k,
            v,
            q_start_loc=cu_seqlens[:-1],
            q_seqlens=cu_seqlens[1:] - cu_seqlens[:-1],
        )

        attn_output = attn_output.reshape(seq_length, -1)

        # o proj
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self,
                 config: Qwen2_5_VLVisionConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):

        super().__init__()
        self.layer_idx = layer_idx
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = RMSNorm(config.hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.attn = Qwen2_5_VLVisionAttention(config, dtype=dtype, device=device)
        self.mlp = Qwen2_5_VLMLP(config, bias=True, dtype=dtype, device=device)

    def forward(self, hidden_states:torch.Tensor,
                cu_seqlens:torch.Tensor,
                rotary_pos_emb:tuple[torch.Tensor],
                residual: Optional[torch.Tensor] = None) -> torch.Tensor:

        if residual is None:
            residual = hidden_states
            hidden_states = self.norm1(hidden_states)
        else:
            hidden_states, residual = self.norm1(hidden_states, residual)

        hidden_states = self.attn(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        hidden_states, residual = self.norm2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class Qwen2_5_VisionTransformerPretrainedModel(nn.Module):

    def __init__(self, config: Qwen2_5_VLVisionConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            dtype=dtype,
            device=device
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2, device=device)

        self.blocks = nn.ModuleList(
            [Qwen2_5_VLVisionBlock(config, layer_idx, dtype=dtype, device=device) for layer_idx in range(config.depth)]
        )
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            dtype=dtype,
            device=device
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def slide_rotary_pos_emb(self,window_index:list,
                             rotary_pos_emb: torch.Tensor,
                             seq_len: int):
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        return rotary_pos_emb

    def forward(self, hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor,
                cu_window_seqlens: torch.Tensor,
                window_index: list, ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        # window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        # cu_window_seqlens = torch.tensor(
        #     cu_window_seqlens,
        #     device=hidden_states.device,
        #     dtype=torch.int32,
        # )
        # cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        # rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        # rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        # rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        # cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        #     dim=0,
        #     # Select dtype based on the following factors:
        #     #  - FA2 requires that cu_seqlens_q must have dtype int32
        #     #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        #     # See https://github.com/huggingface/transformers/pull/34852 for more information
        #     dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        # )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        residual=None
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states, residual = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb=rotary_pos_emb,
                residual=residual
            )
            hidden_states = hidden_states+residual

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class Qwen2_5_VLAttention(nn.Module):
    """Rewrite module of Qwen2Attention."""

    def __init__(self, config: Qwen2_5_VLConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)

        # packed qkv
        self.qkv_proj = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
            v_head_size=head_dim,
            sliding_window=config.sliding_window,
        )

        # o_proj
        self.o_proj = build_rowwise_linear(num_heads * head_dim,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )

        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self,
                 config: Qwen2_5_VLConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        self.self_attn = Qwen2_5_VLAttention(config, dtype, device)
        self.mlp = Qwen2_5_VLMLP(config,dtype,device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps,
                                                quant_config=quantization_config,
                                                dtype=dtype,
                                                device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class Qwen2_5_VLModel(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None,):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.mrope_section = config.rope_scaling['mrope_section']

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx, dtype=dtype, device=device) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=dtype, device=device)
        
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        rope_init_map={
            "default": RopeType.Default,
            "linear": RopeType.LinearScaling,
            "dynamic": RopeType.DynamicNTKScaling,
            # "yarn": RopeType.Yarn,
            # "longrope": RopeType.LongRoPEScaling,
            # "llama3": RopeType.Llama3,
        }
        self.rotary_emb = build_rotary_embedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            emb_type=rope_init_map.get(self.rope_type,RopeType.Default)
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mrope_position_ids: torch.LongTensor = None,
    ):
        """Rewrite of LlamaModel.forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        if mrope_position_ids is None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            cos, sin = cos[0], sin[0]
        else:
            cos, sin = _apply_mrope_selection(hidden_states, mrope_position_ids, self.mrope_section, position_ids,
                                              self.rotary_emb)
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class Qwen2_5_VLForConditionalGeneration(nn.Module, DeployModelMixin, CudaGraphMixin):

    packed_modules_mapping = {
        'qkv_proj': [
            'q_proj',
            'k_proj',
            'v_proj',
        ],
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    def __init__(self,
                 config:Qwen2_5_VLConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr

        self.input_processor = Qwen2_5VLInputProcessor(config)

        self.visual = Qwen2_5_VisionTransformerPretrainedModel(
            config.vision_config,
            dtype=dtype,
            device=device,
        )

        self.model = Qwen2_5_VLModel(config, dtype=dtype, device=device)

        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        mrope_position_ids: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        vis_cu_seqlens: torch.Tensor = None,
        vis_pos_emb: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        cu_window_seqlens: torch.Tensor = None,
        window_index: List = None,
        **kwargs,
    ):
        """model forward, return logits."""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                dtype = inputs_embeds.dtype
                pixel_values = pixel_values.to(dtype)
                vis_pos_emb = (vis_pos_emb[0].to(dtype), vis_pos_emb[1].to(dtype))
                image_embeds = self.visual(pixel_values,
                                           cu_seqlens=vis_cu_seqlens,
                                           rotary_pos_emb=vis_pos_emb,
                                           cu_window_seqlens=cu_window_seqlens,
                                           window_index=window_index)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask[..., None], image_embeds)

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=mrope_position_ids,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.lm_head(hidden_states)

    def update_weights(self):
        """update weights."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """prepare input."""

        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        pixel_values = None
        vis_cu_seqlens = None
        vis_pos_emb = None
        image_mask = None
        cu_window_seqlens = None
        window_index = None
        if context.input_multimodals is not None:
            image_data = [input_mm['image'] for input_mm in context.input_multimodals]

            if len(image_data) > 0:
                # flatten batch
                image_data = [data for im_data in image_data for data in im_data]
                pixel_values = torch.cat([data.data for data in image_data])
                image_token_id = image_data[0].meta['image_token_id']
                image_mask = input_ids == image_token_id
                grid_thw = torch.cat([data.meta['grid_thw'] for data in image_data]).cpu()
                vis_pos_emb = self.visual.rot_pos_emb(grid_thw)
                window_index, cu_window_seqlens = self.visual.get_window_index(grid_thw)
                cu_window_seqlens = torch.tensor(
                    cu_window_seqlens,
                    device=pixel_values.device,
                    dtype=torch.int32,
                )
                cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
                
                vis_cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                                         grid_thw[:, 0]).to(pixel_values.device)
                vis_cu_seqlens = vis_cu_seqlens.cumsum(dim=0, dtype=torch.int32)
                seq_len, _ = pixel_values.size()
                vis_pos_emb = self.visual.slide_rotary_pos_emb(window_index, vis_pos_emb, seq_len)
                vis_pos_emb = vis_pos_emb.repeat(1, 2)
                vis_pos_emb = (vis_pos_emb.cos(), vis_pos_emb.sin())

        mrope_position_ids = getattr(context, 'mrope_position_ids', None)

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=mrope_position_ids,
            pixel_values=pixel_values,
            vis_cu_seqlens=vis_cu_seqlens,
            vis_pos_emb=vis_pos_emb,
            image_mask=image_mask,
            cu_window_seqlens=cu_window_seqlens,
            window_index=window_index,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                if '.qkv.' in name:
                    param = params_dict[name]
                    q, k, v = param.weight_spliter(loaded_weight)
                    load_weight(param, q, shard_id='q')
                    load_weight(param, k, shard_id='k')
                    load_weight(param, v, shard_id='v')
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """make cudagraph buffers from forward inputs."""
        max_tokens = graph_meta.max_tokens

        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers['mrope_position_ids'] = mrope_position_ids.new_zeros(3, max_tokens)

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """fill cudagraph buffers from forward inputs."""

        new_inputs = super().fill_buffers_cudagraph(graph_meta=graph_meta, **kwargs)

        input_ids = kwargs.get('input_ids')
        attn_metadata = kwargs.get('attn_metadata')
        block_offsets = attn_metadata.block_offsets
        num_tokens = input_ids.size(-1)
        batch_size, _ = block_offsets.size()
        new_batch_size = next_power_of_2(batch_size)

        is_decoding = graph_meta.is_decoding
        input_buffers = graph_meta.input_buffers
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers['mrope_position_ids'][:, :num_tokens] = mrope_position_ids
            if is_decoding:
                new_inputs['mrope_position_ids'] = input_buffers['mrope_position_ids'][:, :new_batch_size]
            else:
                new_inputs['mrope_position_ids'] = input_buffers['mrope_position_ids']

        return new_inputs

    def _update_model_meta_decoding(self, context: StepContext):
        """update model meta for decoding."""
        model_metas = context.model_metas
        position_ids = context.position_ids

        mrope_deltas = [meta['mrope_delta'] for meta in model_metas]
        mrope_deltas = position_ids.new_tensor(mrope_deltas)
        mrope_position_ids = position_ids + mrope_deltas[None]
        mrope_position_ids = mrope_position_ids.expand(3, -1)

        context.mrope_position_ids = mrope_position_ids
        return model_metas

    def _get_multimodal_pos_ids(self, grid_thw: list, device: torch.device):
        """get mrope ids."""
        t, h, w = grid_thw
        h //= 2
        w //= 2
        stride = torch.tensor([h * w, w, 1], device=device)[:, None]
        size = torch.tensor([t, h, w], device=device)[:, None]
        pos_ids = torch.arange(t * h * w, device=device)[None].expand(3, -1)
        pos_ids = pos_ids // stride % size
        return pos_ids

    def _update_model_meta_prefilling(self, context: StepContext):
        """update model meta for prefilling."""
        model_metas = context.model_metas
        input_multimodals = context.input_multimodals
        if input_multimodals is None:
            input_multimodals = [None] * len(model_metas)
        position_ids = context.position_ids
        batched_pos_ids = position_ids[0].split(context.q_seqlens.tolist())
        mrope_position_ids = []
        new_model_metas = []
        for pos_ids, model_meta, input_mm in zip(batched_pos_ids, model_metas, input_multimodals):
            images = []
            if input_mm is not None:
                images = input_mm['image']
            if model_meta is None or 'mrope_delta' not in model_meta:
                mrope_delta = 0
            else:
                mrope_delta = model_meta['mrope_delta']

            pos_start = pos_ids[0].item()
            mrope_pos_ids = pos_ids + mrope_delta
            mrope_pos_ids = mrope_pos_ids[None].expand(3, -1).clone()
            for img in images:
                grid_thw = img.meta['grid_thw'][0].tolist()
                _, h, w = grid_thw
                h //= 2
                w //= 2
                num_pad = img.end - img.start - max(h, w)
                mrope_delta -= num_pad
                fill_start = img.start - pos_start
                fill_end = img.end - pos_start
                img_pos_ids = self._get_multimodal_pos_ids(grid_thw, pos_ids.device)
                img_pos_ids += mrope_pos_ids[:, fill_start:fill_start + 1]
                mrope_pos_ids[:, fill_end:] -= num_pad
                mrope_pos_ids[:, fill_start:fill_end] = img_pos_ids

            mrope_position_ids.append(mrope_pos_ids)
            new_model_metas.append(dict(mrope_delta=mrope_delta))

        mrope_position_ids = torch.cat(mrope_position_ids, dim=1)
        context.mrope_position_ids = mrope_position_ids

        return new_model_metas

    def update_model_metas(self,
                           past_key_values: List[List[torch.Tensor]],
                           inputs_embeds: Optional[torch.Tensor] = None,
                           context: StepContext = None):
        """update model meta."""
        if context.is_decoding:
            return self._update_model_meta_decoding(context)
        else:
            return self._update_model_meta_prefilling(context)

    def get_input_processor(self) -> BaseModelInputProcessor:
        """get input processor."""
        return self.input_processor


class Qwen2_5VLInputProcessor(BaseModelInputProcessor):
    """qwen2.5 input processor."""

    def __init__(self, config: Qwen2_5_VLConfig):
        self.config = config
    
    def preprocess_input(self,
                         input_ids: List[int],
                         input_multimodals: List[Dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values']
            image_grid_thw = input_mm['image_grid_thw']
            offset = input_mm['offset']
            start = offset
            image_token_id = input_mm.get('image_token_id', 0)
            num_pad = input_mm['image_tokens']
            if isinstance(num_pad, torch.Tensor):
                num_pad = num_pad.item()

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=start,
                                       end=start + num_pad,
                                       meta=dict(grid_thw=image_grid_thw, image_token_id=image_token_id))
            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )
        return result
