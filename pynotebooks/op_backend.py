import torch
from typing import Tuple
from abc import ABC, abstractmethod
from configs import OpType, CacheConfig, BackendConfig, ModelConfig

class OpsBackend(ABC):
    """Layer backend abstract."""

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """get backend name."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """get builder of given layer type."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_attention_metadata_cls():
        """get attention metadata class."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """get block shape of k."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """get block shape of v."""
        raise NotImplementedError

    @classmethod
    def update_step_context(cls, step_context):
        """update StepContext for inference.

        attention meta should be built here.
        """
        return step_context

    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig,
                           cache_config: CacheConfig,
                           backend_config: BackendConfig,
                           device: torch.device):
        """build graph runner."""
        from cuda.graph_runner import GraphRunner
        return GraphRunner(model, model_config, cache_config, backend_config,
                           device)


class DefaultOpsBackend(OpsBackend):

    @staticmethod
    def get_name() -> str:
        return 'default'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """get builder of given layer type."""
        if layer_type == OpType.Linear:
            from cuda.linear import DefaultLinearBuilder
            return DefaultLinearBuilder
        elif layer_type == OpType.RotaryEmbedding:
            from cuda.rotary_embedding import DefaultRotaryEmbeddingBuilder
            return DefaultRotaryEmbeddingBuilder
        elif layer_type == OpType.ApplyRotaryEmb:
            from cuda.apply_rotary_emb import DefaultApplyRotaryEmbBuilder
            return DefaultApplyRotaryEmbBuilder
        elif layer_type == OpType.SiluAndMul:
            from cuda.activation import DefaultSiluAndMulBuilder
            return DefaultSiluAndMulBuilder
        elif layer_type == OpType.GeluAndMul:
            from cuda.activation import DefaultGeluAndMulBuilder
            return DefaultGeluAndMulBuilder
        elif layer_type == OpType.RMSNorm:
            from cuda.norm import DefaultRMSNormBuilder
            return DefaultRMSNormBuilder
        elif layer_type == OpType.LayerNorm:
            from cuda.norm import DefaultLayerNormBuilder
            return DefaultLayerNormBuilder
        elif layer_type == OpType.MultinomialSampling:
            from cuda.multinomial_sampling import DefaultMultinomialSamplingBuilder
            return DefaultMultinomialSamplingBuilder
        elif layer_type == OpType.LinearW4A16:
            from cuda.awq_modules import DefaultLinearW4A16Builder
            return DefaultLinearW4A16Builder
        elif layer_type == OpType.SoftmaxTopK:
            from cuda.moe import DefaultSoftmaxTopKBuilder
            return DefaultSoftmaxTopKBuilder
        else:
            raise RuntimeError(f'{layer_type} not supported.')

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """get block shape of k."""
        return (
            block_size,
            num_heads,
            head_size,
        )

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """get block shape of v."""
        return (
            block_size,
            num_heads,
            head_size,
        )


"pytorch op_backend"
class CudaOpsBackend(DefaultOpsBackend):
    """cuda layer backend."""

    @staticmethod
    def get_name() -> str:
        """backend name."""
        return 'cuda'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """get cuda layer builder."""
        if layer_type == OpType.Attention:
            from cuda.attention import TritonAttentionBuilder
            return TritonAttentionBuilder
        elif layer_type == OpType.ApplyRotaryEmb:
            from cuda.apply_rotary_emb import TritonApplyRotaryEmbBuilder
            return TritonApplyRotaryEmbBuilder
        elif layer_type == OpType.RMSNorm:
            from cuda.norm import TritonRMSNormBuilder
            return TritonRMSNormBuilder
        elif layer_type == OpType.LoRA:
            from cuda.lora import TritonLoRABuilder
            return TritonLoRABuilder
        elif layer_type == OpType.LinearW8A8:
            from cuda.qmodules import TritonLinearW8A8Builder
            return TritonLinearW8A8Builder
        elif layer_type == OpType.RMSNormW8A8:
            from cuda.qmodules import TritonRMSNormBuilder
            return TritonRMSNormBuilder
        elif layer_type == OpType.MultinomialSampling:
            from cuda.multinomial_sampling import TritonMultinomialSamplingBuilder
            return TritonMultinomialSamplingBuilder
        elif layer_type == OpType.SiluAndMul:
            from cuda.activation import TritonSiluAndMulBuilder
            return TritonSiluAndMulBuilder
        # elif layer_type == OpType.LinearW4A16:
        #     from awq.modules.linear.gemm import AWQ_INSTALLED
        #     if AWQ_INSTALLED:
        #         from cuda.awq_modules import AwqLinearW4A16Builder
        #         return AwqLinearW4A16Builder
        #     else:
        #         logger.debug(
        #             f'Op {layer_type} fallback to default implementation.')
        #         return super().get_layer_impl_builder(layer_type)
        elif layer_type == OpType.FusedMoE:
            from cuda.moe import TritonFusedMoEBuilder
            return TritonFusedMoEBuilder
        else:
            print(
                f'Op {layer_type} fallback to default implementation.')
            return super().get_layer_impl_builder(layer_type)

    @staticmethod
    def get_attention_metadata_cls():
        """get attention metadata class."""
        from cuda.attention import TritonAttentionMetadata
        return TritonAttentionMetadata

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """get k block shape."""
        return (
            block_size,
            num_heads,
            head_size,
        )

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """get v block shape."""
        return (
            block_size,
            num_heads,
            head_size,
        )

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        attn_meta_cls = cls.get_attention_metadata_cls()
        q_seqlens = step_context.q_seqlens
        q_start_loc = q_seqlens.cumsum(0) - q_seqlens
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_seqlens=step_context.kv_seqlens,
        )

        step_context.attn_metadata = attn_metadata
        return step_context

    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig,
                           cache_config: CacheConfig,
                           backend_config: BackendConfig,
                           device: torch.device):
        """build graph runner."""
        from cuda.graph_runner import CUDAGraphRunner
        return CUDAGraphRunner(model, model_config, cache_config,
                               backend_config, device)
