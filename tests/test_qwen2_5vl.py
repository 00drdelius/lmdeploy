from lmdeploy import PytorchEngineConfig, GenerationConfig, pipeline
from lmdeploy.vl import load_image
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from pathlib import Path

# model_path='/models/Qwen/Qwen2-VL-2B-Instruct'
model_path='/models/Qwen/Qwen2.5-VL-3B-Instruct'
img_path = str(Path(__file__).parent.joinpath("jinzhe_1024.jpg").absolute())

save_dir="/home/gmcc/workspace/repositories/site-packages/delius-lmdeploy/tests"

backend_config = PytorchEngineConfig(
    dtype="bfloat16",
    tp=1,
    cache_max_entry_count=0.4,
    device_type="cuda",
    eager_mode=True,
    max_prefill_token_num=10240
)
gen_config = GenerationConfig(
    n=1,
    do_sample=False,
    top_p=1,
    temperature=0,
    skip_special_tokens=False
)
image = load_image(img_path)
conversation = [
    {"role":"system","content":"你是一名高级人工智能助手。"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "描述下图片这个女孩"},
            {"type": "image_data","image_data":{"data":image}},
        ],
    }
]
#TODO: 模型貌似没有获取到图像特征。模型看不到我传进去的图片
if __name__ == '__main__':
    engine=pipeline(
    # engine = VLAsyncEngine(
        # backend='pytorch',
        model_path,
        backend_config=backend_config,
        log_level="INFO",
    )
    # resp = engine(("描述下图片这个女孩",image),gen_config=gen_config)
    resp = engine(conversation,gen_config=gen_config)
    print(resp)