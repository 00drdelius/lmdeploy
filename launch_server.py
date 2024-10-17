import os
from lmdeploy.api import serve
from lmdeploy import PytorchEngineConfig
# glm-4v-9b only supports tensor parallelism = 2
os.environ.setdefault('CUDA_VISIBLE_DEVICES',"2,3")
model="/data/zszhangyk/Delius/models/glm-4v-9b"

  # --session-len 2560\ useless
  # --enable-prefix-caching\ faster than 2s but allocate memory more

# lmdeploy serve api_server $model\
#   --log-level INFO\
#   --dtype bfloat16\
#   --tp 2\
#   --cache-max-entry-count 0.8\
#   --model-name "glm-4v-9b"\
#   --server-name "0.0.0.0"\
#   --server-port 23333
# if __name__ == '__main__':
client = serve(model_path=model,
      model_name=None,
      backend='pytorch',
      server_name = '0.0.0.0',
      server_port = 23333,
      log_level= 'INFO',
      backend_config=PytorchEngineConfig(
          dtype="bfloat16",
          tp=2,
          cache_max_entry_count=0.4,
      ))