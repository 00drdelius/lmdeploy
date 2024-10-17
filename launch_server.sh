# glm-4v-9b only supports tensor parallelism = 2
export CUDA_VISIBLE_DEVICES="2,3"
model="/data/zszhangyk/Delius/models/glm-4v-9b"


lmdeploy serve api_server $model\
  --log-level INFO\
  --dtype bfloat16\
  --tp 2\
  --enable-prefix-caching\
  --cache-max-entry-count 0.4\
  --model-name "glm-4v-9b"\
  --server-name "0.0.0.0"\
  --server-port 23333