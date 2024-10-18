# glm-4v-9b only supports tensor parallelism = 2
export CUDA_VISIBLE_DEVICES="2,3"
#model="/data/zszhangyk/Delius/models/glm-4v-9b"
model="/models/glm-4v-9b"

## cache-max-entry-count=0.4 is optimal, balancing the cost between of time and memory.
lmdeploy serve api_server $model\
  --dtype bfloat16\
  --tp 2\
  --enable-prefix-caching\
  --cache-max-entry-count 0.4\
  --model-name "glm-4v-9b"\
  --server-name "0.0.0.0"\
  --server-port 23333
#  --log-level INFO
