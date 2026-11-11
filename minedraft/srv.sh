#!/usr/bin/env bash
# srv.sh <gpu> <port> <model> <tag> [extra...]
GPU=$1; PORT=$2; M=$3; TAG=$4; shift 4
source ~/sglang/.venv/bin/activate
set -a; [ -f ~/.radix_env ] && . ~/.radix_env; set +a
export HF_HUB_OFFLINE=1 CUDA_HOME=$HOME/sglang/.venv/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH CPATH=$CUDA_HOME/include/cccl:${CPATH:-} LIBRARY_PATH=$CUDA_HOME/lib:${LIBRARY_PATH:-}
mkdir -p ~/md/mdlogs
for p in $(pgrep -f "launch_server.*--port $PORT"); do kill -9 $p; done; sleep 2
CUDA_VISIBLE_DEVICES=$GPU nohup python -m sglang.launch_server --model-path $M --port $PORT \
  --mem-fraction-static 0.8 --max-running-requests 128 --cuda-graph-max-bs-decode 128 "$@" > ~/md/mdlogs/$TAG.log 2>&1 &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health_generate >/dev/null && { echo "up $TAG"; exit 0; }; sleep 3; done
echo "FAILED $TAG"; tail -5 ~/md/mdlogs/$TAG.log
