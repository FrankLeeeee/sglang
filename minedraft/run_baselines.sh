#!/usr/bin/env bash
# Usage: run_baselines.sh <gpu> <port> <specfile>
# specfile lines: tag|target|extra server flags
set -u
GPU=$1; PORT=$2; SPEC=$3
cd ~/md
source ~/sglang/.venv/bin/activate
set -a; [ -f ~/.radix_env ] && . ~/.radix_env; set +a
export HF_HUB_OFFLINE=1
export CUDA_HOME=$HOME/sglang/.venv/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH CPATH=$CUDA_HOME/include/cccl:${CPATH:-}
export LIBRARY_PATH=$CUDA_HOME/lib:${LIBRARY_PATH:-}
export PYTHONPATH=$HOME/mdtiming:${PYTHONPATH:-} MD_TIMING_OUT=$HOME/md/timing/TAGPLACEHOLDER
mkdir -p timing ${TIMDIR:-timing}
mkdir -p results logs ${RESDIR:-results}
while IFS='|' read -r TAG TARGET EXTRA; do
  [[ -z "$TAG" || "$TAG" == \#* ]] && continue
  [[ -f ${RESDIR:-results}/$TAG.json ]] && { echo "skip $TAG"; continue; }
  echo "=== $TAG ($(date +%T))"
  export MD_TIMING_OUT=$HOME/md/${TIMDIR:-timing}/$TAG
  CUDA_VISIBLE_DEVICES=$GPU python -m sglang.launch_server --model-path "$TARGET" --port $PORT \
    --mem-fraction-static 0.8 --max-running-requests ${MAXRUN:-64} --cuda-graph-max-bs-decode ${MAXRUN:-64} $EXTRA \
    > logs/$TAG.server.log 2>&1 &
  SPID=$!
  ok=0
  for i in $(seq 1 180); do
    if curl -sf http://127.0.0.1:$PORT/health_generate >/dev/null 2>&1; then ok=1; break; fi
    kill -0 $SPID 2>/dev/null || break
    sleep 5
  done
  if [[ $ok == 1 ]]; then
    timeout 1500 python ~/bench_closed_loop.py --url http://127.0.0.1:$PORT --tokenizer "$TARGET" \
      --prompts ~/prompts.jsonl --conc ${CONC:-1,4,16,64} --max-new ${MAXNEW:-256} --rounds ${ROUNDS:-2} --out ${RESDIR:-results}/$TAG.json \
      2>&1 | tee logs/$TAG.bench.log
  else
    echo "server failed: $TAG"; tail -20 logs/$TAG.server.log
  fi
  kill $SPID 2>/dev/null; sleep 3; kill -9 $SPID 2>/dev/null
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5
done < "$SPEC"
echo "DONE $(date +%T)"
