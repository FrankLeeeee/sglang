#!/usr/bin/env bash
# Level 2 — end-to-end for one full MiniMax-M3 attention layer, all three
# granularities, on the released 64 Q / 4 KV / 4 index head shape.
#
#   CUDA_VISIBLE_DEVICES=0 bash benchmarks/minimax_m3_sparse_attn/run_e2e.sh
#   GRANS="block token" bash .../run_e2e.sh
#   bash .../run_e2e.sh --context-lens 4096,32768
#
# Runs the real MiniMaxM3Attention.forward() with dummy weights against a real
# MiniMaxSparseKVPool, so the measurement includes the fused QKV + index-QKV
# projection, per-head Gemma RMS-norm, partial RoPE, the fused KV + index-K
# cache store, the full selection pipeline, and o_proj.
#
# End-to-end *for the attention layer*, not the model: no MoE, no TP all-reduce
# after o_proj, no scheduling. A true full-model run needs the ~435B checkpoint
# and a server — see the README.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PYTHON="${PYTHON:-python}"
GRANS="${GRANS:-block token dense}"
GRANS="${GRANS//,/ }"
read -r -a GRAN_LIST <<< "$GRANS"
if [ ${#GRAN_LIST[@]} -eq 0 ]; then
  echo "error: GRANS must contain at least one of: block token dense" >&2
  exit 2
fi
for GRAN in "${GRAN_LIST[@]}"; do
  case "$GRAN" in
    block|token|dense) ;;
    *)
      echo "error: unsupported granularity '$GRAN' (expected block, token, or dense)" >&2
      exit 2
      ;;
  esac
done


# Each run stands up its own world-size-1 TP group; reusing a port fails with
# EADDRINUSE if the previous one has not torn down, so keep counting.
PORT="${PORT:-29531}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "error: PORT must be an integer in [1, 65535], got '$PORT'" >&2
  exit 2
fi
TOTAL_RUNS=${#GRAN_LIST[@]}
if [ $((PORT + TOTAL_RUNS - 1)) -gt 65535 ]; then
  echo "error: PORT range for $TOTAL_RUNS runs exceeds 65535 (start: $PORT)" >&2
  exit 2
fi

echo "==> level 2 full-layer benchmark -> $LAYER_OUT"
echo "    granularities: ${GRAN_LIST[*]} | first port: $PORT"

OUT_DIR="$LAYER_OUT/results"

for GRAN in "${GRAN_LIST[@]}"; do
  echo "--> layer: $GRAN"
  "$PYTHON" "$HERE/bench_layer.py" "$@" \
    --granularity "$GRAN" --port "$PORT" \
    --wait-for-idle "$WAIT_IDLE" -o "$OUT_DIR" --tag "layer_$GRAN"
  PORT=$((PORT + 1))
done

echo "--> plots"
replot "$OUT_DIR" "$LAYER_OUT/plots"

echo
echo "==> done. CSV/JSON under $LAYER_OUT/results/, plots under $LAYER_OUT/plots/"
