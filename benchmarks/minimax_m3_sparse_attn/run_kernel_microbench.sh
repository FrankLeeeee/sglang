#!/usr/bin/env bash
# Level 1 — the sparse attention kernels in isolation.
#
#   CUDA_VISIBLE_DEVICES=0 bash benchmarks/minimax_m3_sparse_attn/run_kernel_microbench.sh
#   MODES="context" bash .../run_kernel_microbench.sh           # skip the sweeps
#   GRANS="block token" bash .../run_kernel_microbench.sh
#   bash .../run_kernel_microbench.sh --context-lens 4096,32768
#
# Pin to an *idle* GPU. These kernels launch grid-wide, so a co-tenant inflates
# every number here (2-3x is typical) and the per-stage breakdown with it.
#
# One GPU, one shape: the released M3 layer, 64 Q / 4 KV / 4 index heads. Shape
# sensitivity is the job of the `num_q_heads` / `num_kv_heads` dimension sweeps,
# which cover the same axis a TP sweep did without pretending to be a topology.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PYTHON="${PYTHON:-python}"
# memory: static KV accounting (no timing) | context: 4k..1M | sweeps: shape dims
MODES="${MODES:-memory context sweeps}"
# Space- or comma-separated. Normalize once for bench_kernels.py.
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
GRAN_CSV="$(IFS=,; echo "${GRAN_LIST[*]}")"


echo "==> level 1 kernel microbenchmarks -> $KERNEL_OUT"
echo "    modes: $MODES | granularities: $GRAN_CSV"

OUT_DIR="$KERNEL_OUT/results"

for MODE in $MODES; do
  case "$MODE" in
    memory|context|sweeps) ;;
    *)
      echo "error: unsupported mode '$MODE' (expected memory, context, or sweeps)" >&2
      exit 2
      ;;
  esac
  echo "--> mode: $MODE"
  # `memory` is pure arithmetic, so it needs no idle window.
  if [ "$MODE" = "memory" ]; then
    "$PYTHON" "$HERE/bench_kernels.py" "$@" \
      --mode memory --granularity "$GRAN_CSV" \
      -o "$OUT_DIR" --tag kernels_memory
  else
    "$PYTHON" "$HERE/bench_kernels.py" "$@" \
      --mode "$MODE" --granularity "$GRAN_CSV" \
      --wait-for-idle "$WAIT_IDLE" -o "$OUT_DIR" --tag "kernels_$MODE"
  fi
done

echo "--> plots"
replot "$OUT_DIR" "$KERNEL_OUT/plots"

echo
echo "==> done. CSV/JSON under $KERNEL_OUT/results/, plots under $KERNEL_OUT/plots/"
