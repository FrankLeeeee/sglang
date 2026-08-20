#!/usr/bin/env bash
# Everything, in order: accuracy tests, level-1 kernel microbenchmarks, then the
# level-2 full-layer benchmark. Each benchmark stage plots its own results.
#
#   CUDA_VISIBLE_DEVICES=0 bash benchmarks/minimax_m3_sparse_attn/run_all.sh
#   SKIP_TESTS=1 CUDA_VISIBLE_DEVICES=0 bash .../run_all.sh   # benchmarks only
#
# The three stages are also runnable on their own:
#   run_accuracy_test.sh      correctness only, no idle GPU needed
#   run_kernel_microbench.sh  level 1
#   run_e2e.sh                level 2
#
# Pin to an *idle* GPU for the benchmark stages; see run_kernel_microbench.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SKIP_TESTS="${SKIP_TESTS:-0}"

if [ "$SKIP_TESTS" != "1" ]; then
  banner "stage 1/3: accuracy tests"
  bash "$HERE/run_accuracy_test.sh"
fi

banner "stage 2/3: level-1 kernel microbenchmarks"
bash "$HERE/run_kernel_microbench.sh"

banner "stage 3/3: level-2 full-layer benchmark"
bash "$HERE/run_e2e.sh"

echo
echo "==> done."
if [ "$SKIP_TESTS" != "1" ]; then
  echo "    accuracy: $ACCURACY_OUT/results/"
fi
echo "    level 1:  $KERNEL_OUT/{results,plots}/"
echo "    level 2:  $LAYER_OUT/{results,plots}/"
