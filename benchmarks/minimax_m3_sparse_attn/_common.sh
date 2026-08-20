#!/usr/bin/env bash
# Shared configuration for the run_*.sh scripts. Sourced, never executed.
#
# Knobs (all overridable from the environment):
#   OUT        where results land                  (default: <this dir>/results)
#   WAIT_IDLE  seconds to wait for an idle GPU     (default: 180; 0 on a dedicated box)
#   METRIC     plot estimator: gpu | min | median  (default: gpu)
#
# Each stage owns a subtree under $OUT, with data and plots kept apart:
#
#   $OUT/bench_accuracy/results/            correctness (run_accuracy_test.sh)
#   $OUT/bench_kernels/{results,plots}/     level 1 (run_kernel_microbench.sh)
#   $OUT/bench_layers/{results,plots}/      level 2 (run_e2e.sh)
#
# The accuracy stage has no plots — it emits pytest logs, JUnit XML and a
# summary. Everything runs the released M3 shape (64 Q / 4 KV / 4 index heads)
# on a single GPU; shape sensitivity lives in the dimension sweeps.
set -euo pipefail

# Key off this file's own path, not the caller's: robust however it is sourced.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

OUT="${OUT:-$HERE/results}"
ACCURACY_OUT="$OUT/bench_accuracy"
KERNEL_OUT="$OUT/bench_kernels"
LAYER_OUT="$OUT/bench_layers"
WAIT_IDLE="${WAIT_IDLE:-180}"
METRIC="${METRIC:-gpu}"

# Regenerate every plot for one results directory into a separate plots
# directory. Idempotent, and it reads whatever JSON is present.
#   replot <results-dir> <plots-dir>
replot() {
  python "$HERE/plot_results.py" --results "$1" --out "$2" --metric "$METRIC"
}


banner() {
  echo
  echo "############################################################"
  echo "# $*"
  echo "############################################################"
}
