#!/usr/bin/env bash
# Correctness of the MiniMax-M3 sparse attention kernels, against PyTorch
# references. No benchmarking, no GPU-idleness requirement — these only check
# numbers, so a busy GPU is fine.
#
#   bash benchmarks/minimax_m3_sparse_attn/run_accuracy_test.sh
#   bash .../run_accuracy_test.sh -k chunkedprefill
#   PYTEST_ARGS="-k chunkedprefill" bash .../run_accuracy_test.sh
#   PYTEST_ARGS='-k "decode and not padding"' bash .../run_accuracy_test.sh
#
# Needs pytest:  uv pip install --python .venv/bin/python pytest
#
# Every suite runs even if an earlier one fails, so one invocation shows the
# whole picture; the script exits non-zero if any suite failed.
#
# Results are saved alongside the benchmark ones, under
# $OUT/bench_accuracy/results/: one <suite>.log and <suite>.junit.xml per suite,
# plus summary.json / summary.txt for the whole run. Each invocation overwrites
# them, exactly as the benchmark runners overwrite their JSON/CSV.
#
# This is *kernel* accuracy, not model accuracy. Everything in this directory
# runs on dummy weights and says nothing about generation quality — in
# particular, token granularity changes what M3 attends to and its weights were
# trained against block selection. For a real eval see the README.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PYTHON="${PYTHON:-python}"
PYTEST_ARGS="${PYTEST_ARGS:-}"
# Parse into an array so quoted expressions survive: PYTEST_ARGS='-k "a or b"'
# must reach pytest as two arguments, not four.
if ! "$PYTHON" -c \
  'import shlex, sys
try:
    shlex.split(sys.argv[1])
except ValueError:
    raise SystemExit(1)' \
  "$PYTEST_ARGS"; then
  echo "error: PYTEST_ARGS contains invalid shell-style quoting" >&2
  exit 2
fi
mapfile -d '' -t PYTEST_ARGV < <(
  "$PYTHON" -c \
    'import shlex, sys; [sys.stdout.write(arg + "\0") for arg in shlex.split(sys.argv[1])]' \
    "$PYTEST_ARGS"
)
RESULT_DIR="$ACCURACY_OUT/results"
mkdir -p "$RESULT_DIR"
STARTED_AT="$(date -Is)"
cd "$ROOT"

echo "==> accuracy tests -> $RESULT_DIR"

FAILED=()
SKIPPED=()
# One tab-separated record per suite: slug, name, status, exit code, log, xml.
RECORDS=()

run_suite() {
  local slug="$1" name="$2" path="$3" rc=0 status
  shift 3
  local log="$RESULT_DIR/$slug.log" xml="$RESULT_DIR/$slug.junit.xml"
  banner "$name"
  # tee keeps the run watchable while the log is captured; pipefail is already
  # set, and tee never fails, so the pipeline's status is pytest's own.
  "$PYTHON" -m pytest "$path" -q --no-header --junit-xml "$xml" \
    "${PYTEST_ARGV[@]}" "$@" 2>&1 | tee "$log" || rc=$?
  case "$rc" in
    0) status=passed ;;
    # pytest exit 5 = nothing collected. Expected when PYTEST_ARGS filters a
    # suite down to nothing; not a failure.
    5) echo "    (no tests matched the filter — skipped)"
       status=skipped; SKIPPED+=("$name") ;;
    *) status=failed; FAILED+=("$name") ;;
  esac
  RECORDS+=("$slug	$name	$status	$rc	$log	$xml")
}

run_suite harness_units \
  "CPU harness units (stage attribution, memory accounting, budget cap)" \
  benchmarks/minimax_m3_sparse_attn/test_harness_units.py "$@"
run_suite block_kernels \
  "block-granularity kernels (sparse GQA + indexer vs reference)" \
  python/sglang/srt/layers/attention/minimax_sparse_ops/tests/ "$@"
run_suite block_topk \
  "block-level top-k selectors (decode radix + prefill radix path)" \
  test/registered/kernels/ops/attention/test_minimax_decode_topk.py "$@"
run_suite token_dense \
  "token-granularity + dense kernels" \
  test/registered/kernels/ops/attention/test_minimax_token_sparse.py "$@"

# summary.json is the machine-readable record: per-suite status plus the
# test counts parsed back out of each JUnit XML. summary.txt is the same
# thing as the pass/fail table printed below.
printf '%s\n' "${RECORDS[@]}" | "$PYTHON" \
  "$HERE/summarize_accuracy.py" --out "$RESULT_DIR" --started-at "$STARTED_AT" \
  --pytest-args "$PYTEST_ARGS"

echo
if [ ${#SKIPPED[@]} -gt 0 ]; then
  echo "==> skipped (filter matched nothing): ${#SKIPPED[@]}"
  for s in "${SKIPPED[@]}"; do echo "      - $s"; done
fi
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "==> FAILED suites: ${#FAILED[@]}"
  for s in "${FAILED[@]}"; do echo "      - $s"; done
  echo "==> logs: $RESULT_DIR"
  exit 1
fi
echo "==> all accuracy tests passed. Logs + summary under $RESULT_DIR"
