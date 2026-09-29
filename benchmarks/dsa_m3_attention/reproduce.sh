# Run from the repository root, using the benchmark environment.
# Correctness (includes causal isolation and CUDA graph replay).
CUDA_VISIBLE_DEVICES=0 BENCH_PORT=29670 .venv/bin/python -m pytest \
  benchmarks/dsa_m3_attention/test_implementations.py -q

# Full DSA sweep: 3 warmups, then 15 recorded samples per point.
CUDA_VISIBLE_DEVICES=0 BENCH_PORT=29671 .venv/bin/python \
  benchmarks/dsa_m3_attention/benchmark.py --architecture dsa \
  --configs 1,2 --contexts 16384,32768,65536,131072 --batches 1,2,4,8,16,32 \
  --phases prefill,decode --variants dense_fa2,dense_fa3,sparse \
  --prefill-budget 2048 --topk 2048 --warmup 3 --iterations 15 \
  --output benchmarks/dsa_m3_attention/results/dsa.json

# Full M3 sweep, same grid and timing contract.
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/dsa_m3_attention/benchmark.py --architecture m3 \
  --configs 1,2 --contexts 16384,32768,65536,131072 --batches 1,2,4,8,16,32 \
  --phases prefill,decode --variants dense_fa2,dense_fa3,sparse \
  --prefill-budget 2048 --topk 2048 --warmup 3 --iterations 15 \
  --output benchmarks/dsa_m3_attention/results/m3.json

# Independent component timings and kernel diagnostics (also warmed up).
for arch in dsa m3; do
  CUDA_VISIBLE_DEVICES=0 BENCH_PORT=29672 .venv/bin/python \
    benchmarks/dsa_m3_attention/benchmark.py --architecture "$arch" \
    --contexts 16384,131072 --batches 1,32 --variants sparse --stages \
    --prefill-budget 2048 --topk 2048 --warmup 3 --iterations 15 \
    --output "benchmarks/dsa_m3_attention/results/${arch}_stages.json"
  CUDA_VISIBLE_DEVICES=0 BENCH_PORT=29673 .venv/bin/python \
    benchmarks/dsa_m3_attention/profile_stages.py --architecture "$arch" \
    --output "benchmarks/dsa_m3_attention/results/${arch}_profile.json"
done

# Validate the full grid; generate tables, Chinese report, and labeled bars.
.venv/bin/python benchmarks/dsa_m3_attention/report.py
