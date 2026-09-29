#!/usr/bin/env bash
set -euo pipefail
# 在仓库根目录执行；两块空闲 H200，各跑一个配置。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q \
  benchmarks/gqa_lightning_attention/test_attention.py \
  benchmarks/gqa_lightning_attention/test_two_stage.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/gqa_lightning_attention/benchmark_two_stage.py \
  --experiments 2 --configs 1 --contexts 16384,32768,65536,131072 \
  --batches 1,2,4,8,16,32 --phases prefill,decode \
  --variants dense_fa2,dense_fa3,sparse,two_stage --candidate-blocks 64 \
  --prefill-budget 2048 --warmup 3 --iterations 15 &
PID1=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/gqa_lightning_attention/benchmark_two_stage.py \
  --experiments 2 --configs 2 --contexts 16384,32768,65536,131072 \
  --batches 1,2,4,8,16,32 --phases prefill,decode \
  --variants dense_fa2,dense_fa3,sparse,two_stage --candidate-blocks 64 \
  --prefill-budget 2048 --warmup 3 --iterations 15 &
PID2=$!
wait "$PID1"
wait "$PID2"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/gqa_lightning_attention/recall_two_stage.py --experiments 2
.venv/bin/python benchmarks/gqa_lightning_attention/report_two_stage.py --experiment 2
