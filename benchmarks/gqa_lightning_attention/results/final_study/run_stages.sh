#!/usr/bin/env bash
# Usage: run_stages.sh <gpu> <comma-separated main indices>; stages run in order.
set -u
cd "$(dirname "$0")/../.."
gpu=$1; mains=$2
for method in dense token block two_stage; do
  CUDA_VISIBLE_DEVICES=$gpu ../../.venv/bin/python run_final_study.py \
    --methods $method --mains $mains \
    --output results/final_study/${method}_gpu${gpu}.json \
    > results/final_study/${method}_gpu${gpu}.log 2>&1
  echo "$method done (exit $?)" >> results/final_study/progress_gpu${gpu}.txt
done
