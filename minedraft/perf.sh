#!/usr/bin/env bash
# EAGLE-3 MineDraft perf, all eager without overlap scheduling so the three modes compare like for like.
cd ~/md; source ~/sglang/.venv/bin/activate; export HF_HUB_OFFLINE=1
E3="--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --disable-overlap-schedule --disable-cuda-graph"
CONC=${CONC:-16,32,64,128}
run() { tag=$1; shift; bash ~/srv.sh "$@" && timeout 1500 python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer Qwen/Qwen3-8B --prompts ~/prompts.jsonl --conc $CONC --rounds 3 --out ~/md/mdres/perf_$tag.json 2>&1 | grep conc | sed "s/^/$tag /"; }
run coloc_eager 0 30000 Qwen/Qwen3-8B perf_coloc $E3
SGLANG_MINEDRAFT_DRAFT_GPU=1 SGLANG_MINEDRAFT_COHORTS=0 run disagg_serial 0,1 30000 Qwen/Qwen3-8B perf_serial $E3
SGLANG_MINEDRAFT_DRAFT_GPU=1 run minedraft 0,1 30000 Qwen/Qwen3-8B perf_md $E3
grep "\[minedraft\] cohorts" ~/md/mdlogs/perf_md.log | tail -2
echo PERFDONE
