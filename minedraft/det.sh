#!/usr/bin/env bash
# Deterministic-mode correctness: AR, colocated EAGLE-3, disaggregated serial, disaggregated MineDraft.
cd ~/md; source ~/sglang/.venv/bin/activate; export HF_HUB_OFFLINE=1
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
E3="--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --disable-overlap-schedule --disable-cuda-graph"
run() { tag=$1; shift; bash ~/srv.sh "$@" && timeout 900 python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer Qwen/Qwen3-8B --prompts ~/prompts.jsonl --conc 16 --rounds 2 --out ~/md/mdres/det_$tag.json 2>&1 | grep conc; }
run ar 0 30000 Qwen/Qwen3-8B det_ar --enable-deterministic-inference --disable-cuda-graph
run coloc 0 30000 Qwen/Qwen3-8B det_coloc --enable-deterministic-inference $E3
SGLANG_MINEDRAFT_DRAFT_GPU=1 SGLANG_MINEDRAFT_COHORTS=0 run serial 0,1 30000 Qwen/Qwen3-8B det_serial --enable-deterministic-inference $E3
SGLANG_MINEDRAFT_DRAFT_GPU=1 run md 0,1 30000 Qwen/Qwen3-8B det_md --enable-deterministic-inference $E3
python3 - <<'PY'
import json
d={t:json.load(open(f"/data/frankleeeee/home/md/mdres/det_{t}.json"))["16"]["outputs"] for t in ["ar","coloc","serial","md"]}
for t in ["coloc","serial","md"]:
    ks=[k for k in d["ar"] if k in d[t]]
    print(t,"== ar:",sum(d[t][k]==d["ar"][k] for k in ks),"/",len(ks))
ks=[k for k in d["serial"] if k in d["md"]]
print("md == serial:",sum(d["md"][k]==d["serial"][k] for k in ks),"/",len(ks))
PY
echo DETDONE
