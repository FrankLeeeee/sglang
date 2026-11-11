cd ~/md; source ~/sglang/.venv/bin/activate; export HF_HUB_OFFLINE=1 SGLANG_MINEDRAFT_DRAFT_GPU=1 SGLANG_MINEDRAFT_DRAFT_DECODE_GRAPHS=1 SGLANG_MINEDRAFT_TIMING=1
E3="--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
run() { tag=$1; shift; bash ~/srv.sh "$@" && timeout 1200 python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer Qwen/Qwen3-8B --prompts ~/prompts.jsonl --conc 32,64,128 --rounds 3 --out ~/md/mdres/$tag.json 2>&1 | grep conc | sed "s/^/$tag /"; }
run g_nooverlap_minedraft_v2 0,1 30000 Qwen/Qwen3-8B m5 $E3 --disable-overlap-schedule
grep "host time" ~/md/mdlogs/m5.log | tail -1
SGLANG_MINEDRAFT_COHORTS=0 run g_nooverlap_disagg_serial_v2 0,1 30000 Qwen/Qwen3-8B s5 $E3 --disable-overlap-schedule
SGLANG_MINEDRAFT_COHORTS=0 run g_overlap_disagg_serial_v2 0,1 30000 Qwen/Qwen3-8B o5 $E3
echo P5DONE
