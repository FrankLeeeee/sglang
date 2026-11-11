# EAGLE-3, Qwen3-8B, CUDA graphs on (target cuda:0; draft decode graphs on cuda:1), overlap scheduler OFF.
cd ~/md; source ~/sglang/.venv/bin/activate; export HF_HUB_OFFLINE=1
E3="--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --disable-overlap-schedule"
run() { tag=$1; conc=$2; shift 2; bash ~/srv.sh "$@" && timeout 1200 python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer Qwen/Qwen3-8B --prompts ~/prompts.jsonl --conc $conc --rounds 3 --out ~/md/mdres/$TAG_PREFIX$tag.json 2>&1 | grep conc | sed "s/^/$tag /"; }
TAG_PREFIX=g_nooverlap_
run coloc 16,32,64,128 0 30000 Qwen/Qwen3-8B c3 $E3
SGLANG_MINEDRAFT_DRAFT_GPU=1 SGLANG_MINEDRAFT_DRAFT_DECODE_GRAPHS=1 SGLANG_MINEDRAFT_COHORTS=0 run disagg_serial 32,64,128 0,1 30000 Qwen/Qwen3-8B s3 $E3
SGLANG_MINEDRAFT_DRAFT_GPU=1 SGLANG_MINEDRAFT_DRAFT_DECODE_GRAPHS=1 run minedraft 32,64,128 0,1 30000 Qwen/Qwen3-8B m3 $E3
grep -h "cuda graph" ~/md/mdlogs/s3.log | tail -1 | cut -c1-200
grep -h "Capture draft decode CUDA graph end" ~/md/mdlogs/m3.log | cut -c1-120
grep -h "cohorts" ~/md/mdlogs/m3.log | tail -1
echo P3DONE
