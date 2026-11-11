# EAGLE-3, Qwen3-8B, CUDA graphs on, overlap scheduler ON.
cd ~/md; source ~/sglang/.venv/bin/activate; export HF_HUB_OFFLINE=1
E3="--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
run() { tag=$1; conc=$2; shift 2; bash ~/srv.sh "$@" && timeout 1200 python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer Qwen/Qwen3-8B --prompts ~/prompts.jsonl --conc $conc --rounds 3 --out ~/md/mdres/g_overlap_$tag.json 2>&1 | grep conc | sed "s/^/$tag /"; }
run coloc 16,32,64,128 0 30000 Qwen/Qwen3-8B c4 $E3
SGLANG_MINEDRAFT_DRAFT_GPU=1 SGLANG_MINEDRAFT_DRAFT_DECODE_GRAPHS=1 run disagg_serial 32,64,128 0,1 30000 Qwen/Qwen3-8B s4 $E3
grep -c Traceback ~/md/mdlogs/s4.log
echo P4DONE
