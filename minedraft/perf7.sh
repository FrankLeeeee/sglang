# Heavier EAGLE-3 draft (7 steps, 8 draft tokens), CUDA graphs on (draft decode+prefill graphs on cuda:1), overlap ON.
cd ~/md; source ~/sglang/.venv/bin/activate; export HF_HUB_OFFLINE=1
E3="--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8"
run() { tag=$1; conc=$2; shift 2; bash ~/srv.sh "$@" && timeout 1200 python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer Qwen/Qwen3-8B --prompts ~/prompts.jsonl --conc $conc --rounds 3 --out ~/md/mdres/k7_overlap_$tag.json 2>&1 | grep conc | sed "s/^/$tag /"; }
run coloc 16,32,64,128 0 30000 Qwen/Qwen3-8B k7c $E3
export SGLANG_MINEDRAFT_DRAFT_GPU=1 SGLANG_MINEDRAFT_DRAFT_DECODE_GRAPHS=1 SGLANG_MINEDRAFT_DRAFT_PREFILL_GRAPHS=1
SGLANG_MINEDRAFT_COHORTS=0 run disagg_serial 32,64,128 0,1 30000 Qwen/Qwen3-8B k7s $E3
run minedraft 32,64,128 0,1 30000 Qwen/Qwen3-8B k7m $E3
echo P7DONE
