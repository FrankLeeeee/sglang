cd ~/md; source ~/sglang/.venv/bin/activate; export HF_HUB_OFFLINE=1 SGLANG_MINEDRAFT_DRAFT_GPU=1 SGLANG_MINEDRAFT_DRAFT_DECODE_GRAPHS=1
E3="--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
bash ~/srv.sh 0,1 30000 Qwen/Qwen3-8B m6 $E3 && timeout 1200 python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer Qwen/Qwen3-8B --prompts ~/prompts.jsonl --conc 32,64,128 --rounds 3 --out ~/md/mdres/g_overlap_minedraft.json 2>&1 | grep conc | sed "s/^/g_overlap_minedraft /"
grep cohorts ~/md/mdlogs/m6.log | tail -1
echo P6DONE
