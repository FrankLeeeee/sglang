cd ~/md; source ~/sglang/.venv/bin/activate; export HF_HUB_OFFLINE=1 SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
E3="--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --disable-overlap-schedule"
[ -f ~/md/mdres/det2_ar.json ] || { bash ~/srv.sh 0 30000 Qwen/Qwen3-8B d2ar --enable-deterministic-inference && python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer Qwen/Qwen3-8B --prompts ~/prompts.jsonl --conc 16 --rounds 2 --out ~/md/mdres/det2_ar.json 2>&1 | grep conc; }
SGLANG_MINEDRAFT_DRAFT_GPU=1 SGLANG_MINEDRAFT_DRAFT_DECODE_GRAPHS=1 bash ~/srv.sh 0,1 30000 Qwen/Qwen3-8B d2md$1 --enable-deterministic-inference $E3 && python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer Qwen/Qwen3-8B --prompts ~/prompts.jsonl --conc 16 --rounds 2 --out ~/md/mdres/det2_md$1.json 2>&1 | grep conc
python3 -c "
import json
a=json.load(open(\"/data/frankleeeee/home/md/mdres/det2_ar.json\"))[\"16\"][\"outputs\"]; b=json.load(open(\"/data/frankleeeee/home/md/mdres/det2_md$1.json\"))[\"16\"][\"outputs\"]
ks=[k for k in a if k in b]; print(\"md$1 == ar:\", sum(a[k]==b[k] for k in ks), \"/\", len(ks))"
grep cohorts ~/md/mdlogs/d2md$1.log | tail -1
