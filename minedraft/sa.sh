# Standalone draft (Llama-3.1-8B target + Llama-3.2-1B draft, (3,1,4)), CUDA graphs on, overlap ON.
cd ~/md; source ~/sglang/.venv/bin/activate; export HF_HUB_OFFLINE=1
SA="--speculative-algorithm STANDALONE --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
T=meta-llama/Llama-3.1-8B-Instruct
run() { tag=$1; conc=$2; shift 2; bash ~/srv.sh "$@" && timeout 1200 python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer $T --prompts ~/prompts.jsonl --conc $conc --rounds 3 --out ~/md/mdres/sa_overlap_$tag.json 2>&1 | grep conc | sed "s/^/$tag /"; }
run coloc 16,32,64,128 0 30000 $T sac $SA
export SGLANG_MINEDRAFT_DRAFT_GPU=1 SGLANG_MINEDRAFT_DRAFT_DECODE_GRAPHS=1 SGLANG_MINEDRAFT_DRAFT_PREFILL_GRAPHS=1
SGLANG_MINEDRAFT_COHORTS=0 run disagg_serial 32,64,128 0,1 30000 $T sas $SA
run minedraft 32,64,128 0,1 30000 $T sam $SA
# deterministic correctness at C=16
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
unset SGLANG_MINEDRAFT_DRAFT_GPU
bash ~/srv.sh 0 30000 $T sadar --enable-deterministic-inference && python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer $T --prompts ~/prompts.jsonl --conc 16 --rounds 2 --out ~/md/mdres/sa_det_ar.json 2>&1 | grep conc | sed "s/^/det_ar /"
SGLANG_MINEDRAFT_DRAFT_GPU=1 bash ~/srv.sh 0,1 30000 $T sadmd --enable-deterministic-inference $SA && SGLANG_MINEDRAFT_DRAFT_GPU=1 python ~/bench_closed_loop.py --url http://127.0.0.1:30000 --tokenizer $T --prompts ~/prompts.jsonl --conc 16 --rounds 2 --out ~/md/mdres/sa_det_md.json 2>&1 | grep conc | sed "s/^/det_md /"
python3 -c "
import json
a=json.load(open(\"/data/frankleeeee/home/md/mdres/sa_det_ar.json\"))[\"16\"][\"outputs\"]; b=json.load(open(\"/data/frankleeeee/home/md/mdres/sa_det_md.json\"))[\"16\"][\"outputs\"]
ks=[k for k in a if k in b]; print(\"standalone md == ar:\", sum(a[k]==b[k] for k in ks), \"/\", len(ks))"
echo SADONE
