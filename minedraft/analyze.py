"""Colocated SGLang measurements -> MineDraft (draft GPU1, target GPU0) step-model predictions.

Per concurrency c (colocated, one GPU), measured:
  wall(c)  = c * tau(c) / tok_s(c)          wall-clock time of one spec step
  T_v(c)   = target verify forward GPU time  (CUDA events around TpModelWorker.forward_batch_generation)
  T_d(c)   = wall(c) - T_v(c)                everything that is not the verify: draft, draft-extend,
                                             accept/glue, host gaps (attributed to the draft side)
  X(c)     = GPU0->GPU1 copy of hidden states + tokens for c*(k+1) rows (measured, upper bound)
MineDraft at total concurrency C runs two sub-batches of m = C/2 and overlaps draft(one) with
verify(other) on separate GPUs, so one sub-batch step costs
  step_md(m) = max(T_v(m), T_d(m) + X(m))
and a round (both sub-batches advance once) is 2 * step_md(m), giving C * tau(C) tokens.
These are model predictions, not measurements of a running MineDraft.
"""
import glob, json, os, sys

R = sys.argv[1] if len(sys.argv) > 1 else "."
xfer = json.load(open(os.path.join(R, "xfer.json")))
RES_T = os.environ.get("RES_T", "results_t")  # spec runs with phase timing
TIM = os.environ.get("TIM", "timing")
ARDIR = os.environ.get("ARDIR", "results")  # AR runs
BASEDIR = os.environ.get("BASEDIR", "results")  # spec runs used for exact match vs AR
CS = [int(c) for c in os.environ.get("CS", "1,4,16,64").split(",")]
OUT = os.environ.get("OUT", "summary.json")

CFG = {  # (model, algo) -> (verify rows per request, hidden width carried to the draft)
    ("qwen", "eagle3_314"): (4, "3x4096 (EAGLE-3)"),
    ("llama", "eagle3_314"): (4, "3x4096 (EAGLE-3)"),
    ("qwen", "dspark"): (8, "5x4096"),  # block 7 + bonus, 5 target layers
    ("qwen", "dflash"): (16, "5x4096"),  # block 16, 5 target layers
    ("llama", "dflash"): (10, "5x4096"),  # block 10, 5 target layers
}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def load_timing(tag):
    agg = {}
    for p in glob.glob(os.path.join(R, TIM, f"{tag}.*.json")):
        for k, v in json.load(open(p)).items():
            kind, bs = k.split("|")
            a = agg.setdefault((kind, int(bs)), [0, 0.0])
            a[0] += v["n"]; a[1] += v["n"] * v["mean_ms"]
    return {k: v[1] / v[0] for k, v in agg.items() if v[0] >= 10}


def xfer_ms(width, rows):
    pts = sorted((r["rows"], r["median_ms"]) for r in xfer if r["width"] == width)
    for r, t in pts:
        if r >= rows:
            return t
    r, t = pts[-1]
    return t * rows / r


def phase(sumr, tim, c):
    s = sumr.get(str(c), {}).get("summary")
    tv = tim.get(("target", c))
    if s is None or tv is None:
        return None
    wall = c * s["accept_len"] / s["tok_s"] * 1e3
    return {"wall": wall, "tv": tv, "td": max(0.0, wall - tv), "tau": s["accept_len"], "tok_s": s["tok_s"]}


rows_out = []
for (model, algo), (k1, width) in CFG.items():
    tag = f"{model}_{algo}"
    tres = load(os.path.join(R, RES_T, f"{tag}.json"))
    base = load(os.path.join(R, BASEDIR, f"{tag}.json"))
    ar = load(os.path.join(R, ARDIR, f"{model}_ar.json"))
    tim = load_timing(tag)
    if not tres:
        continue
    for C in CS:
        pc = phase(tres, tim, C)
        if pc is None:
            continue
        ar_tps = ar.get(str(C), {}).get("summary", {}).get("tok_s")
        exact = None
        if str(C) in ar and str(C) in base:
            o1, o2 = ar[str(C)]["outputs"], base[str(C)]["outputs"]
            keys = [k for k in o1 if k in o2]
            exact = round(sum(o1[k] == o2[k] for k in keys) / max(1, len(keys)), 2)
        row = {"model": model, "algo": algo, "C": C, "ar_tok_s": round(ar_tps, 1) if ar_tps else None,
               "coloc_tok_s": round(pc["tok_s"], 1), "accept_len": round(pc["tau"], 2),
               "coloc_x_ar": round(pc["tok_s"] / ar_tps, 2) if ar_tps else None, "exact_vs_ar": exact,
               "T_v_ms": round(pc["tv"], 2), "T_d_ms": round(pc["td"], 2),
               "Td_over_Tv": round(pc["td"] / pc["tv"], 2)}
        m = C // 2
        if m >= 1:
            pm = phase(tres, tim, m)
            if pm is not None:
                x = xfer_ms(width, m * k1)
                step = max(pm["tv"], pm["td"] + x)
                md = C * pc["tau"] / (2 * step) * 1e3
                row.update({"m": m, "T_v_m": round(pm["tv"], 2), "T_d_m": round(pm["td"], 2),
                            "X_ms": round(x, 3), "md_step_ms": round(step, 2),
                            "md_tok_s_pred": round(md, 1),
                            "md_vs_coloc": round(md / pc["tok_s"], 2),
                            "md_per_gpu_vs_coloc": round(md / 2 / pc["tok_s"], 2),
                            "md_bound": "verify" if pm["tv"] >= pm["td"] + x else "draft",
                            "dp2_tok_s": round(2 * pm["tok_s"], 1),
                            "md_vs_dp2": round(md / (2 * pm["tok_s"]), 2)})
        rows_out.append(row)

json.dump(rows_out, open(os.path.join(R, OUT), "w"), indent=1)
cols = ["model", "algo", "C", "ar_tok_s", "coloc_tok_s", "coloc_x_ar", "accept_len", "exact_vs_ar",
        "T_v_ms", "T_d_ms", "Td_over_Tv", "m", "X_ms", "md_step_ms", "md_tok_s_pred", "md_vs_coloc",
        "md_per_gpu_vs_coloc", "dp2_tok_s", "md_vs_dp2", "md_bound"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in rows_out:
    print("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
