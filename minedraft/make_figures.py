"""Figures from summary.json: colocated SD vs MineDraft estimate, and the T_d / T_v ratio."""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "."
rows = json.load(open(os.path.join(R, "summary.json")))
os.makedirs(os.path.join(R, "figures"), exist_ok=True)
spec = [r for r in rows if r.get("md_tok_s_pred")]
allr = [r for r in rows if r.get("Td_over_Tv") is not None]
groups = sorted({(r["model"], r["algo"]) for r in spec})
Cs = sorted({r["C"] for r in spec})

fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3.0), sharey=False, squeeze=False)
for ax, (m, a) in zip(axes[0], groups):
    rs = {r["C"]: r for r in spec if r["model"] == m and r["algo"] == a}
    xs = [c for c in Cs if c in rs]
    ax.plot(xs, [rs[c]["md_vs_coloc"] for c in xs], "o-", color="#1f77b4", label="MineDraft (2 GPUs) / colocated")
    ax.plot(xs, [rs[c]["md_vs_dp2"] for c in xs], "s--", color="#d62728", label="MineDraft / 2 colocated replicas")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels(xs)
    ax.set_title(f"{m} + {a}", fontsize=9); ax.set_xlabel("concurrency C")
axes[0][0].set_ylabel("throughput ratio"); axes[0][0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "minedraft_vs_colocated.png"), dpi=200)

fig, ax = plt.subplots(figsize=(4.5, 3.0))
for (m, a) in groups:
    rs = sorted((r for r in allr if r["model"] == m and r["algo"] == a), key=lambda r: r["C"])
    ax.plot([r["C"] for r in rs], [r["Td_over_Tv"] for r in rs], "o-", label=f"{m}+{a}")
ax.axhline(1.0, color="gray", lw=0.8)
ax.set_xscale("log", base=2); ax.set_xlabel("concurrency C"); ax.set_ylabel("T_draft / T_verify (GPU time)")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "draft_verify_ratio.png"), dpi=200)
print("ok")
