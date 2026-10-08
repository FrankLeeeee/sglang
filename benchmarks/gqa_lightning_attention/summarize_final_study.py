"""Join the final-study runs into one comparison CSV and a markdown summary.

Speedup is best dense median (FlashInfer FA2 or SGLang FA3, per point) divided
by the sparse median; above 1 means sparse is faster.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

METHODS = ["token", "block", "two_stage"]
MAINS = ["48q4kv", "64q8kv", "80q8kv", "64q4kv"]
CONTEXTS = [16384, 65536, 131072, 524288, 1048576]
BATCHES = [1, 8, 32]
VARIANTS = [("identical", 64), ("identical", 128), ("reduced", 64), ("reduced", 128)]


def load(directory):
    rows = []
    for path in sorted(directory.glob("*_gpu*.json")):
        rows += json.loads(path.read_text())["rows"]
    return rows


def paired(rows):
    dense = defaultdict(dict)
    for r in rows:
        if r["method"] == "dense" and r["status"] == "ok":
            key = (r["main"], r["phase"], r["context"], r["batch"])
            dense[key][r["dense_backend"]] = r["median_ms"]
    out = []
    for r in rows:
        if r["method"] == "dense":
            continue
        key = (r["main"], r["phase"], r["context"], r["batch"])
        row = {k: r[k] for k in ("main", "phase", "context", "batch", "method")}
        row.update(indexer=r["indexer"], dim=r["dim"], status=r["status"])
        if key in dense:
            backend = min(dense[key], key=dense[key].get)
            row.update(dense_backend=backend, dense_ms=dense[key][backend])
            row["dense_fa2_ms"] = dense[key].get("flashinfer_fa2")
            row["dense_sgl_fa3_ms"] = dense[key].get("sgl_fa3")
        if r["status"] == "ok":
            row["sparse_ms"] = r["median_ms"]
            row["speedup"] = row["dense_ms"] / r["median_ms"]
            row["stage_ms"] = r["stage_ms"]
        out.append(row)
    return out


def win_table(rows):
    lines = ["| Method | Indexer | Dim | Prefill wins | Decode wins |", "|---|---|---:|---:|---:|"]
    for method in METHODS:
        for indexer, dim in VARIANTS:
            cells = []
            for phase in ("prefill", "decode"):
                sel = [
                    r for r in rows if r["method"] == method and r["indexer"] == indexer
                    and r["dim"] == dim and r["phase"] == phase and r["status"] == "ok"
                ]
                cells.append(f"{sum(r['speedup'] > 1 for r in sel)}/{len(sel)}")
            lines.append(f"| {method} | {indexer} | {dim} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def speedup_grid(rows, main, phase, batch):
    header = "| Method / indexer | " + " | ".join(f"{c // 1024}k" for c in CONTEXTS) + " |"
    lines = [header, "|---|" + "---:|" * len(CONTEXTS)]
    for method in METHODS:
        for indexer, dim in VARIANTS:
            cells = []
            for ctx in CONTEXTS:
                hit = [
                    r for r in rows if (r["main"], r["phase"], r["context"], r["batch"], r["method"], r["indexer"], r["dim"])
                    == (main, phase, ctx, batch, method, indexer, dim)
                ]
                ok = hit and hit[0]["status"] == "ok"
                cells.append(f"{hit[0]['speedup']:.2f}" if ok else "OOM")
            lines.append(f"| {method} {indexer}/{dim} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_table(rows):
    """Best sparse variant per main config / phase / context at each batch."""
    lines = ["| Main | Phase | Batch | " + " | ".join(f"{c // 1024}k" for c in CONTEXTS) + " |", "|---|---|---:|" + "---:|" * len(CONTEXTS)]
    for main in MAINS:
        for phase in ("prefill", "decode"):
            for batch in BATCHES:
                cells = []
                for ctx in CONTEXTS:
                    sel = [
                        r for r in rows if (r["main"], r["phase"], r["context"], r["batch"]) == (main, phase, ctx, batch)
                        and r["status"] == "ok"
                    ]
                    best = max(sel, key=lambda r: r["speedup"])
                    tag = {"token": "T", "block": "B", "two_stage": "2"}[best["method"]]
                    cells.append(f"{best['speedup']:.2f}{tag}")
                lines.append(f"| {main} | {phase} | {batch} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def breakdown(rows, points):
    lines = []
    for main, phase, ctx, batch in points:
        lines.append(f"\n**{main}, {phase}, {ctx // 1024}k, batch {batch}** (stage GPU ms)\n")
        names = ["main_projection", "index_projection", "block_pool", "score", "coarse_score_select", "fine_score", "select", "expand", "sparse_attention"]
        lines += ["| Method / indexer | Wall | " + " | ".join(names) + " |", "|---|---:|" + "---:|" * len(names)]
        for method in METHODS:
            for indexer, dim in [("identical", 64), ("reduced", 64)]:
                hit = [r for r in rows if (r["main"], r["phase"], r["context"], r["batch"], r["method"], r["indexer"], r["dim"]) == (main, phase, ctx, batch, method, indexer, dim) and r["status"] == "ok"]
                if not hit:
                    continue
                s = hit[0]["stage_ms"]
                lines.append(f"| {method} {indexer}/{dim} | {hit[0]['sparse_ms']:.3f} | " + " | ".join(f"{s[n]:.3f}" if n in s else "-" for n in names) + " |")
    return "\n".join(lines)


def export_json(raw, rows, path):
    """One self-contained file: every measurement, its pairing and its speedup."""
    speedups = {
        (r["main"], r["phase"], r["context"], r["batch"], r["method"], r["indexer"], r["dim"]): r
        for r in rows
    }
    measurements = []
    for r in raw:
        entry = dict(r)
        if r["method"] != "dense":
            key = (r["main"], r["phase"], r["context"], r["batch"], r["method"], r["indexer"], r["dim"])
            paired_row = speedups[key]
            if "speedup" in paired_row:
                entry["dense_best_ms"] = paired_row["dense_ms"]
                entry["dense_best_backend"] = paired_row["dense_backend"]
                entry["speedup"] = paired_row["speedup"]
        measurements.append(entry)
    path.write_text(
        json.dumps(
            {
                "description": "Dense vs token / block / two-stage sparse GQA attention, one layer, H200, BF16, TP=1",
                "speedup": "best dense median (FlashInfer FA2 or SGLang FA3 per point) / sparse median",
                "prefill": "final scheduled chunk of 2048 total new tokens across the batch, eager",
                "decode": "one new token per request, CUDA graph replay",
                "timing": "3 warmups, 15 CUDA-event samples, L2 flushed; stage_ms are GPU windows from one extra eager pass",
                "main_configs": MAINS,
                "indexer_variants": [{"type": t, "head_dim": d} for t, d in VARIANTS],
                "contexts": CONTEXTS,
                "batches": BATCHES,
                "measurements": measurements,
            },
            indent=1,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path(__file__).parent / "results/final_study")
    args = p.parse_args()
    raw = load(args.results)
    rows = paired(raw)
    export_json(raw, rows, args.results / "final_study_results.json")
    fields = ["main", "phase", "context", "batch", "method", "indexer", "dim", "status", "dense_backend", "dense_fa2_ms", "dense_sgl_fa3_ms", "dense_ms", "sparse_ms", "speedup"]
    with (args.results / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    parts = ["# Final sparse-attention study\n", f"{len(rows)} sparse measurements.\n", "## Wins over the best dense baseline\n", win_table(rows), "\n## Best sparse speedup per point (T token, B block, 2 two-stage)\n", best_table(rows)]
    for main_cfg in MAINS:
        for phase in ("prefill", "decode"):
            parts += [f"\n## {main_cfg} {phase}, batch 8\n", speedup_grid(rows, main_cfg, phase, 8)]
    parts += ["\n## Stage breakdowns"] + [breakdown(rows, [("48q4kv", "prefill", 131072, 8), ("48q4kv", "decode", 131072, 8), ("64q8kv", "prefill", 524288, 8)])]
    (args.results / "summary.md").write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
