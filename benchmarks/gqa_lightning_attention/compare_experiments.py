"""Compare the three named experiments without rewriting historical raw data."""

import csv
import json
import statistics
from pathlib import Path


def main():
    root = Path(__file__).parent
    registry = json.loads((root / "experiments.json").read_text())
    data = {}
    for eid, spec in registry.items():
        values = {}
        for ci in (1, 2):
            run = json.loads((root / spec["results"] / f"config{ci}.json").read_text())
            assert len(run["rows"]) == 96
            for r in run["rows"]:
                assert r["status"] == "ok"
                for key, field in [
                    ("main_q_heads", "num_q_heads"),
                    ("main_kv_heads", "num_k_heads"),
                    ("indexer_q_heads", "indexer_q_heads"),
                    ("indexer_kv_heads", "indexer_kv_heads"),
                ]:
                    assert r[field] == spec[key][ci - 1], (eid, ci, field)
                assert r["median_ms"] == statistics.median(r["samples_ms"])
                key = (ci, r["phase"], r["context"], r["batch"], r["variant"])
                assert key not in values
                values[key] = r
        data[eid] = values
    assert data["1"].keys() == data["2"].keys() == data["3"].keys()
    comparisons = []
    for (ci, phase, length, batch, variant), r in sorted(data["3"].items()):
        if variant != "dense":
            continue
        item = {"config": ci, "phase": phase, "context": length, "batch": batch}
        for eid in registry:
            dense = data[eid][(ci, phase, length, batch, "dense")]
            sparse = data[eid][(ci, phase, length, batch, "sparse")]
            for field in (
                "query_tokens",
                "cuda_graph",
                "head_dim",
                "hidden_dim",
                "page_size",
            ):
                assert dense[field] == r[field] == sparse[field]
            assert sparse["top_k"] == 2048
            item[f"e{eid}_dense_ms"] = dense["median_ms"]
            item[f"e{eid}_sparse_ms"] = sparse["median_ms"]
            item[f"e{eid}_dense_over_sparse"] = dense["median_ms"] / sparse["median_ms"]
        item["e2_over_e3_dense"] = item["e2_dense_ms"] / item["e3_dense_ms"]
        item["e2_over_e3_sparse"] = item["e2_sparse_ms"] / item["e3_sparse_ms"]
        comparisons.append(item)
    out = root / registry["3"]["results"]
    (out / "experiments_comparison.json").write_text(json.dumps(comparisons, indent=2))
    with (out / "experiments_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    lines = [
        "# Experiment 3: main attention with one KV head",
        "",
        "Only the main attention KV head count changes from experiment 2. Indexer query heads remain 4/8 and indexer KV heads remain 1. Main Q heads remain 48/64.",
        "",
        "| Experiment | Main Q/KV (config 1; config 2) | Indexer Q/KV (config 1; config 2) |",
        "|---|---|---|",
        "| 1 | 48/4; 64/8 | 48/4; 64/8 |",
        "| 2 | 48/4; 64/8 | 4/1; 8/1 |",
        "| 3 | 48/1; 64/1 | 4/1; 8/1 |",
        "",
        "H200, BF16, TP=1, hidden dimension 8192, main head dimension 128, indexer head dimension 64, top-2048, page size 128. Prefill is a final chunk of 2,048 total new tokens across the batch. Decode is one new token/request with CUDA graph replay. Latencies include projections, normalization, cache writes, scores, selection and attention. Each median uses 15 CUDA-event samples.",
        "",
        "In experiment 3 there is one attention KV group: the 4/8 weighted indexer heads are summed into one score per token, and one top-2048 set serves all 48/64 main Q heads. Experiment 2 instead has 4/8 separate scores and selections. This reduces both cache size and selection count; it also changes model behavior.",
        "",
        "Dense MQA uses the same FlashInfer paged backend, with a 1 GiB workspace required by its tensor-core decode planner (128 MiB for the previous GQA experiments). Workspace allocation and planning are outside timing.",
        "",
        "Experiment 1 and 2 use their existing measured runs. Experiment 3 is a fresh complete 192-measurement run, not an extrapolation. These synthetic random-weight timings do not establish model quality. Different shapes also change random weight/cache realizations, and cross-run comparisons include timing noise.",
        "",
        "[Experiment 3 dense vs sparse tables](comparison.md) · [All three experiments CSV](experiments_comparison.csv)",
        "",
    ]
    for ci in (1, 2):
        for phase in ("prefill", "decode"):
            section = [
                r for r in comparisons if r["config"] == ci and r["phase"] == phase
            ]
            wins = sum(r["e3_dense_over_sparse"] > 1 for r in section)
            lines += [
                f"## Configuration {ci}: {phase}",
                "",
                f"Experiment 3 sparse is faster than its dense baseline at {wins}/24 measured points.",
                "",
                "| Context | Batch | E1 dense | E1 sparse | E2 dense | E2 sparse | E3 dense | E3 sparse | E3 dense/sparse |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in section:
                values = " | ".join(
                    f"{r[f'e{e}_{v}_ms']:.3f}"
                    for e in (1, 2, 3)
                    for v in ("dense", "sparse")
                )
                lines.append(
                    f"| {r['context']//1024}k | {r['batch']} | {values} | {r['e3_dense_over_sparse']:.2f}× |"
                )
            lines += [
                "",
                "All latencies are milliseconds; ratios above 1 mean sparse is faster.",
                "",
            ]
            for length in (16, 32, 64, 128):
                lines += [
                    f"![Experiment 3 config {ci} {phase} {length}k](charts/config{ci}_{phase}_{length}k.png)",
                    "",
                ]
    (out / "experiments_comparison.md").write_text("\n".join(lines))
    print(
        f"Verified 576 measurements across 3 experiments; wrote {len(comparisons)} six-way comparisons."
    )


if __name__ == "__main__":
    main()
