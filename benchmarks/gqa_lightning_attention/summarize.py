"""Build paired latency tables from benchmark JSON (no third-party packages)."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results/comparison"
    )
    args = parser.parse_args()
    grouped = {}
    metadata = []
    for path in args.inputs:
        data = json.loads(path.read_text())
        metadata.append(data["metadata"])
        for r in data["rows"]:
            key = (r["config"], r["phase"], r["context"], r["batch"])
            variants = grouped.setdefault(key, {})
            if r["variant"] in variants:
                raise ValueError(f"Duplicate measurement {key} {r['variant']}")
            variants[r["variant"]] = r
    rows = []
    settings = metadata[0]["arguments"]
    shape_rows = [
        next(
            v["sparse"] for key, v in grouped.items() if key[0] == ci and "sparse" in v
        )
        for ci in sorted({key[0] for key in grouped})
    ]
    for run in metadata[1:]:
        if run["arguments"].get("indexer_q_heads", "main") != settings.get(
            "indexer_q_heads", "main"
        ):
            raise ValueError("Cannot combine runs with different indexer query heads")
        for name in (
            "top_k",
            "prefill_budget",
            "full_prefill",
            "prefill_graph",
            "page_size",
            "indexer_kv_heads",
        ):
            if run["arguments"][name] != settings[name]:
                raise ValueError(f"Cannot combine runs with different {name}")
    prefill_mode = "CUDA graph" if settings["prefill_graph"] else "eager"
    prefill_scope = (
        "full-prompt prefill"
        if settings["full_prefill"]
        else f"a final chunk of **{settings['prefill_budget']:,} total new tokens across the batch**, not full-prompt latency"
    )
    lines = [
        "# Dense vs Lightning sparse GQA latency",
        "",
        f"Median milliseconds per attention-layer forward. Sparse includes indexer projections, scores and exact top-{settings['top_k']} selection.",
        "",
        " | ".join(
            f"Configuration {r['config']}: main **{r['num_q_heads']}Q/{r['num_k_heads']}KV**, indexer **{r['indexer_q_heads']}Q/{r['indexer_kv_heads']}KV**"
            for r in shape_rows
        ),
        "",
        f"BF16, TP=1, {metadata[0]['gpu']}, shuffled {settings['page_size']}-token pages. Decode uses CUDA graph replay. Prefill uses {prefill_mode} execution and measures {prefill_scope}. Planning and scheduler overhead are excluded.",
        "",
    ]
    for (ci, phase, length, batch), variants in sorted(grouped.items()):
        dense, sparse = variants.get("dense"), variants.get("sparse")
        if (
            not dense
            or not sparse
            or dense["status"] != "ok"
            or sparse["status"] != "ok"
        ):
            raise ValueError(f"Missing or failed pair {(ci, phase, length, batch)}")
        assert dense["query_tokens"] == sparse["query_tokens"]
        assert dense["cuda_graph"] == sparse["cuda_graph"]
        for name in (
            "num_q_heads",
            "num_k_heads",
            "indexer_kv_heads",
            "indexer_q_heads",
        ):
            if dense[name] != sparse[name]:
                raise ValueError(f"Mismatched {name} for {(ci, phase, length, batch)}")
        expected_indexer_heads = (
            dense["num_k_heads"] if settings["indexer_kv_heads"] == "grouped" else 1
        )
        if sparse["indexer_kv_heads"] != expected_indexer_heads:
            raise ValueError("Row indexer heads disagree with run metadata")
        rows.append(
            {
                "config": ci,
                "num_q_heads": dense["num_q_heads"],
                "num_k_heads": dense["num_k_heads"],
                "indexer_kv_heads": sparse["indexer_kv_heads"],
                "indexer_q_heads": sparse["indexer_q_heads"],
                "phase": phase,
                "context": length,
                "batch": batch,
                "dense_ms": dense["median_ms"],
                "sparse_ms": sparse["median_ms"],
                "speedup_dense_over_sparse": dense["median_ms"] / sparse["median_ms"],
            }
        )
    for ci in sorted({r["config"] for r in rows}):
        for phase in ("prefill", "decode"):
            section = [r for r in rows if r["config"] == ci and r["phase"] == phase]
            if not section:
                continue
            lines += [
                f"## Configuration {ci}: {phase}",
                "",
                "| Context | Batch | Dense (ms) | Sparse (ms) | Dense / sparse |",
                "|---:|---:|---:|---:|---:|",
            ]
            for r in section:
                lines.append(
                    f"| {r['context'] // 1024}k | {r['batch']} | {r['dense_ms']:.3f} | {r['sparse_ms']:.3f} | {r['speedup_dense_over_sparse']:.2f}× |"
                )
            lines.append("")
    lines += [
        "Ratios above 1 mean sparse is faster. These are synthetic layer timings, not whole-model serving latency or an accuracy evaluation of sparsification.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".md").write_text("\n".join(lines))
    args.output.with_suffix(".json").write_text(
        json.dumps({"runs": metadata, "comparisons": rows}, indent=2)
    )
    with args.output.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} paired comparisons to {args.output}")


if __name__ == "__main__":
    main()
