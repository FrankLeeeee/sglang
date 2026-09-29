"""Reproducible dense versus Lightning token-sparse layer latency sweep."""

import argparse
import csv
import gc
import json
import platform
import statistics
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import flashinfer
import torch
import triton
from attention import CONFIGS, Attention, PagedState


def ints(value):
    return [int(x) for x in value.split(",")]


@torch.no_grad()
def measure(fn, graph, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    captured = None
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            output = fn()
        run = captured.replay
    else:
        run = fn
    # H200 L2 is 50 MiB. Flush outside the measured interval to approximate
    # intervening model layers, especially for small decode caches.
    flush = torch.empty(64 * 1024**2, device="cuda", dtype=torch.uint8)
    samples = []
    for _ in range(iterations):
        flush.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    if captured is not None:
        captured.reset()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "p90_ms": sorted(samples)[min(len(samples) - 1, int(0.9 * len(samples)))],
        "samples_ms": samples,
    }


def save(rows, metadata, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2)
    )
    fields = sorted({k for r in rows for k in r if k != "samples_ms"})
    with out.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", type=ints, default=[1, 2])
    p.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3],
        help="1: original grouped indexer; 2: reduced-query shared-key indexer; 3: experiment 2 with main KV heads=1",
    )
    p.add_argument(
        "--main-kv-heads",
        type=int,
        default=None,
        help="Override main attention KV heads; indexer-q-heads=kv retains the base configuration's 4/8 indexer query heads",
    )
    p.add_argument("--contexts", type=ints, default=[16384, 32768, 65536, 131072])
    p.add_argument("--batches", type=ints, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--phases", default="prefill,decode")
    p.add_argument("--sparse", action="store_true", help="Run only sparse attention")
    p.add_argument("--dense-only", action="store_true")
    p.add_argument(
        "--indexer-q-heads",
        choices=["main", "kv"],
        default="main",
        help="main: match main Q heads; kv: match base configuration KV heads (4/8), independent of --main-kv-heads",
    )
    p.add_argument(
        "--indexer-kv-heads",
        choices=["1", "grouped"],
        default="grouped",
        help="Default: one indexer KV head per main attention KV head; 1 shares one indexer key across groups",
    )
    p.add_argument("--top-k", type=int, default=2048)
    p.add_argument(
        "--prefill-budget",
        type=int,
        default=2048,
        help="Total new tokens across the batch",
    )
    p.add_argument(
        "--full-prefill",
        action="store_true",
        help="All B*L prompt tokens in one forward (potentially expensive)",
    )
    p.add_argument(
        "--prefill-graph",
        action="store_true",
        help="Capture prefill too; default is eager as in typical SGLang serving",
    )
    p.add_argument("--page-size", type=int, default=128)
    p.add_argument("--score-budget-mib", type=int, default=256)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results/latency"
    )
    args = p.parse_args()
    if args.experiment is not None:
        if (
            args.main_kv_heads is not None
            or args.indexer_q_heads != "main"
            or args.indexer_kv_heads != "grouped"
        ):
            p.error("Use either --experiment or individual head overrides")
        if args.experiment in (2, 3):
            args.indexer_q_heads = "kv"
            args.indexer_kv_heads = "1"
        if args.experiment == 3:
            args.main_kv_heads = 1
    if args.sparse and args.dense_only:
        p.error("--sparse and --dense-only are mutually exclusive")
    if args.iterations < 1 or args.warmup < 1:
        p.error("iterations and warmup must be positive")
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "scope": "One attention layer, BF16, TP=1; QKV projections, per-head RMSNorm, cache writes, attention; sparse includes indexer projection + score + exact top-k. Excludes scheduler, metadata planning, full model and output projection.",
        "prefill": (
            "Full prompt"
            if args.full_prefill
            else "Final chunk at target context length, fixed total query budget across batch; NOT full prompt latency"
        ),
        "dense_kernel": "FlashInfer paged FA2 prefill / tensor-core paged decode",
        "sparse_kernel": "SGLang autotuned Triton token-sparse GQA + fused grouped Lightning scores + FlashInfer exact top-k",
        "cache": "BF16, shuffled physical pages, independent request caches, L2 flushed before each timed sample",
        "dense_workspace": "1 GiB for main KV heads=1; 128 MiB otherwise; allocated outside timing",
    }
    rows = []
    for ci in args.configs:
        cfg = CONFIGS[ci]
        cfg = replace(
            cfg,
            num_k_heads=(
                cfg.num_k_heads if args.main_kv_heads is None else args.main_kv_heads
            ),
            indexer_q_heads=(
                cfg.num_k_heads if args.indexer_q_heads == "kv" else cfg.num_q_heads
            ),
            indexer_kv_heads=(
                (cfg.num_k_heads if args.main_kv_heads is None else args.main_kv_heads)
                if args.indexer_kv_heads == "grouped"
                else 1
            ),
        )
        for phase in args.phases.split(","):
            if phase not in ("prefill", "decode"):
                p.error(f"Invalid phase: {phase}")
            for length in args.contexts:
                for batch in args.batches:
                    qpr = (
                        1
                        if phase == "decode"
                        else (
                            length
                            if args.full_prefill
                            else min(length, args.prefill_budget // batch)
                        )
                    )
                    if phase == "prefill" and (qpr < 4 or qpr % 4):
                        p.error(
                            "prefill budget / batch must be a positive multiple of 4"
                        )
                    for sparse in (
                        [True]
                        if args.sparse
                        else [False] if args.dense_only else [False, True]
                    ):
                        row = {
                            "experiment": args.experiment,
                            "config": ci,
                            **asdict(cfg),
                            "indexer_q_heads": cfg.indexer_q_heads,
                            "phase": phase,
                            "context": length,
                            "batch": batch,
                            "variant": "sparse" if sparse else "dense",
                            "query_tokens": qpr * batch,
                            "query_tokens_per_request": qpr,
                            "top_k": args.top_k if sparse else 0,
                            "cuda_graph": phase == "decode" or args.prefill_graph,
                            "page_size": args.page_size,
                            "status": "ok",
                        }
                        layer = state = x = None
                        try:
                            torch.manual_seed(args.seed)
                            layer = Attention(
                                cfg,
                                sparse=sparse,
                                top_k=args.top_k,
                                score_budget_mib=args.score_budget_mib,
                            )
                            # Same main cache, input and QKV weights for both variants.
                            torch.manual_seed(args.seed + 1)
                            state = PagedState(
                                cfg, batch, length, qpr, args.page_size, sparse, phase
                            )
                            torch.manual_seed(args.seed + 2)
                            x = torch.randn(
                                batch * qpr,
                                cfg.hidden_dim,
                                device="cuda",
                                dtype=torch.bfloat16,
                            )
                            torch.cuda.reset_peak_memory_stats()
                            row.update(
                                measure(
                                    lambda: layer.forward(x, state),
                                    row["cuda_graph"],
                                    args.warmup,
                                    args.iterations,
                                )
                            )
                            row["peak_allocated_gib"] = (
                                torch.cuda.max_memory_allocated() / 1024**3
                            )
                        except torch.OutOfMemoryError as error:
                            row.update(status="oom", error=str(error))
                        rows.append(row)
                        save(rows, metadata, args.output)
                        print(
                            f"config={ci} {phase} L={length} B={batch} {row['variant']}: "
                            f"{row.get('median_ms', 'OOM')} ms",
                            flush=True,
                        )
                        del layer, state, x
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
