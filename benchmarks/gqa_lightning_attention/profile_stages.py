"""Dense vs Lightning-sparse latency plus a per-kernel stage profile, as JSON.

Experiment 1: grouped indexer (48/4 and 64/8 indexer Q/KV heads).
Experiment 2: MSA-like indexer (one index Q head per main KV group, one shared
index KV head: 4Q/1KV and 8Q/1KV). Main attention is unchanged in both.
Latency uses the benchmark.py contract (CUDA graph decode, eager prefill, L2
flush). The stage profile sums CUDA kernel time of eager forwards, so it is
diagnostic and excludes launch gaps.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from attention import CONFIGS, Attention, PagedState
from benchmark import ints, measure
from torch.profiler import ProfilerActivity, profile

STAGES = [
    ("candidate_score", ("_candidate_score",)),  # two-stage fine scores
    ("block_pool", ("_pool",)),  # two-stage block summaries
    ("indexer_score", ("_score",)),  # flat scores, or two-stage coarse scores
    ("topk", ("TopK", "top_k")),
    ("sparse_attn", ("_gqa_token_sparse", "_merge_chunks")),
    ("dense_attn", ("BatchPrefill", "BatchDecode", "PrefillWith", "DecodeWith")),
    ("norm_cache_write", ("_norm", "_cache_write")),
    ("projection_gemm", ("nvjet", "gemm", "cutlass", "splitKreduce", "cublas")),
]


def stage_of(name):
    for stage, needles in STAGES:
        if any(n in name for n in needles):
            return stage
    return "other"


def kernel_breakdown(fn, repeats):
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    stages, kernels = {}, {}
    for event in prof.key_averages():
        ms = event.device_time_total / repeats / 1000
        if ms == 0:
            continue
        stages[stage_of(event.key)] = stages.get(stage_of(event.key), 0) + ms
        kernels[event.key[:120]] = ms
    top = dict(sorted(kernels.items(), key=lambda kv: -kv[1])[:8])
    return {k: round(v, 5) for k, v in stages.items()}, top


def experiment_config(base, experiment):
    if experiment == 1:
        return base
    return replace(
        base, indexer_q_heads=base.num_k_heads, indexer_kv_heads=1
    )  # 4Q/1KV, 8Q/1KV


@torch.no_grad()
def run_point(cfg, phase, length, batch, sparse, args):
    two_stage = sparse and args.two_stage
    qpr = 1 if phase == "decode" else args.prefill_budget // batch
    torch.manual_seed(args.seed)
    layer = Attention(cfg, sparse=sparse, two_stage=two_stage)
    torch.manual_seed(args.seed + 1)
    state = PagedState(
        cfg, batch, length, qpr, 128, sparse, phase, two_stage=two_stage
    )
    torch.manual_seed(args.seed + 2)
    x = torch.randn(batch * qpr, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)
    row = measure(
        lambda: layer.forward(x, state),
        phase == "decode",
        args.warmup,
        args.iterations,
    )
    row["stage_ms"], row["top_kernels_ms"] = kernel_breakdown(
        lambda: layer.forward(x, state), args.profile_repeats
    )
    row["kernel_sum_ms"] = round(sum(row["stage_ms"].values()), 5)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=int, choices=[1, 2], required=True)
    p.add_argument(
        "--two-stage",
        action="store_true",
        help="Use the mean-pooled 64-block shortlist selector for the sparse variant",
    )
    p.add_argument("--configs", type=ints, default=[1, 2])
    p.add_argument("--contexts", type=ints, default=[16384, 32768, 65536, 131072])
    p.add_argument("--batches", type=ints, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--phases", default="prefill,decode")
    p.add_argument("--prefill-budget", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--profile-repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows = []
    for ci in args.configs:
        cfg = experiment_config(CONFIGS[ci], args.experiment)
        for phase in args.phases.split(","):
            for length in args.contexts:
                for batch in args.batches:
                    for sparse in (False, True):
                        row = {
                            "experiment": args.experiment,
                            "selector": (
                                "two_stage" if sparse and args.two_stage else "flat"
                            ),
                            "config": ci,
                            **asdict(cfg),
                            "phase": phase,
                            "context": length,
                            "batch": batch,
                            "variant": "sparse" if sparse else "dense",
                            "status": "ok",
                        }
                        try:
                            row.update(run_point(cfg, phase, length, batch, sparse, args))
                        except torch.OutOfMemoryError as error:
                            row.update(status="oom", error=str(error))
                        rows.append(row)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(
                            json.dumps({"arguments": {k: str(v) for k, v in vars(args).items()}, "device": torch.cuda.get_device_name(), "rows": rows}, indent=1)
                        )
                        print(
                            f"exp{args.experiment} cfg{ci} {phase} L={length} B={batch} "
                            f"{row['variant']}: {row.get('median_ms', 'OOM')} ms",
                            flush=True,
                        )
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
