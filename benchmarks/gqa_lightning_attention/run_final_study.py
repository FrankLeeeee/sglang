"""Final sparse-attention study: dense, token, block and two-stage selection.

Every main configuration x indexer variant x method x phase x context x batch
point is timed (CUDA-event wall time, CUDA-graph decode, eager prefill, L2
flushed) and then broken into GPU-time stage windows (see stages.py). Prefill
is the final scheduled chunk of 2048 total new tokens; decode is one token per
request. Results are appended to one JSON file and the run can be resumed.
"""

import argparse
import gc
import json
from pathlib import Path

import stages
import torch
from attention import Attention, Config, PagedState
from benchmark import measure
from study_config import BATCHES, CONTEXTS, INDEXER_DIMS, INDEXER_TYPES, MAIN, make_cfg
from torch.profiler import ProfilerActivity, profile

METHODS = ["dense", "token", "block", "two_stage"]


def ints(value):
    return [int(x) for x in value.split(",")]


def key_of(row):
    fields = ("main", "indexer", "dim", "method", "phase", "context", "batch")
    return tuple(row[f] for f in fields) + (row.get("dense_backend"),)


class Writer:
    def __init__(self, path, arguments):
        self.path, self.arguments = path, arguments
        self.rows = json.loads(path.read_text())["rows"] if path.exists() else []
        self.done = {key_of(r) for r in self.rows if r["status"] == "ok"}

    def add(self, row):
        self.rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "arguments": self.arguments,
                    "device": torch.cuda.get_device_name(),
                    "rows": self.rows,
                },
                indent=1,
            )
        )


def top_kernels(fn, limit=10):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)
    return {e.key[:100]: round(e.device_time_total / 1000, 5) for e in events[:limit]}


def profile_point(fn, flush):
    fn()
    flush.zero_()
    row = {"stage_ms": {}}
    row["stage_ms"] = {k: round(v, 5) for k, v in stages.record_stages(fn).items()}
    row["stage_sum_ms"] = round(sum(row["stage_ms"].values()), 5)
    flush.zero_()
    row["top_kernels_ms"] = top_kernels(fn)
    return row


def run_row(fn, phase, args, flush, base):
    row = dict(base, status="ok")
    torch.cuda.reset_peak_memory_stats()
    timing = measure(fn, phase == "decode", args.warmup, args.iterations)
    row.update(timing)
    row.update(profile_point(fn, flush))
    if phase == "prefill":
        # Eager launch gaps and L2 effects are what separate wall from windows.
        row["launch_gap_ms"] = round(timing["median_ms"] - row["stage_sum_ms"], 5)
    row["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    return row


def sgl_fa3_runner(state, q, qpr):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    cfg, batch = state.cfg, state.batch
    table = (state.req_to_token[:, :: state.page_size] // state.page_size).int()
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * qpr
    lengths = torch.full((batch,), state.length, device="cuda", dtype=torch.int32)
    k, v = (
        t.view(-1, state.page_size, cfg.num_k_heads, cfg.head_dim)
        for t in (state.k, state.v)
    )

    def run():
        return flash_attn_with_kvcache(
            q,
            k,
            v,
            page_table=table,
            cache_seqlens=lengths,
            cu_seqlens_q=cu_q,
            max_seqlen_q=qpr,
            causal=True,
            softmax_scale=cfg.head_dim**-0.5,
        )

    return run


def dense_rows(state, base, phase, qpr, args, flush, writer):
    cfg = state.cfg
    x = torch.randn(state.batch * qpr, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(args.seed)
    layer = Attention(cfg, sparse=False)
    q = torch.randn(len(x), cfg.num_q_heads, cfg.head_dim, device="cuda", dtype=torch.bfloat16)
    backends = {
        "flashinfer_fa2": lambda: layer.forward(x, state),
        "sgl_fa3": None,
    }
    fa3 = sgl_fa3_runner(state, q, qpr)

    def forward_fa3():
        # Same projections, norms and cache writes; only the attention differs.
        state.dense, original = (lambda qq: fa3()), state.dense
        try:
            return layer.forward(x, state)
        finally:
            state.dense = original

    backends["sgl_fa3"] = forward_fa3
    for name, fn in backends.items():
        row = dict(base, method="dense", dense_backend=name, indexer="none", dim=0)
        if key_of(row) in writer.done:
            continue
        writer.add(run_row(fn, phase, args, flush, row))
        print_row(writer.rows[-1])


def sparse_rows(state, main, base, phase, qpr, methods, args, flush, writer):
    x = torch.randn(
        state.batch * qpr, state.cfg.hidden_dim, device="cuda", dtype=torch.bfloat16
    )
    for indexer in INDEXER_TYPES:
        for dim in INDEXER_DIMS:
            wanted = [
                m
                for m in methods
                if key_of(dict(base, method=m, indexer=indexer, dim=dim))
                not in writer.done
            ]
            if not wanted:
                continue
            cfg = make_cfg(main, indexer == "reduced", dim)
            try:
                state.attach_indexer(cfg)
            except torch.OutOfMemoryError as error:
                state.release_indexer()
                torch.cuda.empty_cache()
                for m in wanted:
                    row = dict(base, method=m, indexer=indexer, dim=dim)
                    writer.add(dict(row, status="oom", error=str(error)[:200]))
                    print_row(writer.rows[-1])
                continue
            for method in wanted:
                row = dict(
                    base,
                    method=method,
                    indexer=indexer,
                    dim=dim,
                    indexer_q_heads=cfg.indexer_q_heads,
                    indexer_kv_heads=cfg.indexer_kv_heads,
                )
                try:
                    if method == "two_stage":
                        from two_stage import prepare_block_cache

                        prepare_block_cache(state)
                    torch.manual_seed(args.seed)
                    layer = Attention(
                        cfg,
                        sparse=True,
                        two_stage=method == "two_stage",
                        block=method == "block",
                        candidate_blocks=args.candidate_blocks,
                    )
                    writer.add(
                        run_row(lambda: layer.forward(x, state), phase, args, flush, row)
                    )
                except torch.OutOfMemoryError as error:
                    writer.add(dict(row, status="oom", error=str(error)[:200]))
                finally:
                    state.block_ik = None
                    layer = None
                    gc.collect()
                    torch.cuda.empty_cache()
                print_row(writer.rows[-1])
            state.release_indexer()
            torch.cuda.empty_cache()


def print_row(row):
    label = f"{row['main']} {row['method']}/{row.get('indexer')}/{row.get('dim')} {row['phase']} L={row['context']} B={row['batch']}"
    print(label, f"{row['median_ms']:.3f} ms" if row["status"] == "ok" else row["status"], flush=True)


def run_point(main, phase, length, batch, methods, args, flush, writer):
    qpr = 1 if phase == "decode" else args.prefill_budget // batch
    base = {
        "main": f"{main[0]}q{main[1]}kv",
        "phase": phase,
        "context": length,
        "batch": batch,
        "query_tokens": batch * qpr,
    }
    torch.manual_seed(args.seed + 1)
    try:
        state = PagedState(Config(*main), batch, length, qpr, 128, False, phase)
    except torch.OutOfMemoryError as error:
        for m in methods:
            for indexer, dim in (
                [("none", 0)]
                if m == "dense"
                else [(i, d) for i in INDEXER_TYPES for d in INDEXER_DIMS]
            ):
                writer.add(
                    dict(
                        base,
                        method=m,
                        indexer=indexer,
                        dim=dim,
                        status="oom",
                        error=str(error)[:200],
                    )
                )
                print_row(writer.rows[-1])
        return
    if "dense" in methods:
        dense_rows(state, base, phase, qpr, args, flush, writer)
    sparse = [m for m in methods if m != "dense"]
    if sparse:
        sparse_rows(state, main, base, phase, qpr, sparse, args, flush, writer)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--mains", type=ints, default=list(range(len(MAIN))))
    p.add_argument("--contexts", type=ints, default=CONTEXTS)
    p.add_argument("--batches", type=ints, default=BATCHES)
    p.add_argument("--phases", default="prefill,decode")
    p.add_argument("--prefill-budget", type=int, default=2048)
    p.add_argument("--candidate-blocks", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    methods = args.methods.split(",")
    writer = Writer(args.output, {k: str(v) for k, v in vars(args).items()})
    flush = torch.empty(64 * 1024**2, device="cuda", dtype=torch.uint8)
    for index in args.mains:
        for phase in args.phases.split(","):
            for length in args.contexts:
                for batch in args.batches:
                    run_point(MAIN[index], phase, length, batch, methods, args, flush, writer)
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
