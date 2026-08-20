#!/usr/bin/env python3
"""Level 1 — kernel-level microbenchmarks for MiniMax-M3 block-sparse attention.

Benchmarks the three-stage sparse attention pipeline that
``MiniMaxSparseAttnBackend`` runs for every sparse layer:

    1. indexer_score  lightweight attention (index heads, index KV cache) that
                      emits one score per 128-token KV block
    2. topk_select    select the top-k blocks per query (+ forced init/local)
    3. sparse_attn    main GQA attention restricted to the selected blocks

``--granularity token`` runs the DeepSeek-style variant instead: the indexer
keeps a score per *key* (no block pooling) and the selection picks individual
token positions. The token budget defaults to the block path's
(topk_blocks x block_size), so the two are directly comparable.

Reports, per configuration: end-to-end latency, the per-stage runtime
breakdown, the transient workspace footprint, and the static KV-cache
footprint. Sweeps context length, Q/KV head counts, head dim, top-k, sparse
block size, KV page size and decode batch size.

Usage:
    python benchmarks/minimax_m3_sparse_attn/bench_kernels.py --help
    python benchmarks/minimax_m3_sparse_attn/bench_kernels.py --mode context
    CUDA_VISIBLE_DEVICES=1 python .../bench_kernels.py --mode all -o results/
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (
    emit_plots,
    STAGE_ORDER,
    DecodeInputs,
    PrefillInputs,
    analytic_memory,
    bench_cuda,
    build_decode_inputs,
    build_prefill_inputs,
    fmt_bytes,
    gpu_info,
    measure_transient_bytes,
    profile_breakdown,
    wait_for_idle,
    warn_if_contended,
    write_results,
)
from m3_config import (
    DEFAULT_CONTEXT_LENS,
    DEFAULT_PREFILL_CHUNK,
    M3_MAX_POSITION,
    SparseAttnConfig,
    m3_config,
)

from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_decode,
    minimax_sparse_prefill,
)
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_dense import (
    minimax_dense_decode,
    minimax_dense_prefill,
)
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_token_sparse import (
    minimax_token_sparse_decode,
    minimax_token_sparse_prefill,
)

# The backend always passes block_size_q=1: every query token picks its own
# block set (MiniMaxSparseAttnBackend.__init__).
BLOCK_SIZE_Q = 1


# ---------------------------------------------------------------------------
# callables under test
# ---------------------------------------------------------------------------


def make_prefill_fn(cfg: SparseAttnConfig, inp: PrefillInputs) -> Callable[[], object]:
    if cfg.granularity == "dense":

        def run_dense():
            return minimax_dense_prefill(
                inp.q,
                inp.k_cache,
                inp.v_cache,
                None,  # sink
                None,
                None,
                None,
                None,  # no indexer
                inp.req_to_token,
                inp.slot_ids,
                inp.cu_seqlens,
                inp.seq_lens,
                inp.prefix_lens,
                inp.max_seqlen_q,
                k_extend=inp.k_extend,
                v_extend=inp.v_extend,
                prefix_lens_cpu=inp.prefix_lens_cpu,
            )

        return run_dense

    if cfg.granularity == "token":

        def run_token():
            return minimax_token_sparse_prefill(
                inp.q,
                inp.k_cache,
                inp.v_cache,
                None,  # sink
                inp.idx_q,
                inp.idx_k_cache,
                inp.idx_v_cache,
                None,  # idx_sink
                inp.req_to_token,
                inp.slot_ids,
                inp.cu_seqlens,
                inp.seq_lens,
                inp.prefix_lens,
                inp.max_seqlen_q,
                inp.max_seqlen_k,
                cfg.effective_topk_tokens,
                cfg.init_tokens,
                cfg.local_tokens,
                disable_index_value=cfg.disable_index_value,
                seqlens_cpu=inp.seqlens_cpu,
                prefix_lens_cpu=inp.prefix_lens_cpu,
            )

        return run_token

    def run():
        return minimax_sparse_prefill(
            inp.q,
            inp.k_cache,
            inp.v_cache,
            None,  # sink
            inp.idx_q,
            inp.idx_k_cache,
            inp.idx_v_cache,
            None,  # idx_sink
            inp.req_to_token,
            inp.slot_ids,
            inp.cu_seqlens,
            inp.seq_lens,
            inp.prefix_lens,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            BLOCK_SIZE_Q,
            cfg.block_size,
            cfg.topk_blocks,
            cfg.init_blocks,
            cfg.local_blocks,
            score_type=cfg.score_type,
            disable_index_value=cfg.disable_index_value,
            use_msa=False,
            cu_seqblocks_q=inp.cu_seqblocks_q,
            max_seqblock_q=inp.max_seqblock_q,
            all_seqblock_q=inp.all_seqblock_q,
            seqlens_cpu=inp.seqlens_cpu,
            page_size=cfg.page_size,
        )

    return run


def make_decode_fn(cfg: SparseAttnConfig, inp: DecodeInputs) -> Callable[[], object]:
    if cfg.granularity == "dense":
        total_kv = int(inp.seq_lens.shape[0]) * inp.max_seqlen

        def run_dense():
            return minimax_dense_decode(
                inp.q,
                None,  # sink
                inp.k_cache,
                inp.v_cache,
                None,
                None,
                None,
                None,  # no indexer
                inp.req_to_token,
                inp.slot_ids,
                inp.seq_lens,
                total_kv=total_kv,
            )

        return run_dense

    if cfg.granularity == "token":

        def run_token():
            return minimax_token_sparse_decode(
                inp.q,
                None,  # sink
                inp.k_cache,
                inp.v_cache,
                inp.idx_q,
                None,  # idx_sink
                inp.idx_k_cache,
                inp.idx_v_cache,
                inp.req_to_token,
                inp.slot_ids,
                inp.seq_lens,
                inp.max_seqlen,
                cfg.effective_topk_tokens,
                cfg.init_tokens,
                cfg.local_tokens,
                disable_index_value=cfg.disable_index_value,
            )

        return run_token

    def run():
        return minimax_sparse_decode(
            inp.q,
            None,  # sink
            inp.k_cache,
            inp.v_cache,
            inp.idx_q,
            None,  # idx_sink
            inp.idx_k_cache,
            inp.idx_v_cache,
            inp.req_to_token,
            inp.slot_ids,
            inp.seq_lens,
            inp.max_seqlen,
            BLOCK_SIZE_Q,
            cfg.block_size,
            cfg.topk_blocks,
            cfg.init_blocks,
            cfg.local_blocks,
            score_type=cfg.score_type,
            disable_index_value=cfg.disable_index_value,
            page_size=cfg.page_size,
            use_msa=False,
        )

    return run


# ---------------------------------------------------------------------------
# one measurement point
# ---------------------------------------------------------------------------


def _base_row(
    cfg: SparseAttnConfig,
    *,
    phase: str,
    sweep: str,
    batch_size: int,
    context_len: int,
    chunk_len: int,
    num_query_tokens: int,
) -> dict:
    # Dense has no fixed selection budget: every query attends to its causal
    # context. ``token_budget == -1`` is only an internal sentinel and must not
    # leak into result rows as negative attended-token counts or sparsity.
    attended = (
        context_len
        if cfg.granularity == "dense"
        else min(context_len, cfg.token_budget)
    )
    return {
        "phase": phase,
        "sweep": sweep,
        "context_len": context_len,
        "batch_size": batch_size,
        "chunk_len": chunk_len,
        "num_query_tokens": num_query_tokens,
        "num_q_heads": cfg.num_q_heads,
        "num_kv_heads": cfg.num_kv_heads,
        "gqa_group_size": cfg.gqa_group_size,
        "head_dim": cfg.head_dim,
        "num_idx_heads": cfg.num_idx_heads,
        "idx_head_dim": cfg.idx_head_dim,
        "idx_group_size": cfg.idx_group_size,
        "granularity": cfg.granularity,
        "block_size": cfg.block_size,
        "topk_blocks": cfg.topk_blocks,
        "topk_tokens": cfg.effective_topk_tokens,
        "page_size": cfg.page_size,
        "dtype": cfg.dtype,
        "token_budget": cfg.token_budget,
        "attended_tokens": attended,
        "sparsity": round(attended / context_len, 6),
    }


def _fill_measurements(
    row: dict,
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    profile_iters: int,
    show_unmapped: bool,
) -> dict:
    timing = bench_cuda(fn, warmup=warmup, iters=iters)
    row.update(
        {
            "latency_mean_ms": round(timing.mean_ms, 6),
            "latency_median_ms": round(timing.median_ms, 6),
            "latency_min_ms": round(timing.min_ms, 6),
            "latency_p90_ms": round(timing.p90_ms, 6),
        }
    )

    stages, kernels = profile_breakdown(fn, iters=profile_iters, warmup=3)
    for stage in STAGE_ORDER:
        row[f"stage_{stage}_ms"] = round(stages.get(stage, 0.0), 6)
    stage_sum = sum(stages.values())
    row["stage_sum_ms"] = round(stage_sum, 6)
    # Wall time the GPU spends outside kernels: launch gaps, host-side glue.
    row["launch_gap_ms"] = round(max(0.0, timing.median_ms - stage_sum), 6)

    if show_unmapped:
        from harness import classify_kernel

        unmapped = {n: v for n, v in kernels.items() if classify_kernel(n) == "other"}
        if unmapped:
            print(f"    [unmapped kernels] {unmapped}")

    row["transient_bytes"] = measure_transient_bytes(fn)
    return row


def run_prefill_point(
    cfg: SparseAttnConfig,
    *,
    sweep: str,
    batch_size: int,
    context_len: int,
    chunk_len: Optional[int],
    iters: int,
    profile_iters: int,
    show_unmapped: bool,
) -> dict:
    inp = build_prefill_inputs(
        cfg, batch_size=batch_size, context_len=context_len, chunk_len=chunk_len
    )
    row = _base_row(
        cfg,
        phase="prefill",
        sweep=sweep,
        batch_size=batch_size,
        context_len=context_len,
        chunk_len=inp.max_seqlen_q,
        num_query_tokens=inp.num_query_tokens,
    )
    row.update(
        analytic_memory(
            cfg,
            batch_size=batch_size,
            context_len=context_len,
            num_query_tokens=inp.num_query_tokens,
            is_prefill=True,
        )
    )
    _fill_measurements(
        row,
        make_prefill_fn(cfg, inp),
        warmup=max(3, iters // 4),
        iters=iters,
        profile_iters=profile_iters,
        show_unmapped=show_unmapped,
    )
    row["throughput_tok_per_s"] = round(
        inp.num_query_tokens / (row["latency_median_ms"] / 1000.0), 1
    )

    del inp
    torch.cuda.empty_cache()
    return row


def run_decode_point(
    cfg: SparseAttnConfig,
    *,
    sweep: str,
    batch_size: int,
    context_len: int,
    iters: int,
    profile_iters: int,
    show_unmapped: bool,
) -> dict:
    inp = build_decode_inputs(cfg, batch_size=batch_size, context_len=context_len)
    row = _base_row(
        cfg,
        phase="decode",
        sweep=sweep,
        batch_size=batch_size,
        context_len=context_len,
        chunk_len=1,
        num_query_tokens=batch_size,
    )
    row.update(
        analytic_memory(
            cfg,
            batch_size=batch_size,
            context_len=context_len,
            num_query_tokens=batch_size,
            is_prefill=False,
        )
    )
    _fill_measurements(
        row,
        make_decode_fn(cfg, inp),
        warmup=max(10, iters // 4),
        iters=iters,
        profile_iters=profile_iters,
        show_unmapped=show_unmapped,
    )
    row["throughput_tok_per_s"] = round(
        batch_size / (row["latency_median_ms"] / 1000.0), 1
    )

    del inp
    torch.cuda.empty_cache()
    return row


def prime_runtime(base: SparseAttnConfig, decode_batch_sizes: list[int]) -> None:
    """Burn the one-time costs before any timed point.

    First-touch of the Triton launchers and the JIT radix top-k compile land on
    whichever configuration happens to run first, inflating it by 10x or more.
    The decode kernels specialize on a batch-size bucket, so prime every batch
    size the run will actually use, not just one.
    """
    prefill_inp = build_prefill_inputs(base, batch_size=1, context_len=4096)
    prefill_fn = make_prefill_fn(base, prefill_inp)
    for _ in range(3):
        prefill_fn()
    del prefill_inp, prefill_fn

    for bs in sorted(set(decode_batch_sizes)):
        decode_inp = build_decode_inputs(base, batch_size=bs, context_len=4096)
        decode_fn = make_decode_fn(base, decode_inp)
        for _ in range(3):
            decode_fn()
        del decode_inp, decode_fn
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _guarded(label: str, thunk: Callable[[], dict]) -> Optional[dict]:
    try:
        row = thunk()
        row["status"] = "ok"
        print(
            f"  {label:<58} {row['latency_median_ms']:9.4f} ms "
            f"(min {row['latency_min_ms']:8.4f})  "
            + (
                f"(dense {row['stage_dense_attn_ms']:.4f}) "
                if row.get("stage_dense_attn_ms")
                else ""
            )
            + f"(index {row['stage_indexer_score_ms']:.4f} | "
            f"topk {row['stage_topk_select_ms']:.4f} | "
            f"sparse {row['stage_sparse_attn_ms']:.4f})  "
            f"ws={fmt_bytes(row['transient_bytes'])}"
        )
        return row
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  {label:<58} OOM")
        return {"label": label, "status": "oom"}
    except Exception as err:  # keep the sweep going, record why
        torch.cuda.empty_cache()
        print(f"  {label:<58} FAILED: {type(err).__name__}: {err}")
        traceback.print_exc(limit=3)
        return {"label": label, "status": f"error: {type(err).__name__}: {err}"}


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def sweep_context(base: SparseAttnConfig, args) -> list[dict]:
    rows: list[dict] = []
    print("\n=== context-length sweep: PREFILL ===")
    for ctx in args.context_lens:
        chunk = None if args.prefill_chunk == 0 else min(args.prefill_chunk, ctx)
        label = f"prefill bs={args.prefill_batch_size} ctx={ctx} chunk={chunk or ctx}"
        row = _guarded(
            label,
            lambda ctx=ctx, chunk=chunk: run_prefill_point(
                base,
                sweep="context",
                batch_size=args.prefill_batch_size,
                context_len=ctx,
                chunk_len=chunk,
                iters=args.prefill_iters,
                profile_iters=args.profile_iters,
                show_unmapped=args.show_unmapped,
            ),
        )
        if row:
            rows.append(row)

    print("\n=== context-length sweep: DECODE ===")
    for bs in args.decode_batch_sizes:
        for ctx in args.context_lens:
            label = f"decode bs={bs} ctx={ctx}"
            row = _guarded(
                label,
                lambda bs=bs, ctx=ctx: run_decode_point(
                    base,
                    sweep="context",
                    batch_size=bs,
                    context_len=ctx,
                    iters=args.decode_iters,
                    profile_iters=args.profile_iters * 5,
                    show_unmapped=args.show_unmapped,
                ),
            )
            if row:
                rows.append(row)
    return rows


def _dimension_sweeps(
    base: SparseAttnConfig, args
) -> dict[str, list[SparseAttnConfig]]:
    """Named sweeps, each a list of configs varying exactly one dimension."""
    sweeps: dict[str, list[SparseAttnConfig]] = {}

    # Q heads (KV heads fixed): raises arithmetic intensity of the sparse kernel.
    sweeps["num_q_heads"] = [
        base.replace(num_q_heads=h) for h in args.q_heads if h % base.num_kv_heads == 0
    ]

    # KV heads: index heads track KV heads so idx_group_size stays 1, exactly as
    # M3 ships it. Q heads are raised with KV heads to keep GQA >= 1.
    kv_rows = []
    for kv in args.kv_heads:
        q = max(base.num_q_heads, kv)
        q = q - (q % kv)
        kv_rows.append(base.replace(num_kv_heads=kv, num_idx_heads=kv, num_q_heads=q))
    sweeps["num_kv_heads"] = kv_rows

    sweeps["head_dim"] = [
        base.replace(head_dim=d, idx_head_dim=d) for d in args.head_dims
    ]

    # Budget sweep. Both granularities walk the *same* token budgets so their
    # curves are comparable: the block path varies top-k blocks, the token path
    # varies top-k tokens.
    if base.granularity == "dense":
        # No selection knob to sweep: dense reads the whole context by definition.
        pass
    elif base.granularity == "token":
        sweeps["topk_blocks"] = [
            base.replace(topk_tokens=k * base.block_size) for k in args.topks
        ]
    else:
        sweeps["topk_blocks"] = [base.replace(topk_blocks=k) for k in args.topks]

    if base.granularity == "block":
        # Block granularity itself. Meaningless for the token path, which has no
        # blocks, so it is only swept for the block path.
        sweeps["block_size"] = [
            base.replace(
                block_size=b,
                # keep the token budget constant so this isolates block
                # granularity rather than also changing how much KV is read
                topk_blocks=max(
                    base.init_blocks + base.local_blocks + 1, base.token_budget // b
                ),
            )
            for b in args.block_sizes
        ]

    sweeps["page_size"] = [base.replace(page_size=p) for p in args.page_sizes]
    return sweeps


def sweep_dimensions(base: SparseAttnConfig, args) -> list[dict]:
    rows: list[dict] = []
    sweeps = _dimension_sweeps(base, args)
    for name, cfgs in sweeps.items():
        if args.sweeps and name not in args.sweeps:
            continue
        print(f"\n=== dimension sweep: {name} (ctx={args.sweep_context_len}) ===")
        for cfg in cfgs:
            tag = cfg.shape_tag()
            row = _guarded(
                f"prefill {name} {tag}",
                lambda cfg=cfg: run_prefill_point(
                    cfg,
                    sweep=name,
                    batch_size=args.prefill_batch_size,
                    context_len=args.sweep_context_len,
                    chunk_len=None if args.prefill_chunk == 0 else args.prefill_chunk,
                    iters=args.prefill_iters,
                    profile_iters=args.profile_iters,
                    show_unmapped=args.show_unmapped,
                ),
            )
            if row:
                rows.append(row)
            row = _guarded(
                f"decode  {name} {tag}",
                lambda cfg=cfg: run_decode_point(
                    cfg,
                    sweep=name,
                    batch_size=args.sweep_decode_batch_size,
                    context_len=args.sweep_context_len,
                    iters=args.decode_iters,
                    profile_iters=args.profile_iters * 5,
                    show_unmapped=args.show_unmapped,
                ),
            )
            if row:
                rows.append(row)

    if not args.sweeps or "batch_size" in args.sweeps:
        print(f"\n=== dimension sweep: batch_size (ctx={args.sweep_context_len}) ===")
        for bs in args.sweep_batch_sizes:
            row = _guarded(
                f"decode  batch_size bs={bs}",
                lambda bs=bs: run_decode_point(
                    base,
                    sweep="batch_size",
                    batch_size=bs,
                    context_len=args.sweep_context_len,
                    iters=args.decode_iters,
                    profile_iters=args.profile_iters * 5,
                    show_unmapped=args.show_unmapped,
                ),
            )
            if row:
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# memory report
# ---------------------------------------------------------------------------


def memory_report(base: SparseAttnConfig, args) -> list[dict]:
    """Static KV-cache accounting for a whole M3 server, not just one layer."""
    from m3_config import M3_NUM_DENSE_LAYERS, M3_NUM_SPARSE_LAYERS

    per_tok = base.kv_bytes_per_token_per_layer()
    dense_per_tok = 2 * base.num_kv_heads * base.head_dim * base.torch_dtype.itemsize

    rows = []
    print("\n=== static KV-cache footprint (one GPU) ===")
    print(
        f"  sparse layer: main {per_tok['main_kv']} B/token + index "
        f"{per_tok['index_kv']} B/token = {per_tok['total']} B/token"
    )
    print(f"  dense  layer: main {dense_per_tok} B/token")
    for ctx in args.context_lens:
        sparse_total = per_tok["total"] * ctx * M3_NUM_SPARSE_LAYERS
        dense_total = dense_per_tok * ctx * M3_NUM_DENSE_LAYERS
        model_total = sparse_total + dense_total
        # what the same model would cost with no index cache at all
        baseline = dense_per_tok * ctx * (M3_NUM_SPARSE_LAYERS + M3_NUM_DENSE_LAYERS)
        rows.append(
            {
                "phase": "memory",
                "sweep": "kv_footprint",
                "context_len": ctx,
                "kv_bytes_per_token_sparse_layer": per_tok["total"],
                "kv_bytes_per_token_dense_layer": dense_per_tok,
                "kv_bytes_sparse_layers": sparse_total,
                "kv_bytes_dense_layers": dense_total,
                "kv_bytes_model_per_gpu": model_total,
                "kv_bytes_model_no_index": baseline,
                "index_overhead_ratio": round(model_total / baseline, 4),
                "status": "ok",
            }
        )
        print(
            f"  ctx={ctx:>7}: {fmt_bytes(model_total):>10} / GPU for one request "
            f"({fmt_bytes(baseline):>10} without the index cache, "
            f"+{100 * (model_total / baseline - 1):.1f}%)"
        )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode",
        default="all",
        choices=["all", "context", "sweeps", "memory"],
        help="which measurement groups to run",
    )
    p.add_argument("--context-lens", type=_int_list, default=DEFAULT_CONTEXT_LENS)
    p.add_argument("--prefill-batch-size", type=int, default=1)
    p.add_argument(
        "--prefill-chunk",
        type=int,
        default=DEFAULT_PREFILL_CHUNK,
        help=f"extend-chunk size (default: {DEFAULT_PREFILL_CHUNK}); "
        "0 = single-shot prefill of the whole context",
    )
    p.add_argument("--decode-batch-sizes", type=_int_list, default=[1, 32])

    p.add_argument("--sweep-context-len", type=int, default=32768)
    p.add_argument("--sweep-decode-batch-size", type=int, default=32)
    p.add_argument(
        "--sweeps",
        type=lambda s: [x for x in s.split(",") if x],
        default=None,
        help="subset of: num_q_heads,num_kv_heads,head_dim,topk_blocks,block_size,page_size,batch_size",
    )
    p.add_argument("--q-heads", type=_int_list, default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--kv-heads", type=_int_list, default=[1, 2, 4, 8])
    p.add_argument("--head-dims", type=_int_list, default=[64, 128, 192, 256])
    p.add_argument("--topks", type=_int_list, default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--block-sizes", type=_int_list, default=[16, 32, 64, 128])
    p.add_argument("--page-sizes", type=_int_list, default=[1, 16, 32, 64, 128, 256])
    p.add_argument(
        "--sweep-batch-sizes", type=_int_list, default=[1, 4, 16, 64, 128, 256]
    )

    p.add_argument("--prefill-iters", type=int, default=20)
    p.add_argument("--decode-iters", type=int, default=200)
    p.add_argument("--profile-iters", type=int, default=10)
    p.add_argument(
        "--granularity",
        type=lambda s: [x for x in s.split(",") if x],
        default=["block", "token", "dense"],
        help="granularities to measure: block, token, dense (default: all three)",
    )
    p.add_argument("--show-unmapped", action="store_true")
    p.add_argument(
        "--wait-for-idle",
        type=float,
        default=0.0,
        help="seconds to wait for the GPU to go idle before starting (shared boxes)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="skip the plots this benchmark emits when it finishes",
    )
    p.add_argument("--tag", default="kernels")
    args = p.parse_args(argv)
    if not args.context_lens:
        p.error("--context-lens must contain at least one value")
    if any(ctx <= 0 or ctx > M3_MAX_POSITION for ctx in args.context_lens):
        p.error(
            f"every context length must be in [1, {M3_MAX_POSITION}] "
            "(MiniMax-M3's max_position_embeddings)"
        )
    if args.prefill_chunk < 0:
        p.error("--prefill-chunk must be non-negative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.mode != "memory" and not torch.cuda.is_available():
        print("CUDA is required for these benchmarks.")
        return 1

    base = m3_config()
    print("MiniMax-M3 block-sparse attention — kernel microbenchmarks")
    if args.mode == "memory":
        info = {"gpu": "static-accounting", "torch": torch.__version__}
        print("  device : not used (static memory accounting)")
        print(f"  torch  : {info['torch']}")
    else:
        info = gpu_info()
        print(f"  device : {info['gpu']} (sm{info['sm']}, {info['memory_gb']} GB)")
        print(f"  torch  : {info['torch']}")
    print(f"  config : {base.shape_tag()}")
    print(
        f"  budget : top-{base.topk_blocks} x {base.block_size} = "
        f"{base.token_budget} tokens attended per query, regardless of context"
    )
    print(
        "  note   : MSA (fmha_sm100) is Blackwell-only; this runs the Triton "
        "sparse path used on Hopper/ROCm."
    )
    # Static memory accounting is pure arithmetic. Do not wait for GPU idleness,
    # inspect utilization, allocate benchmark inputs, or JIT-prime kernels.
    if args.mode == "memory":
        gpu_util = None
    else:
        wait_for_idle(args.wait_for_idle)
        gpu_util = warn_if_contended()
        decode_bs = (
            args.decode_batch_sizes
            + args.sweep_batch_sizes
            + [args.sweep_decode_batch_size]
        )
        for granularity in args.granularity:
            prime_runtime(base.replace(granularity=granularity), decode_bs)

    rows: list[dict] = []
    if args.mode in ("all", "memory"):
        # The KV cache is identical either way — selection granularity changes
        # what is read, not what is stored.
        rows += memory_report(base, args)
    if args.mode != "memory":
        for granularity in args.granularity:
            cfg = base.replace(granularity=granularity)
            print(f"\n{'#' * 70}\n# granularity = {granularity}\n{'#' * 70}")
            if args.mode in ("all", "context"):
                rows += sweep_context(cfg, args)
            if args.mode in ("all", "sweeps"):
                rows += sweep_dimensions(cfg, args)

    for row in rows:
        row.setdefault("gpu", info["gpu"])
        row.setdefault("gpu_util_before_pct", gpu_util)
    json_path, csv_path = write_results(rows, args.out, args.tag)
    print(f"\nwrote {json_path}\n      {csv_path}")
    if not args.no_plots:
        import plot_results

        emit_plots(
            plot_results.main,
            ["--results", str(args.out), "--out", str(args.out / "plots")],
            label="plots",
        )
        print(f"      {args.out / 'plots'}/*.png")
    errors = [row for row in rows if str(row.get("status", "")).startswith("error:")]
    if errors:
        print(f"\nFAILED: {len(errors)} benchmark point(s) ended in an error.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
