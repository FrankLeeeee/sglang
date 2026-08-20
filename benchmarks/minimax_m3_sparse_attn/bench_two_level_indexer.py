#!/usr/bin/env python3
"""Two-level (block -> token) indexer vs the one-level indexers M3 ships.

Prototype benchmark for ``two_level_indexer.py``. Three implementations of the
same selection stage, at MiniMax-M3 shapes, on the same paged index KV cache:

  block       shipped M3: score all L keys, pool to L/128 block scores, take
              top-16 blocks. Emits *blocks*, so its accuracy is not comparable —
              it is here as the latency line to beat.
  flat        token granularity: score all L keys, take top-2048 positions.
              Exact, and the accuracy reference for everything below.
  two_level   coarse pass over L/P mean-pooled block keys -> top-M blocks, fine
              pass over those M*P candidates -> top-2048 positions.

The two-level pass reads O(L/P + M*P) index keys per query row against the flat
pass's O(L), so its cost flattens out as context grows while the flat one keeps
climbing. What it gives up is exactness: a token whose block pools badly is
never scored. ``--mode recall`` measures that directly against the flat
selection, as a function of M.

Modes:

    latency   selection latency vs context length, prefill and decode
    stages    where the two-level time goes: pool / coarse+top-M / fine+top-k
    recall    recall vs exact flat top-k, sweeping M (the accuracy/cost curve)
    coverage  all three paths at the same 2048-token budget: which 2048?
    pooling   stage 1 pooling before the dot (LongCat) vs after it (M3 block)
    e2e       selection + sparse GQA attention, flat vs two-level

    CUDA_VISIBLE_DEVICES=0 python bench_two_level_indexer.py
    CUDA_VISIBLE_DEVICES=0 python bench_two_level_indexer.py --mode recall
    python bench_two_level_indexer.py --context-lens 262144,1048576 --mode latency,e2e

Recall is measured on *clustered* index keys (see
``two_level_indexer.make_clustered_index_keys``): mean pooling only summarises a
block whose keys resemble each other, which real KV does and the harness's iid
Gaussians do not. Pass ``--iid`` to see the floor instead.

Outputs (default --out results/two_level_indexer): raw.json, summary.csv, plots.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    bench_cuda,
    build_decode_inputs,
    build_prefill_inputs,
    measure_transient_bytes,
    profile_breakdown,
    wait_for_idle,
    warn_if_contended,
    write_results,
)
from m3_config import m3_config  # noqa: E402
from two_level_indexer import (  # noqa: E402
    TwoLevelConfig,
    build_pooled_index_keys,
    make_clustered_index_keys,
    make_clustered_index_queries,
    reference_exact_topk,
    selection_recall,
    two_level_select_decode,
    two_level_select_prefill,
)

from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (  # noqa: E402
    flash_decode_with_topk_idx,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (  # noqa: E402
    flash_prefill_with_topk_index,
)
from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (  # noqa: E402
    token_select_decode,
    token_select_prefill,
)
from sglang.kernels.ops.attention.minimax_sparse.token.sparse_attn import (  # noqa: E402
    gqa_token_sparse_attn,
)

DEFAULT_CTXS = [16384, 65536, 262144, 1048576]
DEFAULT_RECALL_CTX = 131072
DEFAULT_MS = [8, 16, 32, 64, 128, 256]


def _ctx_label(n: int) -> str:
    return f"{n // 1024}k" if n < 1 << 20 else f"{n >> 20}M"


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------


def _flat_prefill(cfg, inp) -> Callable:
    def run():
        return token_select_prefill(
            inp.idx_q,
            inp.idx_k_cache,
            inp.req_to_token,
            inp.slot_ids,
            inp.cu_seqlens,
            inp.seq_lens,
            inp.prefix_lens,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            cfg.block_token_budget,
            cfg.init_tokens,
            cfg.local_tokens,
            seqlens_cpu=inp.seqlens_cpu,
            prefix_lens_cpu=inp.prefix_lens_cpu,
        )

    return run


def _flat_decode(cfg, inp) -> Callable:
    def run():
        return token_select_decode(
            inp.idx_q,
            inp.idx_k_cache,
            inp.req_to_token,
            inp.slot_ids,
            inp.seq_lens,
            inp.max_seqlen,
            cfg.block_token_budget,
            cfg.init_tokens,
            cfg.local_tokens,
        )

    return run


def _block_prefill(cfg, inp) -> Callable:
    def run():
        return flash_prefill_with_topk_index(
            q=inp.idx_q,
            k_cache=inp.idx_k_cache,
            v_cache=None,
            sink=None,
            req_to_token=inp.req_to_token,
            slot_ids=inp.slot_ids,
            cu_seqlens=inp.cu_seqlens,
            seq_lens=inp.seq_lens,
            prefix_lens=inp.prefix_lens,
            max_seqlen_q=inp.max_seqlen_q,
            max_seqlen_k=inp.max_seqlen_k,
            block_size_q=1,
            block_size_k=cfg.block_size,
            topk=cfg.topk_blocks,
            init_blocks=cfg.init_blocks,
            local_blocks=cfg.local_blocks,
            disable_index_value=True,
            score_type=cfg.score_type,
        )

    return run


def _block_decode(cfg, inp) -> Callable:
    def run():
        return flash_decode_with_topk_idx(
            q=inp.idx_q,
            k_cache=inp.idx_k_cache,
            v_cache=None,
            sink=None,
            req_to_token=inp.req_to_token,
            slot_ids=inp.slot_ids,
            seq_lens=inp.seq_lens,
            max_seqlen=inp.max_seqlen,
            block_size=cfg.block_size,
            topk=cfg.topk_blocks,
            init_blocks=cfg.init_blocks,
            local_blocks=cfg.local_blocks,
            disable_index_value=True,
            score_type=cfg.score_type,
        )

    return run


def _two_level_prefill(cfg, tl_cfg, inp, pooled: Optional[torch.Tensor]) -> Callable:
    def run():
        return two_level_select_prefill(
            inp.idx_q,
            inp.idx_k_cache,
            inp.req_to_token,
            inp.slot_ids,
            inp.cu_seqlens,
            inp.seq_lens,
            inp.prefix_lens,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            num_kv_heads=cfg.num_kv_heads,
            cfg=tl_cfg,
            pooled=pooled,
        )

    return run


def _two_level_decode(cfg, tl_cfg, inp, pooled: Optional[torch.Tensor]) -> Callable:
    def run():
        return two_level_select_decode(
            inp.idx_q,
            inp.idx_k_cache,
            inp.req_to_token,
            inp.slot_ids,
            inp.seq_lens,
            inp.max_seqlen,
            num_kv_heads=cfg.num_kv_heads,
            cfg=tl_cfg,
            pooled=pooled,
        )

    return run


def _pooled_for(tl_cfg, inp, max_seqlen_k: int) -> torch.Tensor:
    return build_pooled_index_keys(
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.seq_lens,
        pool_block_size=tl_cfg.pool_block_size,
        max_seqlen_k=max_seqlen_k,
    )


def _tl_config(
    cfg, args, ctx: int, coarse_blocks: Optional[int] = None
) -> TwoLevelConfig:
    return TwoLevelConfig(
        pool_block_size=args.pool_block,
        coarse_blocks=coarse_blocks or args.coarse_blocks,
        topk_tokens=cfg.block_token_budget,
        query_tile=args.query_tile,
        pool_position=args.pool_position,
        share_candidates=args.share_candidates,
        init_tokens=cfg.init_tokens,
        local_tokens=cfg.local_tokens,
        head_reduce=args.head_reduce,
    )


# ---------------------------------------------------------------------------
# mode: latency
# ---------------------------------------------------------------------------


def _index_keys_per_row(cfg, tl_cfg, ctx: int, impl: str) -> int:
    """How many index keys one query row's selection depends on.

    The algorithmic quantity the whole design turns on, and the reason two-level
    wins: the one-level indexers make every query row's answer depend on all L
    keys — the block path pools *after* the dot, so its small output buys it
    nothing here — while two-level makes it depend on L/P pooled keys plus the
    M*P it recalls, which stops growing with L.

    Deliberately not a DRAM-traffic estimate. Real traffic is this times a
    per-kernel factor that varies: the block and flat *prefill* kernels loop the
    context once per index head (grid `(q_tiles, batch * heads)`), their decode
    kernels put the heads in the dot and read K once, and every kernel amortises
    over its own query tile. Those factors are similar enough across the three
    paths that they do not move the comparison, and pretending to model them
    precisely would be worse than not modelling them.
    """
    if impl.startswith("two_level"):
        # Post-pooling scores every key in stage 1; pre-pooling reads L/P of them.
        coarse_keys = (
            ctx if impl.endswith("_post") else math.ceil(ctx / tl_cfg.pool_block_size)
        )
        keys = coarse_keys + tl_cfg.candidate_width
        if impl == "two_level_rebuild":
            keys += ctx  # stage 0 re-reads the whole cache to rebuild the pool
        return keys
    return ctx


def run_latency(cfg, args, rows: list[dict]) -> None:
    for ctx in args.context_lens:
        tl_cfg = _tl_config(cfg, args, ctx)
        chunk = min(args.prefill_chunk, ctx)
        print(
            f"--- ctx={_ctx_label(ctx)} "
            f"(coarse {math.ceil(ctx / tl_cfg.pool_block_size)} blocks -> "
            f"top-{tl_cfg.coarse_blocks} -> {tl_cfg.candidate_width} candidates "
            f"-> top-{tl_cfg.topk_tokens})"
            + ("  [degenerate: recall is total]" if tl_cfg.is_degenerate(ctx) else "")
        )

        # prefill
        inp = build_prefill_inputs(
            cfg, batch_size=args.prefill_batch_size, context_len=ctx, chunk_len=chunk
        )
        if not args.iid:
            make_clustered_index_keys(
                inp.idx_k_cache, inp.req_to_token, inp.seq_lens, block=args.pool_block
            )
        pooled = _pooled_for(tl_cfg, inp, ctx)
        post_cfg = tl_cfg.replace(pool_position="post")
        runners = {
            "block": _block_prefill(cfg, inp),
            "flat": _flat_prefill(cfg, inp),
            "two_level": _two_level_prefill(cfg, tl_cfg, inp, pooled),
            "two_level_post": _two_level_prefill(cfg, post_cfg, inp, None),
            "two_level_rebuild": _two_level_prefill(cfg, tl_cfg, inp, None),
        }
        for impl, fn in runners.items():
            rows.append(
                _time_one(
                    fn,
                    impl=impl,
                    phase="prefill",
                    ctx=ctx,
                    cfg=cfg,
                    tl_cfg=tl_cfg,
                    batch=args.prefill_batch_size,
                    rows_per_req=chunk,
                    iters=args.prefill_iters,
                )
            )
        del inp, pooled, runners
        torch.cuda.empty_cache()

        # decode
        for bs in args.decode_batch_sizes:
            inp = build_decode_inputs(cfg, batch_size=bs, context_len=ctx)
            if not args.iid:
                make_clustered_index_keys(
                    inp.idx_k_cache,
                    inp.req_to_token,
                    inp.seq_lens,
                    block=args.pool_block,
                )
            pooled = _pooled_for(tl_cfg, inp, ctx)
            post_cfg = tl_cfg.replace(pool_position="post")
            runners = {
                "block": _block_decode(cfg, inp),
                "flat": _flat_decode(cfg, inp),
                "two_level": _two_level_decode(cfg, tl_cfg, inp, pooled),
                "two_level_post": _two_level_decode(cfg, post_cfg, inp, None),
                "two_level_rebuild": _two_level_decode(cfg, tl_cfg, inp, None),
            }
            for impl, fn in runners.items():
                rows.append(
                    _time_one(
                        fn,
                        impl=impl,
                        phase="decode",
                        ctx=ctx,
                        cfg=cfg,
                        tl_cfg=tl_cfg,
                        batch=bs,
                        rows_per_req=1,
                        iters=args.decode_iters,
                    )
                )
            del inp, pooled, runners
            torch.cuda.empty_cache()


def _time_one(
    fn, *, impl, phase, ctx, cfg, tl_cfg, batch, rows_per_req, iters, extra=None
) -> dict:
    timing = bench_cuda(fn, warmup=max(3, iters // 4), iters=iters)
    # Kernel time separately from wall clock: at decode these paths issue enough
    # small launches that the host, not the GPU, sets the wall clock — and a
    # server captures decode in a CUDA graph, where that host cost is gone.
    _, per_kernel = profile_breakdown(fn, iters=max(5, iters // 4))
    gpu_ms = sum(per_kernel.values())
    keys_per_row = _index_keys_per_row(cfg, tl_cfg, ctx, impl)
    key_bytes = (
        batch
        * rows_per_req
        * keys_per_row
        * cfg.idx_head_dim
        * cfg.torch_dtype.itemsize
    )
    row = {
        "impl": impl,
        "phase": phase,
        "context_len": ctx,
        "batch_size": batch,
        "rows_per_req": rows_per_req,
        "pool_block": tl_cfg.pool_block_size,
        "coarse_blocks": tl_cfg.coarse_blocks,
        "candidate_width": tl_cfg.candidate_width,
        "topk": tl_cfg.topk_tokens,
        "latency_median_ms": round(timing.median_ms, 6),
        "latency_min_ms": round(timing.min_ms, 6),
        "gpu_kernel_ms": round(gpu_ms, 6),
        "launch_bound_ratio": round(timing.median_ms / max(gpu_ms, 1e-9), 3),
        "index_keys_per_row": keys_per_row,
        "index_key_bytes_logical": key_bytes,
        "transient_bytes": measure_transient_bytes(fn),
    }
    if extra:
        row.update(extra)
    if impl.startswith("two_level") and "pool_position" not in row:
        row["pool_position"] = "post" if impl.endswith("_post") else "pre"
    print(
        f"  {phase:<10} {impl:<18} ctx={_ctx_label(ctx):<5} bs={batch:<4} "
        f"{timing.median_ms:9.4f} ms wall  {gpu_ms:8.4f} ms gpu   "
        f"idxK {key_bytes / 2**20:9.1f} MiB   "
        f"ws {row['transient_bytes'] / 2**20:7.1f} MiB"
    )
    return row


# ---------------------------------------------------------------------------
# mode: stages
# ---------------------------------------------------------------------------


def _stage_of(kernel: str) -> str:
    """Map a kernel name to the pipeline stage it belongs to."""
    if "pool_index_keys" in kernel:
        return "stage0_pool"
    if "coarse_block_score" in kernel:
        return "stage1_coarse_score"
    if "fine_token_score" in kernel:
        return "stage2_fine_score"
    if "map_columns" in kernel:
        return "stage2_map"
    if "TopK" in kernel or "topk" in kernel or "Sort" in kernel:
        return "select"
    return "other"


def run_stages(cfg, args, rows: list[dict]) -> None:
    """Attribute the two-level decode path by kernel.

    Attribution is by kernel name out of the profiler, not by timing truncated
    variants of the pipeline: the driver's fixed cost (allocations, the two
    selector calls, launch overhead) is large enough at decode that a
    prefix-timing split charges all of it to whichever stage runs first.
    """
    for ctx in args.context_lens:
        tl_cfg = _tl_config(cfg, args, ctx)
        for bs in args.decode_batch_sizes:
            inp = build_decode_inputs(cfg, batch_size=bs, context_len=ctx)
            if not args.iid:
                make_clustered_index_keys(
                    inp.idx_k_cache,
                    inp.req_to_token,
                    inp.seq_lens,
                    block=args.pool_block,
                )
            pooled = _pooled_for(tl_cfg, inp, ctx)

            wall = bench_cuda(
                _two_level_decode(cfg, tl_cfg, inp, pooled),
                warmup=5,
                iters=args.decode_iters,
            ).median_ms
            _, per_kernel = profile_breakdown(
                _two_level_decode(cfg, tl_cfg, inp, pooled),
                iters=args.decode_iters // 2,
            )
            # Stage 0 runs outside the selection call, so time it on its own.
            pool_ms = bench_cuda(
                lambda: _pooled_for(tl_cfg, inp, ctx), warmup=5, iters=args.decode_iters
            ).median_ms

            per_stage: dict[str, float] = {"stage0_pool_rebuild": pool_ms}
            for kernel, ms in per_kernel.items():
                stage = _stage_of(kernel)
                per_stage[stage] = per_stage.get(stage, 0.0) + ms
            gpu_ms = sum(v for k, v in per_stage.items() if k != "stage0_pool_rebuild")
            per_stage["host_launch_gap"] = max(0.0, wall - gpu_ms)

            for name, ms in sorted(per_stage.items(), key=lambda kv: -kv[1]):
                rows.append(
                    {
                        "impl": "two_level",
                        "phase": "decode",
                        "stage": name,
                        "context_len": ctx,
                        "batch_size": bs,
                        "pool_block": tl_cfg.pool_block_size,
                        "coarse_blocks": tl_cfg.coarse_blocks,
                        "latency_median_ms": round(ms, 6),
                        "share_of_wall": round(ms / wall, 4),
                    }
                )
                print(
                    f"  ctx={_ctx_label(ctx):<5} bs={bs:<4} {name:<22} "
                    f"{ms:9.4f} ms  ({ms / wall * 100:5.1f}% of the {wall:.3f} ms call)"
                )
            del inp, pooled
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# mode: recall
# ---------------------------------------------------------------------------


def run_recall(cfg, args, rows: list[dict]) -> None:
    ctx = args.recall_context
    bs = args.recall_batch
    inp = build_decode_inputs(cfg, batch_size=bs, context_len=ctx)
    if not args.iid:
        make_clustered_index_keys(
            inp.idx_k_cache,
            inp.req_to_token,
            inp.seq_lens,
            block=args.pool_block,
            noise=args.cluster_noise,
        )

    scale = cfg.idx_head_dim**-0.5
    # The exact flat top-k, per (request, kv head), computed once in torch.
    exact: dict[tuple[int, int], torch.Tensor] = {}
    base_cfg = _tl_config(cfg, args, ctx)
    for b in range(bs):
        slots = inp.req_to_token[b, :ctx].long()
        k = inp.idx_k_cache[slots, 0].float()
        for kh in range(cfg.num_kv_heads):
            group = cfg.num_idx_heads // cfg.num_kv_heads
            q_row = inp.idx_q[b, kh * group : (kh + 1) * group]
            exact[(b, kh)] = reference_exact_topk(
                q_row,
                k,
                ctx - 1,
                topk=base_cfg.topk_tokens,
                sm_scale=scale,
                cfg=base_cfg,
            ).cpu()
        del k
    torch.cuda.empty_cache()

    print(
        f"recall vs exact flat top-{base_cfg.topk_tokens} at ctx={_ctx_label(ctx)}, "
        f"{'iid' if args.iid else f'clustered (noise {args.cluster_noise})'} keys"
    )
    for m in args.coarse_sweep:
        if m * args.pool_block < base_cfg.topk_tokens:
            continue  # cannot even fill the budget
        tl_cfg = _tl_config(cfg, args, ctx, coarse_blocks=m)
        pooled = _pooled_for(tl_cfg, inp, ctx)
        fn = _two_level_decode(cfg, tl_cfg, inp, pooled)
        got = fn().cpu()
        timing = bench_cuda(fn, warmup=5, iters=args.decode_iters)

        recalls = [
            selection_recall(got[kh, b], exact[(b, kh)])[0]
            for b in range(bs)
            for kh in range(cfg.num_kv_heads)
        ]
        mean_recall = sum(recalls) / len(recalls)
        rows.append(
            {
                "impl": "two_level",
                "phase": "decode",
                "context_len": ctx,
                "batch_size": bs,
                "pool_block": args.pool_block,
                "coarse_blocks": m,
                "candidate_width": tl_cfg.candidate_width,
                "candidate_fraction": round(tl_cfg.candidate_width / ctx, 5),
                "topk": tl_cfg.topk_tokens,
                "mean_recall": round(mean_recall, 5),
                "min_recall": round(min(recalls), 5),
                "latency_median_ms": round(timing.median_ms, 6),
                "keys": "iid" if args.iid else "clustered",
            }
        )
        print(
            f"  M={m:<5} candidates={tl_cfg.candidate_width:<8} "
            f"({tl_cfg.candidate_width / ctx * 100:5.2f}% of ctx)  "
            f"recall {mean_recall * 100:6.2f}%  (min {min(recalls) * 100:6.2f}%)  "
            f"{timing.median_ms:8.4f} ms"
        )
        del pooled
        torch.cuda.empty_cache()


def run_recall_prefill(cfg, args, rows: list[dict]) -> None:
    """Does sharing one stage-1 recall across a query tile cost recall?

    The tile's candidate set is the union over its rows *capped at M blocks*, so
    a row loses a block whenever its own preference is crowded out by the rest of
    the tile. How often that happens is entirely a question of how much
    neighbouring queries agree, which the harness does not know — its query rows
    are iid Gaussians and agree about nothing. Both bounds are therefore
    measured: `iid` queries (floor) and `clustered` queries (rows within a tile
    drawn around one centre, the premise every query-block-granular selector
    already relies on).
    """
    ctx, chunk = args.recall_context, 256
    scale = cfg.idx_head_dim**-0.5
    group = cfg.num_idx_heads // cfg.num_kv_heads
    base = _tl_config(cfg, args, ctx)
    sample = list(range(0, chunk, 32))  # one row per tile position

    print(
        f"prefill recall vs exact flat top-{base.topk_tokens} at "
        f"ctx={_ctx_label(ctx)}, {chunk}-token extend, M={base.coarse_blocks}"
    )
    for queries in ("iid", "clustered"):
        inp = build_prefill_inputs(cfg, batch_size=1, context_len=ctx, chunk_len=chunk)
        if not args.iid:
            make_clustered_index_keys(
                inp.idx_k_cache,
                inp.req_to_token,
                inp.seq_lens,
                block=args.pool_block,
                noise=args.cluster_noise,
            )
        if queries == "clustered":
            make_clustered_index_queries(
                inp.idx_q, block=args.query_cluster, noise=args.cluster_noise
            )
        prefix = ctx - chunk

        k = inp.idx_k_cache[inp.req_to_token[0, :ctx].long(), 0].float()
        exact = {
            (i, kh): reference_exact_topk(
                inp.idx_q[i, kh * group : (kh + 1) * group],
                k,
                prefix + i,
                topk=base.topk_tokens,
                sm_scale=scale,
                cfg=base,
            ).cpu()
            for i in sample
            for kh in range(cfg.num_kv_heads)
        }
        del k
        torch.cuda.empty_cache()

        for qt in args.query_tile_sweep:
            tl_cfg = base.replace(query_tile=qt)
            pooled = _pooled_for(tl_cfg, inp, ctx)
            fn = _two_level_prefill(cfg, tl_cfg, inp, pooled)
            got = fn().cpu()
            timing = bench_cuda(fn, warmup=3, iters=max(3, args.prefill_iters // 2))
            recalls = [
                selection_recall(got[kh, i], exact[(i, kh)])[0]
                for i in sample
                for kh in range(cfg.num_kv_heads)
            ]
            mean_recall = sum(recalls) / len(recalls)
            rows.append(
                {
                    "impl": "two_level",
                    "phase": "prefill",
                    "context_len": ctx,
                    "batch_size": 1,
                    "rows_per_req": chunk,
                    "pool_block": args.pool_block,
                    "coarse_blocks": base.coarse_blocks,
                    "candidate_width": base.candidate_width,
                    "query_tile": qt,
                    "topk": base.topk_tokens,
                    "mean_recall": round(mean_recall, 5),
                    "min_recall": round(min(recalls), 5),
                    "latency_median_ms": round(timing.median_ms, 6),
                    "keys": "iid" if args.iid else "clustered",
                    "queries": queries,
                }
            )
            print(
                f"  {queries:<10} query_tile={qt:<5} recall {mean_recall * 100:6.2f}%  "
                f"(min {min(recalls) * 100:6.2f}%)  {timing.median_ms:8.4f} ms"
            )
            del pooled
            torch.cuda.empty_cache()
        del inp
        torch.cuda.empty_cache()


def run_coverage(cfg, args, rows: list[dict]) -> None:
    """The one axis on which all three paths are directly comparable.

    Each spends the *same* 2048-token budget; they differ in which 2048 they
    spend it on. Measured against the exact flat top-2048, and against its
    top-256 head — a path can capture every peak and still score badly on the
    full set if it spends the rest of the budget on the peaks' neighbours, which
    is exactly what block granularity does.
    """
    ctx, bs = args.recall_context, args.recall_batch
    scale = cfg.idx_head_dim**-0.5
    budget = cfg.block_token_budget
    pool = cfg.block_size

    inp = build_decode_inputs(cfg, batch_size=bs, context_len=ctx)
    if not args.iid:
        make_clustered_index_keys(
            inp.idx_k_cache,
            inp.req_to_token,
            inp.seq_lens,
            block=pool,
            noise=args.cluster_noise,
        )
    base = _tl_config(cfg, args, ctx)

    exact, head = {}, {}
    group = cfg.num_idx_heads // cfg.num_kv_heads
    for b in range(bs):
        k = inp.idx_k_cache[inp.req_to_token[b, :ctx].long(), 0].float()
        for kh in range(cfg.num_kv_heads):
            q_row = inp.idx_q[b, kh * group : (kh + 1) * group]
            for store, topk in ((exact, budget), (head, 256)):
                store[(b, kh)] = set(
                    reference_exact_topk(
                        q_row, k, ctx - 1, topk=topk, sm_scale=scale, cfg=base
                    )
                    .cpu()
                    .tolist()
                )
        del k
    torch.cuda.empty_cache()

    selections: dict[str, torch.Tensor] = {}

    # block: top-16 blocks, expanded to the tokens they cover
    _, blk, _ = flash_decode_with_topk_idx(
        q=inp.idx_q,
        k_cache=inp.idx_k_cache,
        v_cache=None,
        sink=None,
        req_to_token=inp.req_to_token,
        slot_ids=inp.slot_ids,
        seq_lens=inp.seq_lens,
        max_seqlen=inp.max_seqlen,
        block_size=pool,
        topk=cfg.topk_blocks,
        init_blocks=cfg.init_blocks,
        local_blocks=cfg.local_blocks,
        disable_index_value=True,
        score_type=cfg.score_type,
    )
    off = torch.arange(pool, device=blk.device)
    tok = blk[..., None].long() * pool + off
    live = (blk >= 0)[..., None].expand(*blk.shape, pool)
    tok = torch.where(live & (tok < ctx), tok, torch.full_like(tok, -1))
    selections["block"] = tok.reshape(*blk.shape[:2], -1).cpu()

    selections["flat"] = token_select_decode(
        inp.idx_q,
        inp.idx_k_cache,
        inp.req_to_token,
        inp.slot_ids,
        inp.seq_lens,
        inp.max_seqlen,
        budget,
        cfg.init_tokens,
        cfg.local_tokens,
    ).cpu()

    for position in ("pre", "post"):
        for m in args.coarse_sweep:
            if m * args.pool_block < budget:
                continue
            tl_cfg = base.replace(coarse_blocks=m, pool_position=position)
            pooled = _pooled_for(tl_cfg, inp, ctx) if position == "pre" else None
            selections[f"two_level_{position}_M{m}"] = two_level_select_decode(
                inp.idx_q,
                inp.idx_k_cache,
                inp.req_to_token,
                inp.slot_ids,
                inp.seq_lens,
                inp.max_seqlen,
                num_kv_heads=cfg.num_kv_heads,
                cfg=tl_cfg,
                pooled=pooled,
            ).cpu()
            del pooled

    keys = "iid" if args.iid else "clustered"
    print(
        f"same {budget}-token budget at ctx={_ctx_label(ctx)}, {keys} keys — "
        f"what fraction of the exact top-{budget} does each attend to?"
    )
    print(
        f"  {'path':<18} {'attended':>9} {'of top-' + str(budget):>10} "
        f"{'of top-256':>11}"
    )
    for name, sel in selections.items():
        counts, recall, head_recall = [], [], []
        for b in range(bs):
            for kh in range(cfg.num_kv_heads):
                row = {int(x) for x in sel[kh, b].tolist() if x >= 0}
                counts.append(len(row))
                recall.append(len(row & exact[(b, kh)]) / max(1, len(exact[(b, kh)])))
                head_recall.append(
                    len(row & head[(b, kh)]) / max(1, len(head[(b, kh)]))
                )
        row_out = {
            "impl": name,
            "phase": "decode",
            "metric": "coverage",
            "coarse_blocks": int(name.split("_M")[-1]) if "_M" in name else None,
            "pool_position": (
                ("post" if "_post" in name else "pre")
                if name.startswith("two_level")
                else None
            ),
            "context_len": ctx,
            "batch_size": bs,
            "budget_tokens": budget,
            "attended_tokens": round(sum(counts) / len(counts), 1),
            "recall_of_exact": round(sum(recall) / len(recall), 5),
            "recall_of_top256": round(sum(head_recall) / len(head_recall), 5),
            "keys": keys,
        }
        rows.append(row_out)
        print(
            f"  {name:<18} {row_out['attended_tokens']:9.0f} "
            f"{row_out['recall_of_exact'] * 100:9.2f}% "
            f"{row_out['recall_of_top256'] * 100:10.2f}%"
        )
    del inp, selections
    torch.cuda.empty_cache()


def run_pooling(cfg, args, rows: list[dict]) -> None:
    """Pool before the dot (LongCat) or after it (M3's block indexer)?

    The two differ in what stage 1 costs and in how well it ranks blocks, so
    both have to be measured together: latency at each context, and coverage of
    the exact top-k at the same M.
    """
    budget = cfg.block_token_budget
    scale = cfg.idx_head_dim**-0.5
    group = cfg.num_idx_heads // cfg.num_kv_heads
    bs = args.recall_batch

    for ctx in args.context_lens:
        base = _tl_config(cfg, args, ctx)
        chunk = min(args.prefill_chunk, ctx)
        print(
            f"--- ctx={_ctx_label(ctx)} "
            f"(stage 1 reads {math.ceil(ctx / base.pool_block_size):,} pooled keys "
            f"if pre, {ctx:,} real keys if post)"
        )

        # latency, both phases
        pre_in = build_prefill_inputs(
            cfg, batch_size=args.prefill_batch_size, context_len=ctx, chunk_len=chunk
        )
        if not args.iid:
            make_clustered_index_keys(
                pre_in.idx_k_cache,
                pre_in.req_to_token,
                pre_in.seq_lens,
                block=args.pool_block,
                noise=args.cluster_noise,
            )
        for position in ("pre", "post"):
            tl_cfg = base.replace(pool_position=position)
            pooled = _pooled_for(tl_cfg, pre_in, ctx) if position == "pre" else None
            rows.append(
                _time_one(
                    _two_level_prefill(cfg, tl_cfg, pre_in, pooled),
                    impl=f"two_level_{position}",
                    phase="prefill",
                    ctx=ctx,
                    cfg=cfg,
                    tl_cfg=tl_cfg,
                    batch=args.prefill_batch_size,
                    rows_per_req=chunk,
                    iters=args.prefill_iters,
                    extra={"pool_position": position},
                )
            )
            del pooled
        del pre_in
        torch.cuda.empty_cache()

        dec_in = build_decode_inputs(cfg, batch_size=bs, context_len=ctx)
        if not args.iid:
            make_clustered_index_keys(
                dec_in.idx_k_cache,
                dec_in.req_to_token,
                dec_in.seq_lens,
                block=args.pool_block,
                noise=args.cluster_noise,
            )
        # the exact selection this context's coverage is scored against
        exact = {}
        for b in range(bs):
            k = dec_in.idx_k_cache[dec_in.req_to_token[b, :ctx].long(), 0].float()
            for kh in range(cfg.num_kv_heads):
                exact[(b, kh)] = set(
                    reference_exact_topk(
                        dec_in.idx_q[b, kh * group : (kh + 1) * group],
                        k,
                        ctx - 1,
                        topk=budget,
                        sm_scale=scale,
                        cfg=base,
                    )
                    .cpu()
                    .tolist()
                )
            del k
        torch.cuda.empty_cache()

        for position in ("pre", "post"):
            tl_cfg = base.replace(pool_position=position)
            pooled = _pooled_for(tl_cfg, dec_in, ctx) if position == "pre" else None
            fn = _two_level_decode(cfg, tl_cfg, dec_in, pooled)
            row = _time_one(
                fn,
                impl=f"two_level_{position}",
                phase="decode",
                ctx=ctx,
                cfg=cfg,
                tl_cfg=tl_cfg,
                batch=bs,
                rows_per_req=1,
                iters=args.decode_iters,
                extra={"pool_position": position},
            )
            sel = fn().cpu()
            recalls = [
                len({int(x) for x in sel[kh, b].tolist() if x >= 0} & exact[(b, kh)])
                / max(1, len(exact[(b, kh)]))
                for b in range(bs)
                for kh in range(cfg.num_kv_heads)
            ]
            row["mean_recall"] = round(sum(recalls) / len(recalls), 5)
            row["min_recall"] = round(min(recalls), 5)
            rows.append(row)
            print(
                f"    {position:<4} coverage of the exact top-{budget}: "
                f"{row['mean_recall'] * 100:6.2f}%  (worst row "
                f"{row['min_recall'] * 100:6.2f}%)"
            )
            del pooled
        del dec_in, exact
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# mode: e2e (selection + sparse GQA attention)
# ---------------------------------------------------------------------------


def _kv_pool_bytes(cfg, ctx: int, batch: int) -> int:
    """Device bytes the harness's main + index KV pools need for one point."""
    per_token = cfg.kv_bytes_per_token_per_layer()["total"]
    return batch * ctx * per_token


def run_e2e(cfg, args, rows: list[dict]) -> None:
    free, _ = torch.cuda.mem_get_info()
    for ctx in args.context_lens:
        tl_cfg = _tl_config(cfg, args, ctx)
        for bs in args.decode_batch_sizes:
            # e2e needs the *main* KV cache too, which is 32 GiB per tensor at
            # 1M x batch 32. Skip rather than OOM mid-sweep, and say so: a
            # silently dropped point reads as "measured and unremarkable".
            need = _kv_pool_bytes(cfg, ctx, bs)
            if need > 0.6 * free:
                print(
                    f"  skipped ctx={_ctx_label(ctx)} bs={bs}: its KV pools need "
                    f"{need / 2**30:.1f} GiB of {free / 2**30:.1f} GiB free"
                )
                continue
            inp = build_decode_inputs(cfg, batch_size=bs, context_len=ctx)
            if not args.iid:
                make_clustered_index_keys(
                    inp.idx_k_cache,
                    inp.req_to_token,
                    inp.seq_lens,
                    block=args.pool_block,
                )
            pooled = _pooled_for(tl_cfg, inp, ctx)
            q_slot_ids = inp.slot_ids.to(torch.int32)

            def _attn(select: Callable) -> Callable:
                def run():
                    topk_idx = select()
                    return gqa_token_sparse_attn(
                        inp.q,
                        inp.k_cache,
                        inp.v_cache,
                        inp.req_to_token,
                        q_slot_ids,
                        topk_idx,
                    )

                return run

            for impl, select in (
                ("flat", _flat_decode(cfg, inp)),
                ("two_level", _two_level_decode(cfg, tl_cfg, inp, pooled)),
            ):
                rows.append(
                    _time_one(
                        _attn(select),
                        impl=impl,
                        phase="decode_e2e",
                        ctx=ctx,
                        cfg=cfg,
                        tl_cfg=tl_cfg,
                        batch=bs,
                        rows_per_req=1,
                        iters=args.decode_iters,
                    )
                )
            del inp, pooled
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


# Validated categorical slots, assigned by entity and held across every plot
# here and in plot_indexer_comparison.py: a reader who learns "two-level is blue"
# keeps that. `two_level_rebuild` is the same path under a different pool policy,
# so it takes a lighter step of the same hue rather than a fourth identity.
COLORS = {
    "block": "#eb6834",  # slot 2, orange
    "flat": "#1baf7a",  # slot 3, aqua
    "two_level": "#2a78d6",  # slot 1, blue
    "two_level_rebuild": "#86b6ef",  # blue, lighter step
}


def make_plots(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _style(ax, ctxs, ylabel, title):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ctxs)
        ax.set_xticklabels([_ctx_label(c) for c in ctxs])
        ax.minorticks_off()
        ax.set_xlabel("context length")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for phase in ("prefill", "decode", "decode_e2e"):
        # latency rows only: the recall sweep shares the phase names but carries
        # no comparison series, and would draw a one-point line per impl.
        sub = [r for r in rows if r.get("phase") == phase and "gpu_kernel_ms" in r]
        if not sub:
            continue
        for bs in sorted({r["batch_size"] for r in sub}):
            pts = [r for r in sub if r["batch_size"] == bs]
            ctxs = sorted({r["context_len"] for r in pts})
            fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
            for impl, colour in COLORS.items():
                series = sorted(
                    (r["context_len"], r["latency_median_ms"])
                    for r in pts
                    if r["impl"] == impl
                )
                if not series:
                    continue
                ax.plot(
                    [p[0] for p in series],
                    [p[1] for p in series],
                    marker="o",
                    markersize=5,
                    linewidth=1.8,
                    color=colour,
                    linestyle=(0, (6, 3)) if impl.endswith("rebuild") else "-",
                    label=impl,
                )
            _style(
                ax, ctxs, "median ms (log)", f"{phase} selection latency, batch {bs}"
            )
            fig.tight_layout()
            fig.savefig(out / f"{phase}_bs{bs}_latency.png")
            plt.close(fig)

    recall = [r for r in rows if "mean_recall" in r and "query_tile" not in r]
    if recall:
        fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
        pts = sorted(
            (r["candidate_width"], r["mean_recall"], r["latency_median_ms"])
            for r in recall
        )
        ax.plot(
            [p[0] for p in pts],
            [p[1] * 100 for p in pts],
            marker="o",
            color="#d1571f",
            linewidth=1.8,
            label="recall",
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("candidate tokens recalled by stage 1 (M x P)")
        ax.set_ylabel("recall of exact top-k (%)")
        ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
        ax2 = ax.twinx()
        ax2.plot(
            [p[0] for p in pts],
            [p[2] for p in pts],
            marker="s",
            markersize=4,
            color="#2a78d6",
            linewidth=1.4,
            linestyle="--",
            label="latency",
        )
        ax2.set_ylabel("median ms")
        ax.set_title(
            f"Coarse recall budget vs accuracy, ctx={_ctx_label(recall[0]['context_len'])}",
            loc="left",
            fontsize=11,
        )
        for s in ("top",):
            ax.spines[s].set_visible(False)
            ax2.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "recall_vs_budget.png")
        plt.close(fig)

    qt_recall = [r for r in rows if "query_tile" in r]
    if qt_recall:
        fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
        for queries, colour in (("clustered", "#2a78d6"), ("iid", "#eb6834")):
            pts = sorted(
                (r["query_tile"], r["mean_recall"], r["min_recall"])
                for r in qt_recall
                if r.get("queries") == queries
            )
            if not pts:
                continue
            ax.plot(
                [p[0] for p in pts],
                [p[1] * 100 for p in pts],
                marker="o",
                color=colour,
                linewidth=1.8,
                label=f"{queries} queries (mean)",
            )
            ax.fill_between(
                [p[0] for p in pts],
                [p[2] * 100 for p in pts],
                [p[1] * 100 for p in pts],
                color=colour,
                alpha=0.12,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({r["query_tile"] for r in qt_recall}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("query_tile — query rows sharing one stage-1 recall")
        ax.set_ylabel("recall of exact top-k (%), band down to worst row")
        ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(
            "Query-tile sharing vs recall — the bet the prefill win rests on",
            loc="left",
            fontsize=11,
        )
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "recall_vs_query_tile.png")
        plt.close(fig)

    stages = [r for r in rows if r.get("stage")]
    if stages:
        fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
        ctxs = sorted({r["context_len"] for r in stages})
        big_batch = max(r["batch_size"] for r in stages)
        for name, colour in (
            ("stage0_pool_rebuild", "#898781"),
            ("stage1_coarse_score", "#2a78d6"),
            ("stage2_fine_score", "#d1571f"),
            ("select", "#1baf7a"),
            ("host_launch_gap", "#e0a458"),
        ):
            series = sorted(
                (r["context_len"], r["latency_median_ms"])
                for r in stages
                if r["stage"] == name and r["batch_size"] == big_batch
            )
            if series:
                ax.plot(
                    [p[0] for p in series],
                    [p[1] for p in series],
                    marker="o",
                    color=colour,
                    linewidth=1.8,
                    label=name,
                )
        _style(
            ax,
            ctxs,
            "median ms (log)",
            f"Two-level indexer, per-stage decode cost (batch {big_batch})",
        )
        fig.tight_layout()
        fig.savefig(out / "stage_breakdown.png")
        plt.close(fig)


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode", default="latency", help="comma-separated: latency,stages,recall,e2e"
    )
    p.add_argument("--context-lens", type=_int_list, default=DEFAULT_CTXS)
    p.add_argument("--decode-batch-sizes", type=_int_list, default=[1, 32])
    p.add_argument("--prefill-batch-size", type=int, default=1)
    p.add_argument("--prefill-chunk", type=int, default=2048)
    p.add_argument(
        "--pool-block",
        type=int,
        default=128,
        help="P: tokens per coarse block (stage 0 pooling granularity)",
    )
    p.add_argument(
        "--coarse-blocks",
        type=int,
        default=128,
        help="M: blocks stage 1 recalls for stage 2",
    )
    p.add_argument("--head-reduce", default="max", choices=("max", "sum", "relu_sum"))
    p.add_argument(
        "--query-tile",
        type=int,
        default=64,
        help="query rows sharing one stage-1 recall (prefill only)",
    )
    p.add_argument(
        "--pool-position",
        default="pre",
        choices=("pre", "post"),
        help="stage 1 pools keys before the q.k dot (LongCat), or scores every "
        "key and pools the scores (M3's block indexer). --mode pooling runs both",
    )
    p.add_argument(
        "--share-candidates",
        action="store_true",
        help="one candidate set for all index heads (LongCat-style)",
    )
    p.add_argument(
        "--coarse-sweep",
        type=_int_list,
        default=DEFAULT_MS,
        help="M values for --mode recall",
    )
    p.add_argument(
        "--query-tile-sweep",
        type=_int_list,
        default=[1, 16, 64, 128],
        help="query_tile values for the prefill half of --mode recall",
    )
    p.add_argument("--recall-context", type=int, default=DEFAULT_RECALL_CTX)
    p.add_argument("--recall-batch", type=int, default=2)
    p.add_argument("--cluster-noise", type=float, default=0.35)
    p.add_argument(
        "--query-cluster",
        type=int,
        default=64,
        help="rows per query cluster in the clustered-query recall bound",
    )
    p.add_argument(
        "--iid",
        action="store_true",
        help="leave the index cache iid Gaussian (recall floor, not a forecast)",
    )
    p.add_argument("--prefill-iters", type=int, default=10)
    p.add_argument("--decode-iters", type=int, default=50)
    p.add_argument("--wait-for-idle", type=float, default=0.0)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "two_level_indexer",
    )
    args = p.parse_args(argv)

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    cfg = m3_config()

    torch.cuda.init()
    wait_for_idle(args.wait_for_idle)
    warn_if_contended()
    args.out.mkdir(parents=True, exist_ok=True)

    print("MiniMax-M3 two-level (block -> token) indexer prototype")
    print(f"  device        : {torch.cuda.get_device_name(0)}")
    print(
        f"  shape         : {cfg.num_q_heads}Q/{cfg.num_kv_heads}KV heads, "
        f"{cfg.num_idx_heads} index heads x {cfg.idx_head_dim}"
    )
    print(
        f"  budget        : top-{cfg.block_token_budget} tokens "
        f"(init {cfg.init_tokens}, local {cfg.local_tokens})"
    )
    print(
        f"  two-level     : P={args.pool_block}, M={args.coarse_blocks} "
        f"-> {args.pool_block * args.coarse_blocks} candidates, "
        f"head reduce '{args.head_reduce}'"
    )
    print(f"  index keys    : {'iid gaussian' if args.iid else 'clustered'}")
    print(
        f"  query tile    : {args.query_tile} rows per stage-1 recall"
        f"{', shared across index heads' if args.share_candidates else ''}"
    )

    rows: list[dict] = []
    for mode in modes:
        print(f"\n== {mode} ==")
        if mode == "latency":
            run_latency(cfg, args, rows)
        elif mode == "stages":
            run_stages(cfg, args, rows)
        elif mode == "recall":
            run_recall(cfg, args, rows)
            run_recall_prefill(cfg, args, rows)
        elif mode == "coverage":
            run_coverage(cfg, args, rows)
        elif mode == "pooling":
            run_pooling(cfg, args, rows)
        elif mode == "e2e":
            run_e2e(cfg, args, rows)
        else:
            raise SystemExit(f"unknown mode {mode!r}")

    # A shared box can hand the GPU to someone else *during* a run, which the
    # start-of-run idle check cannot see. Don't re-sample utilization to detect
    # it — right after the last kernel that reads back our own residual work —
    # use the self-referential signal instead: profiled kernel time is a subset
    # of wall clock, so a point reporting materially more of it was timed while
    # something else held the device. (The 5% band is profiler overhead on the
    # kernel-time side; a contended run overshoots by far more.)
    timed = [r for r in rows if "gpu_kernel_ms" in r]
    suspect = [r for r in timed if r["gpu_kernel_ms"] > 1.05 * r["latency_median_ms"]]
    if suspect:
        print(
            f"\n  !! {len(suspect)} of {len(timed)} timed points report more kernel "
            f"time than wall clock, which only happens when the device was "
            f"shared mid-run.\n"
            f"  !! Worst: {max(suspect, key=lambda r: r['gpu_kernel_ms'] / r['latency_median_ms'])['impl']} "
            f"@ {_ctx_label(max(suspect, key=lambda r: r['gpu_kernel_ms'] / r['latency_median_ms'])['context_len'])}. "
            f"Do not quote these numbers; re-run pinned to an idle GPU.\n"
        )

    write_results(rows, args.out, "raw")
    if not args.no_plots:
        make_plots(rows, args.out)
    print(
        f"\nwrote {args.out}/raw.json + raw.csv" + ("" if args.no_plots else " + plots")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
