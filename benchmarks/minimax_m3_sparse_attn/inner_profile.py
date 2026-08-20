"""Intra-kernel breakdown of the indexer score kernels, via Proton.

``bench_indexer.py`` splits the indexer into whole-kernel stages; this goes one
level deeper and attributes *warp cycles inside the score kernel* to its inner
operations — page-table lookup, index-K load, q.k dot, bias/masking, block
pooling (block granularity only), and the score store — using Triton's Proton
instrumentation backend (``pl.scope`` regions compiled into a bench-local copy
of each kernel).

The block copies are specialized to how the bench (and the M3 backend) drives
them: ``disable_index_value=True``, no sink, no gumbel — those constexpr-dead
branches are stripped rather than transcribed. They also carry a
``WRITE_SCORES`` constexpr like the token kernels' production hook, which is
what upgrades the block compute-vs-write-back split from estimated to measured
(``block_score_write_split``).

Caveats, and why this lives next to the bench instead of in the kernels:

* Proton disables intra-kernel scopes for Triton-DSL kernels by default because
  the compiler pipelines and reorders across them; we opt in with
  ``pl.enable_semantic("triton")``. The numbers are *warp residency per region*,
  not clean serial phases: a warp stalled on a K load that the compiler hoisted
  can be booked to the region where the value is first used.
* The instrumented copies run the same tile config the production autotuner
  picked (read from ``best_config``), so shapes match, but the instrumentation
  itself costs cycles — use the per-region *shares*, scaled onto the
  uninstrumented kernel time, not the raw instrumented wall time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.runtime.autotuner import Autotuner

from sglang.kernels.ops.attention.minimax_sparse.common.utils import robust_allocator
from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    _decode_score_kernel,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
    _flash_attn_fwd_with_block_score_kernel,
)
from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (
    _token_index_score_decode_kernel,
    _token_index_score_prefill_kernel,
    plan_key_window,
    plan_query_chunk,
)
from sglang.kernels.ops.attention.minimax_sparse.token.index_score_seg import (
    _plan_segments,
    _token_index_score_prefill_kernel_seg,
)
from sglang.srt.environ import envs

INIT_BIAS = tl.constexpr(1e30)
LOCAL_BIAS = tl.constexpr(1e29)

# Region names, in kernel order. `q_load` also covers the per-request scalar
# loads (seqlens etc.), which are a handful of cycles. `pool` (the per-block
# max/lse reduction) only exists in the block-granularity kernels.
SCOPES = ("q_load", "page_table", "k_load", "qk_dot", "mask_bias", "pool", "score_store")


@triton.jit
def _instr_token_score_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    score_ptr,
    req_to_token_ptr,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    slot_ids,
    max_slots,
    num_idx_heads,
    qk_head_dim,
    kv_width,
    chunk_cap,
    q_chunk_start,
    batch_start,
    init_tokens,
    local_tokens,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_r2t_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    # Verbatim copy of _token_index_score_prefill_kernel (WRITE_SCORES=True
    # path) with pl.scope regions; keep in sync with index_score.py.
    pid_q, pid_bh, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pid_b_local = pid_bh // num_idx_heads
    pid_b = batch_start + pid_b_local
    pid_h = pid_bh % num_idx_heads

    sm_scale_log2e = sm_scale * 1.4426950408889634

    with pl.scope("q_load"):
        seq_start = tl.load(cu_seqlens + pid_b)
        q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
        seq_len = tl.load(seq_lens + pid_b)
        prefix_len = tl.load(prefix_lens + pid_b)
        sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots

    q_lo = q_chunk_start + pid_q * BLOCK_SIZE_Q
    q_hi = tl.minimum(tl.minimum(q_lo + BLOCK_SIZE_Q, q_len), q_chunk_start + chunk_cap)
    if q_lo >= q_hi:
        return

    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < qk_head_dim

    local_q = q_lo + off_q
    q_mask = local_q < q_hi
    abs_q = prefix_len + local_q

    with pl.scope("q_load"):
        q = tl.load(
            q_ptr
            + (seq_start + local_q)[:, None] * stride_q_n
            + pid_h * stride_q_h
            + off_d[None, :] * stride_q_d,
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

    s_row = pid_b_local * chunk_cap + (local_q - q_chunk_start)
    s_base = score_ptr + pid_h * stride_s_h + s_row[:, None] * stride_s_n

    pos = pid_k * BLOCK_SIZE_K + off_k
    in_width = pos < kv_width
    pos_mask = (pos < seq_len) & in_width
    with pl.scope("page_table"):
        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        slots = (slots + max_slots) % max_slots
    with pl.scope("k_load"):
        k = tl.load(
            k_cache_ptr + slots[None, :] * stride_k_s + off_d[:, None] * stride_k_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if IS_FP8:
            k = k.to(q.dtype)

    with pl.scope("qk_dot"):
        score = tl.dot(q, k) * sm_scale_log2e

    with pl.scope("mask_bias"):
        causal = (abs_q[:, None] >= pos[None, :]) & pos_mask[None, :]
        init_m = causal & (pos[None, :] < init_tokens)
        local_m = causal & (pos[None, :] > (abs_q[:, None] - local_tokens))
        score = tl.where(init_m, INIT_BIAS, score)
        score = tl.where(local_m & (init_m == 0), LOCAL_BIAS, score)
        score = tl.where(causal, score, float("-inf"))

    with pl.scope("score_store"):
        tl.store(
            s_base + pos[None, :] * stride_s_k,
            score,
            mask=q_mask[:, None] & in_width[None, :],
        )


@triton.jit
def _instr_token_score_decode_kernel(
    q_ptr,
    k_cache_ptr,
    score_ptr,
    req_to_token_ptr,
    seq_lens,
    slot_ids,
    max_slots,
    batch_size,
    num_idx_heads,
    qk_head_dim,
    k_base,
    k_limit,
    win_width,
    init_tokens,
    local_tokens,
    sm_scale,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_d,
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_r2t_b,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    # Verbatim copy of _token_index_score_decode_kernel (WRITE_SCORES=True
    # path) with pl.scope regions; keep in sync with index_score.py.
    pid_bc = tl.program_id(0)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size

    sm_scale_log2e = sm_scale * 1.4426950408889634

    with pl.scope("q_load"):
        seq_len = tl.load(seq_lens + pid_b)
        sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots
    abs_q = seq_len - 1

    chunk_size = (win_width + NUM_KV_CHUNKS - 1) // NUM_KV_CHUNKS
    k_start = k_base + pid_c * chunk_size
    k_end = tl.minimum(k_start + chunk_size, k_limit)
    if k_start >= k_end:
        return

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    h_mask = off_h < num_idx_heads
    d_mask = off_d < qk_head_dim

    with pl.scope("q_load"):
        q = tl.load(
            q_ptr
            + pid_b * stride_q_b
            + off_h[:, None] * stride_q_h
            + off_d[None, :] * stride_q_d,
            mask=h_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

    s_base = score_ptr + off_h[:, None] * stride_s_h + pid_b * stride_s_b

    for i in tl.range(k_start, k_end, BLOCK_SIZE_K):
        pos = i + off_k
        in_width = pos < k_end
        pos_mask = (pos < seq_len) & in_width
        with pl.scope("page_table"):
            slots = tl.load(
                req_to_token_ptr + sid * stride_r2t_b + pos,
                mask=pos_mask,
                other=0,
            ).to(tl.int64)
            slots = (slots + max_slots) % max_slots
        with pl.scope("k_load"):
            k = tl.load(
                k_cache_ptr
                + slots[None, :] * stride_k_s
                + off_d[:, None] * stride_k_d,
                mask=d_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if IS_FP8:
                k = k.to(q.dtype)

        with pl.scope("qk_dot"):
            score = tl.dot(q, k) * sm_scale_log2e

        with pl.scope("mask_bias"):
            init_m = pos_mask[None, :] & (pos[None, :] < init_tokens)
            local_m = pos_mask[None, :] & (pos[None, :] > abs_q - local_tokens)
            score = tl.where(init_m, INIT_BIAS, score)
            score = tl.where(local_m & (init_m == 0), LOCAL_BIAS, score)
            score = tl.where(pos_mask[None, :], score, float("-inf"))

        with pl.scope("score_store"):
            tl.store(
                s_base + (pos - k_base)[None, :] * stride_s_k,
                score,
                mask=h_mask[:, None] & in_width[None, :],
            )


@triton.jit
def _instr_token_seg_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    score_ptr,
    req_to_token_ptr,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    slot_ids,
    max_slots,
    num_idx_heads,
    qk_head_dim,
    kv_width,
    chunk_cap,
    q_chunk_start,
    batch_start,
    num_segments,
    init_tokens,
    local_tokens,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_r2t_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    # Verbatim copy of index_score_seg's interleaved segmented scorer with
    # pl.scope regions; keep in sync with index_score_seg.py.
    pid_q, pid_bh, pid_seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pid_b_local = pid_bh // num_idx_heads
    pid_b = batch_start + pid_b_local
    pid_h = pid_bh % num_idx_heads
    sm_scale_log2e = sm_scale * 1.4426950408889634
    pl.enter_scope("q_load")
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots
    pl.exit_scope("q_load")
    q_lo = q_chunk_start + pid_q * BLOCK_SIZE_Q
    q_hi = tl.minimum(tl.minimum(q_lo + BLOCK_SIZE_Q, q_len), q_chunk_start + chunk_cap)
    if q_lo >= q_hi:
        return
    tiles_total = tl.cdiv(kv_width, BLOCK_SIZE_K)
    if pid_seg >= tiles_total:
        return
    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < qk_head_dim
    local_q = q_lo + off_q
    q_mask = local_q < q_hi
    abs_q = prefix_len + local_q
    pl.enter_scope("q_load")
    q = tl.load(
        q_ptr
        + (seq_start + local_q)[:, None] * stride_q_n
        + pid_h * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    pl.exit_scope("q_load")
    s_row = pid_b_local * chunk_cap + (local_q - q_chunk_start)
    s_base = score_ptr + pid_h * stride_s_h + s_row[:, None] * stride_s_n
    for t in tl.range(pid_seg, tiles_total, num_segments):
        pos = t * BLOCK_SIZE_K + off_k
        in_width = pos < kv_width
        pos_mask = (pos < seq_len) & in_width
        pl.enter_scope("page_table")
        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        slots = (slots + max_slots) % max_slots
        pl.exit_scope("page_table")
        pl.enter_scope("k_load")
        k = tl.load(
            k_cache_ptr + slots[None, :] * stride_k_s + off_d[:, None] * stride_k_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if IS_FP8:
            k = k.to(q.dtype)
        pl.exit_scope("k_load")
        pl.enter_scope("qk_dot")
        score = tl.dot(q, k) * sm_scale_log2e
        pl.exit_scope("qk_dot")
        pl.enter_scope("mask_bias")
        causal = (abs_q[:, None] >= pos[None, :]) & pos_mask[None, :]
        init_m = causal & (pos[None, :] < init_tokens)
        local_m = causal & (pos[None, :] > (abs_q[:, None] - local_tokens))
        score = tl.where(init_m, INIT_BIAS, score)
        score = tl.where(local_m & (init_m == 0), LOCAL_BIAS, score)
        score = tl.where(causal, score, float("-inf"))
        pl.exit_scope("mask_bias")
        pl.enter_scope("score_store")
        tl.store(
            s_base + pos[None, :] * stride_s_k,
            score,
            mask=q_mask[:, None] & in_width[None, :],
        )
        pl.exit_scope("score_store")


def _best_config(kernel, defaults: dict) -> tuple[dict, int, int]:
    """(kwargs, num_warps, num_stages) the production autotuner settled on.

    The production kernels are ``heuristics(autotune(jit))`` wrappers, so
    ``best_config`` lives on the inner Autotuner — walk the ``fn`` chain to it.
    The bench always times the production kernel first, so ``best_config`` is
    the config tuned for exactly this point. Fall back to sane defaults if the
    instrumented path somehow runs first.
    """
    obj = kernel
    while obj is not None and not isinstance(obj, Autotuner):
        obj = getattr(obj, "fn", None)
    cfg = getattr(obj, "best_config", None)
    if cfg is None:
        return defaults, 4, 2
    return dict(cfg.kwargs), cfg.num_warps, cfg.num_stages


def _launch_prefill(inp, *, init_tokens: int, local_tokens: int, num_stages: Optional[int]) -> None:
    idx_q, idx_k = inp.idx_q, inp.idx_k_cache
    total_q, num_heads, head_dim = idx_q.shape
    batch_size = inp.cu_seqlens.shape[0] - 1
    chunk_len, context_len = inp.max_seqlen_q, inp.max_seqlen_k
    prefix = max(inp.prefix_lens_cpu)
    kwargs, num_warps, tuned_stages = _best_config(
        _token_index_score_prefill_kernel,
        {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128},
    )
    batch_cap, chunk_cap = plan_query_chunk(
        batch_size=batch_size,
        max_seqlen_q=chunk_len,
        max_seqlen_k=context_len,
        num_idx_heads=num_heads,
    )
    bq, bk = kwargs["BLOCK_SIZE_Q"], kwargs["BLOCK_SIZE_K"]
    for b_start in range(0, batch_size, batch_cap):
        b_count = min(batch_cap, batch_size - b_start)
        for q_start in range(0, chunk_len, chunk_cap):
            kv_width = min(context_len, prefix + q_start + chunk_cap)
            scores = torch.empty(
                (num_heads, b_count * chunk_cap, kv_width),
                dtype=torch.float32,
                device=idx_q.device,
            )
            grid = (
                triton.cdiv(chunk_cap, bq),
                b_count * num_heads,
                triton.cdiv(kv_width, bk),
            )
            _instr_token_score_prefill_kernel[grid](
                idx_q,
                idx_k,
                scores,
                inp.req_to_token,
                inp.cu_seqlens,
                inp.seq_lens,
                inp.prefix_lens,
                inp.slot_ids,
                idx_k.shape[0],
                num_heads,
                head_dim,
                kv_width,
                chunk_cap,
                q_start,
                b_start,
                init_tokens,
                local_tokens,
                head_dim**-0.5,
                idx_q.stride(0),
                idx_q.stride(1),
                idx_q.stride(2),
                idx_k.stride(0),
                idx_k.stride(2),
                scores.stride(0),
                scores.stride(1),
                scores.stride(2),
                inp.req_to_token.stride(0),
                BLOCK_SIZE_Q=bq,
                BLOCK_SIZE_K=bk,
                BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
                IS_FP8=idx_k.dtype != idx_q.dtype,
                num_warps=num_warps,
                num_stages=num_stages if num_stages is not None else tuned_stages,
            )
            del scores


def _launch_prefill_seg(
    inp, *, init_tokens: int, local_tokens: int, num_stages: Optional[int]
) -> None:
    idx_q, idx_k = inp.idx_q, inp.idx_k_cache
    total_q, num_heads, head_dim = idx_q.shape
    batch_size = inp.cu_seqlens.shape[0] - 1
    chunk_len, context_len = inp.max_seqlen_q, inp.max_seqlen_k
    prefix = max(inp.prefix_lens_cpu)
    kwargs, num_warps, tuned_stages = _best_config(
        _token_index_score_prefill_kernel_seg,
        {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128},
    )
    batch_cap, chunk_cap = plan_query_chunk(
        batch_size=batch_size,
        max_seqlen_q=chunk_len,
        max_seqlen_k=context_len,
        num_idx_heads=num_heads,
    )
    bq, bk = kwargs["BLOCK_SIZE_Q"], kwargs["BLOCK_SIZE_K"]
    for b_start in range(0, batch_size, batch_cap):
        b_count = min(batch_cap, batch_size - b_start)
        for q_start in range(0, chunk_len, chunk_cap):
            kv_width = min(context_len, prefix + q_start + chunk_cap)
            num_segments = _plan_segments(
                chunk_cap=chunk_cap,
                b_count=b_count,
                num_idx_heads=num_heads,
                kv_width=kv_width,
            )
            scores = torch.empty(
                (num_heads, b_count * chunk_cap, kv_width),
                dtype=torch.float32,
                device=idx_q.device,
            )
            grid = (
                triton.cdiv(chunk_cap, bq),
                b_count * num_heads,
                num_segments,
            )
            _instr_token_seg_prefill_kernel[grid](
                idx_q,
                idx_k,
                scores,
                inp.req_to_token,
                inp.cu_seqlens,
                inp.seq_lens,
                inp.prefix_lens,
                inp.slot_ids,
                idx_k.shape[0],
                num_heads,
                head_dim,
                kv_width,
                chunk_cap,
                q_start,
                b_start,
                num_segments,
                init_tokens,
                local_tokens,
                head_dim**-0.5,
                idx_q.stride(0),
                idx_q.stride(1),
                idx_q.stride(2),
                idx_k.stride(0),
                idx_k.stride(2),
                scores.stride(0),
                scores.stride(1),
                scores.stride(2),
                inp.req_to_token.stride(0),
                BLOCK_SIZE_Q=bq,
                BLOCK_SIZE_K=bk,
                BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
                IS_FP8=idx_k.dtype != idx_q.dtype,
                num_warps=num_warps,
                num_stages=num_stages if num_stages is not None else tuned_stages,
            )
            del scores


def _launch_decode(
    inp, *, topk: int, init_tokens: int, local_tokens: int, num_stages: Optional[int]
) -> None:
    idx_q, idx_k = inp.idx_q, inp.idx_k_cache
    batch_size, num_heads, head_dim = idx_q.shape
    kv_width = min(inp.max_seqlen, inp.req_to_token.shape[1])
    kwargs, num_warps, tuned_stages = _best_config(
        _token_index_score_decode_kernel, {"BLOCK_SIZE_K": 128}
    )
    window, num_windows = plan_key_window(
        kv_width=kv_width,
        topk=topk,
        num_idx_heads=num_heads,
        batch_size=batch_size,
    )
    scores = torch.empty(
        (num_heads, batch_size, window), dtype=torch.float32, device=idx_q.device
    )
    target_grid = 2048
    want = max(1, min(64, target_grid // max(1, batch_size)))
    num_kv_chunks = 1 << (want.bit_length() - 1)
    for w in range(num_windows):
        base = w * window
        limit = min(base + window, kv_width)
        _instr_token_score_decode_kernel[(batch_size * num_kv_chunks,)](
            idx_q,
            idx_k,
            scores,
            inp.req_to_token,
            inp.seq_lens,
            inp.slot_ids,
            idx_k.shape[0],
            batch_size,
            num_heads,
            head_dim,
            base,
            limit,
            limit - base,
            init_tokens,
            local_tokens,
            head_dim**-0.5,
            idx_q.stride(0),
            idx_q.stride(1),
            idx_q.stride(2),
            idx_k.stride(0),
            idx_k.stride(2),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            inp.req_to_token.stride(0),
            BLOCK_SIZE_H=max(16, triton.next_power_of_2(num_heads)),
            BLOCK_SIZE_K=kwargs["BLOCK_SIZE_K"],
            BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
            NUM_KV_CHUNKS=num_kv_chunks,
            IS_FP8=idx_k.dtype != idx_q.dtype,
            num_warps=num_warps,
            num_stages=num_stages if num_stages is not None else tuned_stages,
        )
    del scores


@triton.jit
def _instr_block_score_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    score_ptr,
    req_to_token_ptr,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    slot_ids,
    max_slots,
    num_heads,
    gqa_group_size,
    qk_head_dim,
    block_size: tl.constexpr,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_r2t_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    INSTRUMENT: tl.constexpr,
    WRITE_SCORES: tl.constexpr,
):
    # Copy of _flash_attn_fwd_with_block_score_kernel specialized to the M3
    # indexer configuration (DISABLE_INDEX_VALUE=True, no sink, no gumbel);
    # keep in sync with prefill/flash_with_topk_idx.py.
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    sm_scale_log2e = sm_scale * 1.4426950409
    tl.static_assert(BLOCK_SIZE_K >= block_size)
    BLOCKS_PER_K_BLOCK: tl.constexpr = BLOCK_SIZE_K // block_size
    pid_q, pid_bh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    pid_kh = pid_h // gqa_group_size
    if INSTRUMENT:
        pl.enter_scope("q_load")
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots
    if INSTRUMENT:
        pl.exit_scope("q_load")
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return
    block_num = (seq_len + block_size - 1) // block_size
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, qk_head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    if WRITE_SCORES:
        s_ptrs = tl.make_block_ptr(
            base=score_ptr + seq_start * stride_s_q + pid_h * stride_s_h,
            shape=(q_len, block_num),
            strides=(stride_s_q, stride_s_k),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK),
            order=(1, 0),
        )
    if INSTRUMENT:
        pl.enter_scope("q_load")
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    if INSTRUMENT:
        pl.exit_scope("q_load")
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    kd_mask = off_kd < qk_head_dim
    if not WRITE_SCORES:
        obs_max = tl.full((1,), value=float("-inf"), dtype=tl.float32)
    diag_start = (prefix_len + pid_q * BLOCK_SIZE_Q) // BLOCK_SIZE_K * BLOCK_SIZE_K
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        if INSTRUMENT:
            pl.enter_scope("page_table")
        pos = i + off_k
        pos_mask = pos < seq_len
        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        slots = (slots + max_slots) % max_slots
        if INSTRUMENT:
            pl.exit_scope("page_table")
        if INSTRUMENT:
            pl.enter_scope("k_load")
        k = tl.load(
            k_cache_ptr
            + slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_kd[:, None] * stride_k_d,
            mask=kd_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if INSTRUMENT:
            pl.exit_scope("k_load")
        if INSTRUMENT:
            pl.enter_scope("qk_dot")
        qk = tl.dot(q, k) * sm_scale_log2e
        if INSTRUMENT:
            pl.exit_scope("qk_dot")
        if INSTRUMENT:
            pl.enter_scope("mask_bias")
        if i >= diag_start:
            qk = tl.where(off_q[:, None] >= (i + off_k)[None, :], qk, float("-inf"))
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        if INSTRUMENT:
            pl.exit_scope("mask_bias")
        if INSTRUMENT:
            pl.enter_scope("pool")
        score = tl.reshape(
            qk, (BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK, block_size), can_reorder=False
        )
        sub_max = tl.max(score, axis=2)
        if SCORE_TYPE == "max":
            score = sub_max
        else:  # "lse"
            score = sub_max + tl.log2(
                tl.sum(tl.exp2(score - sub_max[:, :, None]), axis=2)
            )
            score = tl.where(score != score, float("-inf"), score)
        if INSTRUMENT:
            pl.exit_scope("pool")
        if WRITE_SCORES:
            if INSTRUMENT:
                pl.enter_scope("score_store")
            tl.store(
                s_ptrs, score.to(score_ptr.dtype.element_ty), boundary_check=(0, 1)
            )
            if INSTRUMENT:
                pl.exit_scope("score_store")
            s_ptrs = tl.advance(s_ptrs, (0, BLOCKS_PER_K_BLOCK))
        else:
            obs_max = tl.maximum(obs_max, tl.max(score))
    if not WRITE_SCORES:
        # One scalar per CTA keeps the compute alive; the grid (q tiles x batch
        # x heads) never exceeds the score buffer's numel (total_q x heads x
        # >=1 block column).
        flat_pid = pid_q + tl.num_programs(0).to(tl.int64) * pid_bh
        tl.store(score_ptr + flat_pid + tl.arange(0, 1), obs_max)


@triton.jit
def _instr_block_score_decode_kernel(
    q_ptr,
    k_cache_ptr,
    req_to_token_ptr,
    score_ptr,
    seq_lens,
    slot_ids,
    max_slots,
    batch_size,
    gqa_group_size,
    head_dim,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    sm_scale,
    init_blocks,
    local_blocks,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_r2t_b,
    stride_s_h,
    stride_s_b,
    stride_s_n,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    SKIP_TRIVIAL_TOPK_SCORE: tl.constexpr,
    INSTRUMENT: tl.constexpr,
    WRITE_SCORES: tl.constexpr,
):
    # Copy of _decode_score_kernel (the disable_index_value=True score-only
    # producer); keep in sync with decode/flash_with_topk_idx.py.
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    sm_scale_log2e = sm_scale * 1.4426950409
    tl.static_assert(BLOCK_SIZE_N >= block_size)
    BLOCKS_PER_K_BLOCK: tl.constexpr = BLOCK_SIZE_N // block_size
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size
    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = (seq_len + block_size - 1) // block_size
    if SKIP_TRIVIAL_TOPK_SCORE:
        if num_blocks <= topk:
            return
    chunk_size_blocks = tl.cdiv(num_blocks, NUM_KV_CHUNKS)
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    chunk_start = chunk_start_block * block_size
    chunk_end = tl.minimum(chunk_end_block * block_size, seq_len)
    if chunk_start_block >= chunk_end_block:
        return
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_q_b + pid_h * stride_q_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_q_h, stride_q_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    if WRITE_SCORES:
        s_ptrs = tl.make_block_ptr(
            base=score_ptr + pid_b * stride_s_b + pid_h * stride_s_h,
            shape=(gqa_group_size, chunk_end_block),
            strides=(stride_s_h, stride_s_n),
            offsets=(0, chunk_start_block),
            block_shape=(BLOCK_SIZE_H, BLOCKS_PER_K_BLOCK),
            order=(1, 0),
        )
    if INSTRUMENT:
        pl.enter_scope("q_load")
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    if INSTRUMENT:
        pl.exit_scope("q_load")
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_bpk = tl.arange(0, BLOCKS_PER_K_BLOCK)
    dim_mask = off_d < head_dim
    local_start = tl.maximum(0, num_blocks - local_blocks)
    if not WRITE_SCORES:
        obs_max = tl.full((1,), value=float("-inf"), dtype=tl.float32)
    if INSTRUMENT:
        pl.enter_scope("page_table")
    r2t_base = req_to_token_ptr + sid * stride_r2t_b
    prefetch_pos = chunk_start + off_n
    prefetch_mask = prefetch_pos < seq_len
    prefetched_slots = tl.load(
        r2t_base + prefetch_pos,
        mask=prefetch_mask,
        other=0,
    ).to(tl.int64)
    if INSTRUMENT:
        pl.exit_scope("page_table")
    for i in range(chunk_start, chunk_end, BLOCK_SIZE_N):
        pos_mask = prefetch_mask
        slots = prefetched_slots
        if INSTRUMENT:
            pl.enter_scope("page_table")
        next_i = i + BLOCK_SIZE_N
        if next_i < chunk_end:
            next_pos = next_i + off_n
            prefetch_mask = next_pos < seq_len
            prefetched_slots = tl.load(
                r2t_base + next_pos,
                mask=prefetch_mask,
                other=0,
            ).to(tl.int64)
        slots = (slots + max_slots) % max_slots
        if INSTRUMENT:
            pl.exit_scope("page_table")
        if INSTRUMENT:
            pl.enter_scope("k_load")
        k_off = (
            slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_off,
            mask=dim_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if INSTRUMENT:
            pl.exit_scope("k_load")
        if INSTRUMENT:
            pl.enter_scope("qk_dot")
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)
        qk += tl.where(off_n[None, :] < chunk_end - i, 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        if INSTRUMENT:
            pl.exit_scope("qk_dot")
        if INSTRUMENT:
            pl.enter_scope("pool")
        score = tl.reshape(
            qk,
            (BLOCK_SIZE_H, BLOCKS_PER_K_BLOCK, block_size),
            can_reorder=False,
        )
        sub_max = tl.max(score, axis=2)
        if SCORE_TYPE == "max":
            score = sub_max
        else:  # "lse"
            score = sub_max + tl.log2(
                tl.sum(tl.exp2(score - sub_max[:, :, None]), axis=2)
            )
            score = tl.where(score != score, float("-inf"), score)
        if INSTRUMENT:
            pl.exit_scope("pool")
        if INSTRUMENT:
            pl.enter_scope("mask_bias")
        curr_block_idx = i // block_size + off_bpk
        is_init = curr_block_idx < init_blocks
        is_local = (curr_block_idx >= local_start) & (curr_block_idx < num_blocks)
        score = tl.where(
            is_local[None, :], 1e29, tl.where(is_init[None, :], 1e30, score)
        )
        if INSTRUMENT:
            pl.exit_scope("mask_bias")
        if WRITE_SCORES:
            if INSTRUMENT:
                pl.enter_scope("score_store")
            tl.store(
                s_ptrs, score.to(score_ptr.dtype.element_ty), boundary_check=(0, 1)
            )
            if INSTRUMENT:
                pl.exit_scope("score_store")
            s_ptrs = tl.advance(s_ptrs, (0, BLOCKS_PER_K_BLOCK))
        else:
            obs_max = tl.maximum(obs_max, tl.max(score))
    if not WRITE_SCORES:
        # Racy by design: many CTAs share one row of dummy slots. Bounded by
        # batch_size, which never exceeds the score buffer's numel.
        tl.store(score_ptr + (pid_bc % batch_size) + tl.arange(0, 1), obs_max)


def _launch_block_prefill(
    inp,
    score: torch.Tensor,
    *,
    block_size: int,
    write_scores: bool,
    instrument: bool,
    num_stages: Optional[int],
) -> None:
    idx_q, idx_k = inp.idx_q, inp.idx_k_cache
    total_q, num_heads, head_dim = idx_q.shape
    batch_size = inp.cu_seqlens.shape[0] - 1
    kwargs, num_warps, tuned_stages = _best_config(
        _flash_attn_fwd_with_block_score_kernel,
        {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64},
    )
    bq, bk = kwargs["BLOCK_SIZE_Q"], kwargs["BLOCK_SIZE_K"]
    grid = (triton.cdiv(inp.max_seqlen_q, bq), batch_size * num_heads)
    _instr_block_score_prefill_kernel[grid](
        idx_q,
        idx_k,
        score,
        inp.req_to_token,
        inp.cu_seqlens,
        inp.seq_lens,
        inp.prefix_lens,
        inp.slot_ids,
        idx_k.shape[0],
        num_heads,
        num_heads // idx_k.shape[1],
        head_dim,
        block_size,
        head_dim**-0.5,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        idx_k.stride(0),
        idx_k.stride(1),
        idx_k.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        inp.req_to_token.stride(0),
        BLOCK_SIZE_Q=bq,
        BLOCK_SIZE_K=bk,
        BLOCK_SIZE_KD=triton.next_power_of_2(head_dim),
        SCORE_TYPE="max",
        INSTRUMENT=instrument,
        WRITE_SCORES=write_scores,
        num_warps=num_warps,
        num_stages=num_stages if num_stages is not None else tuned_stages,
    )


def _launch_block_decode(
    inp,
    score: torch.Tensor,
    *,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    write_scores: bool,
    instrument: bool,
    num_stages: Optional[int],
) -> None:
    idx_q, idx_k = inp.idx_q, inp.idx_k_cache
    batch_size, num_heads, head_dim = idx_q.shape
    num_kv_heads = idx_k.shape[1]
    kwargs, num_warps, tuned_stages = _best_config(
        _decode_score_kernel, {"BLOCK_SIZE_N": 128}
    )
    # Mirror flash_decode_with_topk_idx's grid and trivial-topk gate exactly.
    target = max(1, min(256, 4096 // max(1, batch_size * num_kv_heads)))
    num_kv_chunks = 1 << (target.bit_length() - 1)
    use_jit_topk = (
        envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.get()
        and score.shape[2] <= 4096
        and topk <= 32
    )
    grid = (batch_size * num_kv_chunks, num_kv_heads)
    _instr_block_score_decode_kernel[grid](
        idx_q,
        idx_k,
        inp.req_to_token,
        score,
        inp.seq_lens,
        inp.slot_ids,
        idx_k.shape[0],
        batch_size,
        num_heads // num_kv_heads,
        head_dim,
        block_size,
        topk,
        head_dim**-0.5,
        init_blocks,
        local_blocks,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        idx_k.stride(0),
        idx_k.stride(1),
        idx_k.stride(2),
        inp.req_to_token.stride(0),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        BLOCK_SIZE_H=max(16, triton.next_power_of_2(num_heads // num_kv_heads)),
        BLOCK_SIZE_N=kwargs["BLOCK_SIZE_N"],
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
        NUM_KV_CHUNKS=num_kv_chunks,
        SCORE_TYPE="max",
        SKIP_TRIVIAL_TOPK_SCORE=use_jit_topk,
        INSTRUMENT=instrument,
        WRITE_SCORES=write_scores,
        num_warps=num_warps,
        num_stages=num_stages if num_stages is not None else tuned_stages,
    )


def _block_score_buffer(inp, *, phase: str, block_size: int) -> torch.Tensor:
    """The block score matrix, allocated once so timed closures only launch
    the kernel (the production wrapper's -inf fill is a separate stage)."""
    idx_q = inp.idx_q
    if phase == "prefill":
        total_q, num_heads, _ = idx_q.shape
        width = triton.cdiv(inp.max_seqlen_k, block_size)
        return torch.full(
            (num_heads, total_q, width),
            float("-inf"),
            dtype=torch.float32,
            device=idx_q.device,
        )
    batch_size, num_heads, _ = idx_q.shape
    width = triton.cdiv(min(inp.max_seqlen, inp.req_to_token.shape[1]), block_size)
    return torch.empty(
        (num_heads, batch_size, width), dtype=torch.float32, device=idx_q.device
    )


def block_score_write_split(cfg, inp, *, phase: str, iters: int) -> tuple[float, float]:
    """(with_store_ms, without_store_ms) per iteration for the block score kernel.

    Both closures launch the bench-local kernel copy (scopes compiled out) with
    the production autotuner's config; the only difference is the score-matrix
    store, so the gap is the measured HBM write-back cost.
    """
    from harness import profile_breakdown

    triton.set_allocator(robust_allocator)
    score = _block_score_buffer(inp, phase=phase, block_size=cfg.block_size)

    def make_run(write_scores: bool):
        if phase == "prefill":
            return lambda: _launch_block_prefill(
                inp,
                score,
                block_size=cfg.block_size,
                write_scores=write_scores,
                instrument=False,
                num_stages=None,
            )
        return lambda: _launch_block_decode(
            inp,
            score,
            block_size=cfg.block_size,
            topk=cfg.topk_blocks,
            init_blocks=cfg.init_blocks,
            local_blocks=cfg.local_blocks,
            write_scores=write_scores,
            instrument=False,
            num_stages=None,
        )

    times = []
    for write_scores in (True, False):
        _, per_kernel = profile_breakdown(make_run(write_scores), iters=iters, warmup=3)
        times.append(sum(per_kernel.values()))
    del score
    return times[0], times[1]


def _collect_scope_cycles(node, acc: dict[str, float]) -> None:
    frame = node.get("frame") or {}
    name = frame.get("name", "")
    if name in SCOPES:
        metrics = node.get("metrics") or {}
        cycle_keys = [k for k in metrics if "cycle" in k.lower()]
        key = cycle_keys[0] if cycle_keys else None
        if key is None:
            numeric = [k for k, v in metrics.items() if isinstance(v, (int, float))]
            key = numeric[0] if numeric else None
        if key is not None:
            acc[name] = acc.get(name, 0.0) + float(metrics[key])
    for child in node.get("children") or []:
        _collect_scope_cycles(child, acc)


def profile_score_inner(
    cfg,
    inp,
    *,
    granularity: str,
    phase: str,
    iters: int,
    out_dir: Path,
    tag: str,
) -> dict[str, float]:
    """Per-scope warp cycles of one instrumented score-kernel run.

    Returns ``{scope: cycles}`` summed over ``iters`` full passes. Callers
    should use the *shares*; raw cycles include instrumentation overhead.
    Raises on any failure (e.g. shared-memory overflow from the profiling
    buffer); one retry drops ``num_stages`` to 1 to make room.
    """
    pl.enable_semantic("triton")
    triton.set_allocator(robust_allocator)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / tag

    block_score = (
        _block_score_buffer(inp, phase=phase, block_size=cfg.block_size)
        if granularity == "block"
        else None
    )

    def launch(num_stages: Optional[int]) -> None:
        if granularity == "block":
            if phase == "prefill":
                _launch_block_prefill(
                    inp,
                    block_score,
                    block_size=cfg.block_size,
                    write_scores=True,
                    instrument=True,
                    num_stages=num_stages,
                )
            else:
                _launch_block_decode(
                    inp,
                    block_score,
                    block_size=cfg.block_size,
                    topk=cfg.topk_blocks,
                    init_blocks=cfg.init_blocks,
                    local_blocks=cfg.local_blocks,
                    write_scores=True,
                    instrument=True,
                    num_stages=num_stages,
                )
        elif granularity == "token_seg":
            _launch_prefill_seg(
                inp,
                init_tokens=cfg.init_tokens,
                local_tokens=cfg.local_tokens,
                num_stages=num_stages,
            )
        elif phase == "prefill":
            _launch_prefill(
                inp,
                init_tokens=cfg.init_tokens,
                local_tokens=cfg.local_tokens,
                num_stages=num_stages,
            )
        else:
            _launch_decode(
                inp,
                topk=cfg.effective_topk_tokens,
                init_tokens=cfg.init_tokens,
                local_tokens=cfg.local_tokens,
                num_stages=num_stages,
            )

    for num_stages in (None, 1):
        proton.start(str(base), backend="instrumentation")
        try:
            for _ in range(iters):
                launch(num_stages)
            torch.cuda.synchronize()
        except Exception:
            proton.finalize()
            if num_stages == 1:
                raise
            continue
        proton.finalize()
        break

    acc: dict[str, float] = {}
    data = json.loads((Path(str(base) + ".hatchet")).read_text())
    roots = data if isinstance(data, list) else [data]
    for root in roots:
        _collect_scope_cycles(root, acc)
    if not acc:
        raise RuntimeError("proton produced no scope metrics")
    return acc
