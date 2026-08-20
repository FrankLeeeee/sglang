"""Fused token-granularity prefill selectors (two-pass threshold-emit).

Alternatives to ``index_score.token_select_prefill`` with the same contract —
``[idx_heads, total_q, topk]`` int32 positions, -1 padded, exact — that never
materialize the [heads, rows, L] score matrix. Both take two q.k passes: pool
maxima -> per-row threshold tau -> emit scores >= tau into bounded candidate
lists -> exact top-k over the candidates (see the design comment below).

``token_select_prefill_fused`` emits with a Triton kernel;
``token_select_prefill_fused_cuda`` with the CUDA ballot emitter in
``jit/csrc/minimax/minimax_token_emit.cuh``, which is what makes the fused
path competitive. Candidate overflow falls back to the materializing path.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import robust_allocator
from .index_score import (
    _INIT_BIAS,
    _LOCAL_BIAS,
    _select_topk,
    token_select_prefill,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fused variant: never materialize the [heads, rows, L] score matrix.
#
# The materializing scorer writes 32 GiB at 256k and the selector reads it
# back, which is 62% of the measured time. Holding a per-query top-2048
# in SMEM instead is not possible (128 queries x 2048 x 8 B = 2 MB against
# 228 KB), so this takes two passes and never stores a full row:
#
#   pass 1  q.k, reduce each POOL keys to their max in registers, store only
#           the maxima            -> [heads, rows, L/POOL]      (POOL x smaller)
#   host    tau = the k-th largest pool maximum, per query row — computed as
#           min-of-top-k through the verified selector (torch.kthvalue returns
#           the same value but its CUDA path is far slower)
#   pass 2  q.k again, emit only scores >= tau into a bounded per-row list
#   host    exact top-k over those candidates, via the same verified selector
#
# Exactness: every pool holds an element >= its own max, so the top-k pool
# maxima are witnessed by at least k tokens, which puts tau at or below the true
# k-th score. Everything the true top-k contains therefore clears tau and
# reaches the candidate list; the final selection is exact over a superset.
#
# The candidate count is self-limiting: with npool ~ 2k, tau lands near the
# median pool maximum and the survivors number about k. CANDIDATE_SLACK gives
# headroom, and an overflow falls back rather than silently truncating.
#
# Both passes load each K tile once and reuse it for every index head — all
# index heads score against the same single-head index-K cache, so folding the
# head loop into the program divides K-cache traffic by NUM_HEADS.
#
# Measured status (H200, ctx 256k, 8192 rows, 4 heads, topk 2048): pass 1 runs
# in 6.9 ms — vs 16.6 ms for the materializing scorer, so keeping scores in
# registers does pay — and tau costs ~8 ms, but pass 2 costs ~62 ms and makes
# the whole path slower than the materializing baseline (79 vs 45 ms).
# The blocker is emission, not memory traffic and not the atomics (relaxed vs
# acq_rel is within noise, register spills are near zero): every candidate
# epilogue Triton can express runs per score element, and over rows x context
# elements the instruction cost alone (`tl.cumsum` ~14 ms, each masked scatter
# store ~13 ms) exceeds the 32 GiB the materializing path writes and re-reads.
# Compacting ~2k survivors per row out of register tiles needs warp-ballot
# style compaction, which Triton does not expose.
#
# `token_select_prefill_fused_cuda` replaces pass 2 with exactly that CUDA
# ballot emitter (jit csrc/minimax/minimax_token_emit.cuh): 79 -> 53 ms at
# 256k, i.e. parity with the 512 MiB-budget baseline at 1/6th the workspace
# of the materializing path (5.7 vs 34.6 GiB), but still behind `optimized`
# (45 ms). The remaining cost is the emit kernel's wmma GEMM (~36 ms against
# a ~10 ms roofline): wmma::load_matrix_sync from padded (unswizzled) shared
# memory bank-conflicts ~2x and cannot pipeline; closing the rest of the gap
# means raw ldmatrix/mma.sync with XOR swizzle and cp.async — a
# FlashAttention-grade rewrite of the tile pipeline.
# ---------------------------------------------------------------------------

# The pre-filter only narrows anything when there are many more pools than the
# budget: tau is the k-th largest pool maximum, so if npool <= topk it degenerates
# to the smallest maximum and every token clears it. Target npool = POOL_RATIO x
# topk and derive the pool size from the context, rather than fixing it.
POOL_RATIO = 4
CANDIDATE_SLACK = 8


def _choose_pool(kv_width: int, topk: int, block_k: int = 128) -> int:
    """Largest pool that still leaves POOL_RATIO x topk pools, power of two."""
    want = max(1, kv_width // max(1, POOL_RATIO * topk))
    pool = 1 << max(0, want.bit_length() - 1)
    return max(1, min(pool, block_k))


@triton.heuristics({"BLOCK_SIZE_D": lambda a: triton.next_power_of_2(a["qk_head_dim"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_K": bk}, num_warps=nw, num_stages=ns)
        for bq, bk in ((32, 128), (64, 128), (128, 128))
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["qk_head_dim", "POOL_SIZE", "NUM_HEADS", "IS_FP8"],
)
@triton.jit
def _score_poolmax_kernel(
    q_ptr, k_cache_ptr, pool_ptr, req_to_token_ptr,
    cu_seqlens, seq_lens, prefix_lens, slot_ids,
    max_slots, qk_head_dim, kv_width, npool,
    init_tokens, local_tokens, sm_scale,
    stride_q_n, stride_q_h, stride_q_d, stride_k_s, stride_k_d,
    stride_p_h, stride_p_n, stride_p_k, stride_r2t_b,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, POOL_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr, IS_FP8: tl.constexpr,
):
    """Pass 1: q.k, then collapse each POOL_SIZE keys to their max in registers.

    One program serves every index head: the K tile is loaded once and reused
    for NUM_HEADS dots.
    """
    pid_q, pid_b, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tl.static_assert(BLOCK_SIZE_K % POOL_SIZE == 0)
    POOLS_PER_TILE: tl.constexpr = BLOCK_SIZE_K // POOL_SIZE

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots

    q_lo = pid_q * BLOCK_SIZE_Q
    q_hi = tl.minimum(q_lo + BLOCK_SIZE_Q, q_len)
    if q_lo >= q_hi:
        return

    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < qk_head_dim
    local_q = q_lo + off_q
    q_mask = local_q < q_hi
    abs_q = prefix_len + local_q
    g_row = seq_start + local_q  # row in the [heads, total_q, npool] buffer

    tile_lo = pid_k * BLOCK_SIZE_K
    tile_hi = tile_lo + BLOCK_SIZE_K
    pos = tile_lo + off_k
    pos_mask = (pos < seq_len) & (pos < kv_width)

    # Scalar tests, once per program (see the chunked kernel above): deep inside
    # the prefix no element of this tile can be non-causal, forced, or past the
    # sequence end, and then the mask block is provably a no-op.
    min_abs_q = prefix_len + q_lo
    fully_causal = tile_hi <= min_abs_q + 1
    has_init = tile_lo < init_tokens
    has_local = tile_hi > min_abs_q - local_tokens + 1
    within_seq = tile_hi <= seq_len
    needs_mask = (not fully_causal) or has_init or has_local or (not within_seq)

    slots = tl.load(req_to_token_ptr + sid * stride_r2t_b + pos, mask=pos_mask, other=0).to(tl.int64)
    slots = (slots + max_slots) % max_slots
    k = tl.load(
        k_cache_ptr + slots[None, :] * stride_k_s + off_d[:, None] * stride_k_d,
        mask=d_mask[:, None] & pos_mask[None, :], other=0.0,
    )
    if IS_FP8:
        k = k.to(q_ptr.dtype.element_ty)

    sm_scale_log2e = sm_scale * 1.4426950408889634
    off_p = tile_lo // POOL_SIZE + tl.arange(0, POOLS_PER_TILE)
    p_mask = off_p < npool

    for h in tl.static_range(NUM_HEADS):
        q = tl.load(
            q_ptr + g_row[:, None] * stride_q_n
            + h * stride_q_h + off_d[None, :] * stride_q_d,
            mask=q_mask[:, None] & d_mask[None, :], other=0.0,
        )
        score = tl.dot(q, k) * sm_scale_log2e
        if needs_mask:
            causal = (abs_q[:, None] >= pos[None, :]) & pos_mask[None, :]
            init_m = causal & (pos[None, :] < init_tokens)
            local_m = causal & (pos[None, :] > (abs_q[:, None] - local_tokens))
            score = tl.where(init_m, _INIT_BIAS, score)
            score = tl.where(local_m & (init_m == 0), _LOCAL_BIAS, score)
            score = tl.where(causal, score, float("-inf"))

        # the whole point: reduce before anything leaves the SM
        pooled = tl.max(
            tl.reshape(score, (BLOCK_SIZE_Q, POOLS_PER_TILE, POOL_SIZE)), axis=2
        )
        tl.store(
            pool_ptr + h * stride_p_h
            + g_row[:, None] * stride_p_n + off_p[None, :] * stride_p_k,
            pooled,
            mask=q_mask[:, None] & p_mask[None, :],
        )


@triton.heuristics({"BLOCK_SIZE_D": lambda a: triton.next_power_of_2(a["qk_head_dim"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_K": bk}, num_warps=nw, num_stages=ns)
        for bq, bk in ((32, 128), (64, 128), (128, 128))
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["qk_head_dim", "NUM_HEADS", "IS_FP8"],
    # The autotuner launches every config against live buffers; each trial
    # accumulates into the row counters, so without a reset the counts inflate
    # by the number of trials and every autotuned call "overflows" into the
    # fallback. The candidate lists need no reset: every trial writes the same
    # survivor multiset into slots [0, cnt), so the last trial leaves the
    # correct state.
    reset_to_zero=["cnt_ptr"],
)
@triton.jit
def _emit_above_threshold_kernel(
    q_ptr, k_cache_ptr, tau_ptr, cand_val_ptr, cand_pos_ptr, cnt_ptr,
    req_to_token_ptr, cu_seqlens, seq_lens, prefix_lens, slot_ids,
    max_slots, qk_head_dim, kv_width, cap,
    init_tokens, local_tokens, sm_scale,
    stride_q_n, stride_q_h, stride_q_d, stride_k_s, stride_k_d,
    stride_t_h, stride_t_n, stride_c_h, stride_c_n, stride_r2t_b,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, NUM_HEADS: tl.constexpr, IS_FP8: tl.constexpr,
):
    """Pass 2: recompute q.k and keep only what clears the per-row threshold.

    Emission reserves one contiguous block of a row's candidate list per
    (row, tile) with a single atomic, then places survivors by prefix sum.
    The per-element atomics this replaces all hit the same row counter within
    a tile, serializing up to BLOCK_SIZE_K additions per row per tile.
    """
    pid_q, pid_b, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots

    q_lo = pid_q * BLOCK_SIZE_Q
    q_hi = tl.minimum(q_lo + BLOCK_SIZE_Q, q_len)
    if q_lo >= q_hi:
        return

    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < qk_head_dim
    local_q = q_lo + off_q
    q_mask = local_q < q_hi
    abs_q = prefix_len + local_q
    g_row = seq_start + local_q  # row in the [heads, total_q, ...] buffers

    tile_lo = pid_k * BLOCK_SIZE_K
    tile_hi = tile_lo + BLOCK_SIZE_K
    pos = tile_lo + off_k
    pos_mask = (pos < seq_len) & (pos < kv_width)

    min_abs_q = prefix_len + q_lo
    fully_causal = tile_hi <= min_abs_q + 1
    has_init = tile_lo < init_tokens
    has_local = tile_hi > min_abs_q - local_tokens + 1
    within_seq = tile_hi <= seq_len
    needs_mask = (not fully_causal) or has_init or has_local or (not within_seq)

    slots = tl.load(req_to_token_ptr + sid * stride_r2t_b + pos, mask=pos_mask, other=0).to(tl.int64)
    slots = (slots + max_slots) % max_slots
    k = tl.load(
        k_cache_ptr + slots[None, :] * stride_k_s + off_d[:, None] * stride_k_d,
        mask=d_mask[:, None] & pos_mask[None, :], other=0.0,
    )
    if IS_FP8:
        k = k.to(q_ptr.dtype.element_ty)

    sm_scale_log2e = sm_scale * 1.4426950408889634

    for h in tl.static_range(NUM_HEADS):
        q = tl.load(
            q_ptr + g_row[:, None] * stride_q_n
            + h * stride_q_h + off_d[None, :] * stride_q_d,
            mask=q_mask[:, None] & d_mask[None, :], other=0.0,
        )
        # Masked rows read tau = +inf, so nothing they score can ever be kept.
        tau = tl.load(tau_ptr + h * stride_t_h + g_row * stride_t_n,
                      mask=q_mask, other=float("inf"))

        score = tl.dot(q, k) * sm_scale_log2e
        if needs_mask:
            causal = (abs_q[:, None] >= pos[None, :]) & pos_mask[None, :]
            init_m = causal & (pos[None, :] < init_tokens)
            local_m = causal & (pos[None, :] > (abs_q[:, None] - local_tokens))
            score = tl.where(init_m, _INIT_BIAS, score)
            score = tl.where(local_m & (init_m == 0), _LOCAL_BIAS, score)
            score = tl.where(causal, score, float("-inf"))
            keep = causal & (score >= tau[:, None]) & q_mask[:, None]
        else:
            keep = (score >= tau[:, None]) & q_mask[:, None]

        keep_i = keep.to(tl.int32)
        keep_cnt = tl.sum(keep_i, axis=1)  # survivors per row, this tile
        # Relaxed ordering: the counter only hands out disjoint slot ranges;
        # nothing is read back through it, so no fence is needed.
        base = tl.atomic_add(cnt_ptr + h * stride_t_h + g_row * stride_t_n,
                             keep_cnt, mask=q_mask, sem="relaxed")
        # In-tile placement: the j-th survivor of a row lands at base + j.
        slot = base[:, None] + tl.cumsum(keep_i, axis=1) - 1
        fits = keep & (slot < cap)
        crow = h * stride_c_h + g_row[:, None] * stride_c_n
        tl.store(cand_val_ptr + crow + slot, score, mask=fits)
        tl.store(cand_pos_ptr + crow + slot, pos[None, :].to(tl.int32), mask=fits)


def _pool_threshold(
    *,
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    kv_width: int,
    pool: int,
    npool: int,
    topk: int,
    init_tokens: int,
    local_tokens: int,
    sm_scale: float,
    is_fp8: bool,
) -> torch.Tensor:
    """Pass 1 + threshold: ``tau[h, row]`` = the k-th largest pool maximum.

    min-of-top-k rather than torch.kthvalue: the same value, but through the
    selector that is already verified exact and fast at these widths.
    """
    total_q, num_idx_heads, qk_head_dim = idx_q.shape
    batch_size = cu_seqlens.shape[0] - 1
    poolmax = torch.full((num_idx_heads, total_q, npool), float("-inf"),
                         dtype=torch.float32, device=idx_q.device)
    grid1 = lambda m: (triton.cdiv(max_seqlen_q, m["BLOCK_SIZE_Q"]),
                       batch_size,
                       triton.cdiv(kv_width, m["BLOCK_SIZE_K"]))
    _score_poolmax_kernel[grid1](
        idx_q, idx_k_cache, poolmax, req_to_token,
        cu_seqlens, seq_lens, prefix_lens, slot_ids,
        idx_k_cache.shape[0], qk_head_dim, kv_width, npool,
        init_tokens, local_tokens, sm_scale,
        idx_q.stride(0), idx_q.stride(1), idx_q.stride(2),
        idx_k_cache.stride(0), idx_k_cache.stride(2),
        poolmax.stride(0), poolmax.stride(1), poolmax.stride(2),
        req_to_token.stride(0),
        POOL_SIZE=pool, NUM_HEADS=num_idx_heads, IS_FP8=is_fp8,
    )
    kk = min(topk, npool)
    pool_top, _ = _select_topk(poolmax.reshape(-1, npool), kk)
    return pool_top.amin(dim=-1).view(num_idx_heads, total_q)


def _trim_candidates(
    cand_val: torch.Tensor, cand_pos: torch.Tensor, *, cap: int, topk: int
) -> torch.Tensor:
    """Exact top-k positions over the candidate lists, -1 padded."""
    num_idx_heads, total_q = cand_val.shape[:2]
    k2 = min(topk, cap)
    values, sel = _select_topk(cand_val.reshape(-1, cap), k2)
    picked = torch.gather(cand_pos.reshape(-1, cap), -1, sel)
    picked = torch.where(values > float("-inf"), picked,
                         torch.full_like(picked, -1))
    out = picked.view(num_idx_heads, total_q, k2).to(torch.int32)
    if k2 < topk:
        out = torch.cat(
            [out, out.new_full((num_idx_heads, total_q, topk - k2), -1)], dim=-1
        )
    return out


def _materializing_fallback(kwargs: dict, cnt_max: int, cap: int) -> torch.Tensor:
    logger.warning(
        "MiniMax fused token select: candidate overflow (%d > %d); "
        "falling back to the materializing path.", cnt_max, cap
    )
    return token_select_prefill(**kwargs)


@torch.no_grad()
def token_select_prefill_fused(
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    topk: int,
    init_tokens: int,
    local_tokens: int,
    sm_scale: Optional[float] = None,
    pool: Optional[int] = None,
    candidate_slack: int = CANDIDATE_SLACK,
    seqlens_cpu: Optional[list] = None,
    prefix_lens_cpu: Optional[list] = None,
    **_ignored,
) -> torch.Tensor:
    """Two-pass selection that never stores a full score row.

    Falls back to :func:`index_score.token_select_prefill` if the candidate list
    overflows, so the contract is the same exact answer either way.
    """
    triton.set_allocator(robust_allocator)
    total_q, num_idx_heads, qk_head_dim = idx_q.shape
    max_slots = idx_k_cache.shape[0]
    batch_size = cu_seqlens.shape[0] - 1
    is_fp8 = idx_k_cache.dtype != idx_q.dtype
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    dev = idx_q.device

    if total_q == 0:
        return torch.full((num_idx_heads, 0, topk), -1, dtype=torch.int32, device=dev)

    kv_width = min(max_seqlen_k, req_to_token.shape[1])
    if pool is None:
        pool = _choose_pool(kv_width, topk)
    npool = triton.cdiv(kv_width, pool)
    cap = min(kv_width, topk * candidate_slack)

    tau = _pool_threshold(
        idx_q=idx_q, idx_k_cache=idx_k_cache, req_to_token=req_to_token,
        slot_ids=slot_ids, cu_seqlens=cu_seqlens, seq_lens=seq_lens,
        prefix_lens=prefix_lens, max_seqlen_q=max_seqlen_q, kv_width=kv_width,
        pool=pool, npool=npool, topk=topk, init_tokens=init_tokens,
        local_tokens=local_tokens, sm_scale=sm_scale, is_fp8=is_fp8,
    )

    # ---- pass 2: emit only what clears the threshold -------------------
    cand_val = torch.full((num_idx_heads, total_q, cap), float("-inf"),
                          dtype=torch.float32, device=dev)
    cand_pos = torch.full((num_idx_heads, total_q, cap), -1,
                          dtype=torch.int32, device=dev)
    cnt = torch.zeros((num_idx_heads, total_q), dtype=torch.int32, device=dev)
    grid2 = lambda m: (triton.cdiv(max_seqlen_q, m["BLOCK_SIZE_Q"]),
                       batch_size,
                       triton.cdiv(kv_width, m["BLOCK_SIZE_K"]))
    _emit_above_threshold_kernel[grid2](
        idx_q, idx_k_cache, tau, cand_val, cand_pos, cnt, req_to_token,
        cu_seqlens, seq_lens, prefix_lens, slot_ids,
        max_slots, qk_head_dim, kv_width, cap,
        init_tokens, local_tokens, sm_scale,
        idx_q.stride(0), idx_q.stride(1), idx_q.stride(2),
        idx_k_cache.stride(0), idx_k_cache.stride(2),
        tau.stride(0), tau.stride(1), cand_val.stride(0), cand_val.stride(1),
        req_to_token.stride(0),
        NUM_HEADS=num_idx_heads, IS_FP8=is_fp8,
    )

    cnt_max = int(cnt.max())
    if cnt_max > cap:
        del cand_val, cand_pos, cnt, tau
        return _materializing_fallback(
            dict(idx_q=idx_q, idx_k_cache=idx_k_cache, req_to_token=req_to_token,
                 slot_ids=slot_ids, cu_seqlens=cu_seqlens, seq_lens=seq_lens,
                 prefix_lens=prefix_lens, max_seqlen_q=max_seqlen_q,
                 max_seqlen_k=max_seqlen_k, topk=topk, init_tokens=init_tokens,
                 local_tokens=local_tokens, sm_scale=sm_scale,
                 seqlens_cpu=seqlens_cpu, prefix_lens_cpu=prefix_lens_cpu),
            cnt_max, cap,
        )

    return _trim_candidates(cand_val, cand_pos, cap=cap, topk=topk)


@torch.no_grad()
def token_select_prefill_fused_cuda(
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    topk: int,
    init_tokens: int,
    local_tokens: int,
    sm_scale: Optional[float] = None,
    pool: Optional[int] = None,
    candidate_slack: int = CANDIDATE_SLACK,
    seqlens_cpu: Optional[list] = None,
    prefix_lens_cpu: Optional[list] = None,
    **_ignored,
) -> torch.Tensor:
    """Fused selection with the emission pass in CUDA instead of Triton.

    Same structure and contract as :func:`token_select_prefill_fused`; only
    pass 2 differs. The Triton emitter spends ~15 instructions per score
    element on slot assignment and scatter; the CUDA emitter's warp ballot
    does the same compaction in ~2 (see minimax/minimax_token_emit.cuh).
    Problems the CUDA kernel does not cover (fp8 index cache, head dim != 128,
    non-int32 page table, pre-SM80) delegate to the Triton variant.
    """
    from .emit_cuda import emit_above_threshold, emit_supported

    fused_kwargs = dict(
        idx_q=idx_q, idx_k_cache=idx_k_cache, req_to_token=req_to_token,
        slot_ids=slot_ids, cu_seqlens=cu_seqlens, seq_lens=seq_lens,
        prefix_lens=prefix_lens, max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k, topk=topk, init_tokens=init_tokens,
        local_tokens=local_tokens, sm_scale=sm_scale, pool=pool,
        candidate_slack=candidate_slack, seqlens_cpu=seqlens_cpu,
        prefix_lens_cpu=prefix_lens_cpu,
    )
    if not (emit_supported(idx_q, idx_k_cache) and req_to_token.dtype == torch.int32):
        return token_select_prefill_fused(**fused_kwargs)

    triton.set_allocator(robust_allocator)
    total_q, num_idx_heads, qk_head_dim = idx_q.shape
    is_fp8 = idx_k_cache.dtype != idx_q.dtype
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    dev = idx_q.device

    if total_q == 0:
        return torch.full((num_idx_heads, 0, topk), -1, dtype=torch.int32, device=dev)

    kv_width = min(max_seqlen_k, req_to_token.shape[1])
    if pool is None:
        pool = _choose_pool(kv_width, topk)
    npool = triton.cdiv(kv_width, pool)
    cap = min(kv_width, topk * candidate_slack)

    tau = _pool_threshold(
        idx_q=idx_q, idx_k_cache=idx_k_cache, req_to_token=req_to_token,
        slot_ids=slot_ids, cu_seqlens=cu_seqlens, seq_lens=seq_lens,
        prefix_lens=prefix_lens, max_seqlen_q=max_seqlen_q, kv_width=kv_width,
        pool=pool, npool=npool, topk=topk, init_tokens=init_tokens,
        local_tokens=local_tokens, sm_scale=sm_scale, is_fp8=is_fp8,
    )

    cand_val = torch.full((num_idx_heads, total_q, cap), float("-inf"),
                          dtype=torch.float32, device=dev)
    cand_pos = torch.full((num_idx_heads, total_q, cap), -1,
                          dtype=torch.int32, device=dev)
    cnt = torch.zeros((num_idx_heads, total_q), dtype=torch.int32, device=dev)
    emit_above_threshold(
        idx_q=idx_q, idx_k_cache=idx_k_cache, req_to_token=req_to_token,
        cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens,
        slot_ids=slot_ids, tau=tau, cand_val=cand_val, cand_pos=cand_pos,
        cnt=cnt, max_seqlen_q=max_seqlen_q, kv_width=kv_width,
        init_tokens=init_tokens, local_tokens=local_tokens, sm_scale=sm_scale,
    )

    cnt_max = int(cnt.max())
    if cnt_max > cap:
        del cand_val, cand_pos, cnt, tau
        fused_kwargs.pop("pool")
        fused_kwargs.pop("candidate_slack")
        fused_kwargs["sm_scale"] = sm_scale
        return _materializing_fallback(fused_kwargs, cnt_max, cap)

    return _trim_candidates(cand_val, cand_pos, cap=cap, topk=topk)
