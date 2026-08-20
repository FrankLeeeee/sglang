"""tau-emit token prefill selector — benchmark alternative to index_score.

Same contract as ``index_score.token_select_prefill`` (selection verified
set-equivalent with fp64 adjudication of ~1-ulp boundary ties). One q.k pass,
with the (chunk-budgeted) score matrix as the selection medium:

  1. the segmented scorer writes the fp32 logits AND per-pool maxima from the
     same register tile (fused — pooling costs one tl.max per key tile);
  2. tau per row = a radix lower bound on the topk-th largest pool max
     (``minimax_row_kth``), a provable lower bound on the true topk-th score —
     the two-pass fused selector's threshold argument, without a second q.k;
  3. ``minimax_score_filter`` streams the matrix once, appending every score
     >= tau directly into -inf-prefilled global candidate lists (~2-4k/row);
  4. one exact top-k over all candidates at the end (torch.topk — measured
     faster than flashinfer at candidate widths; ties differ from the
     reference only among fp-equal scores).

Both CUDA kernels live in ``jit/csrc/minimax/minimax_score_filter.cuh``.
Overflow and partial/multi-request chunks fall back to the materializing path.

Measured (H200, bs=1, chunk 8192): 6.4 ms at 16k / 23.1 ms at 128k vs the
materializing path's 6.6 / 30 ms — the fastest exact token selector at both
lengths. The matrix round-trip still scales with context, so >=256k remains
the one-pass architecture's regime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit

from ..common.utils import robust_allocator
from .index_score import (
    _INIT_BIAS,
    _LOCAL_BIAS,
    _topk_positions,
    plan_query_chunk,
    token_select_prefill,
)
from .index_score_seg import DEFAULT_TARGET_CTAS, _plan_segments

if TYPE_CHECKING:
    from tvm_ffi.module import Module

DEFAULT_CAP = 6144  # per-(head, row) candidate capacity


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "CHUNK_BUCKET": lambda args: triton.next_power_of_2(args["chunk_cap"]),
    }
)
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_Q": bq, "BLOCK_SIZE_K": bk}, num_warps=nw, num_stages=ns
        )
        for bq, bk in (
            (16, 128),
            (32, 128),
            (64, 64),
            (64, 128),
            (128, 64),
            (128, 128),
        )
        for nw in (4, 8)
        for ns in (2, 3, 4)
    ],
    key=["qk_head_dim", "CHUNK_BUCKET", "IS_FP8", "POOL"],
)
@triton.jit
def _token_score_seg_pool_kernel(
    q_ptr,
    k_cache_ptr,
    score_ptr,
    pool_ptr,  # [num_idx_heads, batch * chunk_cap, npool] float32
    req_to_token_ptr,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    slot_ids,
    max_slots,
    num_idx_heads,
    qk_head_dim,
    kv_width,
    npool,
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
    stride_p_h,
    stride_p_n,
    stride_p_k,
    stride_r2t_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    CHUNK_BUCKET: tl.constexpr,  # autotune key only
    POOL: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    # index_score_seg's interleaved segmented scorer, plus a fused per-POOL max
    # store — the tile is already in registers, so pooling is nearly free.
    pid_q, pid_bh, pid_seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pid_b_local = pid_bh // num_idx_heads
    pid_b = batch_start + pid_b_local
    pid_h = pid_bh % num_idx_heads

    sm_scale_log2e = sm_scale * 1.4426950408889634

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots

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
    p_base = pool_ptr + pid_h * stride_p_h + s_row[:, None] * stride_p_n

    for t in tl.range(pid_seg, tiles_total, num_segments):
        pos = t * BLOCK_SIZE_K + off_k
        in_width = pos < kv_width
        pos_mask = (pos < seq_len) & in_width
        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        slots = (slots + max_slots) % max_slots
        k = tl.load(
            k_cache_ptr + slots[None, :] * stride_k_s + off_d[:, None] * stride_k_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if IS_FP8:
            k = k.to(q.dtype)

        score = tl.dot(q, k) * sm_scale_log2e

        causal = (abs_q[:, None] >= pos[None, :]) & pos_mask[None, :]
        init_m = causal & (pos[None, :] < init_tokens)
        local_m = causal & (pos[None, :] > (abs_q[:, None] - local_tokens))
        score = tl.where(init_m, _INIT_BIAS, score)
        score = tl.where(local_m & (init_m == 0), _LOCAL_BIAS, score)
        score = tl.where(causal, score, float("-inf"))

        tl.store(
            s_base + pos[None, :] * stride_s_k,
            score,
            mask=q_mask[:, None] & in_width[None, :],
        )

        pooled = tl.max(
            tl.reshape(score, (BLOCK_SIZE_Q, BLOCK_SIZE_K // POOL, POOL)), axis=2
        )
        pidx = (t * BLOCK_SIZE_K) // POOL + tl.arange(0, BLOCK_SIZE_K // POOL)
        tl.store(
            p_base + pidx[None, :] * stride_p_k,
            pooled,
            mask=q_mask[:, None] & (pidx[None, :] < npool),
        )


@cache_once
def _jit_filter() -> Module:
    return load_jit(
        "minimax_score_filter",
        cuda_files=["minimax/minimax_score_filter.cuh"],
        cuda_wrappers=[
            ("filter", "minimax_score_filter"),
            ("row_kth", "minimax_row_kth"),
        ],
    )


def _pool_size(kv_width: int, topk: int) -> int:
    """Largest power-of-two pool with npool >= 2 * topk (capped for the tile)."""
    pool = 1
    while pool * 2 <= min(64, kv_width // (2 * topk)):
        pool *= 2
    return pool


@torch.no_grad()
def token_select_prefill_tau_emit(
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
    seqlens_cpu=None,
    prefix_lens_cpu=None,
    cap: int = DEFAULT_CAP,
    target_ctas: int = DEFAULT_TARGET_CTAS,
) -> torch.Tensor:
    """Same contract as token_select_prefill; tau-emit selection."""
    triton.set_allocator(robust_allocator)
    total_q, num_idx_heads, qk_head_dim = idx_q.shape
    max_slots = idx_k_cache.shape[0]
    batch_size = cu_seqlens.shape[0] - 1
    is_fp8 = idx_k_cache.dtype != idx_q.dtype
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5

    topk_idx = torch.full(
        (num_idx_heads, total_q, topk), -1, dtype=torch.int32, device=idx_q.device
    )
    if total_q == 0:
        return topk_idx

    q_lens = list(seqlens_cpu) if seqlens_cpu is not None else (
        (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    )
    cu_cpu = [0]
    for n in q_lens:
        cu_cpu.append(cu_cpu[-1] + n)
    max_prefix = max(prefix_lens_cpu) if prefix_lens_cpu else max_seqlen_k

    batch_cap, chunk_cap = plan_query_chunk(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_idx_heads=num_idx_heads,
    )

    # Candidates accumulate here (stale tail stays at the -inf prefill); the
    # exact top-k runs ONCE at the end (per-chunk top-k calls measured as the
    # dominant cost of the per-chunk variant).
    all_val = torch.full(
        (num_idx_heads, total_q, cap), float("-inf"),
        dtype=torch.float32, device=idx_q.device,
    )
    all_pos = torch.empty(
        (num_idx_heads, total_q, cap), dtype=torch.int32, device=idx_q.device
    )
    all_cnt = torch.zeros(
        (num_idx_heads, total_q), dtype=torch.int32, device=idx_q.device
    )
    overrides: list[tuple[int, int, torch.Tensor]] = []
    module = _jit_filter()

    for b_start in range(0, batch_size, batch_cap):
        b_count = min(batch_cap, batch_size - b_start)
        for q_chunk_start in range(0, max_seqlen_q, chunk_cap):
            kv_width = min(max_seqlen_k, max_prefix + q_chunk_start + chunk_cap)
            num_segments = _plan_segments(
                chunk_cap=chunk_cap,
                b_count=b_count,
                num_idx_heads=num_idx_heads,
                kv_width=kv_width,
                target_ctas=target_ctas,
            )
            pool = _pool_size(kv_width, topk)
            npool = -(-kv_width // pool)
            rows = b_count * chunk_cap
            scores = torch.empty(
                (num_idx_heads, rows, kv_width),
                dtype=torch.float32, device=idx_q.device,
            )
            poolmax = torch.empty(
                (num_idx_heads, rows, npool),
                dtype=torch.float32, device=idx_q.device,
            )

            def _grid(meta):
                return (
                    triton.cdiv(chunk_cap, meta["BLOCK_SIZE_Q"]),
                    b_count * num_idx_heads,
                    num_segments,
                )

            _token_score_seg_pool_kernel[_grid](
                idx_q,
                idx_k_cache,
                scores,
                poolmax,
                req_to_token,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                slot_ids,
                max_slots,
                num_idx_heads,
                qk_head_dim,
                kv_width,
                npool,
                chunk_cap,
                q_chunk_start,
                b_start,
                num_segments,
                init_tokens,
                local_tokens,
                sm_scale,
                idx_q.stride(0),
                idx_q.stride(1),
                idx_q.stride(2),
                idx_k_cache.stride(0),
                idx_k_cache.stride(2),
                scores.stride(0),
                scores.stride(1),
                scores.stride(2),
                poolmax.stride(0),
                poolmax.stride(1),
                poolmax.stride(2),
                req_to_token.stride(0),
                POOL=pool,
                IS_FP8=is_fp8,
            )

            # tau-emit needs every chunk row live and globally contiguous;
            # partial chunks or multi-request chunks fall back per chunk.
            full_rows = b_count == 1 and (
                min(q_lens[b_start], q_chunk_start + chunk_cap) - q_chunk_start
                == chunk_cap
            )
            if full_rows:
                tau = torch.empty(
                    (num_idx_heads, rows), dtype=torch.float32, device=idx_q.device
                )
                module.row_kth(
                    poolmax.view(num_idx_heads * rows, npool),
                    tau.view(-1),
                    min(topk, npool),
                )
                g_lo = cu_cpu[b_start] + q_chunk_start
                module.filter(scores, tau, all_val, all_pos, all_cnt, g_lo)
                del scores, poolmax, tau
            else:
                chunk_idx = _topk_positions(scores, topk)
                del scores, poolmax
                for b in range(b_start, b_start + b_count):
                    q_len = q_lens[b]
                    lo = q_chunk_start
                    hi = min(q_len, q_chunk_start + chunk_cap)
                    if lo >= hi:
                        continue
                    r_lo = (b - b_start) * chunk_cap
                    overrides.append(
                        (cu_cpu[b] + lo, cu_cpu[b] + hi,
                         chunk_idx[:, r_lo : r_lo + (hi - lo)])
                    )

    # One exact top-k over everything. torch.topk beats flashinfer at this
    # width; ties differ from the reference only among fp-equal scores.
    k = min(topk, cap)
    v, i = torch.topk(all_val.view(-1, cap), k, dim=-1, sorted=False)
    pos = torch.gather(all_pos.view(-1, cap), 1, i)
    idx = torch.where(v > float("-inf"), pos, torch.full_like(pos, -1))
    topk_idx[:, :, :k] = idx.to(torch.int32).view(num_idx_heads, total_q, k)
    for g_lo, g_hi, o in overrides:
        topk_idx[:, g_lo:g_hi] = o

    # Checked after all work is queued so the sync adds no pipeline bubble.
    if int(all_cnt.max().item()) > cap:
        return token_select_prefill(
            idx_q=idx_q,
            idx_k_cache=idx_k_cache,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            topk=topk,
            init_tokens=init_tokens,
            local_tokens=local_tokens,
            sm_scale=sm_scale,
            seqlens_cpu=seqlens_cpu,
            prefix_lens_cpu=prefix_lens_cpu,
        )
    return topk_idx
