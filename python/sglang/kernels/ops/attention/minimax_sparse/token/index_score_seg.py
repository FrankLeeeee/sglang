"""Segmented-grid token prefill selector — benchmark alternative to index_score.

Same contract as ``index_score.token_select_prefill`` and verified
bit-identical against it (same scores element-for-element, same deterministic
tie-break, checked at 4k-1M): only the score kernel's launch shape differs.

The production kernel puts the key axis fully in the grid — one CTA per
128-key tile, no inner loop — because the 512 MiB logits budget leaves too few
query rows live for a per-row loop to fill the device. That shape re-loads q
per key tile (~2000x at 128k) and leaves the page-table -> K dependency chain
exposed. This kernel keeps the same query chunking but gives grid dim 2 a key
*segment* per CTA with an inner loop over its tiles: q is loaded once per CTA
and the page-table chase pipelines under ``num_stages``. Segments are
interleaved (grid-stride over tiles), NOT contiguous runs, so concurrently
resident CTAs always touch consecutive key tiles and keep the per-tile grid's
L2/DRAM locality — the contiguous variant regressed 15% at 1M on exactly that.

Measured (H200, bs=1, chunk 8192): score kernel 10.8 -> 8.7 ms at 128k
(1.24x), with page_table 21% -> 1% and q_load ~19% -> ~0 of kernel time by
Proton attribution. Kept separate from production for unified benchmarking;
promote by swapping the wrapper the backend dispatches.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import robust_allocator
from .index_score import (
    _INIT_BIAS,
    _LOCAL_BIAS,
    _topk_positions,
    plan_query_chunk,
)

# Total CTAs to aim for per launch: ~16x an H100/H200's SM count — enough that
# a wave's tail doesn't idle the device, small enough that each segment still
# loops over many tiles (q reuse + pipelining depth).
DEFAULT_TARGET_CTAS = 2048


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
    key=["qk_head_dim", "CHUNK_BUCKET", "IS_FP8"],
)
@triton.jit
def _token_index_score_prefill_kernel_seg(
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
    CHUNK_BUCKET: tl.constexpr,  # autotune key only
    IS_FP8: tl.constexpr,
):
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

    # This CTA's key segment: tiles pid_seg, pid_seg + S, pid_seg + 2S, ...
    # Interleaved (grid-stride), NOT contiguous runs: concurrent CTAs then
    # always touch consecutive tiles, keeping the per-tile grid's L2/DRAM
    # locality, while each CTA still amortizes its q load over many tiles.
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

    # Loaded once per CTA, reused across the whole segment loop — this is the
    # q_load fix relative to the per-tile-grid kernel.
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


def _plan_segments(
    *,
    chunk_cap: int,
    b_count: int,
    num_idx_heads: int,
    kv_width: int,
    target_ctas: int = DEFAULT_TARGET_CTAS,
) -> int:
    """Segments for grid dim 2 so the launch lands near ``target_ctas`` CTAs.

    Columns are counted with a nominal 64-row q tile; the autotuner may pick a
    different BLOCK_SIZE_Q, which only moves total CTAs by a small factor.
    Capped at one 128-key tile per segment — below that a segment has no loop.
    """
    cols = max(1, chunk_cap // 64) * b_count * num_idx_heads
    return max(1, min(target_ctas // max(1, cols), -(-kv_width // 128)))


@torch.no_grad()
def token_select_prefill_seg(
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
    seqlens_cpu: Optional[list] = None,
    prefix_lens_cpu: Optional[list] = None,
    target_ctas: int = DEFAULT_TARGET_CTAS,
) -> torch.Tensor:
    """token_select_prefill with the segmented score kernel; same contract."""
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
            scores = torch.empty(
                (num_idx_heads, b_count * chunk_cap, kv_width),
                dtype=torch.float32,
                device=idx_q.device,
            )

            def _grid(meta):
                return (
                    triton.cdiv(chunk_cap, meta["BLOCK_SIZE_Q"]),
                    b_count * num_idx_heads,
                    num_segments,
                )

            _token_index_score_prefill_kernel_seg[_grid](
                idx_q,
                idx_k_cache,
                scores,
                req_to_token,
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
                req_to_token.stride(0),
                IS_FP8=is_fp8,
            )

            chunk_idx = _topk_positions(scores, topk)
            del scores

            for b in range(b_start, b_start + b_count):
                q_len = q_lens[b]
                lo = q_chunk_start
                hi = min(q_len, q_chunk_start + chunk_cap)
                if lo >= hi:
                    continue
                g_lo = cu_cpu[b] + lo
                g_hi = cu_cpu[b] + hi
                r_lo = (b - b_start) * chunk_cap
                topk_idx[:, g_lo:g_hi] = chunk_idx[:, r_lo : r_lo + (hi - lo)]
            del chunk_idx

    return topk_idx
