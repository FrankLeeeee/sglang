"""Token-granularity sparse GQA attention (MiniMax-M3 + DeepSeek-style selection).

The block-sparse path attends to ``topk`` *blocks* of ``block_size`` contiguous
tokens, so its inner loop walks a block id and then a contiguous run of tokens.
Token-granularity selection removes the block structure entirely: the indexer
picks ``topk`` individual token positions per query, and this kernel attends to
exactly those, in whatever order they land.

One kernel serves both prefill and decode. A query "row" is one (query token,
kv head) pair in both cases — prefill has ``total_extend_tokens`` rows, decode
has ``batch_size`` — so the only real difference is how much parallelism the
query axis already provides. Decode at small batch gets ``NUM_KV_CHUNKS`` > 1,
which splits the top-k list across programs and merges the partial softmax
states afterwards; prefill runs with a single chunk and skips the merge.

Shapes:
    q          [num_q_rows, num_q_heads, head_dim]
    k_cache    [max_slots, num_kv_heads, head_dim]   (paged)
    v_cache    [max_slots, num_kv_heads, head_dim]   (paged)
    topk_idx   [num_kv_heads, num_q_rows, topk]      token positions, -1 = pad
    out        [num_q_rows, num_q_heads, head_dim]

``topk_idx`` holds positions within the request (0 .. seq_len-1), not KV-pool
slots; the kernel resolves them through ``req_to_token`` exactly like the
block-sparse kernels do. Selection is already causal, so this kernel applies no
causal mask of its own.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import check_sparse_kv_fp8, robust_allocator


@triton.heuristics(
    {
        # tl.dot needs >= 16 rows; a GQA group narrower than that is padded.
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": bn}, num_warps=nw, num_stages=ns)
        for bn in (32, 64, 128)
        for nw in (2, 4, 8)
        for ns in (2, 3)
    ],
    key=["gqa_group_size", "head_dim", "NUM_KV_CHUNKS", "IS_FP8"],
)
@triton.jit
def _gqa_token_sparse_fwd_kernel(
    q_ptr,  # [num_q_rows, num_q_heads, head_dim]
    k_cache_ptr,  # [max_slots, num_kv_heads, head_dim]
    v_cache_ptr,  # [max_slots, num_kv_heads, head_dim]
    req_to_token_ptr,  # [max_reqs, max_kv_len]
    idx_ptr,  # [num_kv_heads, num_q_rows, topk]
    q_slot_ids_ptr,  # [num_q_rows] request slot owning each query row
    o_ptr,  # [NUM_KV_CHUNKS, num_q_rows, num_q_heads, head_dim]
    lse_ptr,  # [NUM_KV_CHUNKS, num_q_rows, num_q_heads] (log2-domain)
    # shape
    max_slots,
    gqa_group_size,
    head_dim,
    topk,
    # sm_scale
    sm_scale,
    # strides
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_r2t_b,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    stride_o_c,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_n,
    stride_l_h,
    # META
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    IS_FP8: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    pid_n = tl.program_id(0)  # query row
    pid_kh = tl.program_id(1)  # kv head
    pid_c = tl.program_id(2)  # top-k chunk (decode split-K)

    sm_scale_log2e = sm_scale * 1.4426950408889634

    sid = (tl.load(q_slot_ids_ptr + pid_n).to(tl.int64) + max_slots) % max_slots
    h0 = pid_kh * gqa_group_size

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    h_mask = off_h < gqa_group_size
    d_mask = off_d < head_dim

    q = tl.load(
        q_ptr
        + pid_n * stride_q_n
        + (h0 + off_h)[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )  # [H, D]

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)

    # Per-chunk slice of the top-k list. chunk_size depends on `topk`, a runtime
    # arg, so it must not be marked constexpr.
    chunk_size = (topk + NUM_KV_CHUNKS - 1) // NUM_KV_CHUNKS
    t_start = pid_c * chunk_size
    t_end = tl.minimum(t_start + chunk_size, topk)

    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_n * stride_ti_n

    for t0 in tl.range(t_start, t_end, BLOCK_SIZE_N):
        t = t0 + off_n
        t_mask = t < t_end
        pos = tl.load(idx_base + t * stride_ti_t, mask=t_mask, other=-1)
        # -1 marks a padded slot: fewer than `topk` tokens were available (short
        # context) or the selection ran out of causally-valid candidates.
        valid = t_mask & (pos >= 0)
        pos_safe = tl.where(valid, pos, 0)

        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos_safe,
            mask=valid,
            other=0,
        ).to(tl.int64)
        slots = (slots + max_slots) % max_slots

        k = tl.load(
            k_cache_ptr
            + slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=valid[None, :] & d_mask[:, None],
            other=0.0,
        )  # [D, N], already transposed for tl.dot
        if IS_FP8:
            k = k.to(q.dtype)

        qk = tl.dot(q, k) * sm_scale_log2e  # [H, N]
        qk = tl.where(valid[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        # An all-masked tile leaves m_ij at -inf; shifting by 0 instead keeps
        # exp2(-inf - 0) = 0 rather than exp2(-inf + inf) = NaN. Identical
        # whenever the tile has at least one valid key.
        m_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_cache_ptr
            + slots[:, None] * stride_v_s
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )  # [N, D]
        if IS_FP8:
            v = v.to(q.dtype)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    tl.store(
        o_ptr
        + pid_c * stride_o_c
        + pid_n * stride_o_n
        + (h0 + off_h)[:, None] * stride_o_h
        + off_d[None, :] * stride_o_d,
        acc.to(o_ptr.dtype.element_ty),
        mask=h_mask[:, None] & d_mask[None, :],
    )
    if WRITE_LSE:
        # log2-domain lse; an empty chunk (l_i == 0) yields -inf and therefore
        # zero weight in the merge.
        lse = tl.where(l_i == 0.0, float("-inf"), m_i + tl.log2(l_i))
        tl.store(
            lse_ptr
            + pid_c * stride_l_c
            + pid_n * stride_l_n
            + (h0 + off_h) * stride_l_h,
            lse,
            mask=h_mask,
        )


@triton.jit
def _merge_chunks_kernel(
    o_ptr,  # [NUM_KV_CHUNKS, num_q_rows, num_q_heads, head_dim], merged into chunk 0
    lse_ptr,  # [NUM_KV_CHUNKS, num_q_rows, num_q_heads]
    head_dim,
    stride_o_c,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_n,
    stride_l_h,
    NUM_KV_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    off_c = tl.arange(0, NUM_KV_CHUNKS)

    lse = tl.load(
        lse_ptr + off_c * stride_l_c + pid_n * stride_l_n + pid_h * stride_l_h
    )  # [C]
    lse_max = tl.max(lse, axis=0)
    # Every chunk empty (all -inf) means the query selected no valid token at
    # all; emit the zero output already stored rather than NaN.
    lse_max_safe = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    w = tl.exp2(lse - lse_max_safe)  # [C], zero for empty chunks
    w_sum = tl.sum(w, axis=0)
    w_sum = tl.where(w_sum == 0.0, 1.0, w_sum)

    o = tl.load(
        o_ptr
        + off_c[:, None] * stride_o_c
        + pid_n * stride_o_n
        + pid_h * stride_o_h
        + off_d[None, :] * stride_o_d,
        mask=d_mask[None, :],
        other=0.0,
    ).to(
        tl.float32
    )  # [C, D]
    merged = tl.sum(o * w[:, None], axis=0) / w_sum

    tl.store(
        o_ptr + pid_n * stride_o_n + pid_h * stride_o_h + off_d * stride_o_d,
        merged.to(o_ptr.dtype.element_ty),
        mask=d_mask,
    )


def pick_num_kv_chunks(
    num_q_rows: int, num_kv_heads: int, topk: int, *, target_grid: int = 4096
) -> int:
    """Split-K factor over the top-k list.

    The query axis alone gives ``num_q_rows * num_kv_heads`` programs, which is
    plenty during prefill but only a handful at decode with a small batch. Split
    the top-k list until the grid is big enough to fill the device, capped so a
    chunk still holds a useful number of tokens.
    """
    base = max(1, num_q_rows * num_kv_heads)
    want = max(1, min(topk, target_grid // base))
    # power of two: the merge kernel indexes chunks with tl.arange
    chunks = 1 << (want.bit_length() - 1)
    # keep at least 32 tokens per chunk so the inner tile is not mostly masked
    while chunks > 1 and topk // chunks < 32:
        chunks //= 2
    return chunks


@torch.no_grad()
def gqa_token_sparse_attn(
    q: torch.Tensor,  # [num_q_rows, num_q_heads, head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    q_slot_ids: torch.Tensor,  # [num_q_rows] request slot per query row
    topk_idx: torch.Tensor,  # [num_kv_heads, num_q_rows, topk], -1 padded
    sm_scale: Optional[float] = None,
    num_kv_chunks: Optional[int] = None,
) -> torch.Tensor:
    """GQA attention restricted to per-query token positions in ``topk_idx``."""
    triton.set_allocator(robust_allocator)
    is_fp8 = check_sparse_kv_fp8(q, k_cache, v_cache, label="token sparse")

    num_q_rows, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    assert v_cache.shape[1] == num_kv_heads and v_cache.shape[-1] == head_dim
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by "
        f"num_kv_heads ({num_kv_heads})"
    )
    gqa_group_size = num_q_heads // num_kv_heads
    assert topk_idx.shape[0] == num_kv_heads and topk_idx.shape[1] == num_q_rows, (
        f"topk_idx {tuple(topk_idx.shape)} does not match "
        f"[{num_kv_heads}, {num_q_rows}, topk]"
    )
    assert q_slot_ids.shape[0] == num_q_rows
    topk = topk_idx.shape[2]
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if num_kv_chunks is None:
        num_kv_chunks = pick_num_kv_chunks(num_q_rows, num_kv_heads, topk)
    write_lse = num_kv_chunks > 1

    o = torch.empty(
        num_kv_chunks, num_q_rows, num_q_heads, head_dim, dtype=q.dtype, device=q.device
    )
    if write_lse:
        lse = torch.empty(
            num_kv_chunks, num_q_rows, num_q_heads, dtype=torch.float32, device=q.device
        )
        lse_strides = (lse.stride(0), lse.stride(1), lse.stride(2))
    else:
        lse = None
        lse_strides = (0, 0, 0)

    if num_q_rows == 0:
        return o[0]

    grid = (num_q_rows, num_kv_heads, num_kv_chunks)
    _gqa_token_sparse_fwd_kernel[grid](
        q,
        k_cache,
        v_cache,
        req_to_token,
        topk_idx,
        q_slot_ids,
        o,
        lse,
        max_slots,
        gqa_group_size,
        head_dim,
        topk,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        req_to_token.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        *lse_strides,
        NUM_KV_CHUNKS=num_kv_chunks,
        IS_FP8=is_fp8,
        WRITE_LSE=write_lse,
    )

    if not write_lse:
        return o[0]

    _merge_chunks_kernel[(num_q_rows, num_q_heads)](
        o,
        lse,
        head_dim,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        NUM_KV_CHUNKS=num_kv_chunks,
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
    )
    return o[0]
