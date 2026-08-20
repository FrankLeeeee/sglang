"""Per-token indexer logits + token selection for MiniMax-M3 token-sparse attention.

The block-sparse indexer computes q·k for every (query, key) pair and then
*pools* each 128-key block into one score (max or lse) before selecting blocks.
Token granularity drops the pooling: the logits are kept per key and the
selection picks individual token positions, exactly like DeepSeek's sparse
attention indexer.

That has a direct consequence for prefill. The block path's score buffer is
``[idx_heads, total_q, ceil(L/128)]``; dropping the pooling makes it
``[idx_heads, total_q, L]`` — 128x larger, and quadratic in context. At 128k
that would be 64 GiB, so prefill *must* stream: the query axis is processed in
chunks sized to a memory budget, and each chunk's logits are selected and
discarded before the next is computed. Decode needs no chunking — its logits are
``[idx_heads, batch, L]``.

Forced attention sinks (``init_tokens``) and the sliding window
(``local_tokens``) are applied as score biases inside the scoring kernel, so the
selection stage stays a plain top-k. This mirrors what the block kernel does
with ``MASK_INIT=False`` / ``MASK_LOCAL=False``, which is how the MiniMax-M3
backend drives it.
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import robust_allocator

logger = logging.getLogger(__name__)


# Bias magnitudes that force a position into the top-k. Real logits are
# q·k*sm_scale*log2e — never within many orders of magnitude of these — so the
# ordering is: init tokens first, then local-window tokens, then by score.
# Exported as plain floats for reference implementations; the kernels need the
# tl.constexpr form because @triton.jit cannot read plain module globals.
INIT_BIAS = 1e30
LOCAL_BIAS = 1e29
_INIT_BIAS = tl.constexpr(INIT_BIAS)
_LOCAL_BIAS = tl.constexpr(LOCAL_BIAS)

# Default cap on the prefill logits buffer. 512 MiB keeps the chunked prefill
# well inside activation headroom at 128k while still giving the scoring GEMM a
# query tile worth several hundred rows.
DEFAULT_SCORE_BUDGET_BYTES = 512 * 1024 * 1024

# Same idea for decode, where the buffer is [idx_heads, batch, L]: 512 MiB at
# tp=1 / batch 32 / 1M context. Decode chunks the *key* axis rather than the
# query axis, so unlike prefill it costs no extra pass over the index KV cache —
# every key is still read exactly once. Above this size the selection runs as a
# per-window top-k plus an exact merge; below it, the whole width is selected in
# one shot exactly as before.
DEFAULT_DECODE_SCORE_BUDGET_BYTES = 128 * 1024 * 1024


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        # `chunk_cap` caps the useful rows of the Q tile, so it has to take part
        # in config selection — at 1M the budget drives it to 32 while the
        # smallest tile was 64, masking off half of every dot. Keying on the raw
        # value would re-autotune for every distinct context length a server
        # sees, so bucket it, as the block kernel does with BLOCK_SIZE_T.
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
        for ns in (2, 3)
    ],
    key=["qk_head_dim", "CHUNK_BUCKET", "IS_FP8"],
)
@triton.jit
def _token_index_score_prefill_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, d]
    k_cache_ptr,  # idx K paged: [max_slots, 1, d]
    score_ptr,  # [num_idx_heads, batch * chunk_cap, kv_width] float32
    req_to_token_ptr,  # [max_reqs, max_kv_len]
    cu_seqlens,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
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
    # strides
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_r2t_b,
    # META
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    CHUNK_BUCKET: tl.constexpr,  # autotune key only; see the heuristic above
    IS_FP8: tl.constexpr,
    # Benchmark hook (bench_indexer.py): False replaces the score-matrix store
    # with one scalar per CTA, so the q.k compute stays observable while the
    # HBM write-back all but disappears. Deliberately *not* an autotune key, so
    # both modes run the tile config tuned for the real (writing) kernel and
    # their timing difference is attributable to the store alone.
    WRITE_SCORES: tl.constexpr,
):
    pid_q, pid_bh, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # grid dim 1 covers only this batch slice; pid_b is the global request id.
    pid_b_local = pid_bh // num_idx_heads
    pid_b = batch_start + pid_b_local
    pid_h = pid_bh % num_idx_heads

    sm_scale_log2e = sm_scale * 1.4426950408889634

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots

    # local (within-request) query indices this program owns
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
    abs_q = prefix_len + local_q  # absolute KV position of each query

    q = tl.load(
        q_ptr
        + (seq_start + local_q)[:, None] * stride_q_n
        + pid_h * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )  # [Q, D]

    # row base in the chunk-local score buffer (slice-relative, not global)
    s_row = pid_b_local * chunk_cap + (local_q - q_chunk_start)
    s_base = score_ptr + pid_h * stride_s_h + s_row[:, None] * stride_s_n

    # Split the key axis across the third grid dimension. The previous kernel
    # looped over the entire context inside each query CTA; at a 512 MiB score
    # budget that leaves only a handful of CTAs active (16 at 128K on TP=8).
    # Score tiles are independent, so exposing K tiles in the launch grid gives
    # the GPU enough parallel work without changing the exact logits or top-k.
    pos = pid_k * BLOCK_SIZE_K + off_k
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
    )  # [D, K]
    if IS_FP8:
        k = k.to(q.dtype)

    score = tl.dot(q, k) * sm_scale_log2e  # [Q, K]

    causal = (abs_q[:, None] >= pos[None, :]) & pos_mask[None, :]
    # Forced positions: attention sinks at the front, sliding window at the
    # back. Guarded by `causal` so an out-of-range key is never promoted.
    init_m = causal & (pos[None, :] < init_tokens)
    local_m = causal & (pos[None, :] > (abs_q[:, None] - local_tokens))
    score = tl.where(init_m, _INIT_BIAS, score)
    score = tl.where(local_m & (init_m == 0), _LOCAL_BIAS, score)
    score = tl.where(causal, score, float("-inf"))

    if WRITE_SCORES:
        tl.store(
            s_base + pos[None, :] * stride_s_k,
            score,
            mask=q_mask[:, None] & in_width[None, :],
        )
    else:
        # One scalar per CTA keeps the dot alive under DCE. There are far fewer
        # CTAs than score elements (one per BLOCK_SIZE_Q x BLOCK_SIZE_K tile),
        # so any flat program id is in bounds of the score buffer.
        flat_pid = pid_q + tl.num_programs(0).to(tl.int64) * (
            pid_bh + tl.num_programs(1).to(tl.int64) * pid_k
        )
        tl.store(score_ptr + flat_pid, tl.max(score))


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["num_idx_heads"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": bk}, num_warps=nw, num_stages=ns)
        for bk in (64, 128, 256)
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["num_idx_heads", "qk_head_dim", "NUM_KV_CHUNKS", "IS_FP8"],
)
@triton.jit
def _token_index_score_decode_kernel(
    q_ptr,  # idx_q: [batch, num_idx_heads, d]
    k_cache_ptr,  # idx K paged: [max_slots, 1, d]
    score_ptr,  # [num_idx_heads, batch, win_width] float32
    req_to_token_ptr,
    seq_lens,
    slot_ids,
    # shape
    max_slots,
    batch_size,
    num_idx_heads,
    qk_head_dim,
    # The score buffer holds the key window [k_base, k_limit); columns are
    # window-relative. k_base=0 / k_limit=kv_width is the unchunked case.
    k_base,
    k_limit,
    win_width,
    init_tokens,
    local_tokens,
    sm_scale,
    # strides
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_d,
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_r2t_b,
    # META
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    IS_FP8: tl.constexpr,
    # Benchmark hook; see _token_index_score_prefill_kernel.
    WRITE_SCORES: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size

    sm_scale_log2e = sm_scale * 1.4426950408889634

    seq_len = tl.load(seq_lens + pid_b)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots
    # The decode query sits at the end of the context.
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

    q = tl.load(
        q_ptr
        + pid_b * stride_q_b
        + off_h[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )  # [H, D]

    s_base = score_ptr + off_h[:, None] * stride_s_h + pid_b * stride_s_b

    obs_max = tl.full((1,), value=float("-inf"), dtype=tl.float32)
    for i in tl.range(k_start, k_end, BLOCK_SIZE_K):
        pos = i + off_k
        in_width = pos < k_end
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
        )  # [D, K]
        if IS_FP8:
            k = k.to(q.dtype)

        score = tl.dot(q, k) * sm_scale_log2e  # [H, K]

        init_m = pos_mask[None, :] & (pos[None, :] < init_tokens)
        local_m = pos_mask[None, :] & (pos[None, :] > abs_q - local_tokens)
        score = tl.where(init_m, _INIT_BIAS, score)
        score = tl.where(local_m & (init_m == 0), _LOCAL_BIAS, score)
        score = tl.where(pos_mask[None, :], score, float("-inf"))

        if WRITE_SCORES:
            tl.store(
                s_base + (pos - k_base)[None, :] * stride_s_k,
                score,
                mask=h_mask[:, None] & in_width[None, :],
            )
        else:
            obs_max = tl.maximum(obs_max, tl.max(score))

    if not WRITE_SCORES:
        # One scalar per CTA (grid is batch * NUM_KV_CHUNKS <= the score
        # buffer's numel, whose width is at least topk); see the prefill kernel.
        tl.store(score_ptr + pid_bc + tl.arange(0, 1), obs_max)


_flashinfer_top_k = None
_flashinfer_tie_break = None
_flashinfer_checked = False


def _get_flashinfer_top_k():
    """FlashInfer's exact wide-row top-k, when the optional dep is installed."""
    global _flashinfer_top_k, _flashinfer_tie_break, _flashinfer_checked
    if not _flashinfer_checked:
        try:
            import flashinfer

            _flashinfer_top_k = flashinfer.top_k
            # The forced sink / sliding-window positions all carry the *same*
            # bias, so a row can hold more exact ties than the budget selects.
            # Any tie-consistent set is a correct top-k, but pinning the smaller
            # index keeps the choice deterministic and identical to torch.topk's,
            # which is what the fallback and the reference tests produce. Free.
            _flashinfer_tie_break = flashinfer.TopKTieBreak.SMALL
        except (ImportError, AttributeError):
            _flashinfer_top_k = None
        _flashinfer_checked = True
    return _flashinfer_top_k


def _select_topk(flat: torch.Tensor, k: int):
    """``(values, indices)`` of the exact top-k of every row of a 2D tensor.

    FlashInfer's selector is exact — verified against ``torch.topk`` at 64k, 256k
    and 1M wide, including the two widths where sgl-kernel's approximate one
    diverges — repeatable run to run, and ~3-3.8x faster (0.42 vs 1.57 ms on a
    128x1M float32 top-2048, H200). ``torch.topk`` remains the fallback: it is
    the same answer, only slower, so availability never changes the result.
    """
    top_k = _get_flashinfer_top_k()
    if top_k is not None and flat.is_contiguous() and flat.is_cuda:
        try:
            return top_k(
                flat,
                k,
                sorted=False,
                deterministic=True,
                tie_break=_flashinfer_tie_break,
            )
        except RuntimeError as err:
            # Unsupported dtype/shape for this build: drop to torch for good.
            global _flashinfer_top_k
            logger.warning(
                "flashinfer.top_k unavailable (%s); using torch.topk for "
                "MiniMax token-sparse selection.",
                err,
            )
            _flashinfer_top_k = None
    return torch.topk(flat, k, dim=-1, sorted=False)


def _topk_positions(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Exact top-k over the last axis, returning positions with -1 padding.

    Rows shorter than ``topk`` (short context, or a query early in the sequence)
    become -1 padding, which every consumer of ``topk_idx`` already handles.
    Positions outside a row's causal range carry ``-inf`` from the scoring
    kernel, so the padding falls out of the value test below and needs no
    separate per-row length input.

    This is deliberately ``torch.topk`` and not sgl-kernel's ``fast_topk_v2``.
    That selector is specialized for DeepSeek's topk=2048 and is ~2.5x faster,
    but it is *approximate*: measured against an exact top-k on H200 it returns
    positions below the true k-th value once rows grow past its tested range —
    ~1% of picks at 256k wide and ~6% at 1M (recall 93.9%, worst selected rank
    2200 of 2048) — and it is not run-to-run deterministic. Its own test
    (``sgl-kernel/tests/test_topk.py``) permits up to 5 wrong entries and only
    covers widths up to 64k. M3 runs to 1M, so the token indexer selects exactly.
    """
    width = scores.shape[-1]
    k = min(topk, width)
    # The selector takes 2D rows; the leading axes are just row identity.
    values, indices = _select_topk(scores.reshape(-1, width), k)
    values = values.view(*scores.shape[:-1], k)
    indices = indices.view(*scores.shape[:-1], k)
    indices = torch.where(
        values > float("-inf"),
        indices.to(torch.int32),
        torch.full_like(indices, -1, dtype=torch.int32),
    )
    if k < topk:
        pad = indices.new_full((*indices.shape[:-1], topk - k), -1)
        indices = torch.cat([indices, pad], dim=-1)
    return indices


def plan_query_chunk(
    *,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_idx_heads: int,
    score_budget_bytes: int = DEFAULT_SCORE_BUDGET_BYTES,
    tile: int = 64,
) -> tuple[int, int]:
    """Plan ``(batch_chunk, query_chunk)`` so the logits buffer fits the budget.

    The buffer is ``[idx_heads, batch_chunk * query_chunk, kv_width] float32``,
    so there are two axes to shrink. Query is preferred — it keeps whole requests
    together and the scoring GEMM well shaped — but a large batch at long context
    can drive the query chunk below one tile, and then shrinking batch as well is
    what keeps the cap honest. Both are per-request-independent, so neither
    changes the selection result.

    The budget is a **hard** cap: the return value is never rounded up, and a
    configuration that cannot fit even one query row of one request raises rather
    than over-allocating. That is the only irreducible case — escaping it would
    mean chunking the KV axis too, turning the exact top-k into a multi-pass
    candidate merge.
    """
    # bytes for one query row of one request
    unit = num_idx_heads * max_seqlen_k * 4
    rows_affordable = score_budget_bytes // max(1, unit)
    if rows_affordable < 1:
        raise ValueError(
            f"MiniMax token-sparse prefill cannot honour the "
            f"{score_budget_bytes >> 20} MiB score budget: a single query row of "
            f"[{num_idx_heads} index heads x {max_seqlen_k} context] already needs "
            f"{(unit + (1 << 20) - 1) >> 20} MiB. Raise "
            f"SGLANG_MINIMAX_TOKEN_SPARSE_SCORE_BUDGET_MB, shorten the context, or "
            f"use --granularity block."
        )

    if rows_affordable >= batch_size * tile:
        # Whole batch fits with at least a full query tile each.
        batch_chunk = batch_size
    else:
        # Give each retained request at least one tile where possible.
        batch_chunk = max(1, min(batch_size, rows_affordable // tile))

    query_chunk = max(1, rows_affordable // batch_chunk)
    if query_chunk >= tile:
        query_chunk = (query_chunk // tile) * tile
    query_chunk = min(query_chunk, max_seqlen_q)
    return batch_chunk, query_chunk


@torch.no_grad()
def token_select_prefill(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, d]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, d]
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,  # [batch]
    cu_seqlens: torch.Tensor,  # [batch + 1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    topk: int,
    init_tokens: int,
    local_tokens: int,
    sm_scale: Optional[float] = None,
    seqlens_cpu: Optional[list] = None,
    prefix_lens_cpu: Optional[list] = None,
    score_budget_bytes: int = DEFAULT_SCORE_BUDGET_BYTES,
    score_writeback: bool = True,
) -> torch.Tensor:
    """Select ``topk`` token positions per query token. Returns [heads, total_q, topk].

    ``score_writeback=False`` is a benchmark hook (bench_indexer.py): the scoring
    kernels run with the score-matrix HBM store disabled and the selection is
    skipped entirely, so the returned indices are all -1 and meaningless. Timing
    this against the default run isolates the write-back cost of the logits.
    """
    triton.set_allocator(robust_allocator)
    total_q, num_idx_heads, qk_head_dim = idx_q.shape
    max_slots = idx_k_cache.shape[0]
    batch_size = cu_seqlens.shape[0] - 1
    is_fp8 = idx_k_cache.dtype != idx_q.dtype
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5

    topk_idx = torch.full(
        (num_idx_heads, total_q, topk),
        -1,
        dtype=torch.int32,
        device=idx_q.device,
    )
    if total_q == 0:
        return topk_idx

    # Host-side lengths drive the chunk loop. Callers pass them in (ForwardBatch
    # already keeps both on the host) so the loop costs no device sync; the
    # fallbacks below do sync, which is fine for tests but not for serving.
    if seqlens_cpu is None:
        q_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()  # syncs
    else:
        q_lens = list(seqlens_cpu)
    cu_cpu = [0]
    for n in q_lens:
        cu_cpu.append(cu_cpu[-1] + n)
    if prefix_lens_cpu is not None:
        max_prefix = max(prefix_lens_cpu) if len(prefix_lens_cpu) else 0
    else:
        # Without host-side prefix lengths, fall back to the loosest safe bound
        # rather than syncing. This only widens the per-chunk logits row (more
        # -inf tail to fill), never changes the result.
        max_prefix = max_seqlen_k

    batch_cap, chunk_cap = plan_query_chunk(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_idx_heads=num_idx_heads,
        score_budget_bytes=score_budget_bytes,
    )

    for b_start in range(0, batch_size, batch_cap):
        b_count = min(batch_cap, batch_size - b_start)
        for q_chunk_start in range(0, max_seqlen_q, chunk_cap):
            # No query in this chunk can see past its own position, so the logits
            # row only needs to be this wide.
            kv_width = min(max_seqlen_k, max_prefix + q_chunk_start + chunk_cap)
            scores = torch.empty(
                (num_idx_heads, b_count * chunk_cap, kv_width),
                dtype=torch.float32,
                device=idx_q.device,
            )

            def _grid(meta):
                return (
                    triton.cdiv(chunk_cap, meta["BLOCK_SIZE_Q"]),
                    b_count * num_idx_heads,
                    triton.cdiv(kv_width, meta["BLOCK_SIZE_K"]),
                )

            _token_index_score_prefill_kernel[_grid](
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
                WRITE_SCORES=score_writeback,
            )

            if not score_writeback:
                del scores
                continue

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


def plan_key_window(
    *,
    kv_width: int,
    topk: int,
    num_idx_heads: int,
    batch_size: int,
    score_budget_bytes: int = DEFAULT_DECODE_SCORE_BUDGET_BYTES,
) -> tuple[int, int]:
    """Plan ``(window, num_windows)`` so the decode logits buffer fits the budget.

    Decode shrinks the *key* axis, not the query axis. Every key is still read
    exactly once — only the live score slice gets smaller — so unlike the prefill
    planner this costs no extra traffic against the index KV cache.

    A window narrower than ``topk`` cannot reduce anything (each window still
    contributes up to ``topk`` candidates), so that is the floor. The budget is
    advisory in exactly that one case.
    """
    row_bytes = max(1, num_idx_heads * batch_size * 4)
    window = max(topk, score_budget_bytes // row_bytes)
    if window >= kv_width:
        return kv_width, 1
    # Power of two keeps the scoring kernel's key tiling aligned.
    window = max(topk, 1 << (window.bit_length() - 1))
    if window >= kv_width:
        return kv_width, 1
    return window, (kv_width + window - 1) // window


def _merge_candidates(
    cand_val: torch.Tensor, cand_idx: torch.Tensor, topk: int
) -> torch.Tensor:
    """Exact top-k over the union of every window's candidates.

    Sound because a position in the global top-k is necessarily in the top-k of
    the one window that contains it, so no candidate can be missed by taking
    ``topk`` per window. Rows with fewer than ``topk`` real positions keep the
    ``-inf`` fill and come back as -1, matching the unchunked path.
    """
    values, pos = torch.topk(cand_val, topk, dim=-1, sorted=False)
    picked = torch.gather(cand_idx, -1, pos)
    return torch.where(values > float("-inf"), picked, torch.full_like(picked, -1))


@torch.no_grad()
def token_select_decode(
    idx_q: torch.Tensor,  # [batch, num_idx_heads, d]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, d]
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen: int,
    topk: int,
    init_tokens: int,
    local_tokens: int,
    sm_scale: Optional[float] = None,
    score_budget_bytes: int = DEFAULT_DECODE_SCORE_BUDGET_BYTES,
    score_writeback: bool = True,
) -> torch.Tensor:
    """Select ``topk`` token positions per decode query. Returns [heads, batch, topk].

    ``score_writeback=False`` is the same benchmark hook as on
    ``token_select_prefill``: scoring runs without the logits store, selection is
    skipped, and the returned indices are meaningless.

    The logits are ``[idx_heads, batch, L]`` float32, which is 512 MiB at tp=1
    with batch 32 and a 1M context. Past ``score_budget_bytes`` the key axis is
    walked in windows and the per-window top-k are merged, which is exact and
    keeps every key a single read.
    """
    triton.set_allocator(robust_allocator)
    batch_size, num_idx_heads, qk_head_dim = idx_q.shape
    max_slots = idx_k_cache.shape[0]
    is_fp8 = idx_k_cache.dtype != idx_q.dtype
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5

    kv_width = min(max_seqlen, req_to_token.shape[1])
    if batch_size == 0:
        return torch.full(
            (num_idx_heads, 0, topk), -1, dtype=torch.int32, device=idx_q.device
        )

    window, num_windows = plan_key_window(
        kv_width=kv_width,
        topk=topk,
        num_idx_heads=num_idx_heads,
        batch_size=batch_size,
        score_budget_bytes=score_budget_bytes,
    )
    scores = torch.empty(
        (num_idx_heads, batch_size, window), dtype=torch.float32, device=idx_q.device
    )

    # Split the context so a small decode batch still fills the device.
    target_grid = 2048
    want = max(1, min(64, target_grid // max(1, batch_size)))
    num_kv_chunks = 1 << (want.bit_length() - 1)

    if num_windows > 1 and score_writeback:
        cand_val = torch.full(
            (num_idx_heads, batch_size, num_windows * topk),
            float("-inf"),
            device=idx_q.device,
        )
        cand_idx = torch.full(
            (num_idx_heads, batch_size, num_windows * topk),
            -1,
            dtype=torch.int32,
            device=idx_q.device,
        )

    for w in range(num_windows):
        base = w * window
        limit = min(base + window, kv_width)
        _token_index_score_decode_kernel[(batch_size * num_kv_chunks,)](
            idx_q,
            idx_k_cache,
            scores,
            req_to_token,
            seq_lens,
            slot_ids,
            max_slots,
            batch_size,
            num_idx_heads,
            qk_head_dim,
            base,
            limit,
            limit - base,
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
            NUM_KV_CHUNKS=num_kv_chunks,
            IS_FP8=is_fp8,
            WRITE_SCORES=score_writeback,
        )
        if not score_writeback:
            continue
        if num_windows == 1:
            return _topk_positions(scores, topk)

        live = scores[:, :, : limit - base]
        take = min(topk, limit - base)
        # Exact per window: the merge below trusts each window's candidate
        # list to hold every position of the window that can reach the
        # global top-k.
        val, idx = _select_topk(live.reshape(-1, limit - base), take)
        val = val.view(num_idx_heads, batch_size, take)
        idx = idx.view(num_idx_heads, batch_size, take)
        lo, hi = w * topk, w * topk + take
        cand_val[:, :, lo:hi] = val
        cand_idx[:, :, lo:hi] = (idx + base).to(torch.int32)

    if not score_writeback:
        return torch.full(
            (num_idx_heads, batch_size, topk),
            -1,
            dtype=torch.int32,
            device=idx_q.device,
        )
    return _merge_candidates(cand_val, cand_idx, topk)
