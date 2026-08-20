#!/usr/bin/env python3
"""Two-level (block -> token) GQA indexer for MiniMax-M3, in Triton.

Prototype. Lives in the benchmark directory, not in ``sglang.kernels``, until the
accuracy/latency trade it makes is settled.

## What it is

M3's shipped indexer is *flat*: it scores every one of the L keys in the context
against 4 index heads, pools each 128-key block to one score, and selects 16
blocks. The token variant in ``minimax_sparse/token/`` drops the pooling and
selects 2048 individual positions. Both pay an O(L) score pass **per query row**,
and the token one additionally materialises an O(L)-wide fp32 logits row.

LongCat-2.0's *Hierarchical Indexing* (LongCat Sparse Attention, arXiv 2608.01662)
replaces that with a coarse-to-fine pass:

1. **block level** — each page of P tokens is represented by the *mean* of its
   index keys. Scoring the query against those L/P representatives costs 1/P of
   the flat pass. The top-M pages are recalled (LongCat: P=128, M=1024).
2. **token level** — the flat per-key indexer runs, but only over the M*P
   recalled tokens, and the final top-k is selected out of that candidate set.

Cost goes from O(L) to O(L/P + M*P) per query row: the second term is a
*constant* in context length, so past the crossover the indexer stops growing
with L. LongCat only turns HI on at >=256k for exactly that reason — below it,
M*P covers the whole context and the coarse pass is pure overhead.

This file implements that scheme against M3's shapes and sglang's paged index KV
cache, and makes it **GQA-native**: the selection is produced once per *KV head*
(reduced across the index heads that serve it), so all `gqa_group_size` query
heads sharing a KV head read one token list, which is exactly what
``gqa_token_sparse_attn`` consumes. The output is therefore a drop-in
replacement for ``token_select_prefill`` / ``token_select_decode``
(`[num_kv_heads, rows, topk]`, -1 padded).

## Structure

    build_pooled_index_keys     stage 0  mean-pool index K per P-token block
    _coarse_block_score_kernel  stage 1  q . k_mean -> [heads, tiles, L/P]   (pre)
    _coarse_score_pool_kernel   stage 1  max(q . k) -> [heads, tiles, L/P]   (post)
    _topk_positions (reused)    stage 1  top-M blocks per query tile
    _fine_token_score_kernel    stage 2  q . k      -> [kv_heads, rows, M*P]
    _topk_positions (reused)    stage 2  top-k candidate columns per row
    _map_columns_kernel         stage 2  column -> absolute token position

Stage 0 is a *cache*, not per-call work: a server maintains the pooled block
representation as index K is written, at O(1) amortised cost per token (a running
mean is exact under append-only writes). The benchmark measures it both ways,
because rebuilding is what an unmodified KV pool would force.

Stages 1 and 2 reuse the production selector ``_topk_positions`` — FlashInfer's
exact wide-row top-k with a torch fallback — so nothing here re-litigates
selection exactness; the approximation lives entirely in *what gets scored*.

One kernel pair serves prefill and decode. Rows are described by
``cu_rows``/``prefix_lens`` — prefill gives a request q_len rows starting at
absolute position ``prefix_len``, decode gives it one row at ``seq_len - 1`` —
so there is no second copy of either kernel.

## Where the time goes

H200, released M3 shape, P=128 M=128, clustered keys, GPU kernel time:

    prefill, 2048-token extend       16k     64k    256k      1M
      block (shipped)               0.33    1.34    5.21   22.14 ms
      flat token                    1.69    3.80   12.66   54.60 ms
      two-level                     3.04    3.05    3.09    3.20 ms

    decode selection, batch 32
      flat token                    0.15    0.38    1.32    5.14 ms
      two-level                     0.24    0.26    0.28    0.36 ms

The two-level pass is flat in context — that is the whole result. It loses below
the crossover (~48k vs flat, ~120k vs the shipped block indexer), which is the
same reason LongCat gates HI at >=256k.

Two caveats the benchmark makes visible. Wall clock at decode is 3-7x kernel
time at small batch: the driver is host-launch bound (six launches and two
selector calls for ~0.1 ms of work), which a server largely hides by capturing
decode in a CUDA graph. And the fine kernel's query tile holds one row at decode
against a 16-row minimum, so 15/16 of that dot is masked off — the first thing a
real implementation would fix.

## What it approximates

The flat token indexer is an *exact* top-k over the causal context. This is not,
in two separate ways:

* a token whose block's mean key scores poorly is never seen by stage 2, which
  is the LongCat approximation and is governed by M (at 128k, M=128 recalls
  12.5% of the context and reproduces the exact top-2048 to within 0.05%);
* a token whose block was crowded out of its *query tile's* shared recall is
  likewise never seen, which is this prototype's own approximation and is
  governed by ``query_tile`` — see the note on that field, it is the riskier of
  the two.

Recall against the exact selection is the quantity of interest for both, and
``selection_recall`` / ``test_two_level_indexer.py`` / ``bench_two_level_indexer.py
--mode recall`` measure them. Forced sinks (``init_tokens``) and the sliding
window (``local_tokens``) are exempt from both — they are biased into every
level, so they survive coarse recall and the tile union by construction.
"""

from __future__ import annotations

import math
from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (
    INIT_BIAS,
    LOCAL_BIAS,
    _topk_positions,
)

_INIT_BIAS = tl.constexpr(INIT_BIAS)
_LOCAL_BIAS = tl.constexpr(LOCAL_BIAS)

# How the index heads serving one KV head are folded into that head's single
# selection. "max" keeps a position any one head wants; "sum" and "relu_sum"
# (LongCat's choice) make heads vote. Identity when num_idx_heads == num_kv_heads,
# which is M3's released shape.
_REDUCE_MAX = tl.constexpr(0)
_REDUCE_SUM = tl.constexpr(1)
_REDUCE_RELU_SUM = tl.constexpr(2)
_REDUCE_CODES = {"max": 0, "sum": 1, "relu_sum": 2}

# Live fp32 score bytes the two levels may hold at once. Both buffers are
# per-row-constant in width (L/P and M*P), so unlike the flat token indexer's
# budget this bounds a chunk size that does *not* shrink as context grows.
DEFAULT_SCORE_BUDGET_BYTES = 512 * 1024 * 1024


class TwoLevelConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Hyper-parameters of the two-level indexer."""

    # Stage 1: coarse. `pool_block_size` is the pooling granularity (LongCat's
    # page size P); `coarse_blocks` is M, how many of them stage 2 then sees.
    pool_block_size: int = 128
    coarse_blocks: int = 128

    # Stage 2: fine. The final token budget handed to the attention kernel.
    topk_tokens: int = 2048

    # How many consecutive query rows share one stage-1 recall. This is the
    # prototype's central bet, and it cuts both ways.
    #
    # It is what makes the scheme fast: it lets stage 2 amortise a key tile over
    # a tile of queries, exactly as the flat prefill kernel does. At 1 the fine
    # pass re-reads its candidates once per query row and the whole thing is
    # *slower* than the flat indexer it replaces — 2.09 ms against 0.37 ms for
    # the same 256-token extend at 128k, and slower than flat outright at a
    # 2048-token one.
    #
    # It is also the largest accuracy risk. The tile's candidate set is the union
    # over its rows (rows reduce with max, whatever `head_reduce` is) but still
    # capped at M blocks, so a row loses a block whenever the rest of the tile
    # crowds it out. How often that happens depends entirely on whether
    # neighbouring queries want the same blocks. Measured at 128k, M=128:
    #
    #     query_tile        1       16       64      128
    #     correlated Q   99.9%    98.9%    98.4%    92.5%
    #     iid Q          99.9%    57.2%    31.4%    23.5%
    #
    # 64 is chosen on the correlated row, which is the premise every
    # query-block-granular selector already runs on, and the iid row is what the
    # harness's independent random query vectors give — a floor, not a forecast.
    # Neither is the released model. `make_clustered_index_queries` is how the
    # benchmark produces the first row; a real answer needs the checkpoint.
    #
    # The final top-k stays strictly per row either way. Decode forces 1: there
    # is one query row per request, so there is nothing to share.
    query_tile: int = 64

    # Whether stage 1 recalls one candidate set for *all* index heads, or one
    # per KV head. LongCat shares — its indexer emits a single per-token score,
    # so there is only one ranking to have — and sharing should in principle cut
    # stage 1 by the head count and let the KV heads hit in L2 in stage 2.
    #
    # It does not pay here, so it is off. Measured on H200, 128k, clustered keys:
    # sharing changed latency by under 2% in both phases (decode 0.102 vs 0.109
    # ms of kernel time, prefill 3.10 vs 3.10 ms) — neither stage is bandwidth
    # bound at these sizes — while recall against the exact top-2048 fell from
    # 99.9% to 90.5% at M=128 and from 89.4% to 46.2% at M=32. Four index heads
    # disagree about which blocks matter far more than the L2 argument is worth.
    # Kept as a knob because both numbers move with head count and with M.
    share_candidates: bool = False

    # Where stage 1 pools: before the q.k dot, or after it.
    #
    #   "pre"   score = q . mean(K_block).  LongCat/HISA. Stage 1 reads L/P
    #           pooled keys — the only variant whose first stage is sublinear in
    #           context, and the reason the whole scheme is flat in L. Costs a
    #           maintained pooled cache (stage 0), and scores a block by its
    #           average key rather than its best one.
    #
    #   "post"  score = max over the block of q . k. This *is* MiniMax-M3's
    #           shipped block indexer (`sparse_score_type="max"`), used as a
    #           recall stage rather than as the final answer — at M = topk/P the
    #           two produce the identical token set, which
    #           test_post_pooling_at_block_budget_is_the_shipped_block_indexer
    #           pins exactly. Stage 1 reads all L keys, so the pass is
    #           O(L) + O(M*P) and its asymptote is the block path's, not flat.
    #           In exchange stage 0 disappears — no pooled cache, no maintenance,
    #           no rebuild cliff — and the block ranking uses real per-key scores.
    #
    # Measured on H200, clustered keys, M=128, GPU kernel time:
    #
    #     prefill (2048-token extend)   16k     64k    256k      1M
    #       pre                        3.04    3.05    3.09    3.21 ms
    #       post                       3.40    4.46    8.75   25.83 ms
    #       (block indexer, shipped)   0.33    1.34    5.22   22.15 ms
    #
    # So "post" costs the shipped path plus a roughly constant refine — a +17%
    # premium over it at 1M, +68% at 256k — and buys near-exact token selection
    # for it. "pre" costs neither and stays flat.
    #
    # The ranking-quality argument does *not* go the way the exactness of the
    # scores suggests. Coverage of the exact top-2048 at 128k:
    #
    #     M                 16      32      64     128     256
    #       pre          70.1%   90.4%   98.6%   99.9%    100%
    #       post         67.8%   87.9%   98.5%    100%    100%
    #
    # Mean pooling ranks *better* below M=128 and worse above it, and the reason
    # is what each score is for. Max says "this block holds one excellent key",
    # mean says "this block is collectively aligned with the query". When M is
    # tight the refine stage keeps most of each recalled block, so collective
    # relevance is the better predictor; when M is loose it cherry-picks inside
    # the blocks, so the presence of peaks is. Neither pooling is uniformly the
    # better block score — it depends on how much room stage 2 has.
    pool_position: str = "pre"

    # Forced regions, in tokens, biased into both levels.
    init_tokens: int = 0
    local_tokens: int = 128

    head_reduce: str = "max"

    def __post_init__(self) -> None:
        assert self.pool_block_size in (16, 32, 64, 128, 256), (
            f"pool_block_size must be a power of two in [16, 256], "
            f"got {self.pool_block_size}"
        )
        assert self.pool_position in ("pre", "post"), (
            f"pool_position must be 'pre' (pool keys, then score — LongCat) or "
            f"'post' (score keys, then pool — M3's block indexer), "
            f"got {self.pool_position!r}"
        )
        assert (
            self.query_tile >= 1 and (self.query_tile & (self.query_tile - 1)) == 0
        ), f"query_tile must be a power of two, got {self.query_tile}"
        assert self.coarse_blocks > 0 and self.topk_tokens > 0
        assert self.head_reduce in _REDUCE_CODES, (
            f"head_reduce must be one of {sorted(_REDUCE_CODES)}, "
            f"got {self.head_reduce!r}"
        )
        assert self.candidate_width >= self.topk_tokens, (
            f"stage 1 recalls {self.candidate_width} candidate tokens "
            f"({self.coarse_blocks} x {self.pool_block_size}) but stage 2 must "
            f"select {self.topk_tokens} of them — raise coarse_blocks"
        )

    @property
    def candidate_width(self) -> int:
        """Tokens stage 2 scores per query row. Constant in context length."""
        return self.coarse_blocks * self.pool_block_size

    def coarse_width(self, max_seqlen_k: int) -> int:
        """Blocks stage 1 scores per query row."""
        return math.ceil(max_seqlen_k / self.pool_block_size)

    def is_degenerate(self, max_seqlen_k: int) -> bool:
        """True when stage 1 recalls the whole context, so it only costs.

        This is LongCat's >=256k gate expressed as a property of the shape: below
        the crossover the flat indexer is strictly the better kernel.
        """
        return self.candidate_width >= max_seqlen_k

    def replace(self, **kwargs) -> "TwoLevelConfig":
        return msgspec.structs.replace(self, **kwargs)

    def tag(self) -> str:
        return (
            f"p{self.pool_block_size}_m{self.coarse_blocks}"
            f"_k{self.topk_tokens}_qt{self.query_tile}_{self.head_reduce}"
            + ("_shared" if self.share_candidates else "_perhead")
        )


# ---------------------------------------------------------------------------
# stage 0 — mean-pooled block representation of the index K cache
# ---------------------------------------------------------------------------


@triton.heuristics(
    {
        "BLOCK_SIZE_P": lambda args: triton.next_power_of_2(args["pool_block"]),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _pool_index_keys_kernel(
    k_cache_ptr,  # index K paged: [max_slots, 1, d]
    req_to_token_ptr,  # [max_reqs, max_kv_len]
    seq_lens,
    slot_ids,
    pooled_ptr,  # [batch, num_blocks, d]
    max_slots,
    head_dim,
    pool_block,
    stride_k_s,
    stride_k_d,
    stride_r2t_b,
    stride_p_b,
    stride_p_n,
    stride_p_d,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_n = tl.program_id(0), tl.program_id(1)

    seq_len = tl.load(seq_lens + pid_b)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots

    off_p = tl.arange(0, BLOCK_SIZE_P)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim

    pos = pid_n * pool_block + off_p
    # The tail block of a request is partial; it is pooled over its live tokens
    # only, so a short last block is not diluted toward zero.
    live = (off_p < pool_block) & (pos < seq_len)

    slots = tl.load(req_to_token_ptr + sid * stride_r2t_b + pos, mask=live, other=0)
    slots = (slots.to(tl.int64) + max_slots) % max_slots
    k = tl.load(
        k_cache_ptr + slots[:, None] * stride_k_s + off_d[None, :] * stride_k_d,
        mask=live[:, None] & d_mask[None, :],
        other=0.0,
    )  # [P, D]

    count = tl.sum(live.to(tl.float32))
    mean = tl.sum(k.to(tl.float32), axis=0) / tl.maximum(count, 1.0)
    tl.store(
        pooled_ptr + pid_b * stride_p_b + pid_n * stride_p_n + off_d * stride_p_d,
        mean.to(pooled_ptr.dtype.element_ty),
        mask=d_mask,
    )


@torch.no_grad()
def build_pooled_index_keys(
    idx_k_cache: torch.Tensor,  # [max_slots, 1, d]
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    pool_block_size: int,
    max_seqlen_k: int,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Mean-pool the index K cache into ``[batch, ceil(L/P), d]``.

    Stage 0 of the two-level indexer, and the only piece that is *state* rather
    than compute: a server would update the affected block(s) as index K is
    written (a running mean is exact under append-only writes), so a decode step
    touches exactly one block. Rebuilding the whole thing per call, which is what
    this does when ``out`` is not supplied, is the unmodified-KV-pool worst case.
    """
    batch = int(slot_ids.shape[0])
    head_dim = int(idx_k_cache.shape[-1])
    num_blocks = math.ceil(max_seqlen_k / pool_block_size)
    if out is None:
        out = torch.empty(
            (batch, num_blocks, head_dim),
            dtype=dtype or idx_k_cache.dtype,
            device=idx_k_cache.device,
        )
    if batch == 0 or num_blocks == 0:
        return out

    _pool_index_keys_kernel[(batch, num_blocks)](
        idx_k_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        out,
        idx_k_cache.shape[0],
        head_dim,
        pool_block_size,
        idx_k_cache.stride(0),
        idx_k_cache.stride(2),
        req_to_token.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


# ---------------------------------------------------------------------------
# stage 1 — coarse block scoring
# ---------------------------------------------------------------------------


@triton.heuristics(
    {
        "BLOCK_SIZE_Q": lambda args: max(16, args["QUERY_TILE"]),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": bn}, num_warps=nw, num_stages=ns)
        for bn in (32, 64, 128)
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["qk_head_dim", "QUERY_TILE", "IDX_GROUP", "REDUCE"],
)
@triton.jit
def _coarse_block_score_kernel(
    q_ptr,  # idx_q: [rows, num_idx_heads, d]
    pooled_ptr,  # [batch, num_blocks, d]
    score_ptr,  # [num_kv_heads, b_count * tiles_per_chunk, num_blocks] fp32
    cu_rows,  # [batch + 1] first row of each request
    seq_lens,
    prefix_lens,  # absolute KV position of each request's first row
    # shape
    num_kv_heads,
    qk_head_dim,
    num_blocks,
    pool_block,
    tiles_per_chunk,
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
    stride_p_b,
    stride_p_n,
    stride_p_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    # META
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    IDX_GROUP: tl.constexpr,  # index heads per KV head
    REDUCE: tl.constexpr,
):
    # One program per (query tile, KV head, block tile). The tile's rows reduce
    # to a single score row: stage 1 recalls one candidate set for the tile.
    pid_t, pid_bh, pid_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pid_b_local = pid_bh // num_kv_heads
    pid_b = batch_start + pid_b_local
    pid_kvh = pid_bh % num_kv_heads

    sm_scale_log2e = sm_scale * 1.4426950408889634

    row_start = tl.load(cu_rows + pid_b)
    q_len = tl.load(cu_rows + pid_b + 1) - row_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)

    q_lo = q_chunk_start + pid_t * QUERY_TILE
    q_hi = tl.minimum(tl.minimum(q_lo + QUERY_TILE, q_len), q_chunk_start + chunk_cap)
    if q_lo >= q_hi:
        return

    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < qk_head_dim

    local_q = q_lo + off_q
    q_mask = local_q < q_hi
    # The tile's first and last live query, in absolute KV positions.
    abs_q_min = prefix_len + q_lo
    abs_q_max = prefix_len + q_hi - 1

    blk = pid_n * BLOCK_SIZE_N + off_n
    # A block exists for this request if it holds at least one live token.
    blk_mask = (blk < num_blocks) & (blk * pool_block < seq_len)

    # [D, N] — the pooled key of each candidate block, laid out for the dot.
    kmean = tl.load(
        pooled_ptr
        + pid_b * stride_p_b
        + blk[None, :] * stride_p_n
        + off_d[:, None] * stride_p_d,
        mask=d_mask[:, None] & blk_mask[None, :],
        other=0.0,
    )

    acc = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_N), dtype=tl.float32)
    for g in tl.static_range(IDX_GROUP):
        q = tl.load(
            q_ptr
            + (row_start + local_q)[:, None] * stride_q_n
            + (pid_kvh * IDX_GROUP + g) * stride_q_h
            + off_d[None, :] * stride_q_d,
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        )  # [Q, D]
        s = tl.dot(q, kmean) * sm_scale_log2e
        if REDUCE == _REDUCE_MAX:
            acc = s if g == 0 else tl.maximum(acc, s)
        elif REDUCE == _REDUCE_RELU_SUM:
            acc += tl.maximum(s, 0.0)
        else:
            acc += s

    # Fold the tile's rows into one score row. This is always a max — it is a
    # *union* over the rows' wishes, not the configurable head reduction — so a
    # block wanted by any row of the tile can still be recalled for all of them.
    tile_score = tl.max(tl.where(q_mask[:, None], acc, float("-inf")), axis=0)

    # Block-level causality against the tile's last query; stage 2 masks per row.
    blk_start = blk * pool_block
    visible = (abs_q_max >= blk_start) & blk_mask
    # Forced regions, so coarse recall can never drop a sink or a window token.
    # The window is the union over the tile, hence the *first* row's bound.
    init_m = visible & (blk_start < init_tokens)
    local_m = visible & ((blk_start + pool_block) > (abs_q_min - local_tokens + 1))
    tile_score = tl.where(init_m, _INIT_BIAS, tile_score)
    tile_score = tl.where(local_m & (init_m == 0), _LOCAL_BIAS, tile_score)
    tile_score = tl.where(visible, tile_score, float("-inf"))

    s_row = pid_b_local * tiles_per_chunk + pid_t
    tl.store(
        score_ptr + pid_kvh * stride_s_h + s_row * stride_s_n + blk * stride_s_k,
        tile_score,
        mask=blk < num_blocks,
    )


# Autotune is the *outer* decorator here, unlike the other kernels: BLOCK_SIZE_K
# is derived from the config's K_MULT, so the heuristic has to run after a config
# is chosen, not before.
@triton.autotune(
    configs=[
        triton.Config({"K_MULT": km}, num_warps=nw, num_stages=ns)
        for km in (1, 2, 4)
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["qk_head_dim", "QUERY_TILE", "POOL_BLOCK", "IDX_GROUP", "REDUCE", "IS_FP8"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_Q": lambda args: max(16, args["QUERY_TILE"]),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        # A key tile is a whole number of pooling blocks, so the pool reduction
        # below is a reshape rather than a scan across tile boundaries.
        "BLOCK_SIZE_K": lambda args: args["POOL_BLOCK"] * args["K_MULT"],
    }
)
@triton.jit
def _coarse_score_pool_kernel(
    q_ptr,  # idx_q: [rows, num_idx_heads, d]
    k_cache_ptr,  # index K paged: [max_slots, 1, d]
    score_ptr,  # [heads, b_count * tiles_per_chunk, num_blocks] fp32
    req_to_token_ptr,
    cu_rows,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_kv_heads,
    qk_head_dim,
    num_blocks,
    tiles_per_chunk,
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
    K_MULT: tl.constexpr,  # autotune knob; BLOCK_SIZE_K = POOL_BLOCK * K_MULT
    POOL_BLOCK: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    IDX_GROUP: tl.constexpr,
    REDUCE: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    """Stage 1 with the pooling *after* the dot: block score = max over its keys.

    This is MiniMax-M3's shipped block indexer, re-used as a recall stage. It
    reads every key, which is the price of an exact block ranking; what it buys
    over the shipped path is that its output feeds a token-level refinement
    instead of being the final answer.
    """
    NUM_BLK: tl.constexpr = BLOCK_SIZE_K // POOL_BLOCK
    pid_t, pid_bh, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pid_b_local = pid_bh // num_kv_heads
    pid_b = batch_start + pid_b_local
    pid_kvh = pid_bh % num_kv_heads

    sm_scale_log2e = sm_scale * 1.4426950408889634

    row_start = tl.load(cu_rows + pid_b)
    q_len = tl.load(cu_rows + pid_b + 1) - row_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots

    q_lo = q_chunk_start + pid_t * QUERY_TILE
    q_hi = tl.minimum(tl.minimum(q_lo + QUERY_TILE, q_len), q_chunk_start + chunk_cap)
    if q_lo >= q_hi:
        return

    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < qk_head_dim

    local_q = q_lo + off_q
    q_mask = local_q < q_hi
    abs_q = prefix_len + local_q

    pos = pid_k * BLOCK_SIZE_K + off_k
    pos_live = pos < seq_len
    slots = tl.load(req_to_token_ptr + sid * stride_r2t_b + pos, mask=pos_live, other=0)
    slots = (slots.to(tl.int64) + max_slots) % max_slots
    k = tl.load(
        k_cache_ptr + slots[None, :] * stride_k_s + off_d[:, None] * stride_k_d,
        mask=d_mask[:, None] & pos_live[None, :],
        other=0.0,
    )  # [D, K]

    acc = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
    for g in tl.static_range(IDX_GROUP):
        q = tl.load(
            q_ptr
            + (row_start + local_q)[:, None] * stride_q_n
            + (pid_kvh * IDX_GROUP + g) * stride_q_h
            + off_d[None, :] * stride_q_d,
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        )  # [Q, D]
        kk = k.to(q.dtype) if IS_FP8 else k
        s = tl.dot(q, kk) * sm_scale_log2e  # [Q, K]
        if REDUCE == _REDUCE_MAX:
            acc = s if g == 0 else tl.maximum(acc, s)
        elif REDUCE == _REDUCE_RELU_SUM:
            acc += tl.maximum(s, 0.0)
        else:
            acc += s

    # Per-token causality and forced bias, then two max reductions that commute:
    # over the tile's query rows (the union), then over each block's keys (the
    # pool). A block carries a forced bias iff any of its tokens does, which is
    # what the pre-pooling variant applies at block granularity directly.
    live = q_mask[:, None] & pos_live[None, :] & (pos[None, :] <= abs_q[:, None])
    init_m = live & (pos[None, :] < init_tokens)
    local_m = live & (pos[None, :] > (abs_q[:, None] - local_tokens))
    acc = tl.where(init_m, _INIT_BIAS, acc)
    acc = tl.where(local_m & (init_m == 0), _LOCAL_BIAS, acc)
    acc = tl.where(live, acc, float("-inf"))

    key_score = tl.max(acc, axis=0)  # [K] — union over the tile's rows
    blk_score = tl.max(
        tl.reshape(key_score, (NUM_BLK, POOL_BLOCK)), axis=1
    )  # [NUM_BLK]

    blk = pid_k * NUM_BLK + tl.arange(0, NUM_BLK)
    s_row = pid_b_local * tiles_per_chunk + pid_t
    tl.store(
        score_ptr + pid_kvh * stride_s_h + s_row * stride_s_n + blk * stride_s_k,
        blk_score,
        mask=blk < num_blocks,
    )


# ---------------------------------------------------------------------------
# stage 2 — fine token scoring, restricted to the recalled blocks
# ---------------------------------------------------------------------------


@triton.heuristics(
    {
        "BLOCK_SIZE_Q": lambda args: max(16, args["QUERY_TILE"]),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_C": bc}, num_warps=nw, num_stages=ns)
        for bc in (64, 128, 256)
        for nw in (4, 8)
        for ns in (2, 3)
    ],
    key=["qk_head_dim", "QUERY_TILE", "IDX_GROUP", "REDUCE", "IS_FP8"],
)
@triton.jit
def _fine_token_score_kernel(
    q_ptr,  # idx_q: [rows, num_idx_heads, d]
    k_cache_ptr,  # index K paged: [max_slots, 1, d]
    cand_ptr,  # [num_kv_heads, b_count * tiles_per_chunk, M] int32, -1 padded
    score_ptr,  # [num_kv_heads, b_count * chunk_cap, M * P] fp32
    req_to_token_ptr,
    cu_rows,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_kv_heads,
    qk_head_dim,
    cand_width,  # M * P
    pool_block,  # P
    tiles_per_chunk,
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
    stride_c_h,
    stride_c_n,
    stride_c_k,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_r2t_b,
    # META
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    IDX_GROUP: tl.constexpr,
    REDUCE: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    # One program per (query tile, KV head, candidate tile). The tile shares one
    # recalled block list, so one gather of K serves all QUERY_TILE rows — the
    # same amortisation the flat kernel gets from its query tiling, which is the
    # only reason the candidate pass is cheaper than scoring the whole context.
    pid_t, pid_kvh, pid_c = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    pid_b_local = pid_t // tiles_per_chunk
    pid_b = batch_start + pid_b_local
    tile = pid_t % tiles_per_chunk

    row_start = tl.load(cu_rows + pid_b)
    q_len = tl.load(cu_rows + pid_b + 1) - row_start
    q_lo = q_chunk_start + tile * QUERY_TILE
    q_hi = tl.minimum(tl.minimum(q_lo + QUERY_TILE, q_len), q_chunk_start + chunk_cap)
    if q_lo >= q_hi:
        return

    sm_scale_log2e = sm_scale * 1.4426950408889634
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (tl.load(slot_ids + pid_b).to(tl.int64) + max_slots) % max_slots

    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_c = tl.arange(0, BLOCK_SIZE_C)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < qk_head_dim

    local_q = q_lo + off_q
    q_mask = local_q < q_hi
    abs_q = prefix_len + local_q

    # Candidate column -> (recalled block, offset in block) -> token position.
    col = pid_c * BLOCK_SIZE_C + off_c
    col_in = col < cand_width
    blk = tl.load(
        cand_ptr
        + pid_kvh * stride_c_h
        + pid_t * stride_c_n
        + (col // pool_block) * stride_c_k,
        mask=col_in,
        other=-1,
    )
    pos = blk * pool_block + (col % pool_block)
    col_live = col_in & (blk >= 0) & (pos < seq_len)

    slots = tl.load(req_to_token_ptr + sid * stride_r2t_b + pos, mask=col_live, other=0)
    slots = (slots.to(tl.int64) + max_slots) % max_slots
    k = tl.load(
        k_cache_ptr + slots[None, :] * stride_k_s + off_d[:, None] * stride_k_d,
        mask=d_mask[:, None] & col_live[None, :],
        other=0.0,
    )  # [D, C]

    acc = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_C), dtype=tl.float32)
    for g in tl.static_range(IDX_GROUP):
        q = tl.load(
            q_ptr
            + (row_start + local_q)[:, None] * stride_q_n
            + (pid_kvh * IDX_GROUP + g) * stride_q_h
            + off_d[None, :] * stride_q_d,
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        )  # [Q, D]
        kk = k.to(q.dtype) if IS_FP8 else k
        s = tl.dot(q, kk) * sm_scale_log2e  # [Q, C]
        if REDUCE == _REDUCE_MAX:
            acc = s if g == 0 else tl.maximum(acc, s)
        elif REDUCE == _REDUCE_RELU_SUM:
            acc += tl.maximum(s, 0.0)
        else:
            acc += s

    # Per-row causality and per-row forced regions: the candidate *set* is
    # shared across the tile, the mask and the final top-k are not.
    live = q_mask[:, None] & col_live[None, :] & (pos[None, :] <= abs_q[:, None])
    init_m = live & (pos[None, :] < init_tokens)
    local_m = live & (pos[None, :] > (abs_q[:, None] - local_tokens))
    acc = tl.where(init_m, _INIT_BIAS, acc)
    acc = tl.where(local_m & (init_m == 0), _LOCAL_BIAS, acc)
    acc = tl.where(live, acc, float("-inf"))

    s_row = pid_b_local * chunk_cap + (local_q - q_chunk_start)
    tl.store(
        score_ptr
        + pid_kvh * stride_s_h
        + s_row[:, None] * stride_s_n
        + col[None, :] * stride_s_k,
        acc,
        mask=q_mask[:, None] & col_in[None, :],
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def plan_row_chunk(
    *,
    batch_size: int,
    max_rows_per_req: int,
    num_kv_heads: int,
    coarse_width: int,
    cand_width: int,
    query_tile: int,
    score_budget_bytes: int = DEFAULT_SCORE_BUDGET_BYTES,
) -> tuple[int, int]:
    """Plan ``(batch_chunk, row_chunk)`` so both score buffers fit the budget.

    The flat token indexer's equivalent planner shrinks the chunk as the context
    grows, because its row width *is* the context. Here the fine row width is
    ``M * P`` — fixed — and the coarse buffer holds one row per *query tile* of
    width ``L / P``, so the chunk is nearly context-independent.

    ``row_chunk`` is always a multiple of ``query_tile``, so a tile never
    straddles two chunks and its shared candidate set stays well defined.
    """
    unit = num_kv_heads * (coarse_width // query_tile + cand_width) * 4
    rows_affordable = score_budget_bytes // max(1, unit)
    if rows_affordable < query_tile:
        raise ValueError(
            f"two-level indexer cannot honour the {score_budget_bytes >> 20} MiB "
            f"score budget: one query row of [{num_kv_heads} KV heads x "
            f"({coarse_width} coarse / {query_tile} + {cand_width} candidate)] "
            f"needs {(unit + (1 << 20) - 1) >> 20} MiB. Lower coarse_blocks or "
            f"raise the budget."
        )

    if rows_affordable >= batch_size * query_tile:
        batch_chunk = batch_size
    else:
        batch_chunk = max(1, min(batch_size, rows_affordable // query_tile))
    row_chunk = max(query_tile, rows_affordable // batch_chunk)
    row_chunk = (row_chunk // query_tile) * query_tile
    row_chunk = min(row_chunk, math.ceil(max_rows_per_req / query_tile) * query_tile)
    return batch_chunk, row_chunk


@triton.jit
def _map_columns_kernel(
    cols_ptr,  # [kv_heads, b_count * chunk_cap, topk] int32, -1 padded
    cand_ptr,  # [1 or kv_heads, b_count * tiles_per_chunk, M] int32
    out_ptr,  # [kv_heads, total_rows, topk] int32
    cu_rows,
    topk,
    pool_block,
    tiles_per_chunk,
    chunk_cap,
    q_chunk_start,
    batch_start,
    stride_col_h,
    stride_col_n,
    stride_col_k,
    stride_cand_h,  # 0 when the candidate set is shared across KV heads
    stride_cand_n,
    stride_cand_m,
    stride_out_h,
    stride_out_n,
    stride_out_k,
    QUERY_TILE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Candidate column -> absolute token position, straight into ``topk_idx``.

    Doing this in torch costs eight elementwise launches plus a per-request copy
    loop, which at decode is most of the wall clock — the whole selection is only
    ~0.1 ms of actual kernel time. Fusing it also folds the tile -> row expansion
    (``repeat_interleave``) and the shared-candidate broadcast (``stride_cand_h``
    is simply 0) into the addressing, so neither materialises.
    """
    pid_h, pid_r, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    pid_b_local = pid_r // chunk_cap
    pid_b = batch_start + pid_b_local
    local_q = q_chunk_start + (pid_r % chunk_cap)
    row_start = tl.load(cu_rows + pid_b)
    if local_q >= tl.load(cu_rows + pid_b + 1) - row_start:
        return
    cand_row = pid_b_local * tiles_per_chunk + (pid_r % chunk_cap) // QUERY_TILE

    off_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    k_mask = off_k < topk
    col = tl.load(
        cols_ptr + pid_h * stride_col_h + pid_r * stride_col_n + off_k * stride_col_k,
        mask=k_mask,
        other=-1,
    )
    blk = tl.load(
        cand_ptr
        + pid_h * stride_cand_h
        + cand_row * stride_cand_n
        + (tl.maximum(col, 0) // pool_block) * stride_cand_m,
        mask=k_mask,
        other=-1,
    )
    pos = blk * pool_block + (col % pool_block)
    pos = tl.where((col >= 0) & (blk >= 0), pos, -1)
    tl.store(
        out_ptr
        + pid_h * stride_out_h
        + (row_start + local_q) * stride_out_n
        + off_k * stride_out_k,
        pos,
        mask=k_mask,
    )


@torch.no_grad()
def _two_level_select(
    *,
    idx_q: torch.Tensor,  # [rows, num_idx_heads, d]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, d]
    pooled: Optional[torch.Tensor],  # [batch, num_blocks, d]; None when post-pooling
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_rows: torch.Tensor,  # [batch + 1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    batch_size: int,
    max_rows_per_req: int,
    max_seqlen_k: int,
    num_kv_heads: int,
    cfg: TwoLevelConfig,
    sm_scale: Optional[float],
    score_budget_bytes: int,
) -> torch.Tensor:
    total_rows, num_idx_heads, qk_head_dim = idx_q.shape
    assert num_idx_heads % num_kv_heads == 0, (
        f"num_idx_heads ({num_idx_heads}) must be divisible by num_kv_heads "
        f"({num_kv_heads}); the two-level indexer emits one selection per KV head"
    )
    idx_group = num_idx_heads // num_kv_heads
    max_slots = idx_k_cache.shape[0]
    is_fp8 = idx_k_cache.dtype != idx_q.dtype
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5

    assert pooled is not None or cfg.pool_position == "post", (
        "pre-pooling needs the pooled block cache; pass it or let the wrapper "
        "build it"
    )
    topk = cfg.topk_tokens
    pool_block = cfg.pool_block_size
    coarse_width = cfg.coarse_width(max_seqlen_k)
    topm = min(cfg.coarse_blocks, coarse_width)
    cand_width = topm * pool_block
    reduce_code = _REDUCE_CODES[cfg.head_reduce]
    # Stage 1 runs either once per KV head, or once for the whole index-head set.
    coarse_heads = 1 if cfg.share_candidates else num_kv_heads
    coarse_group = num_idx_heads if cfg.share_candidates else idx_group

    topk_idx = torch.full(
        (num_kv_heads, total_rows, topk), -1, dtype=torch.int32, device=idx_q.device
    )
    if total_rows == 0:
        return topk_idx

    query_tile = min(cfg.query_tile, triton.next_power_of_2(max(1, max_rows_per_req)))
    batch_cap, chunk_cap = plan_row_chunk(
        batch_size=batch_size,
        max_rows_per_req=max_rows_per_req,
        num_kv_heads=num_kv_heads,
        coarse_width=coarse_width,
        cand_width=cand_width,
        query_tile=query_tile,
        score_budget_bytes=score_budget_bytes,
    )
    tiles_per_chunk = chunk_cap // query_tile

    for b_start in range(0, batch_size, batch_cap):
        b_count = min(batch_cap, batch_size - b_start)
        for q_chunk_start in range(0, max_rows_per_req, chunk_cap):
            rows = b_count * chunk_cap
            tiles = b_count * tiles_per_chunk

            # --- stage 1: coarse block scores + top-M recall, per query tile -
            # `coarse_heads == 1` is the shared-candidate case: one head axis
            # entry folding *all* index heads, which stage 2 then reads with a
            # zero stride so every KV head walks the same candidate list.
            coarse = torch.empty(
                (coarse_heads, tiles, coarse_width),
                dtype=torch.float32,
                device=idx_q.device,
            )

            if cfg.pool_position == "pre":

                def _coarse_grid(meta):
                    return (
                        tiles_per_chunk,
                        b_count * coarse_heads,
                        triton.cdiv(coarse_width, meta["BLOCK_SIZE_N"]),
                    )

                _coarse_block_score_kernel[_coarse_grid](
                    idx_q,
                    pooled,
                    coarse,
                    cu_rows,
                    seq_lens,
                    prefix_lens,
                    coarse_heads,
                    qk_head_dim,
                    coarse_width,
                    pool_block,
                    tiles_per_chunk,
                    chunk_cap,
                    q_chunk_start,
                    b_start,
                    cfg.init_tokens,
                    cfg.local_tokens,
                    sm_scale,
                    idx_q.stride(0),
                    idx_q.stride(1),
                    idx_q.stride(2),
                    pooled.stride(0),
                    pooled.stride(1),
                    pooled.stride(2),
                    coarse.stride(0),
                    coarse.stride(1),
                    coarse.stride(2),
                    QUERY_TILE=query_tile,
                    IDX_GROUP=coarse_group,
                    REDUCE=reduce_code,
                )
            else:

                def _coarse_grid(meta):
                    return (
                        tiles_per_chunk,
                        b_count * coarse_heads,
                        triton.cdiv(coarse_width * pool_block, meta["BLOCK_SIZE_K"]),
                    )

                _coarse_score_pool_kernel[_coarse_grid](
                    idx_q,
                    idx_k_cache,
                    coarse,
                    req_to_token,
                    cu_rows,
                    seq_lens,
                    prefix_lens,
                    slot_ids,
                    max_slots,
                    coarse_heads,
                    qk_head_dim,
                    coarse_width,
                    tiles_per_chunk,
                    chunk_cap,
                    q_chunk_start,
                    b_start,
                    cfg.init_tokens,
                    cfg.local_tokens,
                    sm_scale,
                    idx_q.stride(0),
                    idx_q.stride(1),
                    idx_q.stride(2),
                    idx_k_cache.stride(0),
                    idx_k_cache.stride(2),
                    coarse.stride(0),
                    coarse.stride(1),
                    coarse.stride(2),
                    req_to_token.stride(0),
                    POOL_BLOCK=pool_block,
                    QUERY_TILE=query_tile,
                    IDX_GROUP=coarse_group,
                    REDUCE=reduce_code,
                    IS_FP8=is_fp8,
                )
            cand_blocks = _topk_positions(coarse, topm)  # [kvh, tiles, M]
            del coarse

            # --- stage 2: fine token scores over the recalled blocks --------
            fine = torch.empty(
                (num_kv_heads, rows, cand_width),
                dtype=torch.float32,
                device=idx_q.device,
            )

            def _fine_grid(meta):
                return (
                    tiles,
                    num_kv_heads,
                    triton.cdiv(cand_width, meta["BLOCK_SIZE_C"]),
                )

            _fine_token_score_kernel[_fine_grid](
                idx_q,
                idx_k_cache,
                cand_blocks,
                fine,
                req_to_token,
                cu_rows,
                seq_lens,
                prefix_lens,
                slot_ids,
                max_slots,
                num_kv_heads,
                qk_head_dim,
                cand_width,
                pool_block,
                tiles_per_chunk,
                chunk_cap,
                q_chunk_start,
                b_start,
                cfg.init_tokens,
                cfg.local_tokens,
                sm_scale,
                idx_q.stride(0),
                idx_q.stride(1),
                idx_q.stride(2),
                idx_k_cache.stride(0),
                idx_k_cache.stride(2),
                0 if cfg.share_candidates else cand_blocks.stride(0),
                cand_blocks.stride(1),
                cand_blocks.stride(2),
                fine.stride(0),
                fine.stride(1),
                fine.stride(2),
                req_to_token.stride(0),
                QUERY_TILE=query_tile,
                IDX_GROUP=idx_group,
                REDUCE=reduce_code,
                IS_FP8=is_fp8,
            )
            cols = _topk_positions(fine, topk)
            del fine

            # One candidate list per tile, one final selection per row, written
            # straight into the output at its global row.
            block_k = min(1024, triton.next_power_of_2(topk))
            _map_columns_kernel[(num_kv_heads, rows, triton.cdiv(topk, block_k))](
                cols,
                cand_blocks,
                topk_idx,
                cu_rows,
                topk,
                pool_block,
                tiles_per_chunk,
                chunk_cap,
                q_chunk_start,
                b_start,
                cols.stride(0),
                cols.stride(1),
                cols.stride(2),
                0 if cfg.share_candidates else cand_blocks.stride(0),
                cand_blocks.stride(1),
                cand_blocks.stride(2),
                topk_idx.stride(0),
                topk_idx.stride(1),
                topk_idx.stride(2),
                QUERY_TILE=query_tile,
                BLOCK_SIZE_K=block_k,
            )
            del cand_blocks, cols

    return topk_idx


@torch.no_grad()
def two_level_select_prefill(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, d]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, d]
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,  # [batch + 1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_kv_heads: int,
    cfg: TwoLevelConfig,
    pooled: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    score_budget_bytes: int = DEFAULT_SCORE_BUDGET_BYTES,
) -> torch.Tensor:
    """Select ``cfg.topk_tokens`` positions per query row. -> [kv_heads, total_q, k].

    ``pooled`` is stage 0's output; pass the server-maintained one to measure the
    steady state, or leave it None to rebuild it here (and pay for it).

    Unlike ``token_select_prefill`` this takes no host-side sequence lengths: the
    chunk plan depends only on shapes, and the row -> output mapping is resolved
    on device from ``cu_seqlens``, so the call never syncs.
    """
    if pooled is None and cfg.pool_position == "pre":
        pooled = build_pooled_index_keys(
            idx_k_cache,
            req_to_token,
            slot_ids,
            seq_lens,
            pool_block_size=cfg.pool_block_size,
            max_seqlen_k=max_seqlen_k,
        )
    return _two_level_select(
        idx_q=idx_q,
        idx_k_cache=idx_k_cache,
        pooled=pooled,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_rows=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        batch_size=int(cu_seqlens.shape[0]) - 1,
        max_rows_per_req=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_kv_heads=num_kv_heads,
        cfg=cfg,
        sm_scale=sm_scale,
        score_budget_bytes=score_budget_bytes,
    )


@torch.no_grad()
def two_level_select_decode(
    idx_q: torch.Tensor,  # [batch, num_idx_heads, d]
    idx_k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen: int,
    num_kv_heads: int,
    cfg: TwoLevelConfig,
    pooled: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    score_budget_bytes: int = DEFAULT_SCORE_BUDGET_BYTES,
) -> torch.Tensor:
    """Decode form: one row per request, at absolute position ``seq_len - 1``."""
    batch = int(idx_q.shape[0])
    if pooled is None and cfg.pool_position == "pre":
        pooled = build_pooled_index_keys(
            idx_k_cache,
            req_to_token,
            slot_ids,
            seq_lens,
            pool_block_size=cfg.pool_block_size,
            max_seqlen_k=max_seqlen,
        )
    cu_rows = torch.arange(batch + 1, dtype=torch.int32, device=idx_q.device)
    return _two_level_select(
        idx_q=idx_q,
        idx_k_cache=idx_k_cache,
        pooled=pooled,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_rows=cu_rows,
        seq_lens=seq_lens,
        prefix_lens=(seq_lens.to(torch.int32) - 1),
        batch_size=batch,
        max_rows_per_req=1,
        max_seqlen_k=max_seqlen,
        num_kv_heads=num_kv_heads,
        cfg=cfg,
        sm_scale=sm_scale,
        score_budget_bytes=score_budget_bytes,
    )


# ---------------------------------------------------------------------------
# references and accuracy helpers (torch; test/benchmark use only)
# ---------------------------------------------------------------------------


def _depage(
    idx_k_cache: torch.Tensor, req_to_token: torch.Tensor, sid: int, length: int
) -> torch.Tensor:
    """Contiguous ``[L, d]`` index K of one request, gathered out of the pool."""
    slots = req_to_token[sid, :length].to(torch.int64)
    return idx_k_cache[slots, 0, :].to(torch.float32)


@torch.no_grad()
def make_clustered_index_queries(
    idx_q: torch.Tensor,  # [rows, num_idx_heads, d]
    *,
    block: int = 64,
    noise: float = 0.35,
    seed: int = 1,
) -> torch.Tensor:
    """Overwrite the index queries with *locally correlated* rows, in place.

    The mirror of :func:`make_clustered_index_keys`, and it bounds the other
    approximation this prototype makes. ``query_tile`` shares one recall across
    consecutive query rows, which is free only insofar as neighbouring queries
    want the same blocks. Real neighbouring tokens do — that is the premise
    every query-block-granular selector (M3's own prefill path included) is built
    on — while the harness's iid Gaussian queries want unrelated blocks by
    construction, so recall measured on them is a floor and not a forecast.

    ``block`` should be set to the ``query_tile`` under test, or to the scale at
    which real queries are believed to agree.
    """
    device = idx_q.device
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, num_heads, head_dim = idx_q.shape
    num_blocks = math.ceil(rows / block)
    centres = torch.randn(
        num_blocks,
        num_heads,
        head_dim,
        generator=gen,
        device=device,
        dtype=torch.float32,
    )
    block_of = torch.arange(rows, device=device) // block
    jitter = torch.randn(
        rows, num_heads, head_dim, generator=gen, device=device, dtype=torch.float32
    )
    idx_q[:] = (centres[block_of] + noise * jitter).to(idx_q.dtype)
    return idx_q


@torch.no_grad()
def make_clustered_index_keys(
    idx_k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block: int = 128,
    noise: float = 0.35,
    seed: int = 0,
) -> torch.Tensor:
    """Overwrite the index K cache with *locally correlated* keys, in place.

    Mean pooling can only stand in for a block's best key if keys inside a block
    resemble each other. Real index keys do — adjacent tokens share topic,
    position encoding and residual-stream direction — but the iid Gaussians the
    harness fills the cache with do not, and under them the coarse stage recalls
    close to randomly. Any recall number measured on iid data is therefore a
    floor rather than an estimate, and both the test and the benchmark re-draw
    the cache through this: one random centre per ``block`` tokens, plus noise.

    ``noise`` is the only knob that matters. 0 makes every block perfectly
    summarised by its mean (recall 1.0 by construction, an upper bound); large
    values converge back to iid. 0.3-0.4 is the range where a block's mean still
    ranks its neighbours but does not determine them.
    """
    device = idx_k_cache.device
    gen = torch.Generator(device=device).manual_seed(seed)
    head_dim = idx_k_cache.shape[-1]
    for b in range(int(seq_lens.shape[0])):
        length = int(seq_lens[b])
        slots = req_to_token[b, :length].to(torch.int64)
        num_blocks = math.ceil(length / block)
        centres = torch.randn(
            num_blocks, head_dim, generator=gen, device=device, dtype=torch.float32
        )
        block_of = torch.arange(length, device=device) // block
        jitter = torch.randn(
            length, head_dim, generator=gen, device=device, dtype=torch.float32
        )
        keys = centres[block_of] + noise * jitter
        idx_k_cache[slots, 0, :] = keys.to(idx_k_cache.dtype)
    return idx_k_cache


def _biased_scores(
    q: torch.Tensor,  # [H, d] index-head queries of one row
    k: torch.Tensor,  # [L, d]
    abs_q: int,
    *,
    sm_scale: float,
    init_tokens: int,
    local_tokens: int,
    head_reduce: str,
) -> torch.Tensor:
    """Reference per-token score row: reduce over heads, bias, causal-mask."""
    s = (q.to(torch.float32) @ k.T) * sm_scale * 1.4426950408889634  # [H, L]
    if head_reduce == "max":
        row = s.amax(dim=0)
    elif head_reduce == "relu_sum":
        row = s.clamp(min=0).sum(dim=0)
    else:
        row = s.sum(dim=0)

    pos = torch.arange(k.shape[0], device=k.device)
    causal = pos <= abs_q
    init_m = causal & (pos < init_tokens)
    local_m = causal & (pos > abs_q - local_tokens)
    row = torch.where(init_m, torch.full_like(row, INIT_BIAS), row)
    row = torch.where(local_m & ~init_m, torch.full_like(row, LOCAL_BIAS), row)
    return torch.where(causal, row, torch.full_like(row, float("-inf")))


def reference_exact_topk(
    idx_q_row: torch.Tensor,
    k: torch.Tensor,
    abs_q: int,
    *,
    topk: int,
    sm_scale: float,
    cfg: TwoLevelConfig,
) -> torch.Tensor:
    """Exact flat top-k over the causal context — what the two-level pass approximates."""
    row = _biased_scores(
        idx_q_row,
        k,
        abs_q,
        sm_scale=sm_scale,
        init_tokens=cfg.init_tokens,
        local_tokens=cfg.local_tokens,
        head_reduce=cfg.head_reduce,
    )
    k_eff = min(topk, int((row > float("-inf")).sum().item()))
    return torch.topk(row, k_eff, sorted=False).indices


def reference_two_level(
    idx_q_row: torch.Tensor,  # [H, d] the index heads of ONE kv head
    k: torch.Tensor,  # [L, d] float32
    abs_q: int,
    *,
    sm_scale: float,
    cfg: TwoLevelConfig,
    idx_q_coarse: Optional[torch.Tensor] = None,  # [H_all, d] when shared
    pool_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Torch mirror of the two Triton stages, for one (row, kv head).

    ``pool_dtype`` mirrors where the pooled block representation is *stored*.
    ``build_pooled_index_keys`` accumulates in fp32 and writes the cache's dtype,
    so the coarse dot sees a rounded mean; scoring the fp32 mean here instead
    would make this reference disagree with the kernel on blocks that rank close
    together, which is a dtype question and not a kernel one.
    """
    length = k.shape[0]
    p, m = cfg.pool_block_size, cfg.coarse_blocks
    nblk = math.ceil(length / p)
    pad = nblk * p - length
    padded = torch.cat([k, k.new_zeros(pad, k.shape[1])]) if pad else k
    counts = torch.full((nblk,), float(p), device=k.device)
    if pad:
        counts[-1] = length - (nblk - 1) * p
    kmean = (padded.view(nblk, p, -1).sum(dim=1) / counts[:, None]).to(pool_dtype)

    # Stage 1 sees every index head under `share_candidates`, only this KV head's
    # group otherwise; stage 2 always sees just the group.
    q_coarse = idx_q_row if idx_q_coarse is None else idx_q_coarse
    s = (q_coarse.to(torch.float32) @ kmean.float().T) * sm_scale * 1.4426950408889634
    if cfg.head_reduce == "max":
        coarse = s.amax(dim=0)
    elif cfg.head_reduce == "relu_sum":
        coarse = s.clamp(min=0).sum(dim=0)
    else:
        coarse = s.sum(dim=0)

    blk = torch.arange(nblk, device=k.device)
    start = blk * p
    visible = start <= abs_q
    init_m = visible & (start < cfg.init_tokens)
    local_m = visible & ((start + p) > (abs_q - cfg.local_tokens + 1))
    coarse = torch.where(init_m, torch.full_like(coarse, INIT_BIAS), coarse)
    coarse = torch.where(local_m & ~init_m, torch.full_like(coarse, LOCAL_BIAS), coarse)
    coarse = torch.where(visible, coarse, torch.full_like(coarse, float("-inf")))

    topm = min(m, nblk)
    cand = torch.topk(coarse, topm, sorted=False).indices
    cand = cand[coarse[cand] > float("-inf")]

    fine = _biased_scores(
        idx_q_row,
        k,
        abs_q,
        sm_scale=sm_scale,
        init_tokens=cfg.init_tokens,
        local_tokens=cfg.local_tokens,
        head_reduce=cfg.head_reduce,
    )
    keep = (cand[:, None] * p + torch.arange(p, device=k.device)[None, :]).reshape(-1)
    keep = keep[keep < length]
    masked = torch.full_like(fine, float("-inf"))
    masked[keep] = fine[keep]
    k_eff = min(cfg.topk_tokens, int((masked > float("-inf")).sum().item()))
    return torch.topk(masked, k_eff, sorted=False).indices


def selection_recall(
    selected: torch.Tensor, exact: torch.Tensor
) -> tuple[float, float]:
    """``(recall, mass_recall_placeholder)`` of a selection against an exact top-k.

    Recall is |selected ∩ exact| / |exact|, computed over the live (non -1)
    entries only. The second element is the fraction of the *budget* the
    selection actually filled, which separates "picked the wrong tokens" from
    "had nothing left to pick".
    """
    sel = selected[selected >= 0]
    ex = exact[exact >= 0]
    if ex.numel() == 0:
        return 1.0, 1.0
    hit = torch.isin(ex, sel).sum().item()
    return hit / ex.numel(), sel.numel() / max(1, ex.numel())
