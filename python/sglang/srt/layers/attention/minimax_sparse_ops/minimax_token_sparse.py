"""Token-granularity sparse attention entry points for MiniMax-M3.

Mirrors ``minimax_sparse.py`` (the block-granularity path) stage for stage:

    1. indexer scores every key      (no block pooling)
    2. top-k *token positions*       (instead of top-k blocks)
    3. GQA attention over those tokens

The signatures deliberately match ``minimax_sparse_prefill`` /
``minimax_sparse_decode`` so ``MiniMaxSparseAttnBackend`` can switch granularity
without the model layer knowing anything about it.

Only the K-only indexer is supported (``disable_index_value=True``), which is
how every sparse layer of the released MiniMax-M3 is configured — the indexer
emits selection scores and no value output.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.index import topk_index_reduce
from sglang.kernels.ops.attention.minimax_sparse.token import (
    gqa_token_sparse_attn,
    token_select_decode,
    token_select_prefill,
)


def _require_k_only_indexer(disable_index_value: bool) -> None:
    if not disable_index_value:
        raise NotImplementedError(
            "Token-granularity MiniMax sparse attention supports only the K-only "
            "indexer (sparse_disable_index_value=1), which is how MiniMax-M3 "
            "ships every sparse layer. Got disable_index_value=False."
        )


def _align_index_heads(
    topk_idx: torch.Tensor, num_kv_heads: int, topk: int
) -> torch.Tensor:
    """Map per-index-head token sets onto the main attention's KV heads.

    Three cases, matching what the block path does with ``topk_index_reduce``:
      * equal counts            -> use as is (MiniMax-M3's own configuration)
      * more index than KV heads-> union the sets that share a KV head
      * one index head          -> broadcast it, as a stride-0 view (no copy;
                                   a real expand of [kv, total_q, topk] would be
                                   gigabytes at prefill)
    """
    num_idx_heads = topk_idx.shape[0]
    if num_idx_heads == num_kv_heads:
        return topk_idx
    if num_idx_heads > num_kv_heads:
        assert num_idx_heads % num_kv_heads == 0, (
            f"num_idx_heads ({num_idx_heads}) must be a multiple of "
            f"num_kv_heads ({num_kv_heads})"
        )
        group = num_idx_heads // num_kv_heads
        return topk_index_reduce(topk_idx.view(num_kv_heads, group, -1, topk), dim=1)
    if num_idx_heads == 1:
        return topk_idx.expand(num_kv_heads, -1, -1)
    assert num_kv_heads % num_idx_heads == 0, (
        f"num_kv_heads ({num_kv_heads}) must be a multiple of "
        f"num_idx_heads ({num_idx_heads})"
    )
    return topk_idx.repeat_interleave(num_kv_heads // num_idx_heads, dim=0)


def _prefill_query_slot_ids(
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_q: int,
    seqlens_cpu: Optional[List[int]],
) -> torch.Tensor:
    """Request slot owning each query token, [total_q]."""
    if seqlens_cpu is not None:
        lens = torch.tensor(
            list(seqlens_cpu), device=slot_ids.device, dtype=torch.int64
        )
    else:
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
    out = torch.repeat_interleave(slot_ids.to(torch.int64), lens)
    assert (
        out.shape[0] == total_q
    ), f"query slot map has {out.shape[0]} entries for {total_q} query tokens"
    return out


def minimax_token_sparse_prefill(
    q: torch.Tensor,  # [total_extend_tokens, num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    sink: Optional[torch.Tensor],
    idx_q: torch.Tensor,  # [total_extend_tokens, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim]
    idx_v_cache: Optional[torch.Tensor],
    idx_sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    topk_tokens: int,
    init_tokens: int,
    local_tokens: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    disable_index_value: bool = True,
    seqlens_cpu: Optional[List[int]] = None,
    prefix_lens_cpu: Optional[List[int]] = None,
    score_budget_bytes: Optional[int] = None,
) -> Tuple[None, torch.Tensor]:
    """Token-granularity sparse prefill. ``seqlens_cpu`` / ``prefix_lens_cpu``
    are the host copies of the extend and prefix lengths; passing them keeps the
    chunk planner free of device syncs."""
    _require_k_only_indexer(disable_index_value)
    assert (
        sink is None and idx_sink is None
    ), "attention sinks are not supported by the token-granularity path"

    kwargs = {}
    if score_budget_bytes is not None:
        kwargs["score_budget_bytes"] = score_budget_bytes

    # Step 1+2: per-token indexer logits -> top-k token positions.
    topk_idx = token_select_prefill(
        idx_q=idx_q,
        idx_k_cache=idx_k_cache,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        topk=topk_tokens,
        init_tokens=init_tokens,
        local_tokens=local_tokens,
        sm_scale=idx_sm_scale,
        seqlens_cpu=seqlens_cpu,
        prefix_lens_cpu=prefix_lens_cpu,
        **kwargs,
    )
    topk_idx = _align_index_heads(topk_idx, k_cache.shape[1], topk_idx.shape[-1])

    # Step 3: GQA attention over the selected tokens.
    q_slot_ids = _prefill_query_slot_ids(slot_ids, cu_seqlens, q.shape[0], seqlens_cpu)
    o = gqa_token_sparse_attn(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        q_slot_ids=q_slot_ids,
        topk_idx=topk_idx,
        sm_scale=sm_scale,
        # The query axis already saturates the device during prefill.
        num_kv_chunks=1,
    )
    return None, o


def minimax_token_sparse_decode(
    q: torch.Tensor,  # [batch_size, num_q_heads, qk_head_dim]
    sink: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_q: torch.Tensor,  # [batch_size, num_idx_heads, idx_head_dim]
    idx_sink: Optional[torch.Tensor],
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen: int,
    topk_tokens: int,
    init_tokens: int,
    local_tokens: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    disable_index_value: bool = True,
) -> Tuple[None, torch.Tensor]:
    _require_k_only_indexer(disable_index_value)
    assert (
        sink is None and idx_sink is None
    ), "attention sinks are not supported by the token-granularity path"

    topk_idx = token_select_decode(
        idx_q=idx_q,
        idx_k_cache=idx_k_cache,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
        topk=topk_tokens,
        init_tokens=init_tokens,
        local_tokens=local_tokens,
        sm_scale=idx_sm_scale,
    )
    topk_idx = _align_index_heads(topk_idx, k_cache.shape[1], topk_idx.shape[-1])

    o = gqa_token_sparse_attn(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        q_slot_ids=slot_ids.to(torch.int64),
        topk_idx=topk_idx,
        sm_scale=sm_scale,
    )
    return None, o
