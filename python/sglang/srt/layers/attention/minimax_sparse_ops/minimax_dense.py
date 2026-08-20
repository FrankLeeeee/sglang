"""Dense (non-sparse) attention entry points for MiniMax-M3.

The third leg of the granularity comparison: no indexer, no top-k, no sparse
attention kernel — every query attends to its whole causal context. This is not
a synthetic baseline; it is exactly what MiniMax-M3's own 3 dense layers (of 60)
run, and it reuses sglang's production Triton attention rather than a bespoke
kernel, so the comparison is against the real alternative.

    block / token granularity          dense
    ---------------------------        -----
    1. indexer scores the keys         (removed)
    2. top-k select                    (removed)
    3. sparse attention over top-k     causal attention over the full context

Signatures mirror ``minimax_sparse_prefill`` / ``minimax_token_sparse_prefill``
so the backend and the benchmarks can switch paths without special-casing, and
the ``idx_*`` arguments are accepted-and-ignored for the same reason: the caller
(the MiniMax-M3 layer) still projects them when the layer is configured sparse.
Both functions return ``(None, out)`` — there is no index-value output.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import triton

from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton
from sglang.kernels.ops.kvcache.kv_indices import create_flashinfer_kv_indices_triton

# Matches ServerArgs.triton_attention_num_kv_splits, the value the Triton
# backend uses in production.
DEFAULT_MAX_KV_SPLITS = 8


def _kv_indptr_indices(
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    lens: torch.Tensor,
    total_kv: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten the per-request page tables into the (indptr, indices) pair the
    dense kernels consume. ``total_kv`` is passed in rather than summed on the
    device so this stays sync-free."""
    bs = lens.shape[0]
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=lens.device)
    kv_indptr[1:] = torch.cumsum(lens, dim=0)
    kv_indices = torch.empty(max(total_kv, 1), dtype=torch.int64, device=lens.device)
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        slot_ids,
        lens,
        kv_indptr,
        None,
        kv_indices,
        req_to_token.stride(0),
    )
    return kv_indptr, kv_indices


def minimax_dense_prefill(
    q: torch.Tensor,  # [total_extend_tokens, num_q_heads, head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    idx_q: Optional[torch.Tensor],  # ignored — no indexer
    idx_k_cache: Optional[torch.Tensor],  # ignored
    idx_v_cache: Optional[torch.Tensor],  # ignored
    idx_sink: Optional[torch.Tensor],  # ignored
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    *,
    k_extend: torch.Tensor,  # [total_extend_tokens, num_kv_heads, head_dim]
    v_extend: torch.Tensor,
    sm_scale: Optional[float] = None,
    prefix_lens_cpu: Optional[List[int]] = None,
) -> Tuple[None, torch.Tensor]:
    """Causal dense attention for an extend step.

    ``k_extend`` / ``v_extend`` are the current chunk's K/V — in a server these
    come straight out of the projection, so they are an input here rather than
    something gathered from the cache. The prefix is read from the paged cache
    through ``kv_indptr`` / ``kv_indices``.
    """
    assert sink is None, "dense path does not take an attention sink"
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    # extend_attention_fwd reads only the *prefix* through the page table; the
    # current chunk comes from k_extend/v_extend.
    if prefix_lens_cpu is not None:
        total_prefix = int(sum(prefix_lens_cpu))
    else:
        total_prefix = int(
            prefix_lens.sum().item()
        )  # syncs; callers pass the host copy
    kv_indptr, kv_indices = _kv_indptr_indices(
        req_to_token, slot_ids, prefix_lens, total_prefix
    )

    o = torch.empty_like(q)
    extend_attention_fwd(
        q,
        k_extend.contiguous(),
        v_extend.contiguous(),
        o,
        k_cache,
        v_cache,
        cu_seqlens,
        kv_indptr,
        kv_indices,
        None,  # custom_mask
        True,  # is_causal
        None,  # mask_indptr
        max_seqlen_q,
        1.0,  # k_scale
        1.0,  # v_scale
        sm_scale,
    )
    return None, o


def minimax_dense_decode(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_q: Optional[torch.Tensor],  # ignored — no indexer
    idx_sink: Optional[torch.Tensor],  # ignored
    idx_k_cache: Optional[torch.Tensor],  # ignored
    idx_v_cache: Optional[torch.Tensor],  # ignored
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    total_kv: Optional[int] = None,
    sm_scale: Optional[float] = None,
    max_kv_splits: int = DEFAULT_MAX_KV_SPLITS,
    device_core_count: Optional[int] = None,
) -> Tuple[None, torch.Tensor]:
    """Dense attention over the full context for one decode step per request."""
    assert sink is None, "dense path does not take an attention sink"
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    if total_kv is None:
        total_kv = int(seq_lens.sum().item())  # syncs; callers pass the known total

    kv_indptr, kv_indices = _kv_indptr_indices(
        req_to_token, slot_ids, seq_lens, total_kv
    )

    # Split-K workspace, sized exactly as TritonAttnBackend sizes it.
    attn_logits = torch.empty(
        (batch_size, num_q_heads, max_kv_splits, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    attn_lse = torch.empty(
        (batch_size, num_q_heads, max_kv_splits), dtype=torch.float32, device=q.device
    )
    num_kv_splits = torch.empty((batch_size,), dtype=torch.int32, device=q.device)
    if device_core_count is None:
        device_core_count = torch.cuda.get_device_properties(
            q.device
        ).multi_processor_count
    schedule_seq = max(256, triton.next_power_of_2(batch_size))
    get_num_kv_splits_triton[(1,)](
        num_kv_splits,
        seq_lens,
        batch_size,
        1,  # num_group
        num_q_heads,
        num_kv_heads,
        max_kv_splits,
        device_core_count,
        MAX_NUM_SEQ=schedule_seq,
    )

    o = torch.empty_like(q)
    decode_attention_fwd(
        q,
        k_cache,
        v_cache,
        o,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        1.0,  # k_scale
        1.0,  # v_scale
    )
    return None, o
