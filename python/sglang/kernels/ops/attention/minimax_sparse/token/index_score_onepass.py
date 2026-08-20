"""One-pass token prefill selector — benchmark alternative to index_score.

Same contract as ``index_score.token_select_prefill`` (selection verified
set-equivalent with fp64 adjudication of ~1-ulp wmma boundary ties): a single
CUDA kernel (``jit/csrc/minimax/minimax_token_onepass.cuh``) streams the key
axis once per query strip and selects as it scores — per-(head, row,
key-segment) candidate lists guarded by running thresholds raised via
in-kernel radix compactions — so the score matrix never exists in HBM and q.k
is computed exactly once. The exact top-k then runs over the surviving lists;
candidate overflow (a tie mass wider than a list) falls back to the
materializing path.

Measured (H200, bs=1, chunk 8192): 9.4 ms at 16k / 88 ms at 128k vs the
materializing path's 6.6 / 30 ms — selection-state upkeep (compactions,
per-tile barriers) currently costs more than the traffic it saves, and short
contexts cannot compress (segment length ~ topk). Kept for unified
benchmarking and as the long-context (>=256k) research vehicle: it is the
only implementation whose selection traffic does not scale with context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

from .index_score import INIT_BIAS, LOCAL_BIAS, _select_topk, token_select_prefill

if TYPE_CHECKING:
    from tvm_ffi.module import Module

DEFAULT_CAP = 3072  # per-(head, row, segment) candidate capacity
TARGET_WAVES = 2  # resident-CTA waves used to size the key-segment dimension


@cache_once
def _jit_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "minimax_token_onepass",
        *args,
        cuda_files=["minimax/minimax_token_onepass.cuh"],
        cuda_wrappers=[("onepass", f"minimax_token_onepass<{args}>")],
        extra_dependencies=["cutlass"],  # cute/tensor.hpp, cutlass::arch fences
    )


@torch.no_grad()
def token_select_prefill_onepass(
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
    seqlens_cpu=None,  # unused; kept for the shared call signature
    prefix_lens_cpu=None,  # unused
    cap: int = DEFAULT_CAP,
) -> torch.Tensor:
    """Same contract as token_select_prefill; single-pass fused selection."""
    total_q, num_heads, head_dim = idx_q.shape
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    kv_width = min(max_seqlen_k, req_to_token.shape[1])
    dev = idx_q.device

    # Key segments give the grid its parallelism (per-segment lists are
    # CTA-private, and their union stays an exact superset of the top-k).
    q_strips = -(-max_seqlen_q // 64)
    batch_size = cu_seqlens.numel() - 1
    num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    # The kernel uses 76 KiB of dynamic shared memory, so Hopper can keep two
    # CTAs resident per SM. More segments only multiply the per-segment top-k
    # state and the width of the exact final selection.
    target_ctas = TARGET_WAVES * num_sms
    base_ctas = max(1, q_strips * batch_size)
    num_segs = max(1, min(16, -(-target_ctas // base_ctas)))

    cand_val = torch.empty(
        (num_heads, total_q, num_segs, cap), dtype=torch.float32, device=dev
    )
    cand_pos = torch.empty(
        (num_heads, total_q, num_segs, cap), dtype=torch.int32, device=dev
    )
    cnt = torch.zeros((num_heads, total_q, num_segs), dtype=torch.int32, device=dev)
    overflow = torch.zeros(1, dtype=torch.int32, device=dev)

    _jit_module(idx_q.dtype).onepass(
        idx_q,
        idx_k_cache,
        req_to_token,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        slot_ids.to(torch.int32),
        cand_val,
        cand_pos,
        cnt,
        overflow,
        max_seqlen_q,
        kv_width,
        topk,
        init_tokens,
        local_tokens,
        sm_scale,
        INIT_BIAS,
        LOCAL_BIAS,
    )

    # Exact top-k over the union of segment lists. Stale tails were -inf-filled
    # in-kernel, so the buffers are read as-is with no masking pass.
    width = num_segs * cap
    k = min(topk, width)
    v, i = _select_topk(cand_val.view(-1, width), k)
    pos = torch.gather(cand_pos.view(-1, width), 1, i.long())
    idx = torch.where(v > float("-inf"), pos, torch.full_like(pos, -1))
    idx = idx.to(torch.int32).view(num_heads, total_q, k)
    if k < topk:
        idx = torch.cat(
            [idx, idx.new_full((num_heads, total_q, topk - k), -1)], dim=-1
        )

    # Checked after all work is queued so the sync adds no pipeline bubble.
    if int(overflow.item()) != 0:
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
    return idx
