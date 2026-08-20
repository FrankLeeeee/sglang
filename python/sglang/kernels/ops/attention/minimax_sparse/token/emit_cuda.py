"""JIT CUDA emitter for the MiniMax-M3 token-sparse fused prefill selector.

Wraps ``minimax/minimax_token_emit.cuh``: pass 2 of the fused selection —
recompute the indexer q.k scores and append every score that clears the
per-row threshold to a bounded candidate list. See the .cuh header for the
contract; ``flash_with_topk_idx_optimized.token_select_prefill_fused_cuda``
is the only intended caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

from .index_score import INIT_BIAS, LOCAL_BIAS

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def emit_supported(idx_q: torch.Tensor, idx_k_cache: torch.Tensor) -> bool:
    """Whether the CUDA emitter covers this problem (else use the Triton one)."""
    return (
        idx_q.dtype in _SUPPORTED_DTYPES
        and idx_k_cache.dtype == idx_q.dtype
        and idx_q.shape[-1] == 128
        and idx_q.is_cuda
        and torch.cuda.get_device_capability(idx_q.device)[0] >= 8
    )


@cache_once
def _jit_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "minimax_token_emit",
        *args,
        cuda_files=["minimax/minimax_token_emit.cuh"],
        cuda_wrappers=[("emit", f"minimax_token_emit<{args}>")],
    )


def emit_above_threshold(
    *,
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, 128] bf16/fp16
    idx_k_cache: torch.Tensor,  # [max_slots, 1, 128] same dtype
    req_to_token: torch.Tensor,  # [max_reqs, width] int32
    cu_seqlens: torch.Tensor,  # [batch + 1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    slot_ids: torch.Tensor,  # [batch] any int dtype
    tau: torch.Tensor,  # [heads, total_q] fp32
    cand_val: torch.Tensor,  # [heads, total_q, cap] fp32, pre-filled -inf
    cand_pos: torch.Tensor,  # [heads, total_q, cap] int32, pre-filled -1
    cnt: torch.Tensor,  # [heads, total_q] int32, zeroed
    max_seqlen_q: int,
    kv_width: int,
    init_tokens: int,
    local_tokens: int,
    sm_scale: float,
) -> None:
    """Fill ``cand_val`` / ``cand_pos`` / ``cnt`` in place."""
    if not emit_supported(idx_q, idx_k_cache):
        raise RuntimeError(
            f"minimax_token_emit: unsupported problem "
            f"(dtype {idx_q.dtype}/{idx_k_cache.dtype}, head_dim {idx_q.shape[-1]})"
        )
    if slot_ids.dtype != torch.int32:
        slot_ids = slot_ids.to(torch.int32)

    module = _jit_module(idx_q.dtype)
    module.emit(
        idx_q,
        idx_k_cache,
        req_to_token,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        slot_ids,
        tau,
        cand_val,
        cand_pos,
        cnt,
        int(max_seqlen_q),
        int(kv_width),
        int(init_tokens),
        int(local_tokens),
        float(sm_scale),
        float(INIT_BIAS),
        float(LOCAL_BIAS),
    )
