"""Shape/config model for the MiniMax-M3 block-sparse attention benchmarks.

The numbers below mirror ``MiniMaxAI/MiniMax-M3``'s
``config.json:text_config.sparse_attention_config`` as released:

    sparse_index_dim        128
    sparse_num_index_heads  4
    sparse_topk_blocks      16
    sparse_block_size       128
    sparse_init_block       0
    sparse_local_block      1
    sparse_score_type       "max"
    sparse_disable_index_value   1 for every sparse layer (index pool is K-only)
    sparse_attention_freq   57 sparse / 3 dense out of 60 layers

plus the main-attention shape: 64 Q heads : 4 KV heads, head_dim 128.

A KV-cache page is not a sparse-attention block. ``sparse_block_size`` is a
model/algorithm parameter: it controls how token scores are pooled and which
contiguous token regions are selected. ``page_size`` is a serving-runtime
layout parameter: it controls how tokens are allocated in the paged KV cache.
The validated benchmark default is 128 for both, but the values are independent.
Keeping them equal is required by the optional MSA kernel, not by the Triton
sparse kernels benchmarked here.

A serving rank never sees those full-model head counts: attention is sharded by
``attn_tp_size``. The validated H200 recipe is ``--tp 8``, which gives 8 Q heads,
1 KV head and 1 index head per GPU (4 KV heads < 8 ranks, so KV heads are
replicated; 4 index heads < 8 ranks, so index heads are replicated 2x). Configs
are therefore built from the *model* config plus a TP degree, and every kernel
here is benchmarked at the per-rank shape a real server would launch.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch

# --- released MiniMax-M3 model constants ------------------------------------

M3_NUM_LAYERS = 60
M3_NUM_SPARSE_LAYERS = 57
M3_NUM_DENSE_LAYERS = 3
M3_NUM_Q_HEADS = 64
M3_NUM_KV_HEADS = 4
M3_HEAD_DIM = 128
M3_NUM_INDEX_HEADS = 4
M3_INDEX_HEAD_DIM = 128
M3_BLOCK_SIZE = 128
M3_TOPK_BLOCKS = 16
M3_INIT_BLOCKS = 0
M3_LOCAL_BLOCKS = 1
M3_SCORE_TYPE = "max"
M3_MAX_POSITION = 1048576

# Runtime KV-cache layout default. This deliberately has its own constant:
# unlike M3_BLOCK_SIZE, it is not part of the model's sparse-attention config.
# It defaults to 128 to match the validated --page-size 128 serving recipe and
# to satisfy MSA's page_size == block_size requirement when MSA is available.
DEFAULT_KV_PAGE_SIZE = 128

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class SparseAttnConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Per-rank shape of one MiniMax-M3 sparse attention layer."""

    # main attention (per rank)
    num_q_heads: int = M3_NUM_Q_HEADS
    num_kv_heads: int = M3_NUM_KV_HEADS
    head_dim: int = M3_HEAD_DIM

    # lightweight indexer (per rank)
    num_idx_heads: int = M3_NUM_INDEX_HEADS
    idx_head_dim: int = M3_INDEX_HEAD_DIM
    # M3 sets sparse_disable_index_value=1 on every sparse layer: the index pool
    # stores K only and the indexer emits block scores but no value output.
    disable_index_value: bool = True

    # block-sparse selection
    block_size: int = M3_BLOCK_SIZE
    topk_blocks: int = M3_TOPK_BLOCKS
    init_blocks: int = M3_INIT_BLOCKS
    local_blocks: int = M3_LOCAL_BLOCKS
    score_type: str = M3_SCORE_TYPE

    # KV pool layout. Independent of sparse block_size for the Triton path.
    page_size: int = DEFAULT_KV_PAGE_SIZE
    dtype: str = "bfloat16"

    # Selection granularity. "block" is what M3 ships: top-k blocks of
    # `block_size` contiguous tokens, scored by pooling the block. "token" is the
    # DeepSeek-style variant: no pooling, top-k individual token positions.
    # "dense" removes selection entirely — no indexer, no top-k, full causal
    # attention — which is what M3's own 3 dense layers do.
    granularity: str = "block"
    # Token budget for the token path. None keeps it equal to the block path's
    # (`topk_blocks * block_size`), which is what makes the two comparable.
    topk_tokens: Optional[int] = None

    # bookkeeping only (not a kernel input)

    def __post_init__(self) -> None:
        assert self.num_q_heads % self.num_kv_heads == 0, (
            f"num_q_heads ({self.num_q_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads})"
        )
        assert self.num_idx_heads % self.num_kv_heads == 0 or (
            self.num_kv_heads % self.num_idx_heads == 0
        ), "num_idx_heads and num_kv_heads must divide one another"
        assert self.block_size in (16, 32, 64, 128), (
            f"the Triton prefill kernel only accepts block_size in "
            f"{{16, 32, 64, 128}}, got {self.block_size}"
        )
        assert self.head_dim <= 256 and self.idx_head_dim <= 256
        assert self.page_size > 0, f"page_size must be positive, got {self.page_size}"
        assert self.init_blocks + self.local_blocks <= self.topk_blocks, (
            f"init_blocks + local_blocks ({self.init_blocks} + {self.local_blocks}) "
            f"must be <= topk_blocks ({self.topk_blocks})"
        )
        assert self.granularity in ("block", "token", "dense"), (
            f"granularity must be 'block', 'token' or 'dense', "
            f"got {self.granularity!r}"
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]

    @property
    def gqa_group_size(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def idx_group_size(self) -> int:
        """How many index heads vote for one KV head's block set.

        > 1 triggers the pure-PyTorch ``topk_index_reduce`` union pass between the
        indexer and the sparse kernel, which is a real (and costly) extra stage.
        """
        return max(1, self.num_idx_heads // self.num_kv_heads)

    @property
    def block_token_budget(self) -> int:
        """Tokens the block path attends to: top-k blocks x block size."""
        return self.topk_blocks * self.block_size

    @property
    def effective_topk_tokens(self) -> int:
        """Top-k for the token path; defaults to the block path's budget."""
        return self.block_token_budget if self.topk_tokens is None else self.topk_tokens

    @property
    def token_budget(self) -> int:
        """Tokens each query attends to. Unbounded (context-sized) when dense."""
        if self.granularity == "dense":
            return -1  # sentinel: the whole causal context, not a fixed budget
        if self.granularity == "token":
            return self.effective_topk_tokens
        return self.block_token_budget

    @property
    def init_tokens(self) -> int:
        return self.init_blocks * self.block_size

    @property
    def local_tokens(self) -> int:
        return self.local_blocks * self.block_size

    def kv_bytes_per_token_per_layer(self) -> dict[str, int]:
        """Static KV-cache footprint of one sparse layer, per token, per rank."""
        item = self.torch_dtype.itemsize
        main = 2 * self.num_kv_heads * self.head_dim * item
        if self.granularity == "dense":
            # A dense layer has no indexer, so no index KV cache at all.
            return {"main_kv": main, "index_kv": 0, "total": main}
        n_idx_tensors = 1 if self.disable_index_value else 2
        index = n_idx_tensors * 1 * self.idx_head_dim * item
        return {"main_kv": main, "index_kv": index, "total": main + index}

    def replace(self, **kwargs) -> "SparseAttnConfig":
        return msgspec.structs.replace(self, **kwargs)

    def shape_tag(self) -> str:
        if self.granularity == "dense":
            sel = "dense"
        elif self.granularity == "token":
            sel = f"tok{self.effective_topk_tokens}"
        else:
            sel = f"blk{self.block_size}x{self.topk_blocks}"
        return (
            f"q{self.num_q_heads}_kv{self.num_kv_heads}_d{self.head_dim}"
            f"_idx{self.num_idx_heads}x{self.idx_head_dim}"
            f"_{sel}_page{self.page_size}"
        )


def m3_config(**overrides) -> SparseAttnConfig:
    """The released MiniMax-M3 sparse layer: 64 Q / 4 KV / 4 index heads.

    One GPU, one shape. The benchmark used to derive a per-rank shard from a
    ``tp_size`` argument, but TP is a launch topology and this suite never
    launches more than one device — the per-rank shapes it produced are reachable
    directly through the ``num_q_heads`` / ``num_kv_heads`` sweeps, which is
    where shape sensitivity belongs.
    """
    fields = dict(
        num_q_heads=M3_NUM_Q_HEADS,
        num_kv_heads=M3_NUM_KV_HEADS,
        num_idx_heads=M3_NUM_INDEX_HEADS,
        head_dim=M3_HEAD_DIM,
        idx_head_dim=M3_INDEX_HEAD_DIM,
        disable_index_value=True,
        block_size=M3_BLOCK_SIZE,
        topk_blocks=M3_TOPK_BLOCKS,
        init_blocks=M3_INIT_BLOCKS,
        local_blocks=M3_LOCAL_BLOCKS,
        score_type=M3_SCORE_TYPE,
        page_size=DEFAULT_KV_PAGE_SIZE,
    )
    fields.update(overrides)
    return SparseAttnConfig(**fields)


# Cover the released model's full 1M-token position range. Long-context prefill
# should use a bounded extend chunk; materializing a one-shot 1M-token prefill
# has quadratic query-by-key work and is not representative of serving.
DEFAULT_CONTEXT_LENS = [
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    M3_MAX_POSITION,
]
DEFAULT_PREFILL_CHUNK = 4096
