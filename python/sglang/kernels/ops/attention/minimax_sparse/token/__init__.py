"""Token-granularity sparse attention kernels for MiniMax-M3.

Same indexer-driven pipeline as the block-sparse path, with the block pooling
removed: the indexer scores every key individually and the selection picks token
positions rather than 128-token blocks.
"""

from .index_score import (
    DEFAULT_SCORE_BUDGET_BYTES,
    INIT_BIAS,
    LOCAL_BIAS,
    plan_query_chunk,
    token_select_decode,
    token_select_prefill,
)
from .sparse_attn import gqa_token_sparse_attn, pick_num_kv_chunks

__all__ = [
    "DEFAULT_SCORE_BUDGET_BYTES",
    "INIT_BIAS",
    "LOCAL_BIAS",
    "gqa_token_sparse_attn",
    "pick_num_kv_chunks",
    "plan_query_chunk",
    "token_select_decode",
    "token_select_prefill",
]
