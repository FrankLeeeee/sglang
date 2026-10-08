"""Main-attention and indexer configurations of the final sparse-attention study."""

from attention import Config

MAIN = [(48, 4), (64, 8), (80, 8), (64, 4)]
INDEXER_TYPES = ["identical", "reduced"]
INDEXER_DIMS = [64, 128]
CONTEXTS = [16384, 65536, 131072, 524288, 1048576]
BATCHES = [1, 8, 32]


def make_cfg(main, reduced, dim):
    """``reduced``: one indexer KV head and one query head per main KV head."""
    q_heads, kv_heads = main
    return Config(
        q_heads,
        kv_heads,
        indexer_head_dim=dim,
        indexer_q_heads=kv_heads if reduced else q_heads,
        indexer_kv_heads=1 if reduced else kv_heads,
    )
