# MiniMax-M3 block-sparse attention benchmarks

System-performance investigation of the block-sparse attention MiniMax-M3 uses:
prefill/decode latency across context length, attention memory, per-stage
runtime breakdown, and trend sweeps over the shape dimensions.

Three variants of the same layer are measured side by side — the released
**block**-sparse path, a DeepSeek-style **token**-sparse path, and a **dense**
path with the indexer removed entirely. See [Three variants](#three-variants-on-the-same-layer).

Everything here runs **without model weights** — one attention layer with dummy
weights against a real `MiniMaxSparseKVPool`.

## What the algorithm is

M3 replaces full attention on 57 of its 60 layers with block-sparse attention
driven by a *separate lightweight indexer*:

```
                 hidden states
                       |
        +--------------+--------------+
        |                             |
   main QKV proj                index QKV proj      (4 index heads, dim 128, K only)
   64 Q : 4 KV heads, dim 128        |
        |                             v
        |                    (1) indexer attention over the index KV cache
        |                        -> one score per 128-token KV block
        |                             |
        |                             v
        |                    (2) top-16 block select  (+ forced init/local blocks)
        |                             |
        v                             v
   (3) sparse GQA attention over exactly those 16 blocks
                       |
                    o_proj
```

Released config (`MiniMaxAI/MiniMax-M3`, `text_config.sparse_attention_config`):

| field | value | meaning |
|---|---|---|
| `sparse_block_size` | 128 | KV block granularity |
| `sparse_topk_blocks` | 16 | blocks kept per query → **2048-token budget** |
| `sparse_index_dim` | 128 | indexer head dim |
| `sparse_num_index_heads` | 4 | indexer heads |
| `sparse_init_block` | 0 | forced attention sinks |
| `sparse_local_block` | 1 | forced sliding-window blocks |
| `sparse_score_type` | `max` | block score = max over the block |
| `sparse_disable_index_value` | 1 on every sparse layer | index pool stores **K only** |
| `sparse_attention_freq` | 57 of 60 layers | the other 3 are dense |

Main attention is 64 Q : 4 KV heads, `head_dim` 128, `rotary_dim` 64
(partial RoPE), per-head Gemma RMS-norm.

The decisive property: **the attended-token count is constant at 2048
regardless of context length.** A query at 128k context reads the same amount of
KV as a query at 4k. Only the indexer's score pass grows with context.

## Three variants on the same layer

Alongside the released block-sparse path, the repo carries two more variants
built on the *same* `MiniMaxM3Attention` module and the same KV pool, selected by
`--granularity`:

| | **block** (released M3) | **token** (DeepSeek-style) | **dense** (no sparsity) |
|---|---|---|---|
| indexer score | `max` over each 128-key block | one score per key, no pooling | — removed |
| selection | top-16 **blocks** | top-2048 **tokens** | — removed |
| attended tokens | 2048 | 2048 (same budget) | the whole causal context |
| prefill logits buffer | `[idx_h, total_q, ⌈L/128⌉]` | `[idx_h, total_q, L]` — 128x larger | none |
| index KV cache | 256 B/token | 256 B/token | **none** |
| attention kernel | contiguous 128-token runs | scattered individual tokens | full causal |

### Dense — no indexer, no sparse kernel

`--granularity dense` removes selection entirely. This is not a synthetic
baseline: M3's own 3 dense layers (of 60) are built exactly this way, and the
variant reuses **sglang's production Triton attention** — `extend_attention_fwd`
for prefill, `decode_attention_fwd` (split-K) for decode — rather than a bespoke
kernel, so the comparison is against the real alternative. Entry points live in
`minimax_sparse_ops/minimax_dense.py`.

Two consequences beyond latency:

* At the layer level the module is instantiated with
  `is_sparse_attention_layer=False`, so the **index QKV projection disappears
  too** — the indexer is gone from the weights, not just bypassed.
* The KV pool is built with no index cache at all, so a dense layer stores
  512 B/token instead of 768 B/token — the +50% index tax goes away.

This is the only dense reference in the results. An earlier revision also
carried a `scaled_dot_product_attention` baseline over *de-paged* contiguous K/V;
it was removed because nothing in a real server can reach it — it skips the paged
gather entirely — and comparing against it made sparsity look worse than it is
(it pushed the apparent crossover from ~16–32k out to ~65k).

### Token — DeepSeek-style, no pooling

New Triton kernels (`sglang/kernels/ops/attention/minimax_sparse/token/`):

* `_token_index_score_prefill_kernel` — per-key indexer logits with causal
  masking and the forced attention-sink / sliding-window bias applied in-kernel,
  so selection stays a plain top-k. The query axis is chunked to a byte budget:
  unchunked, a 128k prefill's logits would be 64 GiB.
* `_token_index_score_decode_kernel` — the decode equivalent, split-K over the
  context so a small batch still fills the device. Its logits are
  `[idx_heads, batch, L]` — 512 MiB at tp=1 with batch 32 and a 1M context — so
  past `DEFAULT_DECODE_SCORE_BUDGET_BYTES` (128 MiB) the *key* axis is walked in
  windows and the per-window top-k are merged. Windowing the key axis rather
  than the query axis is what makes this cheap: every key is still read exactly
  once, where the prefill planner's query chunking re-reads the whole index KV
  cache per chunk.
* `_gqa_token_sparse_fwd_kernel` — GQA flash attention over a per-query list of
  token positions, resolved through `req_to_token`. One kernel serves prefill
  and decode; `NUM_KV_CHUNKS` splits the top-k list for small-batch decode.
* `_merge_chunks_kernel` — log2-domain softmax-state merge for that split.

Selection uses **`flashinfer.top_k`**, with `torch.topk` as the fallback when
FlashInfer is not installed. Both are exact, so availability changes speed and
never the answer. A register-resident streaming top-k was the original plan but
does not fit: at a 2048-token budget the running heap is ~32 KB per query row,
and the bitonic merge the block path uses costs ~82M compare-exchanges per row at
1M against ~0.5M for block granularity.

`tie_break=SMALL` is pinned. The forced sink / sliding-window positions all carry
the same bias, so a row can hold more exact ties than the budget selects; any
tie-consistent set is a correct top-k, but pinning the smaller index keeps the
choice deterministic and identical to the `torch.topk` fallback's. It is free.

**Prefill previously routed `topk == 2048` to sgl-kernel's `fast_topk_v2`. That
has been removed, because the selector is approximate at M3's context lengths.**
Measured on H200 against an exact top-k over float32 rows:

| row width | recall of true top-2048 | wrong picks/row | worst selected rank |
|---|---|---|---|
| 16k – 128k | 100% | 0 | 2047 |
| 256k | 98.88% | 22.9 | 2071 |
| 512k | 100% | 0 | 2047 |
| 1M | 93.95% | 123.9 | 2200 |

The wrongly admitted positions rank just past the cutoff rather than being
unrelated, and the kernel is upfront that it trades accuracy — its own test
(`sgl-kernel/tests/test_topk.py`) permits up to 5 wrong entries. The problems are
that the error is **not monotonic in width** (512k exact, 256k and 1M not), so no
width gate is reliably safe; the result is **not run-to-run deterministic**; and
that test only covers widths to 64k, so nothing exercised the regime M3 runs in.
`fast_topk_transform_fused` is the same selector with a page-table transform
fused on — measured identical error (1.11% at 256k, 6.05% at 1M) — so it is not
an alternative. FlashInfer's was exact at every width tested, including the two
where sgl-kernel's diverges, and is ~3-3.8x faster than `torch.topk`
(0.42 vs 1.57 ms on a 128x1M top-2048), so exactness here costs nothing.

For the token path the model layer is untouched — granularity is entirely a
backend concern. (Dense is the exception: it changes how the layer itself is
built, since removing the indexer removes its projection.) Correctness for both
new variants is covered by
`test/registered/kernels/ops/attention/test_minimax_token_sparse.py` — see the
accuracy-test section below.

Constraint: the token path supports only the K-only indexer
(`sparse_disable_index_value=1`), which is how M3 ships every sparse layer, and
has no attention-sink input. MSA is force-disabled under it — MSA's sparse unit
is a 128-token block.

## Two-level (block → token) indexer — prototype

`two_level_indexer.py` is a fourth selection path, not wired into the model: a
LongCat-style *hierarchical* indexer, in Triton, at M3's GQA shape. It is a
prototype and lives in this directory rather than in `sglang.kernels`.

Both shipped indexers score **every** key in the context for **every** query row.
LongCat-2.0's Hierarchical Indexing (LongCat Sparse Attention, [arXiv
2608.01662](https://arxiv.org/abs/2608.01662)) makes that two passes:

| | stage 1 — block | stage 2 — token |
|---|---|---|
| scores | `q · mean(K)` per P-token block | `q · k` per token |
| over | `L / P` block representatives | the `M · P` recalled tokens |
| selects | top-M blocks | top-2048 positions |
| cost | `O(L / P)` | `O(M · P)` — **constant in L** |

The prototype is deliberately *not* a transcription of LongCat's kernel. Three
things had to change for M3's shape, and each is a measured decision rather than
a preference:

* **Query tiling.** LongCat and M3 both select per query *block*; a naive
  per-row hierarchical pass loses the key-tile reuse the flat kernel gets from
  its query tiling and comes out **6× slower** than the flat indexer it
  replaces. `query_tile` (default 64) makes one stage-1 recall serve a tile of
  rows — union over the tile, per-row top-k — which is what makes it win.
* **Per-KV-head recall.** LongCat's indexer emits one score per token, so it has
  one ranking. M3 has four index heads, and making them share one candidate set
  (`share_candidates=True`) costs recall heavily — 99.9% → 90.5% at M=128,
  89.4% → 46.2% at M=32 — while changing latency by under 2%, because neither
  stage is bandwidth-bound at these sizes. Off by default; the knob remains.
* **Pooled-key cache.** Stage 1 needs a mean-pooled block representation of the
  index K cache. `build_pooled_index_keys` builds it, but a server would
  maintain it incrementally at KV-write time (a running mean is exact under
  append-only writes) for +2 B/token/layer against the index cache's 256 — a
  0.8% tax on it, 0.26% on the layer's KV. The
  benchmark measures both.

Forced sinks (`init_tokens`) and the sliding window (`local_tokens`) are biased
into *both* levels, so they survive coarse recall by construction — which is
asserted, per row, including for the earliest row of each query tile.

**What it is not:** exact. The flat token indexer is an exact top-k; this one
never scores a token whose block pooled badly. `--mode recall` measures that
against the exact selection as a function of M.

#### What it costs and what it buys

H200, one GPU, released M3 shape, `P=128 M=128` (16,384 candidates for a
2048-token budget), `query_tile=64`, clustered index keys. **GPU kernel time**,
which for decode is the number that carries over to a server — sglang captures
decode in a CUDA graph, and the two-level driver's wall clock is otherwise 3-7x
its kernel time at small batch (six launches and two selector calls for ~0.1 ms
of work).

Prefill, one 2048-token extend chunk:

| context | block (shipped) | flat token | **two-level** |
|---|---|---|---|
| 16k | 0.33 ms | 1.69 ms | 3.04 ms |
| 64k | 1.34 ms | 3.80 ms | 3.05 ms |
| 256k | 5.21 ms | 12.66 ms | **3.09 ms** |
| 1M | 22.14 ms | 54.60 ms | **3.20 ms** |

Decode selection, batch 32:

| context | block | flat | **two-level** |
|---|---|---|---|
| 16k | 0.04 ms | 0.15 ms | 0.24 ms |
| 64k | 0.13 ms | 0.38 ms | 0.26 ms |
| 256k | 0.51 ms | 1.32 ms | **0.28 ms** |
| 1M | 2.11 ms | 5.14 ms | **0.36 ms** |

Selection + sparse GQA attention end to end at 1M, batch 32: 5.21 ms flat →
**0.45 ms**.

The shape of it is the point: **the two-level pass does not grow with context.**
It is a fixed ~3.1 ms of prefill and ~0.3 ms of decode from 16k to 1M, because
only stage 1 sees the context and stage 1 is 1/128th of a pass. It loses below
the crossover — ~48k against the flat indexer, ~120k against the shipped block
one — which is the same reason LongCat only enables HI at ≥256k.

Two dependencies decide whether that holds up:

* **Stage 0 must be incremental.** Rebuilding the pooled block cache per call
  costs 2.70 ms at 1M/batch 32 against 0.36 ms for the whole cached selection —
  it is 7x the thing it feeds. Maintained at KV-write time it is free and costs
  +2 B/token/layer against the index cache's 256 (a 0.8% tax on it).
* **Recall depends on M, and on the query-tile bet.** Both measured at 128k
  against the exact flat top-2048.

  M is a clean knob. Latency is flat in M across this whole range, so being
  generous costs nothing:

  | M | candidate tokens | recall | worst row |
  |---|---|---|---|
  | 16 | 2,048 (1.6% of ctx) | 67.6% | 60.9% |
  | 32 | 4,096 (3.1%) | 88.8% | 81.5% |
  | 64 | 8,192 (6.2%) | 98.2% | 96.3% |
  | 128 | 16,384 (12.5%) | **99.95%** | 99.9% |
  | 256 | 32,768 (25%) | 100% | 100% |

  `query_tile` is not. A tile's candidate set is the union over its rows *capped
  at M blocks*, so a row loses blocks whenever the rest of the tile crowds it
  out — harmless if neighbouring queries want the same blocks, ruinous if they
  do not:

  | `query_tile` | correlated queries | iid queries | prefill latency |
  |---|---|---|---|
  | 1 | 99.9% | 99.9% | 2.09 ms |
  | 16 | 98.9% | 57.2% | 0.45 ms |
  | 64 | **98.4%** | **31.4%** | 0.45 ms |
  | 128 | 92.5% | 23.5% | 0.44 ms |

  `query_tile=1` sidesteps the question and gives the entire win back (5.7x
  slower here; slower than the flat indexer outright at a 2048-token extend).
  Neither column is the released model — the harness's query rows are
  independent random vectors, `make_clustered_index_queries` draws them around a
  per-tile centre instead, and settling it needs the checkpoint. The default of
  64 is chosen on the correlated column, which is the premise every
  query-block-granular selector already runs on.

Measured on iid Gaussian keys (`--iid`) every recall number above is far worse,
because mean pooling can only summarise a block whose keys resemble each other.
That is a property of real KV and not of the harness's default cache, which is
why the recall runs re-draw it — see `make_clustered_index_keys`.

#### Where stage 1 pools: before the dot, or after it

`pool_position` chooses which side of the `q·k` the pooling happens on, and the
two sides are different designs rather than an implementation detail:

| | `"pre"` (default, LongCat/HISA) | `"post"` (M3's block indexer) |
|---|---|---|
| block score | `q · mean(K_block)` | `max` over the block of `q · k` |
| stage 1 reads | `L/P` pooled keys | all `L` keys |
| total cost | `O(L/P + M·P)` — flat in L | `O(L + M·P)` — the block path's asymptote |
| stage 0 | a pooled cache, maintained at KV-write time | **none** |
| at M = topk/P | mean-pooled block selection | **exactly the shipped block indexer** |

That last row is not an analogy. With `pool_position="post"` and `M = topk/P`,
stage 1 scores blocks the way the released kernel does and hands stage 2 exactly
as many candidates as the budget, so it keeps all of them — the selection is
bit-identical to `flash_decode_with_topk_idx`'s, which
`test_post_pooling_at_block_budget_is_the_shipped_block_indexer` asserts. So the
post variant is a strict generalization of what M3 ships: **M dials the released
indexer continuously into the exact token one**, and every intermediate M is a
refinement of the block path rather than a different algorithm.

**The cost argument.** Post pays a full context scan in stage 1, so it inherits
the block path's growth (GPU kernel time, 2048-token extend, clustered keys):

| context | pre | post | block (shipped) | post's premium over shipped |
|---|---|---|---|---|
| 16k | 3.04 ms | 3.40 ms | 0.33 ms | +944% |
| 64k | 3.05 ms | 4.46 ms | 1.34 ms | +234% |
| 256k | 3.09 ms | 8.75 ms | 5.22 ms | +68% |
| 1M | **3.21 ms** | 25.83 ms | 22.15 ms | **+17%** |

Read the last column rather than the middle one: post is not competing with pre,
it is competing with *the indexer M3 already runs*. At 1M it buys near-exact
token selection for 17% more indexer time and **no change to the KV pool** —
where pre needs stage 0 maintained or it loses 7.3x at decode.

**The quality argument does not go the way "exact scores" suggests.** Coverage
of the exact top-2048 at 128k:

| M | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| pre (mean) | **70.1%** | **90.4%** | **98.6%** | 99.9% | 100% |
| post (max) | 67.8% | 87.9% | 98.5% | **100%** | 100% |

Mean pooling ranks blocks *better* below M=128 and worse above it, because the
two scores answer different questions. `max` says "this block holds one excellent
key"; `mean` says "this block is collectively aligned with the query". When M is
tight, stage 2 keeps most of each recalled block, so collective relevance
predicts coverage better; when M is loose, stage 2 cherry-picks inside blocks, so
the presence of peaks does. Neither pooling is uniformly the better block score —
which side wins depends on how much room stage 2 has to be selective.

**Which to use.** `post` is the better *drop-in*: it needs nothing from the KV
pool, it is strictly better than the shipped indexer at equal M, and its premium
over that indexer shrinks as context grows. `pre` is the better *destination*: it
is the only one that stops growing with context, and past ~120k it is the only
one that beats the shipped path outright — but it is only real if the pool can be
taught to maintain the pooled cache.

    CUDA_VISIBLE_DEVICES=0 python bench_two_level_indexer.py --mode pooling

#### The three indexers are one family

All three spend the **same 2048-token budget**; they differ only in which 2048
they spend it on. `--mode coverage` measures that directly — every path's
selection against the exact flat top-2048, at 128k, clustered keys:

| path | of the exact top-2048 | of its top-256 |
|---|---|---|
| block (shipped, top-16 blocks) | 67.8% | 99.9% |
| two-level, M=16 | 70.1% | 99.6% |
| two-level, M=32 | 90.4% | 99.9% |
| two-level, M=64 | 98.6% | 100% |
| two-level, M=128 | 99.9% | 100% |
| flat token | 100% | 100% |

The block path is not a different algorithm from the two-level one — it is its
**M = topk/P corner**. At M=16 stage 1 recalls 16 blocks and stage 2 has 2048
candidates for a 2048 budget, so it keeps all of them: block selection, with mean
pooling instead of max. The measurement agrees (70.1% vs 67.8% — mean pools
marginally better here). Everything above M=16 is stage 2 being given room to
choose, and the flat indexer is the M = L/P end of the same axis. One knob spans
all three.

The `top-256` column is why the block path works at all despite covering only
two thirds of the exact set: it captures **99.9% of the highest-scoring
tokens** and spends the rest of the budget on their block-neighbours. Block
granularity is not choosing badly, it is choosing at a coarser unit — and it
gets contiguous 128-token KV runs for it, which the token paths do not.

It also shows both pooled paths lean on the same assumption. On iid keys
(`--iid`), where a block's keys say nothing about each other, block coverage
falls to 8.4% and two-level M=128 to 23.0%, while the flat indexer is exact by
construction. The shipped indexer already bets on within-block locality; the
two-level one makes the same bet at the same granularity and then refines
inside it.

Correctness lives in `test_two_level_indexer.py` (24 cases, one GPU). Its anchor
is that with M large enough to recall every block, the two-level pass must
reproduce `token_select_*`'s selection **exactly** — same bf16 dot, same
operands — which pins the candidate → column → position mapping, the biases, the
causal masking and the chunking in one assertion. Everything sparser than that
is checked structurally (causality, distinctness, forced regions, -1 padding) or
statistically (recall).

## Which kernels actually run

`MiniMaxSparseAttnBackend` has two paths for stage (3):

* **MSA** (`fmha_sm100`) — Blackwell-only, and additionally requires
  `page_size == block_size == 128` and `topk ∈ {4,8,16,32}`.
* **Triton** (`minimax_sparse_ops`) — everywhere else, including **Hopper
  (H200)** and ROCm.

Stages (1) and (2) are always the Triton/JIT kernels; MSA only replaces (3).
On an H200 box these benchmarks therefore measure the Triton path — which is
what the validated H200 recipe (`--attention-backend triton --page-size 128`,
`--tp 8`) runs in production.

Shapes are the full released model: 64 Q heads, 4 KV heads, 4 index heads. To
measure a narrower shard, sweep it directly — `--q-heads 8 --kv-heads 1`.

## Layout

### Infrastructure

| file | role |
|---|---|
| `m3_config.py` | released M3 constants, shape definition, byte accounting |
| `harness.py` | input builders, CUDA-event timing (L2-flushed), memory probes, per-kernel → stage attribution |
| `test_harness_units.py` | CPU-only unit tests for the harness bookkeeping |
| `test_two_level_indexer.py` | correctness of the two-level indexer prototype (needs one GPU) |
| `run_accuracy_test.sh` | correctness tests (no GPU-idleness needed) |
| `run_kernel_microbench.sh` / `run_e2e.sh` / `run_all.sh` | level-1 / level-2 / everything drivers |
| `_common.sh` | shared config (`OUT`, `WAIT_IDLE`, `METRIC`), sourced by the runners |

### Whole-pipeline benchmarks

| file | question it answers |
|---|---|
| `bench_kernels.py` | **level 1** — indexer / top-k / sparse-attention stage breakdown in isolation, plus dimension sweeps (heads, head_dim, topk, block_size, page_size, batch) |
| `bench_layer.py` | **level 2** — the real `MiniMaxM3Attention.forward()`, so projections, qk-norm/rope and kv-store are included |
| `bench_e2e.py` | **level 3** — TTFT / TPOT / ITL / throughput through a real `sglang.Engine` |

Mind the names: `run_e2e.sh` is the **level-2** driver (it runs `bench_layer.py`
for each granularity). `bench_e2e.py` is level 3 and has no runner — invoke it
directly, see [section 4](#4-serving-level-benchmark-level-3).

### Selection pipeline (indexer + top-k)

| file | question it answers |
|---|---|
| `bench_indexer.py` | token vs block *granularity*: what each writes and then selects over |
| `bench_indexer_topk.py` | every fused indexer+top-k *implementation*, both phases, checked for selection equivalence |
| `bench_indexer_compare.py` | block vs token score kernel, per intermediate op, one subprocess per context |
| `bench_topk_selector.py` | the top-k selector alone, fed pre-materialised score matrices |
| `two_level_indexer.py` | **prototype** — the two-level (block → token) indexer itself: Triton kernels, config, torch references, recall helpers. Library, not a CLI |
| `bench_two_level_indexer.py` | two-level vs flat vs block: latency, per-stage cost, recall vs the coarse budget, coverage at equal budget, and selection + attention end to end |
| `plot_indexer_comparison.py` | the one figure comparing all four indexers (block, token, two-level pre/post-pool) — latency in both phases, what each selects, and why |
| `inner_profile.py` | intra-kernel walltime of the indexer score kernel's inner ops (Proton). **Library, not a CLI** — driven by `bench_indexer.py --inner-iters` and `bench_indexer_compare.py` |

### Sparse attention kernel

| file | question it answers |
|---|---|
| `bench_sparse_attention.py` | the attention kernels alone — no indexer, no selector — fed pre-generated selections |
| `bench_proton_breakdown.py` | intra-kernel walltime of the block-sparse prefill kernel's inner ops (Proton) |

### page_size / block_size studies

| file | question it answers |
|---|---|
| `bench_pagesize_pipeline.py` | page_size × context through the full pipeline, original vs paged step-3 kernel |
| `bench_blocksize_pipeline.py` | block_size × context through the full pipeline, token budget held constant |
| `bench_block_sparse_paged.py` | original / no-modulo / paged at kernel level, prefill and decode, bit-exactness checked |
| `bench_stage1_lookup.py` | **stage 1** — slot-compute cost vs page_size, with a lookup-off control |
| `bench_kv_fragmentation.py` | **stage 2** — KV-load cost as a block splits into more, shorter runs |

### Plotting / reporting

| file | role |
|---|---|
| `plot_results.py` | trend plots from the level-1 / level-2 JSON |
| `plot_indexer_topk.py` | plots for `bench_indexer_topk.py` output |
| `plot_indexer_selector.py` | plots for the indexer+selector comparison |
| `summarize_accuracy.py` | folds the accuracy suites' JUnit XML into `summary.json` / `summary.txt` |

Most benchmarks write `raw.json` + `summary.csv` (+ plots) into their own
subdirectory of `results/`, alongside a `README.md` describing the method and an
`ANALYSIS.md` recording what was found.

Implementation lives outside this directory:
`srt/layers/attention/minimax_sparse_ops/` (`minimax_sparse.py` block,
`minimax_token_sparse.py` token, `minimax_dense.py` dense) and
`kernels/ops/attention/minimax_sparse/` (block kernels, plus `token/`).

## Running

**Every command below is run from this directory:**

```bash
cd benchmarks/minimax_m3_sparse_attn
```

Nothing depends on the working directory — each script inserts its own directory
on `sys.path` and resolves its defaults from `__file__`, and `run_accuracy_test.sh`
`cd`s to the repo root itself — so the same commands work from anywhere if you
spell the paths out. Running from here just keeps them short, and makes the
default output paths (`results/...`) mean what they say.

Everything here uses **exactly one GPU** — device 0 of
whatever `CUDA_VISIBLE_DEVICES` exposes. Nothing here is multi-GPU: the kernel
benchmarks and tests make plain single-device calls, and `bench_layer.py` builds
a world-size-1 TP group purely so `QKVParallelLinear` / `RowParallelLinear` can
be constructed.

Everything runs the **released M3 shape**: 64 Q heads, 4 KV heads, 4 index
heads, on a single GPU. The suite used to take a `--tp` argument that derived a
per-rank shard, but TP is a launch topology and nothing here launches more than
one device — the shard shapes it produced are reachable directly through the
`num_q_heads` / `num_kv_heads` dimension sweeps, which is where shape
sensitivity belongs. The one thing this cannot capture either way is the TP
all-reduce a real multi-GPU server pays after `o_proj`.

**Pin to an idle GPU with `CUDA_VISIBLE_DEVICES`.** Contention does not merely add
noise: a co-tenant inflates launch latency enough to swamp these kernels
entirely (a 15 µs decode step reads as 500 µs). The benchmark scripts sample GPU
utilization at startup and warn if the device is already busy; on a shared box
pass `--wait-for-idle 300` to block for up to 5 minutes waiting for a clean
window before measuring. This does not apply to the correctness tests, which
only check numbers.

```bash
# everything: accuracy tests, level 1, level 2 (each level plots its own results)
CUDA_VISIBLE_DEVICES=0 bash run_all.sh

# or each stage on its own
bash run_accuracy_test.sh                            # no idle GPU needed
CUDA_VISIBLE_DEVICES=0 bash run_kernel_microbench.sh # level 1
CUDA_VISIBLE_DEVICES=0 bash run_e2e.sh               # level 2
```

The runners are thin wrappers: everything after the script name is **forwarded
verbatim** to the underlying Python benchmark, so any flag from the sections
below works through them too.

```bash
# forward flags straight through to bench_kernels.py / bench_layer.py
bash run_kernel_microbench.sh --context-lens 4096,32768
bash run_e2e.sh --decode-batch-sizes 1
```

Env knobs:

| variable | default | honoured by | meaning |
|---|---|---|---|
| `OUT` | `<this dir>/results` | all | root of the results subtree |
| `WAIT_IDLE` | `180` | benchmark runners | seconds to wait for an idle GPU (`0` on a dedicated box) |
| `METRIC` | `gpu` | benchmark runners | plot estimator: `gpu` \| `min` \| `median` |
| `PYTHON` | `python` | all | interpreter for the benchmark/pytest step (the plot step always calls `python`) |
| `MODES` | `memory context sweeps` | `run_kernel_microbench.sh` | which `--mode` groups to run |
| `GRANS` | `block token dense` | `run_kernel_microbench.sh`, `run_e2e.sh` | granularities to measure |
| `PORT` | `29531` | `run_e2e.sh` | first TCP port; incremented per granularity |
| `PYTEST_ARGS` | *(empty)* | `run_accuracy_test.sh` | extra pytest args; quoting preserved |
| `SKIP_TESTS` | `0` | `run_all.sh` | `1` = benchmarks only, skip the accuracy stage |

Each stage owns a subtree, data kept apart from plots, and the two benchmark
levels each keeping data and plots apart:

```
results/
  bench_accuracy/results/       <suite>.log, <suite>.junit.xml, summary.json
  bench_kernels/                level 1 (run_kernel_microbench.sh)
    results/                    *.json, *.csv
    plots/                      *.png
  bench_layers/                 level 2 (run_e2e.sh), same shape
```

The focused studies in [section 5](#5-focused-studies) are not part of that
subtree — each writes `raw.json` / `summary.csv` / `*.png` straight into its own
`results/<study>/` directory, and `bench_e2e.py` writes `results/e2e.{json,csv}`.

---

### 1. Accuracy tests

Numerical correctness of the kernels, against PyTorch references. Needs `pytest`
(`uv pip install --python .venv/bin/python pytest`) and takes ~16 min on one GPU.

Every suite is saved under `results/bench_accuracy/results/`: `<suite>.log` (the
console output, tee'd so the run stays watchable), `<suite>.junit.xml`, and a
`summary.json` / `summary.txt` folding the four suites together with per-test
counts and the ids of anything that failed. A rerun overwrites them, exactly as
the benchmark runners overwrite their JSON/CSV.

```bash
# every suite, with a pass/fail summary at the end
bash run_accuracy_test.sh

# narrow it (quoting is preserved, so `-k` expressions work)
PYTEST_ARGS='-k "chunkedprefill or padding"' bash run_accuracy_test.sh

# trailing args are forwarded to every suite, same effect
bash run_accuracy_test.sh -k chunkedprefill

# or invoke pytest directly (the runner cd's to the repo root; these do not,
# so the suite paths are relative to this directory)
CUDA_VISIBLE_DEVICES=0 python -m pytest \
  ../../python/sglang/srt/layers/attention/minimax_sparse_ops/tests/ \
  ../../test/registered/kernels/ops/attention/test_minimax_token_sparse.py -q

# token path only (~1 min) — the one to run while editing the token kernels
CUDA_VISIBLE_DEVICES=0 python -m pytest \
  ../../test/registered/kernels/ops/attention/test_minimax_token_sparse.py -v

# a single case
CUDA_VISIBLE_DEVICES=0 python -m pytest \
  ../../test/registered/kernels/ops/attention/test_minimax_token_sparse.py \
  -k "chunkedprefill" -v
```

The runner drives four suites, in this order (slug = the `<suite>` in the
filenames it writes):

| slug | suite | cases | what it asserts |
|---|---|---|---|
| `harness_units` | `benchmarks/.../test_harness_units.py` | 31 | CPU-only: the harness's stage attribution, memory accounting and budget cap. No GPU |
| `block_kernels` | `minimax_sparse_ops/tests/` (`test_sparse_gqa.py` + `test_flash_with_topk_idx.py`) | 74 | block-path sparse GQA and indexer vs gather-and-softmax references |
| `block_topk` | `test/.../test_minimax_decode_topk.py` | 120 | block-level top-k selectors — the decode radix path and the prefill radix path |
| `token_dense` | `test/.../test_minimax_token_sparse.py` | 56 | token-path selection as an exact **index set** vs a PyTorch reference; attention output vs gather-and-softmax; split-K equivalence; ragged + chunked prefill; paged layouts; `topk > context`; all-padding rows. Plus the **dense** variant vs a causal reference, and a cross-check that full-budget token-sparse reproduces dense exactly |

Every suite runs even if an earlier one fails, and the script exits non-zero if
any failed. A suite that a `-k` filter narrows to nothing counts as *skipped*,
not failed.

**This is kernel accuracy, not model accuracy.** Everything in this directory
runs on dummy weights, so it says nothing about generation quality. In
particular, swapping to token granularity changes what M3 attends to, and its
weights were trained against block selection — expect a quality delta that these
tests cannot see. A real eval needs the checkpoint; the pattern to copy is
`test/registered/amd/accuracy/mi35x/test_minimax_m3_tp4_eval_mi35x.py`, which
launches a server on `MiniMaxAI/MiniMax-M3-MXFP8` and scores GSM8K
chat+thinking against a 0.95 threshold. To measure the granularity delta, run
that eval twice with `SGLANG_USE_MINIMAX_TOKEN_SPARSE` unset vs `=1`.

---

### 2. Kernel micro-benchmarking (level 1)

`bench_kernels.py` — the attention stages in isolation, driven through the real
`minimax_sparse_prefill` / `minimax_sparse_decode` entry points (and
`minimax_dense_*` for `--granularity dense`).

```bash
# driver: all modes, all granularities, plus plots
CUDA_VISIBLE_DEVICES=0 bash run_kernel_microbench.sh

# or call bench_kernels.py directly:
# full level-1 suite: memory accounting + context sweep + dimension sweeps,
# for all three variants
CUDA_VISIBLE_DEVICES=0 python bench_kernels.py --mode all --wait-for-idle 300

# just the context-length sweep (4k..1M), all three variants
CUDA_VISIBLE_DEVICES=0 python bench_kernels.py --mode context

# dense only (no indexer, no sparse kernel)
CUDA_VISIBLE_DEVICES=0 python bench_kernels.py --mode context --granularity dense

# one granularity, one sweep — e.g. the token-path budget curve
CUDA_VISIBLE_DEVICES=0 python bench_kernels.py \
  --mode sweeps --granularity token --sweeps topk_blocks

# quick smoke run while editing a kernel
CUDA_VISIBLE_DEVICES=0 python bench_kernels.py \
  --mode context --context-lens 4096,16384 --decode-batch-sizes 32 \
  --prefill-iters 8 --decode-iters 50 --granularity block --show-unmapped
```

Flags:

```
--mode all|context|sweeps|memory     which groups to run
--granularity block,token,dense      variants to measure (default: all three)
--context-lens 4096,...,1048576      context sweep (through M3's 1M limit)
--decode-batch-sizes 1,32            decode batch sizes for the context sweep
--prefill-chunk 4096                 fixed extend chunk; 0 = whole-context prefill
--sweeps topk_blocks,page_size       run only some dimension sweeps
--sweep-context-len 32768            context the dimension sweeps are held at
--prefill-iters / --decode-iters     timed iterations per point
--wait-for-idle 300                  wait for a clean GPU before measuring
--show-unmapped                      print kernels that fell into the "other" bucket
-o results/ --tag NAME               where the JSON/CSV lands
```

---

### 3. End-to-end test (level 2)

`bench_layer.py` — the real `MiniMaxM3Attention.forward()` with dummy weights,
against a real `MiniMaxSparseKVPool` and the real `MiniMaxSparseAttnBackend`. So
the measurement includes the fused QKV + index-QKV projection, per-head Gemma
RMS-norm, partial RoPE, the fused KV + index-K cache store, the full selection
pipeline, and `o_proj` — everything a serving layer pays for. Under
`--granularity dense` the module is built with `is_sparse_attention_layer=False`,
so there is no index projection and no index KV cache, and attention runs through
a thin adapter onto sglang's dense Triton kernels.

```bash
# driver: all three granularities, plus plots
CUDA_VISIBLE_DEVICES=0 bash run_e2e.sh

# or call bench_layer.py directly:
# end-to-end layer, block granularity (what M3 ships)
CUDA_VISIBLE_DEVICES=0 python bench_layer.py \
  --granularity block --wait-for-idle 300 --tag layer_block

# same layer, token granularity
CUDA_VISIBLE_DEVICES=0 python bench_layer.py \
  --granularity token --wait-for-idle 300 --tag layer_token

# dense: the layer is built with is_sparse_attention_layer=False, so the index
# QKV projection and the index KV cache are gone entirely
CUDA_VISIBLE_DEVICES=0 python bench_layer.py \
  --granularity dense --wait-for-idle 300 --tag layer_dense
```

Run more than one at a time and give each a distinct `--port` — they each stand
up a world-size-1 TP group.

Takes the same `--context-lens` / `--decode-batch-sizes` / iteration
flags as `bench_kernels.py`, plus:

```
--granularity block|token|dense
                            block/token set SGLANG_USE_MINIMAX_TOKEN_SPARSE;
                            dense builds the layer with no indexer at all
--no-download               use the pinned config copy instead of fetching
                            config.json from the Hub
--port 29531                TCP port for the world-size-1 TP group; change it if
                            you run two instances at once
```

Row `layer_ms_x57` extrapolates one layer to M3's 57 sparse layers.

This is end-to-end *for the attention layer*, not for the model: it excludes the
MoE, the TP all-reduce after `o_proj`, and scheduling. A true full-model run
needs the ~435B checkpoint — launch a server per the
[M3 cookbook](../../docs_new/cookbook/autoregressive/MiniMax/MiniMax-M3.mdx)
(`--tp 8 --attention-backend triton --page-size 128` on H200) and drive it with
`python -m sglang.bench_one_batch_server` or `python -m sglang.bench_serving`.

---

### 4. Serving-level benchmark (level 3)

`bench_e2e.py` — TTFT / TPOT / ITL / throughput through a real `sglang.Engine`,
so the numbers include continuous batching, the paged allocator, CUDA graphs and
the scheduler. The model is M3's own architecture cut to `--num-layers` layers
and loaded with `load_format=dummy`, which is why no 435B checkpoint is needed.
Not part of `run_all.sh` — run it directly.

```bash
# defaults: 1 layer, block granularity, 4k/32k/128k input x batch 1/8/32
CUDA_VISIBLE_DEVICES=0 python bench_e2e.py

# one granularity, one point
CUDA_VISIBLE_DEVICES=0 python bench_e2e.py \
  --granularity token --input-lens 32768 --batch-sizes 1,8

# memory is tight: shrink the MoE and give the KV cache more room
CUDA_VISIBLE_DEVICES=0 python bench_e2e.py \
  --num-experts 16 --mem-fraction-static 0.75
```

```
--num-layers 1              depth of the dummy model
--granularity block|token|dense
--input-lens 4096,32768,131072
--batch-sizes 1,8,32
--output-len 32             decoded tokens per request
--num-experts N             shrink the MoE (default: the released 128)
--page-size 128 / --mem-fraction-static 0.85 / --attention-backend NAME
--project-layers 60         also report the figure scaled to this depth (0 = off)
--no-download               use the pinned config copy instead of the Hub
--warmup 1 / --seed 0 / --wait-for-idle 0
-o results/ --tag e2e       -> <out>/e2e.json + e2e.csv
```

One layer is what is measured; `--project-layers` is arithmetic on top of it and
assumes uniform layers, so it ignores sampling, detokenization and per-step
scheduling overhead.

---

### 5. Focused studies

Each of these answers one narrow question and is run directly — none are wired
into the `run_*.sh` drivers. Each writes into its own `results/` subdirectory and
overwrites on rerun — `raw.json` + `summary.csv` + PNGs, except the two indexer
sweeps, which use the tagged `<tag>.json` / `<tag>.csv` naming the level-1/2
benchmarks use and land in `results/` itself. Every one takes `--wait-for-idle`
except `bench_proton_breakdown.py`. Defaults quoted below are the actual
defaults, so the bare commands reproduce each study as-is.

**Selection pipeline (indexer + top-k)**

```bash
# token vs block granularity — what each writes, then selects over
CUDA_VISIBLE_DEVICES=0 python bench_indexer.py
CUDA_VISIBLE_DEVICES=0 python bench_indexer.py \
  --context-lens 65536,262144 --decode-batch-sizes 32
#   --granularity block,token   --prefill-chunk 8192   --inner-iters 5 (0 = skip
#   the Proton intra-kernel breakdown)   --prefill-iters 10 --decode-iters 50
#   --profile-iters 10   -o results/ --tag indexer   --no-plots
#   -> results/indexer.{json,csv} + results/plots/*.png

# every fused indexer+top-k implementation, both phases, equivalence-checked
CUDA_VISIBLE_DEVICES=0 python bench_indexer_topk.py
CUDA_VISIBLE_DEVICES=0 python bench_indexer_topk.py \
  --context-lens 16384,131072 --phases prefill
#   --prefill-impls block,current,fused,fused_cuda,seg,onepass,tau_emit
#   --decode-impls block,token   --skip-check skips the equivalence pass
#   --profile-iters 8 (0 = no stage breakdown)   -o results/ --tag indexer_topk
#   -> results/indexer_topk.{json,csv} + results/plots/*.png

# block vs token score kernel, per intermediate op, one subprocess per context
CUDA_VISIBLE_DEVICES=0 python bench_indexer_compare.py
CUDA_VISIBLE_DEVICES=0 python bench_indexer_compare.py --context-lens 16384,131072
#   --wait-for-idle defaults to 600 here — pass 0 on a dedicated box
#   -> results/indexer_comparison/
#   (--single-ctx / --json are internal: the parent re-execs itself per context
#    so the block kernel's autotune cache cannot leak across contexts)

# the selector alone, fed pre-materialised score matrices
CUDA_VISIBLE_DEVICES=0 python bench_topk_selector.py
CUDA_VISIBLE_DEVICES=0 python bench_topk_selector.py \
  --context-lens 16384,131072 --iters 20
#   --wait-for-idle defaults to 600   -> results/selector_comparison/

# two-level (block -> token) indexer prototype vs the flat and block indexers
CUDA_VISIBLE_DEVICES=0 python bench_two_level_indexer.py
CUDA_VISIBLE_DEVICES=0 python bench_two_level_indexer.py --mode recall,coverage
CUDA_VISIBLE_DEVICES=0 python bench_two_level_indexer.py \
  --mode latency,stages,recall,e2e --context-lens 262144,1048576
#   --mode latency|stages|recall|coverage|pooling|e2e
#   --pool-block 128 --coarse-blocks 128   --pool-position pre|post
#   --query-tile 64   --share-candidates   --head-reduce max|sum|relu_sum
#   --coarse-sweep 8,16,32,64,128,256 (--mode recall)   --recall-context 131072
#   --iid measures recall on the harness's iid keys instead of clustered ones
#   --prefill-chunk 2048   --decode-batch-sizes 1,32   --no-plots
#   -> results/two_level_indexer/

# the four-way comparison figure, from whatever that benchmark last wrote
CUDA_VISIBLE_DEVICES=0 python bench_two_level_indexer.py \
  --mode latency,coverage -o results/four_way
python plot_indexer_comparison.py \
  -i results/four_way/raw.json -o results/four_way/indexer_comparison.png
#   the latency mode emits all four indexers; coverage adds the M sweep for both
#   pooling positions, which panel C draws as one curve through all four

# its correctness suite (not part of run_accuracy_test.sh)
CUDA_VISIBLE_DEVICES=0 python -m pytest test_two_level_indexer.py -q
```

**Sparse attention kernel**

```bash
# attention only — no indexer, no selector, fed pre-generated selections
CUDA_VISIBLE_DEVICES=0 python bench_sparse_attention.py
CUDA_VISIBLE_DEVICES=0 python bench_sparse_attention.py \
  --context-lens 16384,131072 --phases decode
#   --prefill-iters 10 --decode-iters 30   --wait-for-idle defaults to 600
#   -> results/bench_sparse_attention/

# orig / nomod / paged slot resolution, bit-exactness checked at every point
CUDA_VISIBLE_DEVICES=0 python bench_block_sparse_paged.py
CUDA_VISIBLE_DEVICES=0 python bench_block_sparse_paged.py \
  --context-lens 32768,262144 --phases decode
CUDA_VISIBLE_DEVICES=0 python bench_block_sparse_paged.py \
  --variants orig,paged --no-plots
#   --page-size 128 (the paged fast path needs page_size == block_size)
#   --decode-batch-sizes 1,32,128   --prefill-iters 30 --decode-iters 200
#   -> results/block_sparse_kernel_comparison/

# intra-kernel scopes of the block-sparse prefill kernel (Triton Proton)
CUDA_VISIBLE_DEVICES=0 python bench_proton_breakdown.py
#   --context-len 32768 --chunk-len 4096 --reps 3
#   -> results/block_sparse_kernel_comparison/ (same dir as the study above)
#   Read the printed footer before quoting per-scope numbers: instrumentation
#   perturbs the schedule asymmetrically between the two paths.
```

**page_size / block_size studies**

```bash
# page_size x context through the full pipeline, orig vs paged step-3 kernel
CUDA_VISIBLE_DEVICES=0 python bench_pagesize_pipeline.py --wait-for-idle 5
#   --page-sizes 1,8,16,32,64,128,256   --context-lens 4096..1048576
#   --chunk-len 4096   --plot-only regenerates plots from an existing raw.json
#   -> results/pagesize_context_sweep/

# block_size x context, token budget held constant (topk = budget // block_size)
CUDA_VISIBLE_DEVICES=0 python bench_blocksize_pipeline.py --wait-for-idle 5
#   --block-sizes 16,32,64,128   --context-lens 8192,32768,131072,524288
#   --page-size 128   --chunk-len 4096   --plot-only
#   -> results/blocksize_sweep/

# stage 1 in isolation: slot-compute cost vs page_size, with a lookup-off control
CUDA_VISIBLE_DEVICES=0 python bench_stage1_lookup.py --wait-for-idle 5
#   --page-sizes 1,8,16,32,64,128,256   --context-len 32768 --chunk-len 4096
#   --iters 25   -> results/stage1_slot_compute/

# stage 2 in isolation: KV-load cost as a block splits into more, shorter runs
CUDA_VISIBLE_DEVICES=0 python bench_kv_fragmentation.py --wait-for-idle 5
#   --runs 1,2,4,...,128 (runs the 128-token block is split into)
#   --pool-slots 32768,262144,1048576 (max_slots — decides L2 vs DRAM)
#   --iters 30   -> results/kv_fragmentation/
```

`bench_pagesize_pipeline.py` and `bench_blocksize_pipeline.py` both accept
`--plot-only`, which re-renders from the existing `raw.json` without touching the
GPU — useful after editing the plotting code.

---

### 6. Plots

`plot_results.py` reads one results directory and writes into one plots
directory, so point it at the level you want:

```bash
# regenerate one level's plots
python plot_results.py --metric gpu \
  --results results/bench_kernels/results \
  --out     results/bench_kernels/plots

# same for level 2
python plot_results.py --metric gpu \
  --results results/bench_layers/results \
  --out     results/bench_layers/plots
```

`--metric` picks the latency estimator: `gpu` (kernel time only — the only one
that survives a shared GPU), `min` (fastest wall-clock iteration), or `median`
(needs a dedicated machine).

The other two plotters take a results *file* rather than a directory, and every
benchmark that emits plots already calls its own — run these only to re-render
after editing plotting code, or with `--no-plots` runs. None of them touch the
GPU.

```bash
# bench_indexer_topk.py output
python plot_indexer_topk.py \
  --results results/indexer_topk.json \
  --out     results/plots \
  --stack-ctx 131072,1048576        # contexts for the stage-stack panels

# alternate view of the same rows (latency / stage stacks / workspace),
# written next to --results
python plot_indexer_selector.py --results results/indexer_topk.json
```

`plot_indexer_selector.py` defaults to `results/indexer_selector_comparison/
indexer_selector.json`, a path no current benchmark writes — point `--results`
at a `bench_indexer_topk.py` JSON. Its workspace panel comes out empty there,
since that benchmark does not record `transient_bytes`.

Phase-aware plots are emitted as separate `prefill_*.png` and `decode_*.png`
files. Per-stage runtime comes in two paired forms for each phase:

* `<phase>_breakdown_absolute.png` — stacked **milliseconds** per stage.
* `<phase>_breakdown_share.png` — the same, as a **fraction** of the total.

The whole-model KV-cache footprint remains phase-independent and is emitted as
`kv_memory_vs_context.png` — level 1 only, since it comes from the `memory`
mode; attention workspace memory is split by phase and emitted at both levels.

## What is measured

**Latency** — CUDA-event timed, L2 flushed between iterations so every iteration
re-reads the KV cache from DRAM (a real decode step does). Reported as
mean/median/min/p90 over the iteration set.

Note the deliberate asymmetry in the plots: **decode is measured at two batch
sizes (1 and 32), prefill only at 1.** Decode batching is the real scheduler
knob — 32 concurrent sequences each emitting one token is routine. Prefill
batching is not the same axis: 32 long requests in one extend would need huge
activations per layer. The default prefill measurement is therefore a fixed
4096-token extend chunk over each KV context; the meaningful prefill axis is
chunk size, exposed as `--prefill-chunk`, not batch count.

**Memory** — three separate numbers, because they behave very differently:

* `kv_*_bytes` — static KV cache. Analytic and exact: a sparse layer stores
  `2·H_kv·D` bytes of main KV **plus** `1·D_idx` bytes of index K per token.
  With M3's shape that is 2048 B + 256 B = 2304 B/token/layer, i.e. the
  index cache is a **+50% KV tax** on every sparse layer. A dense layer has no
  indexer, so it stores only the 512 B and the tax disappears.
* `transient_bytes` — measured peak *extra* allocation for one attention call:
  block scores, top-k indices, split-k partials, output.
* `score_buffer_bytes` / `score_buffer_unchunked_bytes` — analytic size of the
  indexer's score buffer, which is where granularity shows up most. Block:
  `num_idx_heads × rows × ⌈L/128⌉ × 4 B`, quadratic in a single-shot prefill.
  Token: 128x that, so prefill chunks the query axis — `score_buffer_bytes` is
  the *live* chunk (capped by the budget) and `score_buffer_unchunked_bytes` the
  full matrix it stands in for (64 GiB at 128k). Dense: both are zero, there is
  no indexer.

**Runtime breakdown** — from `torch.profiler` CUDA kernel self-time, attributed
by kernel name to `indexer_score` / `topk_select` / `select_overhead` /
`sparse_attn` / `merge` for the sparse paths, `dense_attn` for the dense path,
plus `buffer_init` and — at the layer level — `projection_gemm`, `qk_norm_rope`
and `kv_store`.
Anything unrecognised lands in an explicit `other` bucket rather than being
folded away — `--show-unmapped` names it. `launch_gap_ms` is
`median wall time − Σ kernel time`, i.e. how much is launch overhead rather than
GPU work.

**Sweeps** — `num_q_heads`, `num_kv_heads` (index heads tracked to keep
`idx_group_size == 1`, as M3 ships), `head_dim`, `topk_blocks`, `block_size`
(with the token budget held constant so it isolates granularity), `page_size`
and decode `batch_size`.

Note on `page_size`: it reaches the kernels two ways. The layout channel is the
page table itself — where tokens physically sit — and the code-path channel is
the step-3 kernel's slot resolution. Both `minimax_sparse_prefill` and
`minimax_sparse_decode` now take a `page_size` and can route step 3 to the paged
kernel, which resolves a selected block with one `req_to_token` lookup per
page-aligned run instead of a per-token gather (falling back to the gather when
page and block sizes do not divide one another). That routing is on by default
for **prefill** (`SGLANG_OPT_USE_MINIMAX_SPARSE_PAGED_PREFILL`) and off for
**decode** (`..._DECODE`): measured on H200 at page_size 128, prefill gains
1.43x at 32k and 1.16x at 256k while decode runs 0.93–0.97x, since decode is
latency-bound and the scalar lookup only lengthens the dependency chain.
`bench_block_sparse_paged.py` is the A/B that produced those numbers. The layout
channel matters on its own, and the harness models it faithfully — the page table is built with tokens
contiguous *within* a page and pages scattered across the pool, exactly as
sglang's paged allocator lays them out. At `page_size=128` one 128-token sparse
block is a single contiguous DRAM run; at `page_size=1` the same block is 128
scattered gathers. Page size also gates the MSA kernel entirely
(`page_size == 128` required).

## Caveats

* MSA (`fmha_sm100`) is not exercised on Hopper. On a Blackwell box, stage (3)
  numbers will differ; stages (1) and (2) will not.
* The layer-level benchmark instantiates one attention module holding the whole
  64 Q / 4 KV head shape. It therefore excludes the TP all-reduce that a real
  multi-GPU server pays after `o_proj`.
* Dummy weights change nothing about kernel timing (shapes and access patterns
  are identical), but the selected block *sets* are random rather than what a
  trained indexer would pick. Selection content does not affect cost — the
  kernel reads `topk` blocks either way — but it does mean these runs say
  nothing about accuracy. The dense variant has no selection at all, so it is
  unaffected.
* Prefill uses a 4096-token extend chunk by default so the sweep can reach the
  model's 1M-token limit without quadratic one-shot workspaces. Pass
  `--prefill-chunk 0` only when deliberately measuring whole-context prefill.
  Chunking mainly reduces the block-score buffer, not the work per query token.
