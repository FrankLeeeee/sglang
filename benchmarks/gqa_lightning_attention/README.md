# Dense GQA vs Lightning token-sparse GQA

This standalone, synthetic single-layer benchmark compares the two requested
attention configurations using real paged caches and GPU kernels. It does not
launch a model server or change SGLang's model/backend registration.

The corrected experiment uses **4/8 indexer KV heads**, matching the main
attention KV heads. The previous shared-head experiment is archived under
[results/shared_indexer_kv1](results/shared_indexer_kv1/README.md).

<!-- RESULTS_START -->
**Completed on two H200 GPUs:** [all 96 corrected dense/sparse comparisons](results/comparison.md)
([CSV](results/comparison.csv), [JSON and run metadata](results/comparison.json)).
All 192 measurements completed without OOM; 15 samples per measurement.
The GPU correctness suite passed **34 tests** with the corrected defaults.

Dense was faster at every measured chunked-prefill point. Sparse decode becomes
faster as context and batch increase. At **128k context, batch 32**:

| Configuration | Phase | Dense (ms) | Sparse (ms) | Dense / sparse |
|---|---|---:|---:|---:|
| 1 | Chunked prefill | 19.721 | 21.313 | 0.93x |
| 2 | Chunked prefill | 26.843 | 38.600 | 0.70x |
| 1 | Decode | 1.964 | 0.827 | 2.38x |
| 2 | Decode | 3.957 | 1.491 | 2.65x |

The smallest **tested** batch at which sparse decode was faster:

| Configuration | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|
| 1 | 32 | 16 | 8 | 4 |
| 2 | 16 | 8 | 4 | 2 |

These crossovers apply to this implementation and measurement contract.
<!-- RESULTS_END -->

| Parameter | Configuration 1 | Configuration 2 |
|---|---:|---:|
| Query heads | 48 | 64 |
| KV heads | 4 | 8 |
| Head dimension | 128 | 128 |
| Hidden dimension | 8192 | 8192 |
| Indexer query heads | 48 | 64 |
| Indexer KV heads | 4 | 8 |
| Indexer head dimension | 64 | 64 |
| Indexer heads per attention group | 12 | 8 |

Each attention KV group has its own indexer key, subset of indexer query heads,
gates, scores, and top-2048 selection, matching the supplied pseudocode.
`--indexer-kv-heads grouped` is the default. `--indexer-kv-heads 1` is retained
only to reproduce the earlier shared-key experiment.

## Forward and kernels

```python
layer = Attention(CONFIGS[1], sparse=False)  # dense
layer = Attention(CONFIGS[1], sparse=True)   # one argument enables sparsity
out = layer.forward(x, paged_state, x_kv=None)
```

Both paths compute bias-free Q/K/V projections and per-head RMSNorm (epsilon
1e-6, unit gain), then append K/V to a paged BF16 cache. The dense path attends
to every causal token; the sparse path attends to the indexer's selected tokens.
The `sparse_attention` name in the requested dense forward is interpreted as the
switchable attention operation. Main attention uses the usual `1/sqrt(128)`
softmax scale. No output projection, RoPE, MLP, or residual was requested.

Sparse indexing computes exactly:

```python
q_I = linear(x.detach(), W_q).view(T, G, n_g, 64)
w = relu(linear(x.detach(), W_w)).view(T, G, n_g)
k_I = linear(x_kv.detach(), W_k).view(T, G, 64)
I[t, g, s] = sum_j(w[t, g, j] * relu(dot(q_I[t, g, j], k_I[s, g])))
```

There is no indexer normalization, RoPE, or scaling. A new Triton tensor-core
kernel fuses the dot products, ReLU, weighted head reduction and causal mask.
The kernel uses four queries per tile for 12-head indexer groups and 16
queries per tile for eight-head groups (when request boundaries permit it).
These tile sizes were selected in the earlier H200 shared-key experiment.
Decode uses one query per tile. Both configurations are remeasured with the
corrected per-group indexer keys, including their projection and cache costs.
It tiles queries and keys; it never materializes `[T,G,T]` or the per-head dot
tensor. Scores are processed in chunks bounded by `--score-budget-mib` (256 by
default). FlashInfer exact top-k selects each group's tokens. Invalid positions
remain `-1`; valid indices are request-local and are resolved through the page
table, equivalent to valid packed absolute indices without the pseudocode's
`-1 + document_start` padding bug.

The final sparse attention reuses SGLang's existing autotuned Triton
`gqa_token_sparse_attn`, including split-K decode and softmax-state merging.
The dense baseline uses FlashInfer paged FA2 prefill and tensor-core paged
decode. FA2 matches the FlashInfer prefill backend configured in this checkout's
`python/sglang/srt/layers/attention/flashinfer_backend.py`.
Projections use PyTorch/cuBLAS; RMSNorm and cache writes use Triton.

## Measurement contract

- BF16 activations, weights and caches; FP32 indexer scores; one H200 per run,
  tensor parallelism 1. No quantization or communication.
- Contexts: 16,384 / 32,768 / 65,536 / 131,072 tokens **per request**, including
  the newly appended tokens. Batch sizes: 1 / 2 / 4 / 8 / 16 / 32.
- User-confirmed **chunked prefill**: the final 2,048 total new tokens are
  divided evenly across requests (2,048 down to 64 tokens per request).
  Prefill latency is one scheduled step, **not full-prompt latency or TTFT**.
- Decode: one new token per request, captured and measured by CUDA graph
  replay. Prefill runs eagerly, as in typical SGLang chunked prefill; GPU event
  latency can include launch gaps. Optional `--prefill-graph` captures it too.
- Independently allocated request caches, shuffled physical 128-token pages.
  Historical cached activations and weights are synthetic random BF16 values;
  current Q/K/V and indexer activations are actually projected on every forward.
  Dense and sparse use identical main projection weights, inputs and main caches.
- The timed region includes Q/K/V projection, all three per-head norms, cache
  writes and attention; sparse additionally includes indexer projection,
  index-cache writes, grouped scoring, exact top-k and padding conversion.
- Planning, page allocation, input generation, JIT/autotuning, graph capture,
  scheduler, sampling and other model layers are outside timing. Cached prefix
  K/V is not reprojected on decode or an extend step.
- Three warmups; 15 samples; CUDA events; median, minimum and p90 plus every
  raw sample saved. A 64 MiB L2 flush precedes each sample outside timing.
- `--full-prefill` optionally processes every prompt token in a single forward;
  it can be much more expensive and use much more memory. It is not the main run.

This simulates the attention execution path of serving, rather than measuring
whole-server throughput or whole-model latency. It does not assess model quality
after sparsification. Backend choice, real score distributions, precision,
page size and scheduling can change the crossover.

## Reproduce

Run from the repository root using an environment with this checkout installed,
PyTorch, Triton, FlashInfer, and pytest. First-use compilation is excluded from
the measurements.

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest benchmarks/gqa_lightning_attention/test_attention.py -q

CUDA_VISIBLE_DEVICES=0 python benchmarks/gqa_lightning_attention/benchmark.py \
  --configs 1 --output benchmarks/gqa_lightning_attention/results/config1
CUDA_VISIBLE_DEVICES=1 python benchmarks/gqa_lightning_attention/benchmark.py \
  --configs 2 --output benchmarks/gqa_lightning_attention/results/config2

python benchmarks/gqa_lightning_attention/summarize.py \
  benchmarks/gqa_lightning_attention/results/config1.json \
  benchmarks/gqa_lightning_attention/results/config2.json
```

By default the driver compares both variants. Use `--sparse` for sparse only,
`--dense-only` for dense only, or `--help` for sweep and timing options. Each
completed measurement is saved immediately. OOM cases are explicitly marked;
other errors stop the run instead of silently substituting another backend.

`results/config{1,2}.{json,csv}` contain raw samples and environment metadata.
`results/comparison.{md,csv,json}` contain all 96 paired comparisons, with speedup
defined as dense median / sparse median (above 1 means sparse is faster).

The GPU tests compare scores with the independent einsum formula for both key
layouts and shapes; test exact top-k values, uniqueness, causality, negative
padding and multiple documents; check sparse outputs against an independent
PyTorch reference; verify graph replay; and compare dense versus sparse when
the selected set contains every causal token.

## Feishu report and labeled bar charts

A fresh full rerun is stored in `results/lark_rerun_20260929/`, with 192
measurements and 2,880 individual timing samples. This run uses the corrected
4/8 indexer KV heads. Its [comparison tables](results/lark_rerun_20260929/comparison.md)
and [CSV](results/lark_rerun_20260929/comparison.csv) are the data behind the
[Feishu report](https://my.feishu.cn/docx/UiWfdNSWvoFzCixy1wUc58kBnBb).

Recreate the 16 labeled bar charts (PNG and PDF), source bundle and Feishu XML:

```bash
python benchmarks/gqa_lightning_attention/plot_results.py \
  --results benchmarks/gqa_lightning_attention/results/lark_rerun_20260929
```

Each chart compares dense and sparse across all six batch sizes for one
configuration, phase and context. Every bar carries its median latency in
milliseconds. All y-axes start at zero; their ranges vary across charts and are
explicitly labeled as such. `charts/manifest.json` maps each bar to its exact
source value; `validation.json` records complete-grid, sample and label checks.

## Experiment 2: reduced-query, shared-key indexer

The alternative indexer uses 4Q/1KV for configuration 1 and 8Q/1KV for
configuration 2, with the main attention unchanged. This is a separate
experiment; defaults retain the original 48Q/4KV and 64Q/8KV indexers.

[Full three-way comparison and labeled charts](results/reduced_indexer/indexer_comparison.md)
([CSV](results/reduced_indexer/indexer_comparison.csv)). All 192 dense/reduced-sparse
measurements were rerun. Original sparse results are taken from the prior
`lark_rerun_20260929` run, so differences include cross-run variation.

Configuration 1 now beats dense prefill at every tested batch for 64k and 128k
contexts (1.11–1.21x); configuration 2 remains slower than dense prefill. Decode
changes are mixed. The reduced-indexer gate is a single nonnegative scalar per
query/group: a positive gate leaves score rankings unchanged; a zero gate makes
all causal tokens tie. With the random weights used here, prefill zero-gate
fractions are 51.15% and 50.54% respectively (`gate_diagnostics.json`). Exact
smaller-index tie-breaking selects early tokens for these groups, affecting
memory locality. These timings do not establish trained-model quality or isolate
head-count arithmetic savings from changed selection distributions.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/gqa_lightning_attention/benchmark.py \
  --configs 1 --indexer-q-heads kv --indexer-kv-heads 1 \
  --output benchmarks/gqa_lightning_attention/results/reduced_indexer/config1
CUDA_VISIBLE_DEVICES=1 python benchmarks/gqa_lightning_attention/benchmark.py \
  --configs 2 --indexer-q-heads kv --indexer-kv-heads 1 \
  --output benchmarks/gqa_lightning_attention/results/reduced_indexer/config2
python benchmarks/gqa_lightning_attention/compare_indexers.py \
  --baseline benchmarks/gqa_lightning_attention/results/lark_rerun_20260929 \
  --reduced benchmarks/gqa_lightning_attention/results/reduced_indexer
```

## Named experiments

`experiments.json` records the experiment identities and their raw result
locations without relabeling historical timing samples:

| Experiment | Main Q/KV, config 1; config 2 | Indexer Q/KV, config 1; config 2 |
|---|---|---|
| 1: original grouped indexer | 48/4; 64/8 | 48/4; 64/8 |
| 2: reduced-query shared-key indexer | 48/4; 64/8 | 4/1; 8/1 |
| 3: main MQA | 48/1; 64/1 | 4/1; 8/1 |

For experiment 3, only the main attention KV head count changes from experiment
2; the indexer head counts remain 4Q/1KV and 8Q/1KV. All indexer query heads now
contribute to one score and one selected token set shared by every main Q head.
The FlashInfer dense MQA planner requires a 1 GiB workspace at these head ratios;
it is preallocated outside timing. The backend and timing contract are unchanged.


Experiment 3 completed all 192 measurements and 94 correctness tests. Sparse
prefill beats the matched MQA dense baseline in all 48 comparisons (1.26–3.44x
for configuration 1; 1.55–4.32x for configuration 2). Dense MQA wins most decode
comparisons: sparse wins only at 128k/batch 32 for configuration 1, and at
64k/batch 32 plus 128k/batches 16 and 32 for configuration 2.

At 128k/batch 32, configuration 1 dense/sparse prefill is 19.060/5.649 ms and
decode is 0.571/0.483 ms. Configuration 2 prefill is 25.695/6.032 ms and decode
is 0.587/0.377 ms. These are one-layer synthetic timings; changing the main
KV sharing and selection structure requires separate model-quality validation.

[Experiment 3 results, comparisons with experiments 1/2, and labeled bar charts](results/experiment3_mqa/experiments_comparison.md)
([CSV](results/experiment3_mqa/experiments_comparison.csv)).

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/gqa_lightning_attention/benchmark.py \
  --configs 1 --experiment 3 \
  --output benchmarks/gqa_lightning_attention/results/experiment3_mqa/config1
CUDA_VISIBLE_DEVICES=1 python benchmarks/gqa_lightning_attention/benchmark.py \
  --configs 2 --experiment 3 \
  --output benchmarks/gqa_lightning_attention/results/experiment3_mqa/config2
python benchmarks/gqa_lightning_attention/summarize.py \
  benchmarks/gqa_lightning_attention/results/experiment3_mqa/config1.json \
  benchmarks/gqa_lightning_attention/results/experiment3_mqa/config2.json \
  --output benchmarks/gqa_lightning_attention/results/experiment3_mqa/comparison
python benchmarks/gqa_lightning_attention/plot_results.py \
  --results benchmarks/gqa_lightning_attention/results/experiment3_mqa
python benchmarks/gqa_lightning_attention/compare_experiments.py
```

`--experiment 1` and `--experiment 2` reproduce the other named configurations.
The experiment presets cannot be combined with individual head-count overrides.

## Consolidated Feishu report

The [Feishu report](https://my.feishu.cn/docx/UiWfdNSWvoFzCixy1wUc58kBnBb)
contains experiments 1, 2 and 3, with a shared shape/measurement summary, 48
labeled bar charts covering the complete grid, and consolidated CSV/ZIP evidence.
`results/all_experiments_doc/` locally holds the publication drafts, source-data
bundle, and document verification receipts; these are excluded from Git. Original experiment 1 chart/file resources
are retained; experiment 2 and 3 are separate sections. This document update
uses the completed measured runs and does not rerun the GPU benchmark.

## Two-stage block shortlist (2026-09-30)

`benchmark_two_stage.py` reruns all three head presets against **both** paged
FlashInfer FA2 and FA3. Reports choose the faster dense median at each point.
This updates the older FA2-only comparisons above; they are historical results.

- `Attention(cfg, sparse=True, two_stage=True)` enables the new selector;
  create the matching `PagedState(..., sparse=True, two_stage=True)` once.
- Stage 1 scores cached mean-pooled indexer keys for 128-token blocks and
  selects 64 blocks. Stage 2 scores their 8192 tokens and selects 2048 tokens
  per main KV group. Both stages use Triton scores and FlashInfer top-k.
- Pooling is FP32 accumulation with BF16 stored means. Existing prefix block
  summaries initialize outside timing; every forward recomputes the summaries
  of blocks containing newly written indexer keys, **inside timing**.
- The current block is forced into the shortlist. Its potentially future-bearing
  mean never influences other selected blocks; fine scores mask future tokens.
  All indices remain request-local and resolve through shuffled physical pages.
- Mean pooling is an approximate coarse selector: it can omit global top-k
  tokens. `recall_two_stage.py` reports sampled token-ID recall, separates zero
  gates, and does not claim model-quality equivalence. This is not a trained
  hierarchical indexer or a production SGLang backend registration.
- All latency runs enforce at least three warmups, then take 15 CUDA-event
  samples. Decode uses CUDA graphs; prefill is a final 2048-total-token chunk.
  Every timing includes projections, normalization, cache append, both selection
  stages, and final attention; allocation/planning/JIT/graph capture are excluded.
- The correctness suites pass 108 tests, including independent fine-score and
  top-k checks, pooled-cache checks, causal/request isolation, and graph replay.

Run one experiment (1, 2, or 3) with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/gqa_lightning_attention/benchmark_two_stage.py \
  --experiments 1 --configs 1,2 --contexts 16384,32768,65536,131072 \
  --batches 1,2,4,8,16,32 --phases prefill,decode \
  --variants dense_fa2,dense_fa3,sparse,two_stage --candidate-blocks 64 \
  --prefill-budget 2048 --warmup 3 --iterations 15
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/gqa_lightning_attention/recall_two_stage.py --experiments 1
.venv/bin/python benchmarks/gqa_lightning_attention/report_two_stage.py --experiment 1
```

Each completed experiment has raw samples and metadata, paired CSV/JSON,
Chinese report, copyable `reproduce.sh`, recall diagnostics and a 16-chart
manifest under `results/two_stage/experimentN/`. Plot generation labels every
bar. `publish_two_stage.py --experiment N --publish` appends the report, charts
and evidence to Feishu and verifies that prior resources were preserved.
DSA/MSA use [the native adapter benchmark](../dsa_m3_attention/README.md).
