# Final study: dense vs token, block and two-stage sparse GQA attention

One synthetic attention layer on an H200 (BF16, TP=1, real paged caches, no model
server). It asks when sparse attention beats the best available dense kernel, for
four main-attention shapes, two indexer designs and three selection methods,
across context length and batch size. Prefill and decode are measured separately,
and every point carries a stage-level breakdown.

**Headline:** sparse attention loses at 16k and breaks even around 64k. Beyond that
the winner depends on the phase and the indexer: block selection with a *reduced*
indexer is best up to about 128k, and two-stage selection wins at 512k and above.
Token-level selection never wins prefill. Details are under [Results](#results).

## Configurations

**Main attention** (head dim 128, hidden size 8192):

| Name | Q heads | KV heads |
|---|---:|---:|
| `48q4kv` | 48 | 4 |
| `64q8kv` | 64 | 8 |
| `80q8kv` | 80 | 8 |
| `64q4kv` | 64 | 4 |

**Indexer variants** (all four are run for every main config and method):

| Type | Index Q heads | Index KV heads | Head dim |
|---|---|---|---|
| `identical` | same as main Q heads | same as main KV heads | 64 or 128 |
| `reduced` | main KV heads (one per KV group) | 1 | 64 or 128 |

Each KV group scores with its own subset of index Q heads and selects its own
tokens, as in the original Lightning pseudocode. The score is
`sum_j w[t,g,j] * relu(q_I[t,g,j] . k_I[s,g])` with gates `w = relu(linear(x))`,
with no normalization, RoPE or scaling. All three sparse methods use this same
score, so they differ only in how they select.

## Methods

| Method | Selection | Attended tokens |
|---|---|---:|
| `dense` | none; FlashInfer paged FA2 and SGLang FA3 (`sgl_kernel`) are both measured, the faster is the baseline | whole causal context |
| `token` | exact top-2048 over every token score (FlashInfer top-k) | 2048 |
| `block` | MSA-style: max token score per 128-token block, top-16 blocks, query's own block always kept | 16 x 128 = 2048 |
| `two_stage` | mean-pool K per 128-token block, score the L/128 pooled keys, keep the top 128 blocks (own block forced), score their 16,384 tokens, exact top-2048 | 2048 |

All sparse methods feed the same token-sparse attention kernel
(`gqa_token_sparse_attn`), so attention cost differs only through the selected set.

Implementation notes:

- The block selector is **our own** max-pooling of the Lightning score, fused into
  the score kernel (`block_max=True`), not the native MSA kernel. The native kernel
  uses a plain dot product with no gate and supports only one index KV head at
  dim 128, so it cannot cover all variants.
- Pooled block means are FP32-accumulated and stored in BF16. Summaries of blocks
  that receive new tokens are recomputed inside the timed region.
- At 16k context, 128 blocks cover the whole context, so two-stage does extra work
  with no benefit.
- The score kernel's query tile and warp count were tuned per head layout at 128k
  (`score_launch` in `attention.py`).

## Measurement contract

- **Grid:** batch 1, 8, 32; context 16k, 64k, 128k, 512k, 1M per request (the
  context includes the new tokens).
- **Prefill:** the final scheduled chunk of **2,048 total new tokens across the
  batch** (2048, 256 or 64 per request), eager. This is not full-prompt latency
  or TTFT.
- **Decode:** one new token per request, CUDA-graph replay.
- **Timed region:** Q/K/V projections, per-head RMSNorm, cache writes and
  attention; sparse methods add index projection, index-cache write, scoring,
  selection and index expansion. Excluded: output projection, RoPE, MLP,
  communication, scheduler, planning, allocation, JIT/autotuning and graph capture.
- **Timing:** 3 warmups, then 15 CUDA-event samples (median, min, p90 and every
  raw sample are saved). A 64 MiB L2 flush precedes each sample outside the timer.
  Independently allocated request caches with shuffled 128-token pages; synthetic
  random weights and historical cache.
- **Speedup** = best dense median / sparse median, where best dense is the faster
  of FlashInfer FA2 and SGLang FA3 at that point. Above 1 means sparse is faster.
- **Stage breakdown:** one extra eager pass with CUDA events around each stage.
  A spin kernel is enqueued first so the CPU is ahead of the GPU, which makes
  each window pure GPU time. Stage sums match eager wall time to within about 1%
  in prefill. In decode the sums run about 0.04 ms above the CUDA-graph wall
  time. Each breakdown is a single execution, so small stages carry noise.
  `top_kernels_ms` lists the heaviest individual kernels from a profiler pass.
- **OOM:** 24 points did not fit on one 141 GB H200: the 8-KV configs with an
  identical indexer at 1M context and batch 32, for each of the three sparse
  methods and both indexer dims. They are marked `status: "oom"` and omitted from
  plots.

## Results

1,680 measurements (240 dense, 1,440 sparse of which 24 OOM). Full tables:
[`summary.md`](summary.md); every pair: [`comparison.csv`](comparison.csv);
everything raw: [`final_study_results.json`](final_study_results.json).

### Sparse wins over the best dense baseline

Counts of grid points (out of 60, or 58 for identical-indexer rows with OOMs).

| Method | Indexer | Dim | Prefill | Decode |
|---|---|---:|---:|---:|
| token | identical | 64 | 0/58 | 39/58 |
| token | identical | 128 | 0/58 | 29/58 |
| token | reduced | 64 | 0/60 | 35/60 |
| token | reduced | 128 | 0/60 | 32/60 |
| block | identical | 64 | 9/58 | 42/58 |
| block | identical | 128 | 0/58 | 36/58 |
| block | reduced | 64 | 48/60 | 46/60 |
| block | reduced | 128 | 45/60 | 42/60 |
| two-stage | identical | 64 | 31/58 | 40/58 |
| two-stage | identical | 128 | 25/58 | 38/58 |
| two-stage | reduced | 64 | 37/60 | 42/60 |
| two-stage | reduced | 128 | 33/60 | 42/60 |

### Best sparse speedup at batch 8

The best variant per point (any method and indexer).

| Main | Phase | 16k | 64k | 128k | 512k | 1M |
|---|---|---:|---:|---:|---:|---:|
| `48q4kv` | prefill | 0.49 | 1.49 | 2.34 | 5.63 | 10.69 |
| `64q8kv` | prefill | 0.36 | 1.04 | 1.70 | 3.93 | 7.54 |
| `80q8kv` | prefill | 0.44 | 1.32 | 2.15 | 4.81 | 9.20 |
| `64q4kv` | prefill | 0.65 | 2.02 | 3.10 | 7.43 | 14.19 |
| `48q4kv` | decode | 0.64 | 1.27 | 2.03 | 7.05 | 13.48 |
| `64q8kv` | decode | 0.86 | 1.86 | 3.37 | 12.21 | 23.34 |
| `80q8kv` | decode | 0.94 | 1.82 | 3.26 | 11.64 | 21.78 |
| `64q4kv` | decode | 0.66 | 1.26 | 2.02 | 6.88 | 13.17 |

At batch 32 and 1M context, decode reaches 37.9x, 53.7x, 52.3x and 37.1x for the
four main configs (53.7x is for `64q8kv`; its identical-indexer point at 1M is OOM).

### Findings

1. **Short context loses.** At 16k the best prefill speedup is 0.34-0.66x for every
   main config and batch; break-even is around 64k.
2. **Token-level selection never wins prefill.** Exact top-2048 over every token
   costs more than the attention it saves. In decode it wins only 29-39 of 58-60
   points.
3. **Block + reduced indexer is the best prefill method up to about 128k**
   (2.34x at 128k for `48q4kv`). Its scoring still reads every key, so it levels
   off at 5.4x at 1M.
4. **Two-stage wins at 512k and above**, because only its coarse stage sees the
   whole context. At 128k, `48q4kv`, batch 8 block takes 4.6 ms and two-stage 7.4 ms
   (reduced, dim 64); at 512k, `64q8kv`, two-stage takes 14.8 ms against block's 18.5 ms.
5. **The identical indexer is what makes token and block selection lose.** Scoring
   all index heads dominates: at 128k, `48q4kv`, batch 8 the block score takes
   10.8 ms with the identical indexer and 1.6 ms with the reduced one. Over all
   paired points the reduced indexer is faster in 336 of 348 prefill and 287 of
   348 decode comparisons (median time ratio 1.34x and 1.09x).
6. **Dim 64 beats dim 128.** It is faster in all 354 paired prefill points (dim 128
   takes 1.04-1.68x as long, median 1.29x) and in 350 of 354 decode points.
7. **Decode benefits most**, because dense decode grows linearly with context.
8. **4-KV configs speed up more than 8-KV configs** in prefill at every context
   (for example 3.10x vs 1.70x at 128k, batch 8, `64q4kv` vs `64q8kv`).

Example stage breakdown (`48q4kv`, prefill, 128k, batch 8, dim 64, GPU ms):

| Method / indexer | Wall | Score | Select | Fine score | Sparse attention |
|---|---:|---:|---:|---:|---:|
| token / identical | 21.08 | 11.87 | 5.13 | - | 3.10 |
| token / reduced | 16.03 | 2.12 | 10.22 | - | 2.88 |
| block / identical | 13.99 | 10.85 | 0.36 | - | 1.88 |
| block / reduced | 4.62 | 1.58 | 0.46 | - | 1.78 |
| two-stage / reduced | 7.36 | 0.53 (coarse) | 1.71 | 2.09 | 1.88 |

With a reduced indexer, token selection's time moves from scoring into top-k.
Random weights give about half of the gates a value of exactly zero, so many rows
are fully tied, which makes the exact top-k slow (see Caveats).

## Caveats

- **No quality measurement.** Weights and caches are random. In an earlier
  diagnostic, sampled top-k recall for the two-stage selector at 128k was only about
  9%, so its latency advantage does not establish that quality is preserved. The
  same applies to block selection and the reduced indexer.
- **Zero gates inflate token-level top-k.** In a micro-benchmark, rows of tied
  scores roughly doubled FlashInfer's top-k time. Token selection (and the reduced
  indexer's top-k) are therefore probably penalised relative to a trained model.
  A positive-gate ablation has not been run.
- **Dense baseline.** FA2 and FA3 are both kept in the raw data. SGLang FA3 was
  faster than FlashInfer FA2 at all 60 prefill points and 31 of 60 decode points.
  Earlier FA2-only results in this directory overstate prefill speedups by about
  1.8-2x.
- **One layer.** Results exclude the output projection, MLP, communication and
  scheduling, and depend on backend choice, page size and the synthetic setup.
- **Prefill is a final 2048-token chunk**, not a full prompt.
- **Block selection is our own max-pooled Lightning score**, not the native MSA
  kernel, so MSA results from [`../../../dsa_m3_attention`](../../../dsa_m3_attention/README.md)
  are not directly comparable.

## Reproduce

Run from `benchmarks/gqa_lightning_attention` with an environment that has this
checkout, PyTorch, Triton, FlashInfer and `sgl_kernel`. Plotting needs matplotlib
and the tests need pytest.

```bash
# correctness: 134 new tests (plus the 108 earlier ones)
CUDA_VISIBLE_DEVICES=0 python -m pytest test_final.py test_attention.py test_two_stage.py -q

# the full sweep: stages run in order dense, token, block, two_stage
results/final_study/run_stages.sh 0 0,2   # GPU 0: 48q4kv and 80q8kv
results/final_study/run_stages.sh 1 1,3   # GPU 1: 64q8kv and 64q4kv

# join, summarise and plot
python summarize_final_study.py
python plot_final_study.py
```

Any subset can be run directly, and an interrupted run resumes from its JSON:

```bash
CUDA_VISIBLE_DEVICES=0 python run_final_study.py --methods block --mains 0 \
  --contexts 131072 --batches 8 --output results/final_study/block_demo.json
```

## Files

| File | Contents |
|---|---|
| `final_study_results.json` | all measurements with speedups, stage times and raw samples |
| `comparison.csv`, `summary.md` | paired speedups; wins, per-config grids and stage breakdowns |
| `plots/speedup_<main>_<phase>.png` | speedup vs context, per indexer type and batch |
| `{dense,token,block,two_stage}_gpu{0,1}.{json,log}` | raw per-stage run outputs |
| `run_stages.sh` | the sweep driver used for the results above |

Code (one directory up): `run_final_study.py` (sweep), `block_select.py` (block
selector), `two_stage.py`, `attention.py` and `kernels.py` (shared layer and
kernels, generalized to head dim 64/128), `stages.py` (stage windows),
`study_config.py` (configurations), `test_final.py` (correctness).
