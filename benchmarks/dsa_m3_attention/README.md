# DSA and MiniMax M3: matched dense versus sparse attention

Synthetic single-layer latency investigation on H200. This suite uses the native
SGLang sparse pipelines inside explicit projection adapters. It does not load
checkpoint weights or launch an inference server.

## Measurement contract

- Context per request: 16k, 32k, 64k, 128k; batch: 1, 2, 4, 8, 16, 32.
- Prefill: final scheduled chunk, **2,048 new tokens total across the batch**,
  eager execution. This is not full-prompt prefill or TTFT.
- Decode: one token per request, CUDA graph replay.
- Warm up 3 times; retain all 15 CUDA-event samples; report median/min/p90.
  Flush 64 MiB L2 outside each measured interval.
- Include main projections, normalization, cache writes, native indexing,
  selection, and attention. Exclude output projection, MLP, communication,
  scheduler, cache allocation, planning, compilation, and graph capture.
- BF16 main weights/activations/cache, TP=1, shuffled physical cache pages.
  Synthetic random weights and historical caches; no model-quality claim.
- Run dense FA2, dense FA3, and sparse at every point. The primary speedup is
  **min(dense FA2 median, dense FA3 median) / sparse median**. Keep both dense
  measurements in the raw results. Values above 1 favor sparse.

## Architectures and adaptations

| Parameter | DSA | MiniMax M3 |
|---|---|---|
| Main query heads, configs 1/2 | 48 / 64 | 48 / 64 |
| Main KV heads, configs 1/2 | 1 / 1 (MLA) | 4 / 8 (GQA) |
| Hidden size | 8192 | 8192 |
| Main attention representation | 512 latent + 64 RoPE; value dim 512 | head dim 128 |
| Index query heads | 64 | 4 / 8, one per main KV group |
| Index KV heads | 1 | 1 |
| Index head dim | 128 | 128 |
| Sparse selection | exact top-2048 individual tokens | 16 blocks x 128 tokens; includes current local block |
| Index cache | FP8 with per-token scale | BF16 |
| Cache page size | 64 (native DSA requirement) | 128 |
| Sparse attention | FlashMLA BF16 | SGLang paged Triton block-sparse |
| Dense attention | FlashInfer MLA FA2 / FA3 | FlashInfer paged FA2 / FA3 |

DSA calls the **complete production `Indexer`**, including query/key/gate
projections, key LayerNorm, RoPE, FP8 quantization/store, DeepGEMM scoring,
native top-k and physical-index transformation. Main queries use a rank-1536
projection, 128 non-positional + 64 RoPE channels, and absorption into a
512-channel latent cache. The value up-projection and final output projection
are outside this benchmark. Hopper FlashMLA requires 64-head tiles; 48 logical
heads are zero-padded to 64, and this padding cost is included. This directly
invokes the kernel: the stock SGLang backend does not accept the non-divisor
48-head padding case. It is an explicit benchmark adaptation.

M3 uses the previous experiments' Q/K/V projections and per-head RMSNorm,
plus RMS-normalized 128-dimensional index Q/K projections. Its native M3
pipeline performs causal token scoring, max pooling into blocks, block top-k,
and paged sparse GQA attention. Index values and attention sinks are disabled.
Main/index RoPE and the model's learned Gemma norm gains are omitted in this
standardized adapter. This is **M3's native sparse pipeline**, not the complete
`MiniMaxM3Attention` model layer. Both dense and sparse use identical main
weights, cache initialization, and inputs at each point.

These architectural differences are necessary to use the native optimized
kernels. Compare sparse against its own matched dense baseline; do not read
DSA-versus-M3 absolute latency as a same-model comparison. Likewise, compare
with experiments 1–3 only after accounting for changed dimensions, kernels,
indexer semantics, and dense-backend choice.

## Reproduce

Use the repository environment with PyTorch, FlashInfer, SGLang/sgl-kernel,
DeepGEMM, Triton, matplotlib, and pytest installed. DSA initializes a single-rank
NCCL process group; `BENCH_PORT` selects its localhost rendezvous port.

```bash
CUDA_VISIBLE_DEVICES=0 BENCH_PORT=29670 .venv/bin/python -m pytest \
  benchmarks/dsa_m3_attention/test_implementations.py -q
CUDA_VISIBLE_DEVICES=0 BENCH_PORT=29671 .venv/bin/python \
  benchmarks/dsa_m3_attention/benchmark.py --architecture dsa \
  --output benchmarks/dsa_m3_attention/results/dsa.json
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/dsa_m3_attention/benchmark.py --architecture m3 \
  --output benchmarks/dsa_m3_attention/results/m3.json
```

Each completed measurement is saved immediately. `--resume` skips completed
points. `--stages` measures projection/cache and indexer/attention components
separately; these independently flushed measurements are diagnostic and are
not additive. For M3, `sparse_pipeline` includes score, select, and attention.

Primary references:
[DSA release](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp),
[FlashMLA](https://github.com/deepseek-ai/FlashMLA),
[M3 configuration](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json).

## Results

[Chinese report](results/report_zh.md), [complete comparison CSV](results/comparison.csv),
and [summary/crossover data](results/summary.json) cover all 576 measurements
and 8,640 primary timing samples. All 12 adapter correctness tests passed.

- DSA: sparse wins 48/48 prefill and 20/48 decode comparisons.
- MSA (M3): sparse wins 38/48 prefill and 42/48 decode comparisons.
- At 128k, MSA exceeds 1.10x in both phases at every tested batch for both
  configurations; DSA does so at batches 4, 8, 16, and 32.
- Neither implementation wins both phases across the entire grid.

[reproduce.sh](reproduce.sh) contains the full commands, including warmups,
component timings, profiler diagnostics, validation, and graph generation.
The compiler was NVCC 12.8; DeepGEMM recommends 12.9+ for best performance.
Results apply to these synthetic BF16-main-cache adapters, not arbitrary models.
