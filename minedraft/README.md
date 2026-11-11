# MineDraft in SGLang: experiments

Experiments for running MineDraft (batch-parallel speculative decoding, arXiv 2603.18016, reference code `electron-shaders/MineDraft@272c855`) inside SGLang, with the target model on one GPU and an EAGLE-3 draft on another. Hardware: 2x H200 (NVLink).

The code changes live in this branch under `python/sglang/srt/`:
- `speculative/eagle_worker_v2.py`: draft worker on its own GPU, the device boundary for every draft entry point, cohort drafting, CUDA graphs on the draft GPU.
- `speculative/minedraft_device.py`: tensor moves across devices, attribute write-back, slot-allocator proxy.
- `managers/scheduler.py`: two alternating cohorts (normal and overlap event loops).

Turn it on with environment variables:

| Variable | Meaning |
|---|---|
| `SGLANG_MINEDRAFT_DRAFT_GPU=1` | build the EAGLE draft on this GPU index (launch with `CUDA_VISIBLE_DEVICES=0,1`) |
| `SGLANG_MINEDRAFT_COHORTS=0` | cross-GPU serial SD (no cohort alternation); default 1 = MineDraft |
| `SGLANG_MINEDRAFT_DRAFT_DECODE_GRAPHS=1` | capture the draft's decode CUDA graphs on the draft GPU |
| `SGLANG_MINEDRAFT_DRAFT_PREFILL_GRAPHS=1` | capture the draft's prefill CUDA graph on the draft GPU |
| `SGLANG_MINEDRAFT_TIMING=1` | log host time per MineDraft phase |
| `SGLANG_MINEDRAFT_PREDRAFT=1` | experimental pre-draft in the overlap loop (gives wrong outputs; keep off) |

## Layout

- `docs/`: `REPORT.md` (colocated baselines, transfer cost, step model), `DESIGN-minedraft-in-sglang.md` (port design, status and in-SGLang results in sections 6-9), `summary.md`, figures.
- Scripts: `srv.sh` (launch a server), `bench_closed_loop.py` (closed-loop client), `perf*.sh` / `det*.sh` (the measured runs), `run_baselines.sh` + `analyze.py` + `make_figures.py` + `xfer_bench.py` + `mdtiming/` (colocated baselines, phase timing and the step model), `minedraft_sglang.py` (early client-side port with a standalone draft).
- `data/`: raw results. In-SGLang MineDraft runs are in `data/insglang/` and `data/insglang2/` (`g_overlap_*`, `g_nooverlap_*`, `k7_overlap_*`, `det2_*`).

## Main result (EAGLE-3 (3,1,4), Qwen3-8B, CUDA graphs on, tok/s)

| scheduler | C | colocated 1 GPU | 2 colocated replicas | cross-GPU serial SD | cross-GPU MineDraft |
|---|---|---|---|---|---|
| overlap ON | 32 | 7081 | 8974 | 5468 | 3488 |
| overlap ON | 64 | 10375 | 14162 | 9092 | 5940 |
| overlap ON | 128 | 14331 | 20749 | 12489 | 9144 |

Greedy outputs of cross-GPU MineDraft equal AR on 32/32 prompts under `--enable-deterministic-inference`. See `docs/DESIGN-minedraft-in-sglang.md` for the overlap-off table, the (7,1,8) draft and the analysis.
