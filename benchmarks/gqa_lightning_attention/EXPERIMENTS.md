# Reproducing the final sparse-attention study

This is a step-by-step guide to re-running the dense vs token / block / two-stage
sparse GQA attention study and regenerating its tables and plots. What the study
measures, and what it found, is in
[`results/final_study/README.md`](results/final_study/README.md).

The whole sweep takes about **15 minutes on two H200s** (it took 14 minutes on the
original run: dense 2 min, token 6, block 3, two-stage 3). Most of that is the 512k
and 1M points.

## 1. Requirements

| Need | Notes |
|---|---|
| 2x NVIDIA H200 (141 GB) | One GPU also works, it just takes about twice as long. Smaller GPUs will OOM at long contexts; OOM points are recorded, not fatal. |
| Idle GPUs | Contention inflates launch latency and ruins decode timings. Check `nvidia-smi` first. |
| The repo checkout with its venv at `<repo>/.venv` | `run_stages.sh` calls `../../.venv/bin/python` relative to this directory. |
| PyTorch, Triton, FlashInfer, `sgl_kernel` | The results were produced with torch 2.11.0+cu130, Triton 3.6.0, FlashInfer 0.6.15.post1, sgl-kernel 0.4.5, driver 595.71. |
| `pytest` (tests) and `matplotlib` (plots) | Not in the repo venv. Install them with `uv pip install --python .venv/bin/python pytest matplotlib`, or into a separate `--target` directory and put it on `PYTHONPATH`. |

Everything below runs from `benchmarks/gqa_lightning_attention`:

```bash
cd benchmarks/gqa_lightning_attention
```

## 2. Check correctness first

```bash
CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python -m pytest \
  test_final.py test_attention.py test_two_stage.py -q
```

Expect 242 passed in about a minute: 134 in `test_final.py` (head dim 64/128, block-max
scores, block and two-stage selection against PyTorch references, end-to-end attention
against a gather-and-softmax reference) plus the 108 earlier tests. Do not benchmark
if these fail.

## 3. Run the sweep

```bash
rm -f results/final_study/*_gpu*.json results/final_study/progress_gpu*.txt   # clean rerun
results/final_study/run_stages.sh 0 0,2 &   # GPU 0: main configs 48q4kv and 80q8kv
results/final_study/run_stages.sh 1 1,3 &   # GPU 1: main configs 64q8kv and 64q4kv
wait
```

Each GPU runs four stages in order (`dense`, `token`, `block`, `two_stage`), one
Python process per stage, writing `results/final_study/<stage>_gpu<N>.json` and
`.log`. `progress_gpu<N>.txt` gains a line when each stage ends, so you can follow
progress with `cat results/final_study/progress_gpu*.txt` or `tail -f` a log.

The main-config indices are 0 = `48q4kv`, 1 = `64q8kv`, 2 = `80q8kv`, 3 = `64q4kv`.
The split above balances the heavier 8-KV configs across the two GPUs. With one GPU,
run the script twice in sequence, or pass all indices (`run_stages.sh 0 0,1,2,3`).

### Expected output

| File | Rows | Of which OOM |
|---|---:|---:|
| `dense_gpu{0,1}.json` | 120 each | 0 |
| `token_gpu{0,1}.json` | 240 each | 4 each |
| `block_gpu{0,1}.json` | 240 each | 4 each |
| `two_stage_gpu{0,1}.json` | 240 each | 4 each |

The 24 OOM rows are the 8-KV configs (`64q8kv`, `80q8kv`) with an `identical` indexer
at 1M context and batch 32, for each sparse method and indexer dim. Verify with:

```bash
../../.venv/bin/python - <<'EOF'
import collections, json, pathlib
for path in sorted(pathlib.Path("results/final_study").glob("*_gpu*.json")):
    rows = json.loads(path.read_text())["rows"]
    print(path.name, len(rows), dict(collections.Counter(r["status"] for r in rows)))
EOF
```

## 4. Summarise and plot

```bash
../../.venv/bin/python summarize_final_study.py   # comparison.csv, summary.md, final_study_results.json
../../.venv/bin/python plot_final_study.py        # plots/speedup_<main>_<phase>.png (8 figures)
```

(With matplotlib installed to a separate directory, prefix the plot command with
`PYTHONPATH=<that directory>`.) Both scripts read every `*_gpu*.json` in
`results/final_study/`, so they also work on a partial run.

## 5. Running a subset

`run_final_study.py` can be called directly. All flags:

| Flag | Default | Meaning |
|---|---|---|
| `--methods` | `dense,token,block,two_stage` | comma-separated subset |
| `--mains` | `0,1,2,3` | main-config indices (see above) |
| `--contexts` | `16384,65536,131072,524288,1048576` | context lengths per request |
| `--batches` | `1,8,32` | batch sizes |
| `--phases` | `prefill,decode` | which phases to run |
| `--prefill-budget` | `2048` | total new tokens per prefill chunk, split across the batch |
| `--candidate-blocks` | `128` | two-stage shortlist size in 128-token blocks |
| `--warmup` / `--iterations` | `3` / `15` | warmups and timed samples |
| `--seed` | `42` | weights and inputs |
| `--output` | required | JSON path |

A quick smoke test of every method on one config (a couple of minutes):

```bash
CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python run_final_study.py \
  --mains 0 --contexts 16384,65536 --batches 8 --iterations 5 \
  --output /tmp/smoke_final.json
```

Notes on subsets and reruns:

- A run **resumes**: re-running a command with the same `--output` skips points that
  already finished with `status: "ok"`. OOM rows are retried and appended again, so
  delete the JSON for a clean rerun rather than resuming a finished one.
- `--mains`, `--contexts` and `--batches` change only what is run, not how. Prefill
  needs `--prefill-budget / batch` to be a multiple of the score tile (4 or 16 for
  the grid above, so any power-of-two batch up to 32 works).
- To run on different hardware, expect different absolute numbers and crossovers.
  The score kernel's tile and warp choices (`score_launch` in `attention.py`) were
  tuned for H200 at 128k.

## 6. Interpreting the output

Each row of a `*_gpu*.json` has:

- identifiers: `main`, `phase`, `context`, `batch`, `method`, `indexer`
  (`identical`, `reduced`, or `none` for dense), `dim`, plus `dense_backend` for
  dense rows (`flashinfer_fa2` or `sgl_fa3`);
- `status` (`ok` or `oom`), and for OK rows `median_ms`, `min_ms`, `p90_ms` and all
  15 `samples_ms`;
- `stage_ms` (GPU time per stage), `stage_sum_ms`, `top_kernels_ms`, and for
  prefill `launch_gap_ms` (wall time minus stage sum);
- `peak_allocated_gib`.

Speedup is the faster of the two dense backends at the same point, divided by the
sparse median. `final_study_results.json` joins this for you and adds `speedup`,
`dense_best_ms` and `dense_best_backend` to each sparse row.

## 7. Things that can go wrong

- **Noisy or inflated decode times:** another process is using the GPU. Re-run on an
  idle device.
- **`ModuleNotFoundError: pytest` or `matplotlib`:** see section 1.
- **More OOM rows than expected:** other processes hold GPU memory, or the GPU has
  less than 141 GB. The long-context points need about 100 GB or more.
- **First points are slow:** Triton compiles and autotunes the token-attention kernel
  on first use of each shape. This happens inside the untimed warmups.
- **`sgl_kernel` import errors:** the dense SGLang FA3 baseline needs it. Install
  the version matching your torch and CUDA, or drop the `sgl_fa3` backend and
  accept FA2-only prefill baselines, which overstate prefill speedups by about
  1.8-2x.
- **Different numbers on a rerun:** expect a few percent of run-to-run variation.
  Weights and caches are random, but seeded.

## 8. Earlier experiments in this directory

The runs under `results/lark_rerun_20260929`, `results/reduced_indexer`,
`results/experiment3_mqa`, `results/two_stage` and `results/profile_stages` are
earlier studies with their own reproduction commands in the main
[`README.md`](README.md). They use FlashInfer FA2 (or FA2/FA3 via FlashInfer) as
the dense baseline, a different grid, and, for some, a 64-block two-stage
shortlist, so they are not directly comparable with this study.
