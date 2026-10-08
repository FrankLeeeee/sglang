# Resume checkpoint — 2026-10-08

User requested saving the current results because the session is ending. Work is
paused. All code and raw evidence remain in `/data/frankleeeee/home/sglang`.
At checkpoint creation, changes were uncommitted; no PR or deployment was created.
The user subsequently requested a branch and push; see Git history for publication.

## Completed

- Implemented opt-in `fine_serial` / `fine_overlap` EAGLE3 continuations for
  Llama and GPT-OSS, using the same two H200 GPUs, TP=2 for both models.
- Eager mode yields between draft steps inside target layer continuations and
  overlaps draft extension A with verification B.
- Full graph mode partitions the target captured DAG into eight ordered node
  groups and launches draft B after group two. It retains the extend-A / verify-B
  fence: the unfenced version failed token checks. FA3 draft extension is eager.
- Fixed pooled position-buffer aliasing, isolated graph logits storage, restored
  per-forward flags around nested draft calls, and used private draft NCCL
  communicators, capture streams and graph pools. Default mode stays `off`.
- Deterministic eager AND safe graph matrices pass exact token identity against
  original for both pairs, every repeat, batches 5 and 16, 64 output tokens.
- GPU fork/join graph-cut replay equivalence passed. CPU tests: four ownership /
  split-merge regressions plus 46 existing input-registry / graph-backend tests.
- Ruff selected checks, compilation and `git diff --check` pass.
- Normal-kernel full-graph benchmark completed for both pairs, batches 16 and 32,
  128 output tokens, three timed repeats after two warmups; profiling afterward.
- Eager normal benchmark completed for Llama and GPT-OSS original. GPT-OSS fine
  serial was interrupted during its workload; fine overlap has not run yet.

## Main findings

Actual GEMM/communication overlap exists in both directions in graph traces.
Across six profiled cycles on rank 0 at batch 16, summed bidirectional GEMM/comm
intersections are 1.602 ms for Llama and 1.378 ms for GPT-OSS. Original and fine
serial controls have zero. This is concurrency evidence, not time saved.

Normal graph throughput (tokens/s):

| Pair | Batch | Original | Fine serial | Fine overlap |
|---|---:|---:|---:|---:|
| Llama | 16 | 3405.710 | 2062.640 | 2021.640 |
| Llama | 32 | 5827.630 | 3716.625 | 3581.429 |
| GPT-OSS | 16 | 1975.276 | 1287.093 | 1301.402 |
| GPT-OSS | 32 | 3346.793 | 2159.830 | 2184.724 |

Normal eager batch 16: Llama original 663.545, fine serial 320.691, fine overlap
324.929. GPT-OSS original 238.168; remaining fine modes pending.

No measured full-batch throughput gain. Normal graph overlap/control is -2.0 to
-3.6% for Llama and +1.1% for GPT-OSS. Splitting the batch costs more than the
small amount of hidden work. Normal-kernel outputs vary with batch shapes; even
original Llama repetitions differ. Exact correctness is separately established
under batch-invariant deterministic execution. Do not describe normal outputs as
all identical or infer a statistically significant gain from three trials.

## Important files

- `benchmark/spec_overlap/FINE_REPORT.md`: current report, tables, time model,
  ownership diagnosis and limitations. Its eager GPT-OSS section is unfinished.
- `benchmark/spec_overlap/README.md`: scheduling and reproduction commands.
- `benchmark/spec_overlap/run_matrix.py`: owned server lifecycle, source hashes,
  token gates and graph-chunk trace gate.
- `benchmark/spec_overlap/analyze_results.py`: GPU interval analysis; accounts
  for graph internal streams using target-chunk / captured verify annotations.
- `python/sglang/srt/speculative/microbatch_overlap.py`: pipeline implementation.
- `python/sglang/srt/speculative/graph_chunks.py`: raw CUDA DAG partitioning.
- `test/manual/spec/eagle/test_microbatch_overlap.py`: four CPU regressions.
- `benchmark/spec_overlap/check_graph_chunks.py`: GPU DAG equivalence test.

Evidence directories (ignored by Git, preserved on disk):

- `results/fine-correctness`: successful eager correctness; `analysis.json`.
- `results/fine-graphs-safe`: successful fenced graph correctness;
  `analysis.json`, `llama-triage.md`, `gpt-oss-triage.md` (three-table profiles).
- `results/fine-normal-graphs`: completed normal graph performance, outputs,
  per-launch sources/arguments, per-rank traces; final analysis not generated yet.
- `results/fine-normal-eager`: partial normal eager matrix; see `matrix.log`.
- `results/fine-graphs-correctness`, `fine-graphs-private-logits`: FAILED unfenced
  graph versions. Exclude their timings from valid performance conclusions.
- `results/graph-fence`: fenced Llama batch-5 diagnostic passed.
- `results/activation-*`, `position-writes`: position aliasing diagnostic evidence.
- `results/graph-storage-audit`: interrupted diagnostic, not a valid benchmark.
- `results/checkpoint/tracked-changes.patch`: tracked diff against base HEAD.
- `results/checkpoint/new-files.tar.gz`: snapshot of new untracked source/docs.
- `results/checkpoint/manifest.json`: hashes of saved report/workload artifacts.

Base HEAD: `ba77d30256bfb1d1e0d53cd877c87f0b2e279ef1`.

## Resume next

1. Benchmark work was stopped with SIGINT to matrix PID 2152605. Its finally
   block cleaned up the owned server; the process check found no remaining
   run_matrix, run_workload or SGLang launch_server processes. Recheck GPU
   availability before resuming and do not kill unrelated processes.
2. Finish BOTH GPT-OSS eager fine modes, rerunning fine serial because its last
   attempt was interrupted:

```bash
python benchmark/spec_overlap/run_matrix.py --models gpt-oss \
  --modes fine_serial fine_overlap --batch-sizes 16 --repeats 3 \
  --output-len 128 --results benchmark/spec_overlap/results/fine-normal-eager \
  > benchmark/spec_overlap/results/fine-normal-eager/resume.log 2>&1
```

3. Analyze completed normal matrices AFTER timed GPU work finishes:

```bash
python benchmark/spec_overlap/analyze_results.py \
  --results benchmark/spec_overlap/results/fine-normal-graphs
python benchmark/spec_overlap/analyze_results.py \
  --results benchmark/spec_overlap/results/fine-normal-eager
```

The analyzer can take a minute because attribution scans many kernel spans.
4. Update FINE_REPORT with final eager GPT-OSS rates and normal-profile results.
   Preserve the negative throughput result and graph extend/verify fence.
5. Verify formatting / diff and report completion. Do not repeat passed GPU or
   CPU tests without a new change or unresolved issue. No commit/PR requested.

## Environment and scope

Two H200 GPUs with NV18; PyTorch 2.14.1+cu130; driver 595.71.05; nvcc 12.8.
All four checkpoints are locally cached. Use FA3, greedy EAGLE3 topk=1, steps=3,
draft tokens=4, context=2048, static memory=.55, PP=DP=CP=EP=1, disabled custom
all-reduce, FlashInfer AR fusion and CPU overlap scheduling. Graph buckets
1,2,4,8,16,32. The harness sets `SGLANG_DEEPGEMM_PDL=0`; deterministic runs also
set `SGLANG_ENABLE_JIT_DEEPGEMM=0` and `--enable-deterministic-inference`.
Custom all-reduce cannot compile with the installed tvm_ffi; do not re-enable it.
Normal serving and deterministic performance are different kernel conditions.
EP/all-to-all is not implemented or measured. Prompts are short repeated raw
explanatory completions without chat templates, not a broad workload suite.

Do not spawn agents: current instructions prohibit delegation absent explicit
user / skill authorization. Relevant repository skills were already read:
generate-profile, llm-torch-profiler-analysis, speculative-naming,
sglang-runtime-context, write-sglang-test. No goal was created. Tool permissions
are unrestricted, approval never; do not set sandbox_permissions.
