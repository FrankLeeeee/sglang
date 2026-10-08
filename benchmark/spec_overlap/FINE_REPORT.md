# Fine-grained same-GPU EAGLE3 overlap

This is the historical report for the preceding implementation. The new
GPT-OSS-only matrix, greedy-tree support and ten rounds are documented in
[REVISION_REPORT.md](REVISION_REPORT.md); historical model runs below were not
repeated for this follow-up.

The implementation creates real compute/communication overlap, but does not
improve full-batch throughput on either requested model pair. With normal CUDA
graph kernels, overlap/control ranges from 0.964x to 1.012x at batches 16 and 32.

This follow-up implements cross-request target continuations on the same two H200
GPUs (TP=2 for both models). It preserves EAGLE3's same-request dependencies.
The feature is opt-in through `--speculative-microbatch-mode fine_overlap`;
`fine_serial` executes the same continuation schedule on one stream.

The model pairs are:

- `meta-llama/Llama-3.1-8B-Instruct` with `lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge` (BF16).
- `openai/gpt-oss-20b` (MXFP4) with `zhuyksir/EAGLE3-gpt-oss-20b-bf16` (BF16).

## Schedule and ownership

Split a ready decode batch into request groups A and B. Draft A precedes verify A.
During verify A, release independent draft B. After verify A finishes, eager mode
extends A while verify B runs. Full graph mode fences extension A before verify B.
Extend B follows verify B, then join both streams and return results in the original request order. Neither model runs concurrently with itself.

The eager path yields between EAGLE draft steps and releases them after a quarter
and half of the target layers. The graph path cuts the captured target DAG into
eight ordered topological node groups and releases captured draft B after the
second group. Graph cuts add ordering barriers and are not layer boundaries.
`fine_serial` uses precisely the same cuts and ordering on the target stream.

An activation audit localized the initial continuation failure to target RoPE:
input normalization and QKV matched before the pause, but the first resumed RoPE
changed. A position audit then established that draft buffer loading overwrote
positions still used by the suspended target. The process-wide input pool keyed
buffers by name, size, dtype, and device, which let the two model runners alias.
The experimental modes now bypass that pool. Separate graph pools alone were
insufficient to fix this input aliasing.

A graph audit also found process-wide equal-vocabulary logits storage, now
isolated per model runner. This was insufficient to fix concurrent extension A
and verification B in full graph mode. A fence for that pair passed the Llama
probe. FA3 draft extension remains eager in SGLang; the graph prototype retains
that fence and overlaps only draft B with partitioned verification A. Separate
target/draft graph pools were confirmed by the audit. The unfenced failures are
retained as diagnostic evidence and their timings are excluded.

Additional isolation uses a draft NCCL communicator on the same ranks, scoped
per-forward flags and attention/MoE contexts, separate capture streams and graph
pools, disabled graph-pool borrowing, retained cross-stream inputs, and cloned graph outputs before subsequent
replays reuse their storage. Default `off` retains existing buffer sharing and
whole-graph execution.

## Time model

Let D, V, and E be draft, verify/accept, and draft-extend durations for one half
batch. The full-batch baseline is `D(2b) + V(2b) + E(2b) + H_original`.
The serial split control is `2(D + V + E) + H_serial`.
一般事件模型显式保留每条分支的准备/提交 gap：

```text
F_DA = D_A
F_VA = F_DA + g_VA + V_A
F_DB = F_DA + g_DB + D_B
F_EA = max(F_VA, F_DB) + g_EA + E_A
F_VB = max(F_VA, F_DB) + g_VB + V_B
F_EB = max(F_EA, F_VB) + g_EB + E_B
T = F_EB + H_residual
```

- `g_X >= 0`：依赖就绪至实际开始的间隙；阶段内部 gap 计入阶段时长。
  `H_residual` 仅覆盖尚未计入的轮前/轮后串行开销。
- `D_B -> E_A` 来自同一 draft stream。graph fence 改为
  `F_VB = max(F_VA, F_DB, F_EA) + g_VB + V_B`。
- `delta = g_DB - g_VA` 可为负：立即提交 D_B 不保证 V_A 同时开始。
  分支准备延迟必须置于对应递推中，不能统一加到 `max(...)` 外的 H。
- 仅当 `g_VA = g_EA = g_VB = g_EB = 0`、`g_DB = delta >= 0`，
  且阶段服务时长不因并发改变时，对称理想式才是
  `D + max(V, delta + D) + max(V, E) + E + H_residual`；
  保留 fence 则为 `D + max(V, delta + D) + E + V + E + H_residual`。
- 多 draft chunk：用统一绝对时间递推
  `finish_i = max(release_i, finish_(i-1)) + g_i + duration_i`，
  初始 `finish_0 = F_DA`；最终完成时间直接作为 `F_DB`。
- callback 控制 CPU 提交点，没有 target-release event / draft wait；
  graph 节点比例不等于 GPU 时间比例。trace 需测量实际开始、结束和 gap。
- 单轮模型含启动/排空。并发争用、graph-cut barriers、gap 与事件等待只能
  在阶段/间隙或增量惩罚中计一次；NCCL 驻留不等于可隐藏计算的通信窗口。

## Validation and measurements

Eager deterministic checks use batch-invariant kernels, 64 output tokens,
two measured repeats after two warmups, and batches 5 and 16. Every output token
sequence matches the original full-batch pipeline for both specified model pairs,
in the serial, fine_serial and fine_overlap modes, on every repeat.

| Pair | Batch | Original tok/s | Fine serial tok/s | Fine overlap tok/s |
|---|---:|---:|---:|---:|
| Llama 3.1 8B | 5 | 123.404 | 63.094 | 64.392 |
| Llama 3.1 8B | 16 | 348.977 | 179.435 | 183.026 |
| GPT-OSS 20B | 5 | 63.720 | 33.083 | 32.924 |
| GPT-OSS 20B | 16 | 182.780 | 94.387 | 94.364 |

These correctness-run rates use different GEMM kernels from normal serving;
they do not replace the normal-kernel performance comparison.
The eager fine pipeline saves about 2% against Llama's same-schedule control,
but the full-batch baseline remains substantially faster. GPT-OSS's fine
scheduling benefit is negligible in this run.

Across six profiled steady decode cycles on rank 0 at batch 16, Llama shows
0.008 ms of target noncommunication kernels intersecting draft communication,
and zero GEMM/communication intersection in either direction. GPT-OSS shows zero
in both directions. Stage-span concurrency is roughly 28 ms, but includes host
launch gaps and waits and must not be called useful compute overlap.

Full graph deterministic validation passes the same per-repeat token checks
for both pairs at batches 5 and 16. Traces contain the target chunk annotations;
these runs did not silently fall back to unpartitioned execution.

| Pair | Batch | Original graphs tok/s | Fine serial tok/s | Fine overlap tok/s |
|---|---:|---:|---:|---:|
| Llama 3.1 8B | 5 | 359.283 | 201.747 | 206.734 |
| Llama 3.1 8B | 16 | 1203.539 | 626.142 | 637.099 |
| GPT-OSS 20B | 5 | 433.541 | 250.464 | 258.650 |
| GPT-OSS 20B | 16 | 1231.257 | 744.745 | 762.784 |

Graph replay equivalence also passes on a captured GPU fork/join DAG, including
one, three and four topological chunks and three changed inputs, with exact
output comparison. Four CPU regression checks cover buffer isolation, per-forward
state restoration, and uneven request splitting/merging. The existing input
registry and full graph backend unit suites pass 46 tests.

The deterministic graph traces establish actual compute/communication overlap
in both directions. On rank 0 across six steady decode cycles at batch 16:

| Pair | Draft GEMM / target communication | Target GEMM / draft communication | All noncomm / opposite comm |
|---|---:|---:|---:|
| Llama 3.1 8B | 0.898 ms | 0.703 ms | 1.954 ms |
| GPT-OSS 20B | 0.353 ms | 1.026 ms | 2.113 ms |

Original and fine_serial graph controls show zero for both GEMM intersections.
Use these interval intersections as evidence of simultaneous GPU execution,
not as critical-path time saved. CUDA graph kernels can execute on internal
streams; attribution includes target-chunk and captured verify annotations.

Profiler triage artifacts contain the kernel, overlap-opportunity and fusion
pattern tables: [Llama](results/fine-graphs-safe/llama-triage.md) and
[GPT-OSS](results/fine-graphs-safe/gpt-oss-triage.md). Source-stack mapping is
unresolved in these stack-free traces; phase ownership comes from explicit GPU
annotations, not inferred source locations. For Llama, graph execution reduced
summed NCCL AllReduce residency from 443.6 ms in the eager trace to 114.9 ms in the
graph trace, while summed GEMM residency stayed near 158 ms. These totals include
prefill and decode and are not wall-clock sums. GPT-OSS's triage identifies an
already available fused MoE activation/requantization path; this experiment does
not change its kernels.

## Normal-kernel throughput

Full decode graphs, 128 output tokens per request, three measured repeats after
two warmups. Rates include the serving request elapsed time; profiling happens
after the timed trials. All modes use the same TP=2 GPUs and common flags.

| Pair | Batch | Original tok/s | Fine serial tok/s | Fine overlap tok/s | Overlap/control |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B | 16 | 3405.7 | 2062.6 | 2021.6 | 0.9801x |
| Llama 3.1 8B | 32 | 5827.6 | 3716.6 | 3581.4 | 0.9636x |
| GPT-OSS 20B | 16 | 1975.3 | 1287.1 | 1301.4 | 1.0111x |
| GPT-OSS 20B | 32 | 3346.8 | 2159.8 | 2184.7 | 1.0115x |

Llama overlap regresses by 2.0–3.6% against its matching serial control.
GPT-OSS improves by about 1.1%. Neither recovers the full-batch throughput:
Llama is 38.5–40.6% below original and GPT-OSS is 34.1–34.7% below original.
These small control differences come from three trials, not a significance test.

Normal serving outputs are not exactly reproducible under all batch shapes.
Even the original Llama batch-16 run matches only 15/16 requests between some
repeats. Fine serial also differs from original, so these rate probes are not
the exact-token correctness gate. The separate batch-invariant matrices above
pass every request and repeat. Per-repeat token IDs, timings and acceptance
statistics are retained in the JSONs. At batch 16 the mean accept length is
1.70–1.71 for Llama and 1.09 for GPT-OSS across these modes, so splitting did not
produce a large acceptance-rate benefit.

The eager batch-16 comparison has completed for Llama: original 663.5 tok/s,
fine_serial 320.7 tok/s, fine_overlap 324.9 tok/s (1.0132x control). GPT-OSS is
pending after the user requested a session checkpoint. See [SESSION_STATE.md](SESSION_STATE.md).

A compact, versioned snapshot of completed rates and repeatability counts is in
[measurements.json](measurements.json). Raw traces and request outputs stay in the
local ignored results directories.

## Reproduction and limits

Use [README.md](README.md) for commands. Raw eager validation evidence is in
[results/fine-correctness](results/fine-correctness/). The graph matrix uses
[results/fine-graphs-safe](results/fine-graphs-safe/). Launch records
include source hashes for reproducibility. Normal performance evidence is in
[graphs](results/fine-normal-graphs/) and [eager](results/fine-normal-eager/).

Base commit: `ba77d30256bfb1d1e0d53cd877c87f0b2e279ef1`. Host: two H200 GPUs,
NV18, PyTorch 2.14.1+cu130, driver 595.71.05, system nvcc 12.8. Experiments use
`SGLANG_DEEPGEMM_PDL=0`; correctness additionally uses
`SGLANG_ENABLE_JIT_DEEPGEMM=0` and `--enable-deterministic-inference`.
Common graph buckets are 1, 2, 4, 8, 16, 32. Exact arguments, source hashes,
per-repeat outputs, logs and per-rank traces are retained in each results directory.

Supported targets are Llama and GPT-OSS with FA3, greedy EAGLE3 topk=1, fixed
budget, PP=DP=CP=EP=1, disabled CPU overlap scheduling, custom all-reduce and
FlashInfer all-reduce fusion. Prefill and unsupported requests use the original
path. Expert-parallel all-to-all is not implemented or measured. GPT-OSS at EP=1
uses tensor-parallel communication; being MoE does not imply all-to-all here.

These repeated synthetic explanatory prompts are a decode-oriented probe. They
do not apply the models' chat templates or represent a broad workload suite.
Context length is 2048; each prompt is short. The results cover TP all-reduce and
logits all-gather on two H200 GPUs with NV18, not expert-parallel all-to-all.
