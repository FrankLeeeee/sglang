> Initial coarse-pipeline investigation. The subsequent fine-grained implementation, shared-input-buffer fix, and new benchmarks are documented in [FINE_REPORT.md](FINE_REPORT.md). Statements below about removed variants describe the initial experiment.

# Same-GPU EAGLE3 overlap: initial experiments

Date: 2026-10-08. Repository base: `ba77d30256bfb1d1e0d53cd877c87f0b2e279ef1`, plus the local experimental changes.

The validated eager prototype does **not improve throughput** on these workloads. Splitting the request batch nearly doubles target verification/launch overhead, while the observed compute/communication overlap is tiny. The more aggressive mid-forward and extend/verify variants failed token checks and were removed from the runtime implementation.

The implementation remains opt-in (`--speculative-microbatch-mode off|serial|overlap`, default `off`). This is a reference implementation and research harness, not a production overlap optimization. CUDA-graph execution and expert-parallel all-to-all integration are unfinished.

## Setup and scope

Both target and draft use the same two H200 GPUs, TP=2, NV18 connectivity. The pairs are:

- `meta-llama/Llama-3.1-8B-Instruct` + `lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge`.
- `openai/gpt-oss-20b` + `zhuyksir/EAGLE3-gpt-oss-20b-bf16`. The target is MXFP4 and the draft is BF16.

The correctness matrix uses batch sizes 5 and 16, 64 output tokens, two timing trials after two warmups, greedy sampling, ignored EOS, and a 2048-token context limit. Prompts explain computer execution; repeated prompts can hit the prefix cache. These are short-context, closed-batch, decode-oriented probes. They exercise uneven microbatches and shrinking active batches, but do not constitute a broad accuracy suite.

All retained eager modes disable CUDA graphs, the CPU overlap scheduler, custom all-reduce, and FlashInfer all-reduce fusion. On this host, custom all-reduce failed to build against tvm_ffi headers; nvcc 12.8 required disabling DeepGEMM PDL. Batch-invariant correctness runs additionally disable JIT DeepGEMM. PyTorch is 2.14.1+cu130 and the NVIDIA driver is 595.71.05. Exact server arguments and outputs are in each result JSON.

Normal-kernel performance trials are recorded separately below. Neither matrix measures the fully optimized production configuration with CUDA graphs and all communication fusions enabled.

## Dependencies and time model

For these EAGLE3 pairs, the actual cycle is `draft -> verify/accept -> draft extend -> draft`. Extend consumes accepted target auxiliary features and the new bonus token. That dependency prevents overlapping successive whole cycles of the same request without additional speculation/rollback. Independent request groups provide ready work.

[PEARL's engine](https://github.com/smart-lty/parallelspeculativedecoding/blob/fe7ddefc566d593d45ebe721341004565aeda195/src/engine.py) uses separate workers/accelerators and rollback. Its scheduling idea does not remove EAGLE3's target-feature dependency.

Let D(b), V(b), and E(b) include draft/tree preparation, verification/sampling/commit preparation, and draft extension for b requests. H denotes exposed host work not already included in stage durations. Avoid adding overlapping host and GPU time twice.

- Original full batch: `T_original(2b) = D(2b) + V(2b) + E(2b) + H_original`.
- Split serial control: `T_serial = 2[D(b) + V(b) + E(b)] + H_serial`.
- Fully isolated overlap, with draft B released delta into verify A: `T_ideal = D + max(V, delta + D) + max(V, E) + E + H_overlap`.
- Validated prototype, which fences extend A before verify B: `T_prototype = D + max(V, delta + D) + E + V + E + H_prototype`.

For uneven groups, replace the symmetric ideal expression with `D_A + max(V_A, delta + D_B) + max(V_B, E_A) + E_B + H_overlap`. These are one-round makespans including startup/drain; the prototype rejoins each scheduler iteration.

The practical model is `T_overlap = T_serial - O_feasible + P_contention + P_launch + P_sync`. Only independent, ready operations contribute to O. NCCL kernels consume SMs/HBM and can spend time waiting for peer ranks; their duration is not automatically usable compute slack. Resource demand and the event DAG provide lower bounds, not guaranteed speedup. Throughput is useful generated tokens divided by elapsed time, with measured acceptance included.

The validated schedule submits verify A's host call completely before draft B, allowing pending GPU work to overlap. It uses a separate draft CUDA stream and NCCL communicator on the same ranks. Events retain D_A -> V_A, V_A -> E_A, D_B -> V_B, E_A -> V_B, and V_B -> E_B dependencies. Each model remains serial with itself; results rejoin in request order.

## Normal-kernel throughput

Batch 16, 128 output tokens per request, three trials after two warmups, with the same eager-execution and communication restrictions. Profiling is excluded from these rates.

| Pair | Original tokens/s | Split serial tokens/s | Validated schedule tokens/s | Schedule/original | Schedule/serial |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama | 680.00 | 344.77 | 340.41 | 0.501x | 0.987x |
| GPT-OSS | 240.19 | 122.22 | 120.09 | 0.500x | 0.983x |

Both normal-kernel comparisons also show roughly half the original throughput. Ordinary-kernel split runs changed some token IDs relative to the full batch and had small differences between repeats; these samples do not establish exact-output equivalence under ordinary kernels. The separate batch-invariant matrix supplies the retained schedule's correctness evidence. Repeated token matches: Llama: serial repeat matches [15, 16], overlap repeat matches [15, 15] out of 16; GPT-OSS: serial repeat matches [15, 15], overlap repeat matches [16, 16] out of 16.

Evidence: [normal results](results/normal/), [normal comparison](results/normal/comparison.json).

## Correctness-mode throughput

These rates use batch-invariant execution. `Original` means current sequential EAGLE3, not target-only decoding. Every request in every timing repeat of the retained serial and overlap modes matched the original token IDs.

| Pair | Requests | Original tokens/s | Split serial tokens/s | Validated overlap tokens/s | Overlap/original | Overlap/serial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama | 5 | 125.27 | 65.30 | 64.23 | 0.513x | 0.984x |
| Llama | 16 | 349.69 | 185.18 | 179.06 | 0.512x | 0.967x |
| GPT-OSS | 5 | 64.34 | 33.42 | 33.02 | 0.513x | 0.988x |
| GPT-OSS | 16 | 185.35 | 93.02 | 93.97 | 0.507x | 1.010x |

Evidence: [validated comparison](results/draft-only/comparison.json), [original/serial results](results/deterministic/), [validated overlap results](results/draft-only/). Mean accepted lengths at batch 16 are about 1.535 for Llama and 1.078 for GPT-OSS, identical across the three correctness modes. This low acceptance and the short-context workload limit generalization.

## Profile measurements

The following stage statistics use the batch-invariant correctness traces.

Torch traces contain eight scheduler steps: two prefills and six decode rounds. Decode has six D/V stages in the original path and twelve in the split paths. Extension statistics include two prefill extensions in the original path. Each rank has its own compressed Chrome trace; analysis below uses TP0. Timing trials completed before profiling.

Stack collection stalled export and consumed excessive host RAM, so retained traces use `with_stack=False`, `record_shapes=True`. Kernel-to-Python attribution is therefore incomplete. GPU stage ranges include stream waits and kernel-free launch gaps; they are not isolated compute costs.

| Original pair, batch 16 | Draft median ms | Verify median ms | Extend median ms | Verify kernel union median ms | Verify kernel-free stream median ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama | 8.33 | 67.99 | 3.96 | 12.09 | 55.44 |
| GPT-OSS | 8.25 | 91.70 | 3.81 | 6.84 | 84.87 |

Columns are separate medians and need not add exactly. Kernel-free stream intervals can include copies, host launch gaps, and synchronization; they are not a hardware utilization measurement.

| Validated overlap, batch 16 | Draft/target stage-range intersection ms | Draft noncommunication / target communication kernel intersection ms | Target noncommunication / draft communication kernel intersection ms |
| --- | ---: | ---: | ---: |
| Llama | 0.23030 | 0.00374 | 0.00000 |
| GPT-OSS | 0.41664 | 0.00838 | 0.00000 |

These totals cover the captured trace, not one iteration. Kernel attribution uses matching GPU stream/range boundaries. Noncommunication includes elementwise and bookkeeping kernels, so these intersections are not necessarily GEMM overlap; the analysis JSON also reports GEMM-only intersections. The interval intersection demonstrates a small amount of concurrent execution but does not measure saved latency. Batch splitting has a much larger cost. This TP=2/EP=1 experiment contains all-reduce/all-gather communication; it does not test expert all-to-all.

## Independent GPU resource probe

`bench_stream_overlap.py` compares the fully isolated serial and concurrent DAG using private activations, row-parallel GEMMs, and separate NCCL groups. It has no model weights, attention, KV cache, or acceptance. Timers report the slowest rank and exclude profiling and the timing-result reduction. Both ranks passed exact output checks across 20 alternating trials.

| Synthetic configuration | Serial median ms | Concurrent median ms | Serial/concurrent |
| --- | ---: | ---: | ---: |
| H=4096, 32 layers | 4.7600 | 4.8752 | 0.9764x |
| H=2880, 24 layers | 3.4764 | 3.6112 | 0.9627x |

These probes were also slower under concurrency. They establish a correct independent-stream control, not a prediction for either real model. Shapes, launch overhead, and shared HBM/SM/fabric demand need workload-specific tuning.

## Rejected variants and remaining implementation work

- Initial eligibility mistakenly rejected the default all-false sampling-mask list. Traces exposed silent fallback; the predicate and regression test now cover that case. Preliminary fallback numbers are excluded.
- A full-decode CUDA graph experiment used separate target/draft capture streams and pools, separate NCCL communicators, and cloned draft recurrent outputs. Llama original and split-serial runs passed every deterministic token check at batches 5 and 16. Concurrent execution failed one of five requests in the second batch-5 trial, first diverging at output token 53. The runner stopped before batch 16. The graph path was removed and the retained prototype rejects graphs; its timing is excluded from validated performance results. Sources and failing evidence are retained under [graphs-correctness](results/graphs-correctness/). Separate graph pools alone did not establish correct concurrent execution.
- Concurrent extend A / verify B changed outputs across repeats on both models. Switching the logits gatherer to NCCL did not fix it. The retained runtime fences that pair.
- Draft B launched inside target forward, after the first layer or after a quarter of layers, changed target outputs. A quarter-layer single-stream control still matched only 14/16 Llama requests. Concurrent CUDA execution is therefore not necessary for that failure; mid-forward reentrancy/shared forward state needs isolation. The retained runtime does not nest model forward calls. Failed quarter-layer source and result snapshots are under [fine](results/fine/); the single-stream control is under [reentry-control](results/reentry-control/).

A production implementation needs explicit target continuations outside a live ModelRunner forward scope, an audit of all per-forward metadata and borrowed buffers, and ownership through the last cross-stream consumer. Whole-model eager hooks are insufficient here. CUDA-graph chunks need separate static input/output/workspace leases and captured event edges. Existing fused communication paths need private scratch before re-enabling concurrent use.

The scheduling design must also recover batch efficiency: verification of each half costs almost as much as the full batch in these eager traces. An ideal overlap cannot compensate for that cost merely by hiding the small draft stage. Investigate queued graph chunks and appropriately sized ready-request groups before extending the prototype. Expert-parallel all-to-all requires a separate supported configuration, profiling, and accuracy checks.

## Reproduce and validate

See [README.md](README.md) for formulas, constraints, server commands, and profile analysis. Run the correctness matrix with `--deterministic`; its runner now rejects any repeat that differs from the original token IDs and saves the failing result. The trace check rejects silent fallback. Run normal-kernel throughput separately.

The forward-snapshot regression passed (`test/manual/spec/eagle/test_microbatch_overlap.py`), including uneven split/join, request/sample alignment, missing scheduler penalizer, and default sampling-mask eligibility. The five server-argument namespace tests passed. Python compilation, Ruff F/I checks on new code, and `git diff --check` passed.

## Profiler triage tables

These are the repository skill's single-trace tables for the batch-invariant original batch-16 runs. Kernel shares are fractions of summed kernel durations, including the two prefill steps, rather than fractions of total request wall time. Automatic overlap attribution is inconclusive without mapping/formal traces. The GPT-OSS fusion match points to an existing implementation; it is not a new optimization proposal.

### llama

```text
Triage View
Mode: single-trace
Framework: SGLang
Input traces: /data/frankleeeee/home/sglang/benchmark/spec_overlap/results/deterministic/llama-off-b16-profile/1791429493.1444006-TP-0.trace.json.gz

```

#### Kernel table

| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| matmul_kernel_persistent | gemm | 78.26 ms | 62.9% | 1140 | unresolved | cuLaunchKernelEx |
| ncclDevKernel_AllReduce_Sum_bf16_TREE_LL(ncclDevKernelArgsStorage<4096ul>) | communication | 26.60 ms | 21.4% | 580 | unresolved | cuLaunchKernelEx |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> > | memory | 3.72 ms | 3.0% | 1132 | unresolved | cudaLaunchKernel |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}> | elementwise | 2.30 ms | 1.8% | 552 | unresolved | cudaLaunchKernel |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, false, true, true, true, false, false, cutlass::bfloat16_t, false, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, false, true, true, true, true, true> > > > | gemm | 2.00 ms | 1.6% | 276 | unresolved | cudaLaunchKernelExC |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>, false> | memory | 1.37 ms | 1.1% | 1104 | unresolved | cudaLaunchKernel |
| mean_kernel | other | 1.31 ms | 1.1% | 552 | unresolved | cuLaunchKernelEx |

#### Overlap-opportunity table

| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

#### Fuse-pattern table

| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| No medium-confidence source-backed fusion opportunity matched this trace. | - | - | - | - | - | - | - |

### gpt-oss

```text
Triage View
Mode: single-trace
Framework: SGLang
Input traces: /data/frankleeeee/home/sglang/benchmark/spec_overlap/results/deterministic/gpt-oss-off-b16-profile/1791429870.3442464-TP-0.trace.json.gz

```

#### Kernel table

| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| matmul_kernel_persistent | gemm | 21.73 ms | 30.0% | 500 | unresolved | cuLaunchKernelEx |
| ncclDevKernel_AllReduce_Sum_bf16_TREE_LL(ncclDevKernelArgsStorage<4096ul>) | communication | 17.64 ms | 24.3% | 452 | unresolved | cuLaunchKernelEx |
| _matmul_NNT_bf16xbf16xmxfp4_16x256x128x1_swiglu | gemm | 8.46 ms | 11.7% | 192 | unresolved | cuLaunchKernelEx |
| _matmul_NNT_bf16xbf16xmxfp4_16x256x128x1 | gemm | 5.37 ms | 7.4% | 192 | unresolved | cuLaunchKernelEx |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> > | memory | 2.89 ms | 4.0% | 876 | unresolved | cudaLaunchKernel |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}> | elementwise | 1.67 ms | 2.3% | 424 | unresolved | cudaLaunchKernel |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<192>, cute::C<64> >, 64, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, false, true, true, true, false, false, cutlass::bfloat16_t, false, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<64>, cute::C<192> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<64, 192, 128, 128, false, true, true, true, true, true> > > > | gemm | 1.49 ms | 2.1% | 192 | unresolved | cudaLaunchKernelExC |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>, false> | memory | 1.04 ms | 1.4% | 848 | unresolved | cudaLaunchKernel |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8> > | reduce_topk | 1.00 ms | 1.4% | 192 | unresolved | cudaLaunchKernel |
| mean_kernel | other | 0.88 ms | 1.2% | 424 | unresolved | cuLaunchKernelEx |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}> | elementwise | 0.86 ms | 1.2% | 424 | unresolved | cudaLaunchKernel |
| _all_gather_kernel_inner | communication | 0.80 ms | 1.1% | 28 | unresolved | cuLaunchKernelEx |

#### Overlap-opportunity table

| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

#### Fuse-pattern table

| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| Fused MoE activation + quant / re-quant | Confirmed | 13.87 ms | 19.1% | _matmul_NNT_bf16xbf16xmxfp4_16x256x128x1_swiglu (11.7%)<br>_matmul_NNT_bf16xbf16xmxfp4_16x256x128x1 (7.4%) | unresolved | python/sglang/srt/layers/moe/ep_moe/kernels.py<br>python/sglang/kernels/ops/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py<br>python/sglang/srt/layers/moe/cutlass_w4a8_moe.py | Split kernels in this family take 19.1% of GPU time. This tree already has a matching path. Quantized MoE backends already fuse activation with re-quantization. |
