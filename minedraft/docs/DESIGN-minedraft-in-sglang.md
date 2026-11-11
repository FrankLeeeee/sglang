# Design: MineDraft inside SGLang for hidden-state drafts (EAGLE-3, DSpark, DFlash)

Status: design, 2026-11-11. Target branch on `FrankLeeeee/sglang`: `minedraft-disagg`. Reference: `electron-shaders/MineDraft@272c855` (`plugin/spec_decode/spec_decode_worker.py::ParallelSpecDecodeWorker._run_speculative_decoding_step`, `plugin/core/scheduler.py` batch_flag logic).

## 1. What the reference does (per engine step)

1. The scheduler schedules all 2m running requests. Each carries `batch_flag in {0, 1}`; `_skip_batch_flag` names the cohort that only drafts this step. New requests get the flag of the smaller cohort (`_balance_counter`); finished, aborted or preempted requests return their slot (`_recycle_batch_flag`). The drafting cohort gets no verify KV slots (`_has_deferred_skip`).
2. The worker splits the step into cohort A (verify, with proposals stored from the previous step) and cohort B (draft now).
3. It starts the target scoring of A asynchronously on the target ranks, runs the draft of B on the drafter rank meanwhile, waits for A's scores, accepts on the drafter/driver rank, stores B's proposals, flips `_send_batch_flag`.
4. The target sends hidden states (and prefill hidden states) to the drafter with point-to-point sends every step. With one cohort, or before any proposals exist, it runs a plain SD step.

## 2. Mapping onto SGLang

Single scheduler process, target on `cuda:0` (TP1), draft on `cuda:1`. TP4 targets are out of scope for now (Frank, 2026-11-11).

| Reference piece | SGLang place | Change |
|---|---|---|
| `batch_flag`, `_balance_counter`, recycle | `Req` + scheduler batch selection (`get_next_batch_to_run`) | New `req.minedraft_cohort`; admission picks the smaller cohort; the decode batch is split by cohort into `verify_batch` (A) and `draft_batch` (B). Read `large-class-style` before touching `Scheduler`. |
| `ParallelSpecDecodeWorker` | new `speculative/minedraft_worker.py` wrapping `EAGLEWorkerV2` | `forward_batch_generation(verify_batch, draft_batch)`: (1) launch target verify of A on cuda:0; (2) under `torch.cuda.device(1)`, run draft-extend + draft of B on cuda:1; (3) sync, accept A on cuda:0 (SGLang's existing accept kernels; the reference accepts on the drafter rank, a deliberate difference); (4) store B's draft tokens/tree in B's `spec_info`. |
| drafter rank with its own KV | `EagleDraftWorker` built with `gpu_id=1` | Its own `ReqToTokenPool` and KV allocator on cuda:1, indexed by the same `req_pool_idx`; draft KV slots allocated in lockstep with target slots. Today the draft shares the target's `req_to_token_pool` and allocator (`Scheduler.alloc_memory_pool` -> `draft_worker.alloc_memory_pool`), so this is the main surgery. |
| hidden-state sends | `EagleDraftInput.hidden_states` | After A's verify, copy A's accepted rows' aux hidden states (EAGLE-3: 3 x 4096; DFlash/DSpark: 5 x 4096) and accept tokens to cuda:1 with `non_blocking` P2P on a copy stream plus an event; A's draft-extend next step waits on that event. Measured cost <= 0.15 ms for 1024 rows (REPORT section 4). |
| prefill hidden states | prefill path of each worker | New requests prefill on cuda:0; their prompt hidden states are copied to cuda:1 and materialized into the draft KV (EAGLE-3 draft extend; DFlash/DSpark `_append_target_hidden_to_draft_kv_by_loc`). |
| shared embed / lm_head | `EagleDraftWorker.init_lm_head` / `_resolve_shared_embed_and_head` | When the draft shares the target's embedding or head, keep a copy on cuda:1 instead of aliasing the cuda:0 tensor. |
| fallback to plain SD | worker | If either cohort is empty, run today's EAGLEWorkerV2 step (draft on cuda:1, verify on cuda:0, serial). |

## 3. Device isolation (main risk)

- `get_device().device` is a process-global device; the draft worker and its CUDA graphs, attention backend buffers, `plan_stream`, and sampling buffers must be created and replayed under `torch.cuda.device(1)`. The draft `TpModelWorker` must not call `torch.cuda.set_device` globally or re-init the process group (`is_draft_worker=True` already skips the latter).
- Overlap: both GPUs are driven by one Python thread. With CUDA graphs on both sides, the host launches the cuda:0 verify graph and returns at once, then launches the cuda:1 draft graphs, so the two run concurrently. Eager paths (prefill, draft-extend without graphs) serialize host time; measure it.
- SGLang's overlap scheduler (one batch ahead) and `FutureMap` assume one batch per step; the first version runs with `--disable-overlap-schedule`, then re-enables it.

## 4. Order of work and acceptance tests

1. EAGLE-3 (existing `EAGLEWorkerV2` path), topk 1 chain, Qwen3-8B and Llama-3.1-8B.
   - Correctness: greedy outputs equal to SGLang colocated EAGLE-3 and AR under `--enable-deterministic-inference` on the 90 prompts.
   - Timing: per-step time vs `max(T_v(C/2), T_d(C/2) + X)` from REPORT section 5.
2. DSpark (`dspark_components/dspark_worker_v2.py`), then DFlash (`dflash_worker_v2.py`): the same cohort split; their draft KV injection of target hidden states moves to cuda:1.
3. Benchmarks: C = 1, 4, 16, 64, 128 vs colocated 1 GPU and 2 colocated replicas.

## 5. Already done (side result)

A client-side port of the reference loop with two unmodified SGLang servers and a standalone draft (`minedraft_sglang/minedraft_sglang.py`, Llama-3.1-8B + Llama-3.2-1B, k = 5) runs and matches the reference schedule; first point C = 4: MineDraft 314 tok/s vs sequential SD 280 tok/s on the same harness (1.12x), same accept length (3.54). It cannot serve hidden-state drafts and its absolute speed is bounded by HTTP and extend-path verify, so it is not the deliverable.

## 6. Status 2026-11-11 17:45 (branch `FrankLeeeee/sglang:minedraft-disagg`, head 3bc889f62e)

Implemented (EAGLE-3, `SGLANG_MINEDRAFT_DRAFT_GPU=1`, `--disable-overlap-schedule`):
- Draft worker on cuda:1: own RoPE cache, copied shared embedding, own KV pool, mirrored `req_to_token`, slot-id allocator proxy (`speculative/minedraft_device.py`); every draft entry point crosses a device boundary (inputs incl. target hidden states copied to cuda:1, fields the draft rebinds written back, nested objects included).
- Two persistent cohorts in the scheduler that alternate as the running batch (`_minedraft_before_step` / `_minedraft_after_decode`); the worker verifies the current cohort with drafts made for it in the previous step and launches the parked cohort's draft asynchronously on cuda:1 (stream events, row-wise table sync, buffer clones), finishing it at the end of the step before the scheduler prepares that cohort. About 87-93% of steps reuse a pre-made draft.
- Target CUDA graphs on; draft eager (draft graph capture on cuda:1 still fails).

Correctness (deterministic inference, Qwen3-8B, C = 16, 32 requests): AR, colocated EAGLE-3, disaggregated serial SD and disaggregated MineDraft give identical outputs, 32/32 each (`minedraft_sglang/data/insglang/det_*.json`).

First performance (all `--disable-overlap-schedule`, Qwen3-8B, 256 tokens, 3C requests; `data/insglang/perf_*.json`):

| C | colocated 1 GPU (graphs) | 2 colocated replicas (2 x C/2) | disaggregated serial SD (2 GPUs) | MineDraft in SGLang (2 GPUs) |
|---|---|---|---|---|
| 16 | 3571 | n/a (C = 8 not measured) | 1988 | 1012 |
| 32 | 5701 | 7142 | 3316 | 1726 |
| 64 | 8551 | 11402 | 5468 | 2980 |
| 128 | 11626 | 17102 | 8383 | 4896 |

Two replicas = 2 x the colocated point at C/2 from the same pass.

The implementation is correct but not yet fast: MineDraft is 0.51-0.58x of the disaggregated serial path and 0.26-0.29x of two replicas. The step is host-serialized: the parked cohort's draft runs eager (dozens of launches per step on one Python thread), and the current cohort's draft-extend goes through the synchronous boundary (full device syncs), so GPU0's verify and GPU1's draft barely overlap. Next: capture the draft's CUDA graphs on cuda:1, make draft-extend asynchronous like the draft, and remove the remaining host syncs; then re-measure against the model's 0.85-0.98x of two replicas.

## 7. Status 2026-11-11 18:30 (allocation expired at 18:27; branch head 12a25f7135)

- Draft CUDA graphs now capture and replay on cuda:1: SGLang's CUDA graph memory pool and named streams (capture stream, plan stream) are process-global and were created on cuda:0; every draft call now swaps in a draft-GPU pool and stream set (`_DraftDeviceResources`). Draft decode graphs are captured on cuda:1 (`SGLANG_MINEDRAFT_DRAFT_DECODE_GRAPHS=1`); the draft's breakable prefill graph is still eager (only affects prefill).
- With target and draft decode graphs on (overlap scheduler off), MineDraft measured 1250 tok/s at C = 16 and 3602 tok/s at C = 64 (eager draft: 1012 / 2980), Qwen3-8B, 48 / 192 requests.
- Lost: a graphs-on run of cross-GPU serial SD and MineDraft at C = 32/64/128 was still on the node when the allocation expired (extension was not applied), so those numbers do not exist.

Next session, in order:
1. Re-run graphs-on (both GPUs) at C = 32/64/128 for single-GPU colocated, two replicas, cross-GPU serial and cross-GPU MineDraft (`minedraft_sglang/perf.sh` with `SGLANG_MINEDRAFT_DRAFT_DECODE_GRAPHS=1`), then the same with the overlap scheduler on.
2. Remove the remaining host serialization in the MineDraft step: make the current cohort's draft-extend asynchronous like the parked cohort's draft, drop the device-wide synchronizations in `_call_draft`, and cut the per-step Python cost of the proxy/write-back traversal.
3. Capture the draft's prefill graph on cuda:1 (warm the JIT kernels on cuda:1 before the breakable capture).

## 8. Results 2026-11-11 20:05 (new allocation; branch head ede6c7ee29)

All configs: EAGLE-3 (3, 1, 4), Qwen3-8B, bf16 greedy, 256 tokens, 3C requests per point, **CUDA graphs on** (target on cuda:0; for the cross-GPU configs the draft's decode graphs are captured on cuda:1, its prefill runs eager). "2 colocated replicas" = 2 x the colocated point at C/2 from the same pass. tok/s.

| scheduler | C | colocated 1 GPU | 2 colocated replicas | cross-GPU serial SD | cross-GPU MineDraft | MineDraft / replicas | MineDraft / serial |
|---|---|---|---|---|---|---|---|
| overlap ON | 32 | 7081 | 8974 | 5468 | 3488 | 0.39x | 0.64x |
| overlap ON | 64 | 10375 | 14162 | 9092 | 5940 | 0.42x | 0.65x |
| overlap ON | 128 | 14331 | 20749 | 12489 | 9144 | 0.44x | 0.73x |
| overlap OFF | 32 | 5599 | 5963 | 4350 | 2680 | 0.45x | 0.62x |
| overlap OFF | 64 | 8276 | 11199 | 7240 | 4363 | 0.39x | 0.60x |
| overlap OFF | 128 | 11466 | 16552 | 9823 | 6760 | 0.41x | 0.69x |

Correctness under `--enable-deterministic-inference` (C = 16, 32 requests): cross-GPU MineDraft equals AR on 32/32 with graphs on, both with the overlap scheduler off and on (`data/insglang2/det2_*.json`).

How MineDraft runs in each mode:
- Overlap OFF: the scheduler alternates two cohorts; while cohort A verifies on cuda:0, the worker drafts cohort B on cuda:1 (launched before A's verify; 85-93% of steps reuse that draft).
- Overlap ON: the scheduler alternates cohorts and each step drafts its own cohort inline; because the host runs one step ahead, cohort B's draft on cuda:1 runs while cohort A's verify is still on cuda:0. A pre-draft of the parked cohort from FutureMap-resolved inputs (`SGLANG_MINEDRAFT_PREDRAFT=1`) gives wrong outputs and is off.
- The draft boundary uses stream events, no device-wide host syncs (this alone moved cross-GPU serial from 3699/6343/9370 to 4350/7240/9823 and MineDraft from 2140/3649/5919 to 2680/4363/6760, overlap off).

Why MineDraft loses here: it splits the batch into two cohorts, so each round runs two verifies of C/2 requests plus two sets of per-step host work. The verify is memory-bound up to about 64-128 rows, so a half-batch verify costs nearly as much as a full one, and EAGLE-3's draft on cuda:1 (T_d / T_v about 0.2-0.9, REPORT section 3) is too short for hiding it to pay for the doubled per-step cost. Measured MineDraft is 0.60-0.73x of the cross-GPU serial path and 0.39-0.45x of two colocated replicas, below the step model's 0.85-0.98x (REPORT section 5), the gap being the per-step host cost that the model put on the hidden draft side.

## 9. Follow-up 2026-11-11 23:10 (branch head c079da4e36)

- The draft's breakable prefill graph now also captures on cuda:1 (`SGLANG_MINEDRAFT_DRAFT_PREFILL_GRAPHS=1`; the earlier failure had the same root cause as the decode graphs, process-global streams/pool). Cross-GPU configs now have the same graph coverage as colocated (decode + prefill graphs; draft-extend runs eager in both).
- Host time per cohort step, overlap on, C = 64 (`SGLANG_MINEDRAFT_TIMING=1`): step about 14.7 ms; worker draft 1.8 ms, draft-extend 3.7 ms (about 2 ms of it eager draft-extend kernels, same as colocated), verify launch 0.9 ms, device-boundary copies and field traffic about 2 ms in total. Removing the boundary entirely would save about 10%, not enough to reach the cross-GPU serial path.
- Heavier draft, EAGLE-3 (7, 1, 8), CUDA graphs on, overlap ON (`data/insglang2/k7_overlap_*.json`), tok/s:

| C | colocated 1 GPU | 2 colocated replicas | cross-GPU serial SD | cross-GPU MineDraft | MineDraft / replicas | MineDraft / serial |
|---|---|---|---|---|---|---|
| 32 | 5969 | 7420 | 5444 | 3484 | 0.47x | 0.64x |
| 64 | 8104 | 11938 | 7746 | 5470 | 0.46x | 0.71x |
| 128 | 10017 | 16208 | 9575 | 7708 | 0.48x | 0.80x |

A longer draft moves MineDraft closer to the serial path (0.80x at C = 128, from 0.73x with (3, 1, 4)) but not past it, and it stays at about half of two replicas.
