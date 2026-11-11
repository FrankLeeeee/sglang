# 06-minedraft-sglang--2xh200: with SGLang's hidden-state drafts on an 8B target, MineDraft (draft GPU + target GPU) is verify-bound and is predicted to lose to two colocated replicas

**Stage**: 06 (MineDraft reproduction, measure-then-model) · **Status**: done (model predictions; no running port) · **Date**: 2026-11-11
**Branch**: `research/06-minedraft-sglang/2xh200` · **Forked from**: `research-main` · **Built on**: `02-baselines--placement-2xh200`, `03-profiling--sglang-ssd`, `04-ideation--block-drafts`

## TL;DR
- Measured on 2x H200 in SGLang (v0.5.23 fork), Qwen3-8B with EAGLE-3 / DSpark / DFlash and Llama-3.1-8B with EAGLE-3 / DFlash, C = 1-64, 256 tokens. DFlash now runs: it needed nvcc, crt and nvvm pinned to CUDA 13.0 to match the runtime headers.
- Colocated SD at C = 1: DFlash on Llama 2.21x AR, DSpark on Qwen 2.10x, EAGLE-3 1.58-1.81x. At C = 64: 1.18-1.55x, and Qwen + DFlash (block 16) drops to 0.88x of AR.
- The draft side is always cheaper than the verify: T_d / T_v = 0.21-0.33 at C = 1 and 0.55-0.96 at C = 64. Copying a round's hidden states GPU0 -> GPU1 over NVLink costs 0.04-0.15 ms.
- MineDraft model (two cohorts of C/2, draft on GPU1 overlapping verify on GPU0): verify-bound everywhere. Predicted 0.68-0.88x of one colocated GPU at C = 4; 1.22-1.60x of one GPU at C = 64, which is 0.61-0.80x per GPU and 0.85-0.92x of two colocated replicas on the same GPUs.
- Why it differs from the paper: MineDraft wins when draft time is close to verify time (standalone small-LM drafts, TP4 targets). Hidden-state drafts on an 8B target leave little draft time to hide, and splitting the batch makes GPU0 run two memory-bound verifies instead of one.

## Decisions
| Decision | Reason |
|---|---|
| Measure, then model; no scheduler port | 2-hour allocation; a port needs a separate draft KV/req pool on GPU1, hidden-state relay and a two-cohort batch manager (several days). Frank chose this scope. |
| Compare MineDraft to two colocated replicas, not only to one GPU | Same GPU count; the paper's per-GPU normalization alone hides this baseline |

## Open issues
- Short runs (up to ~20% pass-to-pass variance at C = 64); T_d is a residual that includes host time; Llama EAGLE-3 accept length (2.0-2.25) is below stage 02 (2.87); no deterministic exact-match check.

## Next
- If MineDraft stays on the table: repeat measure-then-model on a 70B TP4 target + 1 draft GPU with a standalone draft (T_d / T_v near 1), before any port.

## Artifacts
- Report: `.research/nodes/06-minedraft-sglang--2xh200/REPORT.md`; figures and `results.json` next to it.
- Code and raw data: `minedraft_sglang/` (runner, closed-loop client, timing `sitecustomize`, transfer bench, analysis, `data/`).
