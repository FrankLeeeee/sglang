# GPT-OSS overlap follow-up — active checkpoint (2026-10-09)

## User scope and deadline

- Only target `openai/gpt-oss-20b`, draft `zhuyksir/EAGLE3-gpt-oss-20b-bf16`.
- Complete 20 original profiling cases, ten refinement rounds, Chinese Feishu
  reports, deterministic correctness checks, commit and push.
- Host will shut down about 03:00 Asia/Shanghai on 2026-10-09. Save all source
  and evidence remotely before that time.
- Branch: `feat/eagle3-fine-grained-overlap`; previous implementation `f8287b1248`.

## Completed

- Both Feishu documents accessible with `lark-cli --as user`.
- All 20 original full-graph cases (five configs, batches 1/8/16/32), two-rank
  torch traces, three throughput repeats and two request warmups.
- First document has Chinese tables, plot, representative trace, analysis JSON,
  kernel triage, and **38 MB full baseline evidence archive**.
- Previous README/REPORT/FINE_REPORT math corrected: asymmetric D/V/E recurrence,
  D_B -> E_A draft-stream dependency, graph fence and CPU enqueue vs GPU release.
- Graph replay equivalence, 46 graph/backend tests, 23 namespace/config tests,
  seven manual microbatch tests and two profiler-analysis tests passed.
- Runtime supports configurable cuts/release/split and greedy tree drafting.
  Private tree masks prevent B proposal overwriting A metadata inputs.
- Source patch + new scripts checkpoint attached to second Feishu document.

## Active and pending

- Normal rounds 1–6 completed. Original round 2 moved into
  `results/revision/diagnostics/02-tree-shared-mask` and excluded: tree-mask
  ownership was not isolated. Corrected round 2 rerun completed.
- Driver `run_refinements.py --rounds 2 7 8 9 10` is active; later rounds adapt to
  the best measured chain schedule. Owned process cleans up each server.
- After normal experiments: select a schedule, run deterministic token gates
  against full-batch original for all five configs; do not treat normal-kernel
  shape-dependent output mismatches as proof of correctness or failure.
- Finish Chinese second document and local follow-up report, archive ALL raw
  evidence (including superseded diagnostics), upload archives, commit/push.

## Evidence and commands

Local root: `benchmark/spec_overlap/results/revision/` (Git ignored).

```bash
python benchmark/spec_overlap/summarize_profiles.py --results benchmark/spec_overlap/results/revision --output benchmark/spec_overlap/results/revision/summary
python benchmark/spec_overlap/summarize_refinements.py --results benchmark/spec_overlap/results/revision/refinements --output benchmark/spec_overlap/results/revision/summary --require-complete
```

Remote docs:
- https://my.feishu.cn/docx/D1uPdOb4gozVGmxIdLQcyCQgnOb (baseline)
- https://my.feishu.cn/docx/BbutdflyNoq5oAx7p9Ycapi3nZg (prototype/refinement)

## Findings so far / guardrails

- Baseline `3-1-4` is fastest at every batch; batch 16 is 1918.4 tok/s.
- Old eight-cut overlap: 1309.6 tok/s; actual bidirectional GEMM/communication
  intersection 0.0361 ms across six profiled cycles, not critical-path time saved.
- Immediate release before target graph gives zero useful overlap in these runs.
- All split schedules remain slower than the original full batch so far.
- EP=1 uses TP AllReduce/AllGather, not All2All; GPT-OSS DeepEP forward unsupported.
- Keep graph extension-A / verification-B fence. No model graph/kernel fusion
  implemented; round 10 launches a whole target graph and separate draft graph.
- Default microbatch mode stays off. No unrelated GPU processes may be killed.
