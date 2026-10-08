# GPT-OSS overlap follow-up — completed record (2026-10-09)

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
  eight manual microbatch tests and two profiler-analysis tests passed.
- Runtime supports configurable cuts/release/split and greedy tree drafting.
  Private tree masks prevent B proposal overwriting A metadata inputs.
- Source patch + new scripts checkpoint attached to second Feishu document.

## Final status

- All ten normal refinement rounds completed. Second round rerun with private
  tree masks; superseded shared-mask data remains under diagnostics.
- Both Chinese reports contain tables/plots, branch-gap equations, GPU overlap
  evidence and accuracy/correctness results. Document readbacks verified.
- Selected experimental schedule: one target graph launch, release zero, split50%.
- All five configs at batches1/5/8/16/32 pass two64-token exact comparisons:
  25cases, 620sequences and100rank traces. Batch1 follows the original path.
- GPT-OSS router tinygemm cutoff caused the old config4 B32 mismatch; split serial
  and overlap agreed. Batch-invariant router bypass fixes the entire matrix.
- Final-source normal recheck: off1986.9, serial1299.7, overlap1315.9 tok/s;
  overlap+1.25% vs serial, -33.8% vs off. Each mode GSM8K19/20.
- Nine raw-evidence archives (321MB) are uploaded and verified, including all
  baseline/refinement/correctness/diagnostic/final-normal traces and outputs.
  REVISION_EVIDENCE.json records names, SHA256 hashes and attachment tokens.
- Source checkpoints e6e38692f2 and1d123103d4 already pushed; final report commit
  follows this record. Complete source bundle/patch and summary archive are also
  attached to the second document. No owned GPU server remains running.

## Evidence and commands

Local root: `benchmark/spec_overlap/results/revision/` (Git ignored).

```bash
python benchmark/spec_overlap/summarize_profiles.py --results benchmark/spec_overlap/results/revision --output benchmark/spec_overlap/results/revision/summary
python benchmark/spec_overlap/summarize_refinements.py --results benchmark/spec_overlap/results/revision/refinements --output benchmark/spec_overlap/results/revision/summary --require-complete
```

Remote docs:
- https://my.feishu.cn/docx/D1uPdOb4gozVGmxIdLQcyCQgnOb (baseline)
- https://my.feishu.cn/docx/BbutdflyNoq5oAx7p9Ycapi3nZg (prototype/refinement)

## Findings / guardrails

- Baseline `3-1-4` is fastest at every batch; batch 16 is 1918.4 tok/s.
- Round-one eight-cut overlap: 1309.6 tok/s; bidirectional GEMM/communication
  intersection 0.0361 ms across seven profiled cycles, not critical-path time saved.
- Immediate release before target graph gives zero useful overlap in these runs.
- All ten split schedules remain slower than the original full batch. Round ten
  is nominally fastest: 1324.2 tok/s (+1.1% vs round one, +1.3% vs own serial,
  -31.0% vs original). Three trials do not establish significance.
- EP=1 uses TP AllReduce/AllGather, not All2All; GPT-OSS DeepEP forward unsupported.
- Keep graph extension-A / verification-B fence. No model graph/kernel fusion
  implemented; round 10 launches a whole target graph and separate draft graph.
- Default microbatch mode stays off. No unrelated GPU processes may be killed.
