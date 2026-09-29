"""Validate a completed experiment, draw labeled bars, and write a Chinese report."""

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
CONTEXTS = (16384, 32768, 65536, 131072)
BATCHES = (1, 2, 4, 8, 16, 32)
VARIANTS = ("dense_fa2", "dense_fa3", "sparse", "two_stage")


def commands(experiment):
    return f"""# 在仓库根目录执行；两块空闲 H200，各跑一个配置。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q \\
  benchmarks/gqa_lightning_attention/test_attention.py \\
  benchmarks/gqa_lightning_attention/test_two_stage.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/gqa_lightning_attention/benchmark_two_stage.py \\
  --experiments {experiment} --configs 1 --contexts 16384,32768,65536,131072 \\
  --batches 1,2,4,8,16,32 --phases prefill,decode \\
  --variants dense_fa2,dense_fa3,sparse,two_stage --candidate-blocks 64 \\
  --prefill-budget 2048 --warmup 3 --iterations 15 &
PID1=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/gqa_lightning_attention/benchmark_two_stage.py \\
  --experiments {experiment} --configs 2 --contexts 16384,32768,65536,131072 \\
  --batches 1,2,4,8,16,32 --phases prefill,decode \\
  --variants dense_fa2,dense_fa3,sparse,two_stage --candidate-blocks 64 \\
  --prefill-budget 2048 --warmup 3 --iterations 15 &
PID2=$!
wait "$PID1"
wait "$PID2"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/gqa_lightning_attention/recall_two_stage.py --experiments {experiment}
.venv/bin/python benchmarks/gqa_lightning_attention/report_two_stage.py --experiment {experiment}"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=int, required=True, choices=[1, 2, 3])
    a = p.parse_args()
    e = a.experiment
    out = BASE / "results/two_stage" / f"experiment{e}"
    charts = out / "charts"
    charts.mkdir(exist_ok=True)
    data = []
    for ci in (1, 2):
        blob = json.loads((out / f"config{ci}.json").read_text())
        rs = blob["rows"]
        assert len(rs) == 192
        assert blob["metadata"]["arguments"]["warmup"] >= 3
        assert all(r["status"] == "ok" and len(r["samples_ms"]) == 15 for r in rs)
        assert (
            len({(r["phase"], r["context"], r["batch"], r["variant"]) for r in rs})
            == 192
        )
        data += rs
    lookup = {
        (r["config"], r["phase"], r["context"], r["batch"], r["variant"]): r
        for r in data
    }
    rows = []
    for ci, phase, context, batch in itertools.product(
        (1, 2), ("prefill", "decode"), CONTEXTS, BATCHES
    ):
        variants = {
            v: lookup[ci, phase, context, batch, v]["median_ms"] for v in VARIANTS
        }
        backend = min(("dense_fa2", "dense_fa3"), key=variants.get)
        dense = variants[backend]
        rows.append(
            dict(
                experiment=e,
                config=ci,
                phase=phase,
                context=context,
                batch=batch,
                dense_backend=backend,
                dense_ms=dense,
                **{v + "_ms": t for v, t in variants.items()},
                sparse_speedup=dense / variants["sparse"],
                two_stage_speedup=dense / variants["two_stage"],
                indexer_speedup=variants["sparse"] / variants["two_stage"],
            )
        )
    (out / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out / "comparison.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    manifest = []
    for ci, phase, context in itertools.product(
        (1, 2), ("prefill", "decode"), CONTEXTS
    ):
        points = [
            r
            for r in rows
            if (r["config"], r["phase"], r["context"]) == (ci, phase, context)
        ]
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(6)
        width = 0.25
        for i, (key, label, color) in enumerate(
            (
                ("dense_ms", "Dense: best FA2/FA3", "#4775ad"),
                ("sparse_ms", "Original sparse", "#d58b35"),
                ("two_stage_ms", "Two-stage sparse", "#42916d"),
            )
        ):
            bars = ax.bar(
                x + (i - 1) * width,
                [r[key] for r in points],
                width,
                label=label,
                color=color,
            )
            ax.bar_label(
                bars,
                labels=[f"{r[key]:.3f}" for r in points],
                padding=3,
                fontsize=8,
                rotation=0,
            )
        ax.set_xticks(x, [str(b) for b in BATCHES])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Median latency (ms)")
        ax.set_title(
            f"Experiment {e} | Config {ci} | {phase} | context {context//1024}k"
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.20)
        ax.legend(loc="upper left", ncol=3, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        path = charts / f"config{ci}_{phase}_{context//1024}k.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        manifest.append(
            dict(
                path=str(path.relative_to(out)),
                bars=18,
                labels=18,
                config=ci,
                phase=phase,
                context=context,
            )
        )
    (charts / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    heads = {
        1: "主注意力 Q/KV=48/4、64/8；索引器 Q/KV=48/4、64/8。",
        2: "主注意力 Q/KV=48/4、64/8；索引器 Q/KV=4/1、8/1。",
        3: "主注意力 Q/KV=48/1、64/1；索引器 Q/KV=4/1、8/1。",
    }[e]
    lines = [
        f"# 我们的两阶段索引器：实验 {e}",
        f"- {heads}",
        "- 新方案：每 128 token 缓存一个均值 key；粗排保留 64 个块（8192 个候选 token），再精排选 2048 token。当前块强制保留，精排逐 token 因果屏蔽。",
        "- 均值粗排为近似选择，不保证保留原始全局 top-k；未训练模型、未测任务质量。不能把性能提升直接等同于等质量加速。",
        "- 计时包含 Q/K/V 投影、归一化、写缓存、索引投影、更新受影响块的均值、两轮评分与 top-k、分页稀疏注意力。历史块初始化不计入当前步骤。",
        "- H200，BF16，hidden=8192，主头维度 128，索引头维度 64；最终 2048-token 总预算分块预填充；解码每请求 1 token、CUDA graph。不是整段 prompt 或整模型测量。",
        "- 每项先预热 3 次，再采样 15 次；每次计时前在计时区外清空 L2。稠密基线逐点取 FlashInfer FA2/FA3 中较快的中位数。",
        "- 本实验完成 384 项测量、5760 个原始样本、96 组比较、16 张柱状图；每根柱顶标出 ms。",
        "## 胜出数量（大于 1×）",
        "| 配置/阶段 | 原始稀疏胜出 | 两阶段胜出 | 两阶段相对稠密范围 | 两阶段快于原始稀疏 |",
        "|---|---:|---:|---:|---:|",
    ]
    summary = []
    for ci, phase in itertools.product((1, 2), ("prefill", "decode")):
        rs = [r for r in rows if r["config"] == ci and r["phase"] == phase]
        s = dict(
            config=ci,
            phase=phase,
            sparse_wins=sum(r["sparse_speedup"] > 1 for r in rs),
            two_stage_wins=sum(r["two_stage_speedup"] > 1 for r in rs),
            vs_original_wins=sum(r["indexer_speedup"] > 1 for r in rs),
            min_speedup=min(r["two_stage_speedup"] for r in rs),
            max_speedup=max(r["two_stage_speedup"] for r in rs),
        )
        summary.append(s)
        lines.append(
            f"| {ci}/{phase} | {s['sparse_wins']}/24 | {s['two_stage_wins']}/24 | {s['min_speedup']:.2f}–{s['max_speedup']:.2f}× | {s['vs_original_wins']}/24 |"
        )
    lines += [
        "## 128k、batch 32",
        "| 配置/阶段 | 稠密 ms | 原始稀疏 ms | 两阶段 ms | 稠密/两阶段 | 原始/两阶段 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["context"] == 131072 and r["batch"] == 32:
            lines.append(
                f"| {r['config']}/{r['phase']} | {r['dense_ms']:.3f} | {r['sparse_ms']:.3f} | {r['two_stage_ms']:.3f} | {r['two_stage_speedup']:.2f}× | {r['indexer_speedup']:.2f}× |"
            )
    lines += ["## 同时加速预填充和解码"]
    for ctx in CONTEXTS:
        good = []
        for b in BATCHES:
            rs = [r for r in rows if r["context"] == ctx and r["batch"] == b]
            if all(r["two_stage_speedup"] > 1.1 for r in rs):
                good.append(b)
        lines.append(
            f"- {ctx//1024}k：两配置、两阶段均超过 1.10× 的 batch："
            + (", ".join(map(str, good)) if good else "无")
            + "。"
        )
    recall = json.loads((out / "recall.json").read_text())["rows"]
    lines += [
        "## 候选召回检查",
        "- 固定随机种子，batch=1，采样最后 16 个 prefill query／1 个 decode query；以下只统计门控非全零的组。指标是候选保留的全局 top-2048 token ID 比例，不是模型准确率。",
    ]
    for ctx in (16384, 131072):
        rr = [r["recall"] for r in recall if r["context"] == ctx and r["active"]]
        lines.append(
            f"- {ctx//1024}k：平均 {statistics.mean(rr):.1%}，范围 {min(rr):.1%}–{max(rr):.1%}，{len(rr)} 个 query/group 样本。"
        )
    lines += [
        "- 零门控使分数全部相同，token ID 召回受并列排序影响，原始逐组数据另存。实际部署需要训练粗排策略并验证模型质量。",
        "## 复现命令",
        "```bash",
        commands(e),
        "```",
    ]
    (out / "report_zh.md").write_text("\n".join(lines) + "\n")
    (out / "reproduce.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + commands(e) + "\n"
    )
    (out / "reproduce.sh").chmod(0o755)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "validation.json").write_text(
        json.dumps(
            dict(
                measurements=len(data),
                samples=sum(len(r["samples_ms"]) for r in data),
                comparisons=len(rows),
                charts=len(manifest),
                bar_labels=sum(m["labels"] for m in manifest),
            ),
            indent=2,
        )
        + "\n"
    )
    print("\n".join(lines[:27]))


if __name__ == "__main__":
    main()
