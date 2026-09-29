"""Validate complete sweeps and generate comparisons, labeled bars and Chinese report."""

import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
CONTEXTS = [16384, 32768, 65536, 131072]
BATCHES = [1, 2, 4, 8, 16, 32]


def main():
    pairs = []
    allrows = []
    datasets = {}
    for arch in ("dsa", "m3"):
        data = json.loads((OUT / f"{arch}.json").read_text())
        datasets[arch] = data
        rows = data["rows"]
        allrows += rows
        expected = set(
            itertools.product(
                [1, 2],
                ["prefill", "decode"],
                CONTEXTS,
                BATCHES,
                ["dense_fa2", "dense_fa3", "sparse"],
            )
        )
        lut = {
            (r["config"], r["phase"], r["context"], r["batch"], r["variant"]): r
            for r in rows
        }
        assert len(lut) == len(rows) == 288 and set(lut) == expected
        assert all(r["status"] == "ok" and len(r["samples_ms"]) == 15 for r in rows)
        for ci, phase, ctx, b in itertools.product(
            [1, 2], ["prefill", "decode"], CONTEXTS, BATCHES
        ):
            fa2, fa3, sp = [
                lut[ci, phase, ctx, b, v] for v in ("dense_fa2", "dense_fa3", "sparse")
            ]
            dense = min((fa2, fa3), key=lambda r: r["median_ms"])
            pairs.append(
                dict(
                    architecture=arch,
                    config=ci,
                    phase=phase,
                    context=ctx,
                    batch=b,
                    dense_fa2_ms=fa2["median_ms"],
                    dense_fa3_ms=fa3["median_ms"],
                    dense_backend=dense["variant"],
                    dense_best_ms=dense["median_ms"],
                    sparse_ms=sp["median_ms"],
                    speedup=dense["median_ms"] / sp["median_ms"],
                    separated_samples=sp["p90_ms"]
                    < min(min(fa2["samples_ms"]), min(fa3["samples_ms"])),
                )
            )
    with (OUT / "comparison.csv").open("w") as f:
        w = csv.DictWriter(f, list(pairs[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(pairs)
    (OUT / "comparison.json").write_text(json.dumps(pairs, indent=2) + "\n")
    summary = []
    both = []
    for arch, ci in itertools.product(("dsa", "m3"), (1, 2)):
        for phase in ("prefill", "decode"):
            rs = [
                r
                for r in pairs
                if (r["architecture"], r["config"], r["phase"]) == (arch, ci, phase)
            ]
            summary.append(
                dict(
                    architecture=arch,
                    config=ci,
                    phase=phase,
                    wins=sum(r["speedup"] > 1 for r in rs),
                    separated_wins=sum(r["separated_samples"] for r in rs),
                    points=len(rs),
                    min_speedup=min(r["speedup"] for r in rs),
                    max_speedup=max(r["speedup"] for r in rs),
                )
            )
        for ctx in CONTEXTS:

            def win(b, threshold):
                return all(
                    next(
                        r
                        for r in pairs
                        if (
                            r["architecture"],
                            r["config"],
                            r["phase"],
                            r["context"],
                            r["batch"],
                        )
                        == (arch, ci, p, ctx, b)
                    )["speedup"]
                    > threshold
                    for p in ("prefill", "decode")
                )

            both.append(
                dict(
                    architecture=arch,
                    config=ci,
                    context=ctx,
                    both_faster_batches=[b for b in BATCHES if win(b, 1)],
                    both_10pct_faster_batches=[b for b in BATCHES if win(b, 1.1)],
                )
            )
    (OUT / "summary.json").write_text(
        json.dumps(dict(phases=summary, both_phases=both), indent=2) + "\n"
    )
    charts = OUT / "charts"
    charts.mkdir(exist_ok=True)
    manifest = []
    for arch, ci, phase, ctx in itertools.product(
        ("dsa", "m3"), (1, 2), ("prefill", "decode"), CONTEXTS
    ):
        rs = [
            r
            for r in pairs
            if (r["architecture"], r["config"], r["phase"], r["context"])
            == (arch, ci, phase, ctx)
        ]
        x = np.arange(6)
        fig, ax = plt.subplots(figsize=(11, 4.8))
        for off, key, label, color, hatch in [
            (-0.19, "dense_best_ms", "Dense: faster of FA2 / FA3", "#3274a1", None),
            (0.19, "sparse_ms", "Sparse: full pipeline", "#dea139", "//"),
        ]:
            bars = ax.bar(
                x + off,
                [r[key] for r in rs],
                0.36,
                label=label,
                color=color,
                hatch=hatch,
            )
            ax.bar_label(
                bars, labels=[f"{r[key]:.3f}" for r in rs], padding=3, fontsize=9
            )
        ax.set_xticks(x, [str(b) for b in BATCHES])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Median latency (ms); lower is better")
        ax.set_ylim(0, max(max(r["dense_best_ms"], r["sparse_ms"]) for r in rs) * 1.3)
        ax.set_title(
            f"{arch.upper()} | Config {ci} | {phase} | {ctx//1024}k context\n"
            + (
                "Final chunk: 2048 total query tokens, eager"
                if phase == "prefill"
                else "1 query / request, CUDA graph replay"
            )
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        name = f"{arch}_config{ci}_{phase}_{ctx//1024}k.png"
        fig.savefig(charts / name, dpi=160)
        plt.close(fig)
        manifest.append(
            dict(
                architecture=arch,
                config=ci,
                phase=phase,
                context=ctx,
                file=name,
                values=rs,
            )
        )
    (charts / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    overall = {}
    for arch in ("dsa", "m3"):
        overall[arch] = {}
        for phase in ("prefill", "decode"):
            rs = [
                r for r in summary if r["architecture"] == arch and r["phase"] == phase
            ]
            overall[arch][phase] = sum(r["wins"] for r in rs)
    long_batches = {}
    for arch in ("dsa", "m3"):
        rows = [r for r in both if r["architecture"] == arch and r["context"] == 131072]
        long_batches[arch] = (
            "、".join(
                map(
                    str,
                    sorted(
                        set(rows[0]["both_10pct_faster_batches"])
                        & set(rows[1]["both_10pct_faster_batches"])
                    ),
                )
            )
            or "无"
        )
    lines = [
        "# 原生 DSA / M3 稀疏流水线调查",
        "",
        "## 结论",
        "",
        "- 稀疏注意力能同时加速预填充和解码，但收益取决于上下文、批量及架构；本次两种实现均未覆盖全部网格。",
        f"- DSA：预填充胜出 {overall['dsa']['prefill']}/48，解码胜出 {overall['dsa']['decode']}/48。",
        f"- M3：预填充胜出 {overall['m3']['prefill']}/48，解码胜出 {overall['m3']['decode']}/48。",
        f"- 128k 下，两配置的两阶段均超过 1.10× 的批量：DSA 为 {long_batches['dsa']}；M3 为 {long_batches['m3']}。",
        "- 小批量解码的索引与投影开销难以摊薄；长上下文／大批量更容易获得稳定收益。",
        "",
        "## 方法与适用范围",
        "",
        "- 历史实验 3 的稀疏预填充已全部胜出；本节进一步检查原生 DSA 与 M3 流水线。",
        "- 新增 576 项测量、8640 个计时样本；两架构、两配置、两阶段、4 个上下文、6 个批量、3 种内核路径。",
        "- 稠密基线逐点取 FA2、FA3 延迟中位数的较小值；加速比 = 最快稠密 ÷ 稀疏。",
        "- 预填充为最后一个分块：全批新增 2048 token、eager；解码每请求 1 token、CUDA Graph。",
        "- 均计入投影、归一化、缓存写入；稀疏还计入实际索引器、评分、top-k、索引转换及注意力。",
        "- 隐藏维度 8192，主 Q 头数 48／64；H200、TP=1、BF16 主缓存、随机物理页；预热 3 次、采样 15 次。",
        "- DSA：原生完整 Indexer，64Q/1KV、维度 128、FP8 索引缓存；MLA 主缓存 512+64、页长 64；选取 2048 token。",
        "- DSA 主路径采用 rank-1536 查询投影、128+64 查询维度及 128→512 权重吸收；48 个主 Q 头填充至 64，开销计入。",
        "- M3：主注意力 48Q/4KV 或 64Q/8KV、维度 128；索引器 4Q/1KV 或 8Q/1KV、维度 128；选 16×128 token 块，含本地块。",
        "- M3 调用原生块评分／选择／注意力流水线；主 Q/K/V 和索引 Q/K 使用标准化投影及 RMSNorm，省略模型 RoPE 与 Gemma 增益。",
        "- 本节是原生内核配合显式投影适配器的合成测试，并非完整模型层；不含输出投影、MLP、通信和调度。",
        "- DSA 与 M3 的维度、页大小及语义不同；仅比较各自稠密／稀疏配对，不把跨架构绝对延迟视为同模型对比。",
        "",
        "## 完整网格结果",
        "",
        "| 架构 | 配置 | 阶段 | 稀疏胜出 | 加速比范围 | 样本明显分离的胜出点 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['architecture'].upper()} | {r['config']} | {'预填充' if r['phase']=='prefill' else '解码'} | {r['wins']}/24 | {r['min_speedup']:.2f}–{r['max_speedup']:.2f}× | {r['separated_wins']}/24 |"
        )
    lines += [
        "",
        "- “样本明显分离”指稀疏 P90 低于两条稠密路径的最小样本；这是保守波动检查，不是统计置信区间。",
        "",
        "## 两阶段同时更快的测试点",
        "",
        "| 架构 | 配置 | 上下文 | 两阶段均加速的批量 | 两阶段均超过 1.10× 的批量 |",
        "|---|---:|---:|---|---|",
    ]
    for r in both:
        fmt = lambda xs: "、".join(map(str, xs)) or "无"
        lines.append(
            f"| {r['architecture'].upper()} | {r['config']} | {r['context']//1024}k | {fmt(r['both_faster_batches'])} | {fmt(r['both_10pct_faster_batches'])} |"
        )
    lines += [
        "",
        "## 128k、批量 32",
        "",
        "| 架构 | 配置 | 阶段 | 最快稠密 ms | 稀疏 ms | 加速比 | 稠密后端 |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for r in pairs:
        if r["context"] == 131072 and r["batch"] == 32:
            lines.append(
                f"| {r['architecture'].upper()} | {r['config']} | {'预填充' if r['phase']=='prefill' else '解码'} | {r['dense_best_ms']:.3f} | {r['sparse_ms']:.3f} | {r['speedup']:.2f}× | {r['dense_backend']} |"
            )
    lines += [
        "",
        "- 不能把某些长上下文／大批量下的优势外推至全部请求；需同时检查预填充和解码。",
        "- 随机权重只用于性能测试；块稀疏、token 稀疏及改变头数均需另做训练模型质量验证。",
        "- 柱状图保留全部批量，柱顶标注毫秒数；纵轴从 0 开始，但图间范围不同。",
        "",
    ]
    lines += [
        "## 开销定位",
        "",
        "- DSA 在 16k／批量 1 下，单独测得索引器约 0.040–0.042 ms、稀疏注意力约 0.061–0.063 ms；解码的固定开销会抵消稀疏收益。",
        "- DSA 在 128k／批量 32 的预填充中，单独测得索引器约 6.6 ms、稀疏注意力约 1.2–1.4 ms；索引器成为主要开销。",
        "- M3 使用连续块，选择空间和不连续读取更少；本次长上下文下能在两个阶段获益。",
        "- DSA 的稠密基线使用 512 维潜在表示；其预填充加速比不能直接套用于原来 128 维 GQA。",
        "- 独立组件计时同样先预热 3 次、采样 15 次；各组件分别清刷 L2，因此不可相加为端到端延迟。",
        "",
        "### 128k／批量 32 的预填充 CUDA 内核分解",
        "",
        "| 架构 | 配置 | 评分 ms | 选择 ms | 稀疏注意力 ms | 其他 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arch in ("dsa", "m3"):
        prof = json.loads((OUT / f"{arch}_profile.json").read_text())
        assert len(prof) == 16
        for r in prof:
            if (r["phase"], r["context"], r["batch"]) == ("prefill", 131072, 32):
                d = r["kernel_stage_ms"]
                lines.append(
                    f"| {arch.upper()} | {r['config']} | {d.get('index_scoring',0):.3f} | {d.get('selection',0):.3f} | {d.get('sparse_attention',0):.3f} | {d.get('projection_cache_other',0):.3f} |"
                )
    lines += [
        "",
        "- 此表为 eager profiler 的 CUDA 内核耗时，仅用于定位开销；预热 3 次后分析 3 次，不替代主表的无 profiler 延迟。",
        "- 其他项包含投影、归一化、缓存读写及转换；不同架构头数／维度不同，不能把差异全部归因于块选择。",
        "- 环境使用 NVCC 12.8；DeepGEMM 提示建议 12.9+，结果限定于本次软件环境与 BF16 主缓存路径。",
        "",
        "## 复现命令",
        "",
        "- 在仓库根目录执行；先安装本仓库及 PyTorch、FlashInfer、sgl-kernel、DeepGEMM、Triton、matplotlib、pytest。",
        "- 每项测量强制至少预热 3 次；预热、JIT、自动调优与 CUDA Graph 捕获均不计入 15 个正式样本。",
        "- 下列两条完整扫描可分别在两张空闲 H200 上运行；单卡环境将 CUDA_VISIBLE_DEVICES 都改为 0，顺序执行。",
        "",
    ]
    commands = (BASE / "reproduce.sh").read_text()
    lines += [
        "```bash",
        commands.rstrip(),
        "```",
        "",
        "- 主扫描输出 DSA／M3 的全部 576 项测量；诊断扫描另存，报告和 32 张带数值柱状图由原始数据生成。",
        "- 原始样本、两个稠密后端及源码哈希见附件；中断后可用 --resume 继续同一套参数。",
        "",
    ]
    (OUT / "report_zh.md").write_text("\n".join(lines))
    validation = dict(
        measurements=len(allrows),
        samples=sum(len(r["samples_ms"]) for r in allrows),
        comparisons=len(pairs),
        charts=len(manifest),
        bar_labels=len(manifest) * 12,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in BASE.glob("*.py")
        },
    )
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(
        json.dumps(
            dict(validation=validation, phases=summary, both_phases=both), indent=2
        )
    )


if __name__ == "__main__":
    main()
