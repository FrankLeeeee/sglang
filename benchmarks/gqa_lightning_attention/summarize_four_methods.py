"""Cross-reference all four methods; each keeps its own matched dense baseline."""

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "results/two_stage"


def publish(cli):
    from publish_two_stage import DOC, markdown_xml

    work = OUT / "document"
    work.mkdir(exist_ok=True)
    draft = work / "overview.xml"
    draft.write_text(markdown_xml((OUT / "four_methods_zh.md").read_text()))

    def call(args):
        result = subprocess.run(
            [cli, *args, "--as", "user"], capture_output=True, text=True, check=True
        )
        d = json.loads(result.stdout)
        assert d.get("ok") and not d.get("data", {}).get("warnings"), d
        return d

    before = call(["docs", "+fetch", "--doc", DOC, "--detail", "full"])
    assert (
        "四类方案最终总览" not in before["data"]["document"]["content"]
    ), "Already published; review existing overview"
    (work / "before.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=2) + "\n"
    )
    result = call(
        [
            "docs",
            "+update",
            "--doc",
            DOC,
            "--command",
            "block_insert_after",
            "--block-id",
            "0",
            "--content",
            "@" + str(draft),
        ]
    )
    assert result["data"]["result"] == "success", result
    (work / "write.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    after = call(["docs", "+fetch", "--doc", DOC, "--detail", "full"])
    (work / "after.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=2) + "\n"
    )
    old = ET.fromstring("<doc>" + before["data"]["document"]["content"] + "</doc>")
    new = ET.fromstring("<doc>" + after["data"]["document"]["content"] + "</doc>")
    for tag, attr in [("img", "src"), ("source", "token")]:
        assert [x.get(attr) for x in old.iter(tag)] == [
            x.get(attr) for x in new.iter(tag)
        ]
    assert "四类方案最终总览" in after["data"]["document"]["content"]
    assert len(list(new.iter("img"))) == 128
    assert len(list(new.iter("source"))) == 13
    print("Published overview; verified all 128 charts and 13 attachments preserved.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publish", action="store_true")
    p.add_argument("--cli", default="lark-cli")
    args = p.parse_args()
    native = json.loads(
        (BASE.parent / "dsa_m3_attention/results/comparison.json").read_text()
    )
    groups = []
    for arch, name in [("dsa", "DSA"), ("m3", "MSA（MiniMax M3）")]:
        groups.append(
            (
                name,
                [
                    dict(r, speed=r["speedup"])
                    for r in native
                    if r["architecture"] == arch
                ],
            )
        )
    for experiment in (1, 2, 3):
        rows = json.loads((OUT / f"experiment{experiment}/comparison.json").read_text())
        for variant, label in [("sparse", "原始稀疏"), ("two_stage", "两阶段索引")]:
            groups.append(
                (
                    f"{label}／实验 {experiment}",
                    [dict(r, speed=r[variant + "_speedup"]) for r in rows],
                )
            )
    lines = [
        "# 四类方案最终总览",
        "- 全网格：16k／32k／64k／128k × batch 1／2／4／8／16／32；每类均覆盖两个配置及 prefill/decode。",
        "- DSA、MSA、新测的原始稀疏与两阶段索引均先预热 3 次，再记录 15 个正式样本；共 1728 项主测量、25920 个计时样本。",
        "- 新增 80 张柱状图均在柱顶标明延迟；文档中另保留此前 48 张历史图。每节包含可复制命令与 CSV/ZIP 数据。",
        "- 最新比较逐点使用 FA2/FA3 中更快的稠密后端；历史 FA2-only 表不能与新胜出数量直接混用。",
        "- 以下加速比各自除以匹配架构的稠密延迟。DSA 为 MLA 512+64，其他为 128 维 GQA/MQA；不表示相同模型的横向绝对性能。",
        "## 128k、batch 32：相对匹配稠密基线的加速比",
        "| 方案 | 配置 1 prefill | 配置 1 decode | 配置 2 prefill | 配置 2 decode |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, rs in groups:
        speeds = []
        for ci, phase in [(1, "prefill"), (1, "decode"), (2, "prefill"), (2, "decode")]:
            r = next(
                r
                for r in rs
                if r["config"] == ci
                and r["phase"] == phase
                and r["context"] == 131072
                and r["batch"] == 32
            )
            speeds.append(f"{r['speed']:.2f}×")
        lines.append("| " + name + " | " + " | ".join(speeds) + " |")
    lines += [
        "## 全网格胜出数量",
        "| 方案 | prefill 胜出/48 | decode 胜出/48 | 128k 两配置、两阶段均 >1.10× 的 batch |",
        "|---|---:|---:|---|",
    ]
    for name, rs in groups:
        counts = [
            sum(r["speed"] > 1 for r in rs if r["phase"] == phase)
            for phase in ("prefill", "decode")
        ]
        batches = []
        for b in (1, 2, 4, 8, 16, 32):
            selected = [r for r in rs if r["context"] == 131072 and r["batch"] == b]
            assert len(selected) == 4
            if all(r["speed"] > 1.1 for r in selected):
                batches.append(str(b))
        lines.append(
            f"| {name} | {counts[0]}/48 | {counts[1]}/48 | {', '.join(batches) or '无'} |"
        )
    lines += [
        "## 结论与限制",
        "- 可以在长上下文／适当 batch 同时获得 prefill 和 decode 加速；本次没有方案在全部测量点均胜出。",
        "- 两阶段把全序列 token 精排缩为 8192 个候选，可降低长序列选择开销；短上下文时额外粗排与索引转换可能更慢。",
        "- 不能把两阶段速度优势直接当作等质量提升：当前粗排只使用未训练的块均值，可能漏掉大量原始 top-k。",
    ]
    for experiment in (1, 2, 3):
        rr = json.loads((OUT / f"experiment{experiment}/recall.json").read_text())[
            "rows"
        ]
        values = [r["recall"] for r in rr if r["context"] == 131072 and r["active"]]
        lines.append(
            f"- 实验 {experiment} 在 128k 的有效门控组平均 top-k ID 召回为 {sum(values)/len(values):.1%}；随机权重、batch=1、少量 query 诊断，不是模型质量指标。"
        )
    lines += [
        "- DSA／MSA／原始稀疏也未做训练模型质量评估；头数、精度、选择策略变化均需独立质量验证。",
        "- 所有结果是单层分页执行路径：prefill 为最终 2048-token 总预算 chunk，decode 使用 CUDA graph；不含输出投影、MLP、网络通信、调度或整模型吞吐。",
        "- 正确性：原始注意力 94 项、两阶段 14 项、DSA/MSA 12 项测试通过；阶段耗时分析只作诊断，不替代主表延迟。",
        "## 统一复现入口",
        "- 在仓库根目录、已安装依赖的环境执行。脚本内包含完整上下文、batch、预热和采样参数；每个脚本结束后再执行下一项，避免 GPU 争用。",
        "```bash",
        "bash benchmarks/dsa_m3_attention/reproduce.sh",
        "bash benchmarks/gqa_lightning_attention/results/two_stage/experiment1/reproduce.sh",
        "bash benchmarks/gqa_lightning_attention/results/two_stage/experiment2/reproduce.sh",
        "bash benchmarks/gqa_lightning_attention/results/two_stage/experiment3/reproduce.sh",
        ".venv/bin/python benchmarks/gqa_lightning_attention/summarize_four_methods.py",
        "```",
    ]
    (OUT / "four_methods_zh.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.publish:
        publish(args.cli)


if __name__ == "__main__":
    main()
