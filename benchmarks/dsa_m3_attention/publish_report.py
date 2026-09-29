"""Append the validated Chinese report to Feishu, preserving existing blocks.

Generating drafts is the default; --publish explicitly enables remote writes.
"""

import argparse
import itertools
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
DOC = "UiWfdNSWvoFzCixy1wUc58kBnBb"


def markdown_xml(markdown):
    lines = markdown.splitlines()
    out = []
    i = 0
    while i < len(lines):
        s = lines[i]
        if s.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            out.append(
                '<pre lang="bash"><code>' + escape("\n".join(code)) + "</code></pre>"
            )
        elif s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            out.append(f"<h{level}>" + escape(s[level:].strip()) + f"</h{level}>")
        elif s.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append("<li>" + escape(lines[i][2:]) + "</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif s.startswith("|"):
            table = []
            while i < len(lines) and lines[i].startswith("|"):
                row = [x.strip() for x in lines[i].strip("|").split("|")]
                if not all(set(x) <= set("-: ") for x in row):
                    table.append(row)
                i += 1

            def tr(row, tag):
                return (
                    "<tr>"
                    + "".join(f"<{tag}><p>" + escape(x) + f"</p></{tag}>" for x in row)
                    + "</tr>"
                )

            out.append(
                "<table><thead>"
                + tr(table[0], "th")
                + "</thead><tbody>"
                + "".join(tr(r, "td") for r in table[1:])
                + "</tbody></table>"
            )
            continue
        elif s.strip():
            out.append("<p>" + escape(s) + "</p>")
        i += 1
    return "".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publish", action="store_true")
    p.add_argument("--cli", default="lark-cli")
    a = p.parse_args()
    work = OUT / "document"
    work.mkdir(exist_ok=True)
    parts = [markdown_xml((OUT / "report_zh.md").read_text())]
    for arch, ci, phase in itertools.product(
        ("dsa", "m3"), (1, 2), ("prefill", "decode")
    ):
        label = f"{arch.upper()}／配置 {ci}／" + (
            "分块预填充" if phase == "prefill" else "解码"
        )
        content = "<h2>" + label + "</h2>"
        for ctx in (16, 32, 64, 128):
            path = OUT / "charts" / f"{arch}_config{ci}_{phase}_{ctx}k.png"
            content += f'<h3>{ctx}k 上下文</h3><img path={quoteattr("@"+str(path))} width="820" caption={quoteattr(label+f'，{ctx}k；柱顶为延迟中位数（ms）。')}/>'
        parts.append(content)
    bundle = OUT / "native_sparse_evidence.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        paths = list(BASE.glob("*.py")) + [BASE / "README.md", BASE / "reproduce.sh"]
        paths += (
            list(OUT.glob("*.json"))
            + list(OUT.glob("*.csv"))
            + [
                OUT / "report_zh.md",
                OUT / "correctness.txt",
                OUT / "charts/manifest.json",
            ]
        )
        for path in paths:
            z.write(path, path.relative_to(BASE))
    parts.append(
        "<h2>DSA／MSA 数据与复现材料</h2><ul><li>CSV 包含全部 192 组配对及两个稠密后端；ZIP 包含 8640 个主扫描计时样本、独立阶段测量、内核分析和源码。</li><li>原生 DSA 使用 BF16 主缓存路径；FP8 主缓存性能不在本次结论范围内。</li></ul>"
        + f'<source path={quoteattr("@"+str(OUT/"comparison.csv"))} name="dsa_msa_comparison.csv"/>'
        + f'<source path={quoteattr("@"+str(bundle))} name="dsa_msa_evidence.zip"/>'
    )
    for i, content in enumerate(parts):
        (work / f"part_{i:02}.xml").write_text(content)
    if not a.publish:
        print(f"Wrote {len(parts)} XML drafts")
        return

    def call(args):
        proc = subprocess.run(
            [a.cli, *args, "--as", "user"], capture_output=True, text=True, check=True
        )
        d = json.loads(proc.stdout)
        assert d.get("ok"), d
        assert not d.get("data", {}).get("warnings"), d
        return d

    def fetch(name):
        d = call(["docs", "+fetch", "--doc", DOC, "--detail", "full"])
        (work / name).write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
        return d

    before = fetch("publish_before.json")
    done_path = work / "published_parts.json"
    done = json.loads(done_path.read_text()) if done_path.exists() else []
    if not done:
        assert (
            "原生 DSA / M3 稀疏流水线调查" not in before["data"]["document"]["content"]
        ), "Report already exists; review before appending"
    for i in range(len(parts)):
        if i in done:
            continue
        d = call(
            [
                "docs",
                "+update",
                "--doc",
                DOC,
                "--command",
                "append",
                "--content",
                "@" + str(work / f"part_{i:02}.xml"),
            ]
        )
        assert d["data"]["result"] == "success", d
        (work / f"write_{i:02}.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=2) + "\n"
        )
        done.append(i)
        done_path.write_text(json.dumps(done))
        fetch("latest.json")
        print(f"Published and fetched {i+1}/{len(parts)}", flush=True)
    after = fetch("after.json")

    def xml(d):
        return ET.fromstring("<doc>" + d["data"]["document"]["content"] + "</doc>")

    old = (
        xml(json.loads((work / "before.json").read_text()))
        if (work / "before.json").exists()
        else xml(before)
    )
    new = xml(after)
    for tag, attr, increase in [("img", "src", 32), ("source", "token", 2)]:
        oldtokens = [x.get(attr) for x in old.iter(tag)]
        newtokens = [x.get(attr) for x in new.iter(tag)]
        assert set(oldtokens) <= set(newtokens)
        assert len(newtokens) == len(oldtokens) + increase, (tag, len(newtokens))
    assert "--warmup 3 --iterations 15" in after["data"]["document"]["content"]
    print(
        "Verified: existing resources preserved, +32 charts, +2 attachments, reproduction commands.",
        flush=True,
    )


if __name__ == "__main__":
    main()
