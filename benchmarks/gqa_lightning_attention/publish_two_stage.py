"""Append one completed two-stage experiment and verify preserved doc resources."""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "dsa_m3_attention"))
from publish_report import markdown_xml

DOC = "UiWfdNSWvoFzCixy1wUc58kBnBb"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--publish", action="store_true")
    p.add_argument("--cli", default="lark-cli")
    a = p.parse_args()
    e = a.experiment
    out = BASE / "results/two_stage" / f"experiment{e}"
    work = out / "document"
    work.mkdir(exist_ok=True)
    validation = json.loads((out / "validation.json").read_text())
    assert validation == dict(
        measurements=384, samples=5760, comparisons=96, charts=16, bar_labels=288
    )
    parts = [markdown_xml((out / "report_zh.md").read_text())]
    for ci in (1, 2):
        for phase in ("prefill", "decode"):
            label = f"两阶段实验 {e}／配置 {ci}／{phase}"
            content = f"<h2>{label}</h2>"
            for ctx in (16, 32, 64, 128):
                path = out / "charts" / f"config{ci}_{phase}_{ctx}k.png"
                content += f'<h3>{ctx}k 上下文</h3><img path={quoteattr("@"+str(path))} width="820" caption={quoteattr(label+f"，{ctx}k；柱顶为延迟中位数（ms）。")}/>'
            parts.append(content)
    bundle = out / f"two_stage_experiment{e}.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for path in list(BASE.glob("*.py")) + [BASE / "README.md"]:
            z.write(path, "source/" + path.name)
        for path in (
            list(out.glob("*.json"))
            + list(out.glob("*.csv"))
            + list(out.glob("*.md"))
            + list(out.glob("*.sh"))
            + [out / "charts/manifest.json"]
        ):
            z.write(path, path.relative_to(out))
    parts.append(
        f'<h2>两阶段实验 {e}：原始数据</h2><source path={quoteattr("@"+str(out/"comparison.csv"))} name="two_stage_experiment{e}.csv"/><source path={quoteattr("@"+str(bundle))} name="two_stage_experiment{e}.zip"/>'
    )
    for i, s in enumerate(parts):
        (work / f"part_{i:02}.xml").write_text(s)
    if not a.publish:
        print(f"Prepared {len(parts)} parts")
        return

    def call(args):
        proc = subprocess.run(
            [a.cli, *args, "--as", "user"], capture_output=True, text=True, check=True
        )
        d = json.loads(proc.stdout)
        assert d.get("ok") and not d.get("data", {}).get("warnings"), d
        return d

    def fetch(path):
        d = call(["docs", "+fetch", "--doc", DOC, "--detail", "full"])
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
        return d

    before_path = work / "before.json"
    before = (
        json.loads(before_path.read_text())
        if before_path.exists()
        else fetch(before_path)
    )
    done_path = work / "published_parts.json"
    done = json.loads(done_path.read_text()) if done_path.exists() else []
    if not done:
        assert (
            f"我们的两阶段索引器：实验 {e}" not in before["data"]["document"]["content"]
        )
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
        fetch(work / "latest.json")
        print(f"Published and fetched {i+1}/{len(parts)}", flush=True)
    after = fetch(work / "after.json")
    old = ET.fromstring("<doc>" + before["data"]["document"]["content"] + "</doc>")
    new = ET.fromstring("<doc>" + after["data"]["document"]["content"] + "</doc>")
    for tag, attr, increase in [("img", "src", 16), ("source", "token", 2)]:
        a0 = [x.get(attr) for x in old.iter(tag)]
        a1 = [x.get(attr) for x in new.iter(tag)]
        assert set(a0) <= set(a1) and len(a1) == len(a0) + increase, (
            tag,
            len(a0),
            len(a1),
        )
    assert "benchmark_two_stage.py" in after["data"]["document"]["content"]
    print(
        "Verified: preserved existing resources; +16 labeled charts, +2 attachments, reproduction commands.",
        flush=True,
    )


if __name__ == "__main__":
    main()
