#!/usr/bin/env python3
"""Collapse one `run_accuracy_test.sh` invocation into summary.json / summary.txt.

Reads one tab-separated record per suite on stdin — as emitted by the runner:

    slug<TAB>name<TAB>status<TAB>exit_code<TAB>log_path<TAB>xml_path

and folds in the test counts pytest wrote to each JUnit XML, so the summary
carries the per-test tally and the names of whatever failed, not just the
per-suite pass/fail the console prints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.etree import ElementTree

COUNT_KEYS = ("tests", "failures", "errors", "skipped")


def _read_junit(path: Path) -> dict:
    """Counts plus failing test ids from one pytest JUnit XML.

    Returns an empty dict when the file is missing or unparsable: a suite that
    died before pytest could write it still belongs in the summary.
    """
    if not path.is_file():
        return {}
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError:
        return {}
    suites = root.iter("testsuite")
    counts = {key: 0 for key in COUNT_KEYS}
    duration = 0.0
    failing: list[str] = []
    for suite in suites:
        for key in COUNT_KEYS:
            counts[key] += int(suite.get(key, 0))
        duration += float(suite.get("time", 0.0))
        for case in suite.iter("testcase"):
            if case.find("failure") is None and case.find("error") is None:
                continue
            classname = case.get("classname", "")
            name = case.get("name", "")
            failing.append(f"{classname}::{name}" if classname else name)
    counts["passed"] = max(
        0, counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    )
    return {**counts, "duration_s": round(duration, 3), "failing_tests": failing}


def _parse_records(text: str) -> list[dict]:
    suites = []
    for line in text.splitlines():
        if not line.strip():
            continue
        slug, name, status, exit_code, log, xml = line.split("\t")
        suites.append(
            {
                "suite": slug,
                "name": name,
                "status": status,
                "exit_code": int(exit_code),
                "log": log,
                "junit_xml": xml,
                **_read_junit(Path(xml)),
            }
        )
    return suites


def _render_table(suites: list[dict]) -> str:
    width = max((len(s["suite"]) for s in suites), default=0)
    lines = []
    for s in suites:
        counts = (
            f"{s.get('passed', 0)} passed"
            f", {s.get('failures', 0) + s.get('errors', 0)} failed"
            f", {s.get('skipped', 0)} skipped"
            if "tests" in s
            else "no JUnit XML"
        )
        lines.append(f"  {s['suite']:<{width}}  {s['status']:<7}  {counts}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--started-at", default="")
    p.add_argument("--pytest-args", default="")
    args = p.parse_args(argv)

    suites = _parse_records(sys.stdin.read())
    failed = [s["suite"] for s in suites if s["status"] == "failed"]
    summary = {
        "started_at": args.started_at,
        "pytest_args": args.pytest_args,
        "status": "failed" if failed else "passed",
        "failed_suites": failed,
        "totals": {
            key: sum(s.get(key, 0) for s in suites) for key in (*COUNT_KEYS, "passed")
        },
        "suites": suites,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = _render_table(suites)
    (args.out / "summary.txt").write_text(
        f"accuracy run {args.started_at}  ->  {summary['status']}\n{table}\n"
    )
    print()
    print(table)
    print(f"  wrote {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
