"""CPU regressions for graph annotations and profiler interval accounting."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from analyze_results import duration, intersection, trace_summary, union

from sglang.test.test_utils import CustomTestCase


def annotation(name, start, duration_us, stream):
    return {
        "name": name,
        "cat": "gpu_user_annotation",
        "pid": 0,
        "tid": stream,
        "ts": start,
        "dur": duration_us,
    }


def kernel(name, start, duration_us, stream):
    event = annotation(name, start, duration_us, stream)
    event["cat"] = "kernel"
    return event


class TestProfileAnalysis(CustomTestCase):
    def test_graph_duplicates_do_not_bias_stage_time_or_hide_overlap(self):
        # Captured graph annotations repeat on internal streams. Use the outer
        # range for stage timing, but retain internal ranges for kernel ownership.
        events = [
            annotation("draft_extend", 100, 200, 94),  # prefill only
            annotation("draft", 1000, 1000, 94),
            annotation("draft", 1600, 200, 516),
            annotation("verify", 1500, 1000, 94),
            annotation("step[VERIFY bs=16]", 1550, 900, 296),
            annotation("draft_extend", 2600, 300, 94),
            kernel("matmul", 1610, 80, 516),
            kernel("matmul", 1650, 100, 516),
            kernel("nccl_AllReduce", 1680, 50, 296),
            kernel("target_elementwise", 1500, 100, 94),
            kernel("target_elementwise", 1550, 100, 94),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump({"traceEvents": events}, stream)
            result = trace_summary(path)
        stages = result["stage_ranges"]
        self.assertEqual(stages["draft"]["count"], 1)
        self.assertEqual(stages["draft"]["median_ms"], 1.0)
        self.assertEqual(stages["draft_extend"]["count"], 1)
        self.assertEqual(stages["draft_extend"]["median_ms"], 0.3)
        self.assertAlmostEqual(stages["verify"]["median_kernel_union_ms"], 0.15)
        self.assertAlmostEqual(stages["verify"]["median_unoccupied_stream_ms"], 0.85)
        self.assertAlmostEqual(
            result["draft_gemm_target_communication_intersection_ms"], 0.05
        )
        self.assertAlmostEqual(
            result["draft_compute_target_communication_intersection_ms"], 0.05
        )
        self.assertAlmostEqual(result["compute_kernel_union_ms"], 0.25)
        self.assertAlmostEqual(result["communication_kernel_union_ms"], 0.05)

    def test_interval_unions_count_overlapping_kernels_once(self):
        # Nested, repeated and adjacent intervals can occur on graph streams.
        intervals = [(8, 12), (0, 5), (2, 4), (4, 8), (8, 12), (15, 20)]
        self.assertEqual(union(intervals), [(0, 12), (15, 20)])
        self.assertEqual(duration(intervals), 17)
        self.assertEqual(intersection(intervals, [(3, 10), (5, 8), (18, 22)]), 9)
        self.assertEqual(intersection(intervals, []), 0)


if __name__ == "__main__":
    unittest.main()
