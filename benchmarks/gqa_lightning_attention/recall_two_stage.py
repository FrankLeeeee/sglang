"""Sample global-top-k retention; synthetic recall is not model accuracy."""

import argparse
import gc
import json
from pathlib import Path

import torch
from attention import LightningIndexer
from benchmark_two_stage import build_case, ints


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiments", type=ints, default=[1, 2, 3])
    p.add_argument("--configs", type=ints, default=[1, 2])
    p.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results/two_stage"
    )
    a = p.parse_args()
    for experiment in a.experiments:
        rows = []
        for ci in a.configs:
            for context in (16384, 131072):
                for phase in ("prefill", "decode"):
                    qpr = 2048 if phase == "prefill" else 1
                    layer, state, x = build_case(
                        experiment, ci, 1, context, qpr, phase, "two_stage"
                    )
                    layer.forward(x, state)
                    q, _, w = layer.indexer.project(x, x)
                    # Last 16 queries (or the decode query), matching tile boundaries.
                    start = max(0, len(x) - 16)
                    exact = LightningIndexer.select(
                        layer.indexer, q, state.ik, w, state, start, len(x), 2048
                    ).cpu()
                    approx = layer.indexer.select(
                        q, state.ik, w, state, start, len(x), 2048
                    ).cpu()
                    active = (
                        w[start:]
                        .view(
                            -1,
                            layer.cfg.num_k_heads,
                            layer.cfg.indexer_q_heads // layer.cfg.num_k_heads,
                        )
                        .sum(-1)
                        > 0
                    ).T.cpu()
                    for g in range(exact.shape[0]):
                        for i in range(exact.shape[1]):
                            expected = set(exact[g, i].tolist()) - {-1}
                            chosen = set(approx[g, i].tolist()) - {-1}
                            rows.append(
                                dict(
                                    experiment=experiment,
                                    config=ci,
                                    context=context,
                                    phase=phase,
                                    query=start + i,
                                    group=g,
                                    active=bool(active[g, i]),
                                    recall=len(expected & chosen) / len(expected),
                                )
                            )
                    del layer, state, x, q, w, exact, approx
                    gc.collect()
                    torch.cuda.empty_cache()
        path = a.output / f"experiment{experiment}" / "recall.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                dict(
                    note="Seed 42 weights / 43 history / 44 input; batch 1; final 16 prefill queries or one decode query. Active means sum of nonnegative gates > 0. Tied zero scores make ID recall ambiguous. Not a quality benchmark.",
                    rows=rows,
                ),
                indent=2,
            )
            + "\n"
        )
        print(path, flush=True)


if __name__ == "__main__":
    main()
