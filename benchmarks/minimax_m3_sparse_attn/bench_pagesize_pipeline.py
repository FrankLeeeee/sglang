#!/usr/bin/env python3
"""page_size x context sweep of the FULL sparse pipeline, two step-3 kernels.

This predates the paged step-3 kernel becoming the default. Back then
``minimax_sparse_prefill`` had no ``page_size`` parameter at all, so on the
prefill path the value only reached ``build_page_table`` (where tokens
physically sit) and never reached a kernel -- which is why
``bench_kernels.py --sweeps page_size`` reported a flat line for block-sparse
attention. That sweep measures the *layout* channel, which is worth ~0 on its
own. ``minimax_sparse_prefill`` now takes ``page_size`` and routes to the paged
kernel by default (``SGLANG_OPT_USE_MINIMAX_SPARSE_PAGED_PREFILL``), so the flat line no
longer holds there; this script still isolates the two channels directly.

This script runs the same three-stage pipeline through bench_kernels' own
``make_prefill_fn`` + ``profile_breakdown``, sweeping page_size AND context
length against two step-3 kernels:

  orig    flash_prefill_with_gqa_share_sparse          (page_size never used)
  paged   flash_prefill_with_gqa_share_sparse_paged    (uses it when
                                                        block_size | page_size)

so the layout channel and the code-path channel can be told apart in the units
bench_kernels reports. Both variants see the *same* input tensors at each cell.

    CUDA_VISIBLE_DEVICES=1 python bench_pagesize_pipeline.py --wait-for-idle 5

Outputs (default --out results/pagesize_context_sweep): raw.json, summary.csv,
speedup_heatmap.png, latency_vs_pagesize.png, latency_vs_context.png,
stage_breakdown.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_kernels import make_prefill_fn  # noqa: E402  (identical call path)
from harness import (  # noqa: E402
    STAGE_ORDER,
    bench_cuda,
    build_prefill_inputs,
    gpu_info,
    profile_breakdown,
    wait_for_idle,
    warn_if_contended,
    write_results,
)
from m3_config import m3_config  # noqa: E402

import sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse as MS  # noqa: E402
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse_paged import (  # noqa: E402
    flash_prefill_with_gqa_share_sparse_paged as PAGED,
)

_ORIG_MAIN = MS.flash_prefill_with_gqa_share_sparse
DEFAULT_PAGES = [1, 8, 16, 32, 64, 128, 256]
DEFAULT_CTXS = [4096, 8192, 16384, 32768, 65536, 131072, 524288, 1048576]
VARIANTS = ("orig", "paged")


def _ctx_label(n: int) -> str:
    return f"{n // 1024}k" if n < 1 << 20 else f"{n >> 20}M"


def _select(variant: str, page_size: int) -> None:
    """Swap step 3 of the pipeline. The call site is all-keyword, so a thin
    wrapper can bind page_size without touching minimax_sparse.py."""
    if variant == "orig":
        MS.flash_prefill_with_gqa_share_sparse = _ORIG_MAIN
    else:
        MS.flash_prefill_with_gqa_share_sparse = lambda **kw: PAGED(
            **kw, page_size=page_size
        )


def _attn_out(x):
    """minimax_sparse_prefill returns (idx_o, o) -- the INDEXER output first and
    the main attention output second (idx_o is None on M3, whose index pool is
    K-only). Take the attention output."""
    return (x[-1] if isinstance(x, tuple) else x).float()


def run_cell(cfg, page_size, ctx, chunk, iters, profile_iters) -> list[dict]:
    """Both variants on one shared set of inputs."""
    inp = build_prefill_inputs(cfg, batch_size=1, context_len=ctx, chunk_len=chunk)
    fast = page_size >= cfg.block_size and page_size % cfg.block_size == 0
    out_rows, base_ms, ref, floor = [], None, None, None

    for variant in VARIANTS:
        _select(variant, page_size)
        fn = make_prefill_fn(cfg, inp)
        first = _attn_out(fn())
        timing = bench_cuda(fn, warmup=max(3, iters // 4), iters=iters)
        stages, _ = profile_breakdown(fn, iters=profile_iters, warmup=3)
        if variant == "orig":
            base_ms = timing.median_ms
            ref = first.clone()
            # The full pipeline is NOT run-to-run deterministic: top-k block
            # selection is unstable under score ties, so two identical calls
            # already differ. Measure that floor so the cross-variant diff below
            # is interpretable rather than alarming.
            floor = float((_attn_out(fn()) - ref).abs().max().item())
        row = {
            "variant": variant,
            "page_size": page_size,
            "context_len": ctx,
            "context_label": _ctx_label(ctx),
            "chunk_len": chunk,
            "block_size": cfg.block_size,
            "fast_path": variant == "paged" and fast,
            "latency_median_ms": round(timing.median_ms, 6),
            "latency_min_ms": round(timing.min_ms, 6),
            "latency_p90_ms": round(timing.p90_ms, 6),
            "speedup_vs_orig": round(base_ms / timing.median_ms, 4),
            "max_abs_diff_vs_orig": float((first - ref).abs().max().item()),
            "nondeterminism_floor": floor,
            "status": "ok",
        }
        for st in STAGE_ORDER:
            row[f"stage_{st}_ms"] = round(stages.get(st, 0.0), 6)
        out_rows.append(row)
        del fn, first
    del inp, ref
    torch.cuda.empty_cache()
    return out_rows


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def make_plots(rows, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pages = sorted({r["page_size"] for r in rows})
    ctxs = sorted({r["context_len"] for r in rows})
    lut = {(r["variant"], r["page_size"], r["context_len"]): r for r in rows}

    # --- 1. speedup heatmap -------------------------------------------------
    grid = np.full((len(pages), len(ctxs)), np.nan)
    for i, ps in enumerate(pages):
        for j, c in enumerate(ctxs):
            r = lut.get(("paged", ps, c))
            if r:
                grid[i, j] = r["speedup_vs_orig"]
    fig, ax = plt.subplots(figsize=(1.15 * len(ctxs) + 3, 0.62 * len(pages) + 2.6))
    vmax = float(np.nanmax(grid))
    im = ax.imshow(grid, cmap="BuGn", vmin=1.0, vmax=max(vmax, 1.02), aspect="auto")
    ax.set_xticks(range(len(ctxs)), [_ctx_label(c) for c in ctxs])
    ax.set_yticks(range(len(pages)), [str(p) for p in pages])
    ax.set_xlabel("context length")
    ax.set_ylabel("page_size")
    ax.set_title(
        "paged vs original — full sparse prefill layer\n"
        "(fast path needs block_size 128 | page_size)",
        fontsize=11,
    )
    for i in range(len(pages)):
        for j in range(len(ctxs)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="#999", fontsize=9)
                continue
            hot = grid[i, j] > 1.05
            ax.text(
                j,
                i,
                f"{grid[i, j]:.2f}×",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if grid[i, j] > 1.22 else "#222",
                fontweight="bold" if hot else "normal",
            )
    fig.colorbar(im, ax=ax, label="speedup vs orig")
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_heatmap.png", dpi=140)
    plt.close(fig)

    # --- 2. latency vs context ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    off = [p for p in pages if not (p >= 128 and p % 128 == 0)]
    on = [p for p in pages if p >= 128 and p % 128 == 0]
    for label, ps_list, color, style in [
        ("orig (any page_size)", pages, "#5145CC", "-"),
        ("paged, fast path OFF", off, "#C25708", "--"),
        ("paged, fast path ON", on, "#0093A3", "-"),
    ]:
        variant = "orig" if label.startswith("orig") else "paged"
        ys = []
        for c in ctxs:
            vals = [
                lut[(variant, p, c)]["latency_median_ms"]
                for p in ps_list
                if (variant, p, c) in lut
            ]
            ys.append(np.median(vals) if vals else np.nan)
        ax.plot(ctxs, ys, style, color=color, marker="o", ms=5, lw=2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ctxs, [_ctx_label(c) for c in ctxs])
    ax.set_xlabel("context length")
    ax.set_ylabel("layer latency (ms, median over page sizes)")
    ax.set_title("full sparse prefill layer latency")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_context.png", dpi=140)
    plt.close(fig)

    # --- 2b. latency vs page_size, one panel per context --------------------
    ncol = 4
    nrow = -(-len(ctxs) // ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.5 * ncol, 3.4 * nrow), squeeze=False
    )
    for idx, c in enumerate(ctxs):
        ax = axes[idx // ncol][idx % ncol]
        for v, col in (("orig", "#5145CC"), ("paged", "#0093A3")):
            ys = [
                lut[(v, p_, c)]["latency_median_ms"]
                for p_ in pages
                if (v, p_, c) in lut
            ]
            ax.plot(
                pages[: len(ys)],
                ys,
                "-o",
                color=col,
                lw=2,
                ms=5,
                label=v,
                alpha=1.0 if v == "paged" else 0.75,
            )
        # mark where the fast path becomes legal
        thr = [p_ for p_ in pages if p_ >= 128 and p_ % 128 == 0]
        if thr:
            ax.axvline(min(thr), color="#C25708", ls=":", lw=1.4)
            ax.text(
                min(thr),
                ax.get_ylim()[1] * 0.06,
                " fast path\n legal →",
                color="#C25708",
                fontsize=7.5,
                va="bottom",
                ha="left",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(pages, [str(p_) for p_ in pages], fontsize=7.5)
        ax.set_ylim(bottom=0)
        ax.set_title(f"ctx {_ctx_label(c)}", fontsize=10)
        ax.grid(alpha=0.3)
        if idx % ncol == 0:
            ax.set_ylabel("layer latency (ms)")
        if idx // ncol == nrow - 1:
            ax.set_xlabel("page_size")
    for k in range(len(ctxs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(fontsize=9)
    fig.suptitle(
        "full sparse prefill layer latency vs page_size  "
        "(block_size 128 — the fast path is legal only at page_size ≥ 128 "
        "and a multiple of it)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_pagesize.png", dpi=140)
    plt.close(fig)

    # --- 3. stage SHARE at page_size 128 ------------------------------------
    # Shares, not absolute ms: stacking absolute stage times on a log axis would
    # be wrong (segment lengths would not represent values), and the share is
    # the point anyway -- it is why the layer speedup decays with context.
    stages = [
        ("stage_indexer_score_ms", "indexer", "#5145CC"),
        ("stage_topk_select_ms", "top-k select", "#C25708"),
        ("stage_sparse_attn_ms", "sparse attn", "#0093A3"),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), width_ratios=[1.45, 1])
    x = np.arange(len(ctxs))
    w = 0.38
    for k, (variant, shift) in enumerate([("orig", -w / 2), ("paged", w / 2)]):
        tot = np.array(
            [
                (
                    sum(lut[(variant, 128, c)][key] for key, _, _ in stages)
                    if (variant, 128, c) in lut
                    else np.nan
                )
                for c in ctxs
            ]
        )
        bottom = np.zeros(len(ctxs))
        for key, name, col in stages:
            vals = (
                np.array(
                    [
                        (
                            lut[(variant, 128, c)][key]
                            if (variant, 128, c) in lut
                            else np.nan
                        )
                        for c in ctxs
                    ]
                )
                / tot
                * 100
            )
            ax.bar(
                x + shift,
                vals,
                w,
                bottom=bottom,
                color=col,
                alpha=1.0 if variant == "paged" else 0.5,
                label=f"{name} ({variant})",
                edgecolor="white",
                linewidth=0.7,
            )
            bottom += np.nan_to_num(vals)
    ax.set_xticks(x, [_ctx_label(c) for c in ctxs])
    ax.set_ylim(0, 100)
    ax.set_xlabel("context length")
    ax.set_ylabel("share of layer time (%)")
    ax.set_title(
        "stage share at page_size 128 — left bar orig, right bar paged", fontsize=11
    )
    ax.legend(fontsize=8, ncol=2, loc="lower left")

    # why the layer speedup decays: attention's share collapses
    share = [
        lut[("orig", 128, c)]["stage_sparse_attn_ms"]
        / sum(lut[("orig", 128, c)][key] for key, _, _ in stages)
        * 100
        for c in ctxs
    ]
    speed = [lut[("paged", 128, c)]["speedup_vs_orig"] for c in ctxs]
    ax2.plot(share, speed, "-o", color="#0093A3", lw=2, ms=7)
    for sh, sp, c in zip(share, speed, ctxs):
        ax2.annotate(
            _ctx_label(c),
            (sh, sp),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=9,
            color="#43505C",
        )
    ax2.axhline(1.0, color="#888", ls="--", lw=1)
    ax2.set_xlabel("sparse-attn share of the layer (%)")
    ax2.set_ylabel("layer speedup (×)")
    ax2.set_title("the speedup tracks Amdahl exactly", fontsize=11)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "stage_breakdown.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    def _ints(s):
        return [int(x) for x in s.split(",") if x]

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--page-sizes", type=_ints, default=DEFAULT_PAGES)
    p.add_argument("--context-lens", type=_ints, default=DEFAULT_CTXS)
    p.add_argument("--chunk-len", type=int, default=4096)
    p.add_argument("--prefill-iters", type=int, default=20)
    p.add_argument("--profile-iters", type=int, default=10)
    p.add_argument("--wait-for-idle", type=float, default=0.0)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate plots from an existing raw.json",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "pagesize_context_sweep",
    )
    args = p.parse_args(argv)

    if args.plot_only:
        import json

        rows = json.loads((args.out / "raw.json").read_text())
        make_plots(rows, args.out)
        print(f"replotted from {args.out / 'raw.json'}")
        return 0

    if not torch.cuda.is_available():
        print("CUDA is required.")
        return 1
    cfg = m3_config().replace(granularity="block")
    info = gpu_info()
    print("full sparse pipeline — page_size x context sweep, two step-3 kernels")
    print(f"  device : {info['gpu']} (sm{info['sm']}, {info['memory_gb']} GB)")
    print(f"  config : {cfg.shape_tag()}  chunk={args.chunk_len}")
    print(
        f"  grid   : {len(args.page_sizes)} page sizes x "
        f"{len(args.context_lens)} contexts x 2 variants = "
        f"{len(args.page_sizes) * len(args.context_lens) * 2} points"
    )
    if args.wait_for_idle > 0:
        wait_for_idle(args.wait_for_idle)
    warn_if_contended()

    rows = []
    for ctx in args.context_lens:
        chunk = min(args.chunk_len, ctx)
        print(f"\n=== ctx {_ctx_label(ctx)} (chunk {chunk}) ===")
        print(
            f"{'page':>6}{'fast':>6}{'orig ms':>10}{'paged ms':>10}{'speedup':>9}"
            f"{'  orig stages (idx|topk|attn)':<30}{'paged attn':>11}"
        )
        for ps in args.page_sizes:
            try:
                pair = run_cell(
                    cfg, ps, ctx, chunk, args.prefill_iters, args.profile_iters
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{ps:>6}   OOM — skipped")
                continue
            o, g = pair
            rows.extend(pair)
            print(
                f"{ps:>6}{('ON' if g['fast_path'] else '-'):>6}"
                f"{o['latency_median_ms']:>10.3f}{g['latency_median_ms']:>10.3f}"
                f"{g['speedup_vs_orig']:>8.3f}x"
                f"  {o['stage_indexer_score_ms']:>6.3f}|"
                f"{o['stage_topk_select_ms']:>6.3f}|"
                f"{o['stage_sparse_attn_ms']:>7.3f}"
                f"{g['stage_sparse_attn_ms']:>16.3f}"
            )
    _select("orig", 0)

    args.out.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = write_results(rows, args.out, "raw")
    summary = args.out / "summary.csv"
    csv_path.replace(summary)
    print(f"\nwrote {json_path}\n      {summary}")
    if not args.no_plots:
        try:
            make_plots(rows, args.out)
            print(f"      {args.out}/*.png")
        except Exception as err:
            print(f"  (plotting failed: {err})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
