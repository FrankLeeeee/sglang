#!/usr/bin/env python3
"""Unified indexer+top-k sweep: every selection implementation, both phases.

Prefill (token granularity, identical contract, checked for selection
equivalence against ``current`` at every point):

  current     index_score.token_select_prefill        materialize + flashinfer
  fused       ...token_select_prefill_fused           two-pass threshold-emit (Triton)
  fused_cuda  ...token_select_prefill_fused_cuda      two-pass threshold-emit (CUDA)
  seg         index_score_seg.token_select_prefill_seg    segmented scorer
  onepass     index_score_onepass....                 matrix-free running-tau
  tau_emit    index_score_tau_emit....                fused-pool tau + filter

plus ``block`` — the block-granularity production path — as the reference
floor (different output contract: block ids, not token ids; not checked).

Decode: ``token`` (token_select_decode) and ``block`` (flash_decode path).

    CUDA_VISIBLE_DEVICES=4 python bench_indexer_topk.py
    python bench_indexer_topk.py --context-lens 16384,131072 --phases prefill
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    emit_plots,
    bench_cuda,
    build_decode_inputs,
    build_prefill_inputs,
    profile_breakdown,
    warn_if_contended,
    write_results,
)

# The stages every selection pipeline decomposes into (see harness.STAGE_RULES
# for the kernel-name mapping). `indexer_score` is the q.k scan — for the
# fused kernels that includes their in-kernel selection work; `topk_select` is
# everything that ranks or filters; the rest is glue.
STAGES = ("indexer_score", "topk_select", "select_overhead", "buffer_init", "other")
from m3_config import m3_config  # noqa: E402
from bench_indexer import BUILDERS  # noqa: E402

from sglang.kernels.ops.attention.minimax_sparse.token.flash_with_topk_idx_optimized import (  # noqa: E402
    token_select_prefill_fused,
    token_select_prefill_fused_cuda,
)
from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (  # noqa: E402
    token_select_prefill,
)
from sglang.kernels.ops.attention.minimax_sparse.token.index_score_onepass import (  # noqa: E402
    token_select_prefill_onepass,
)
from sglang.kernels.ops.attention.minimax_sparse.token.index_score_seg import (  # noqa: E402
    token_select_prefill_seg,
)
from sglang.kernels.ops.attention.minimax_sparse.token.index_score_tau_emit import (  # noqa: E402
    token_select_prefill_tau_emit,
)

PREFILL_IMPLS: dict[str, Callable] = {
    "current": token_select_prefill,
    "fused": token_select_prefill_fused,
    "fused_cuda": token_select_prefill_fused_cuda,
    "seg": token_select_prefill_seg,
    "onepass": token_select_prefill_onepass,
    "tau_emit": token_select_prefill_tau_emit,
}


# ---------------------------------------------------------------------------
# selection-equivalence check (moved here from the merged bench_token_select.py)
# ---------------------------------------------------------------------------

def _index_sets(t: torch.Tensor) -> list[set]:
    return [set(r[r >= 0].tolist()) for r in t.reshape(-1, t.shape[-1]).cpu()]


def _sets_equivalent(out, reference, inp) -> tuple[bool, int]:
    """Set equality, allowing swaps of provably near-tied boundary scores.

    Engines that lower the q.k dot differently (Triton wgmma vs CUDA wmma)
    round it differently by ~1 ulp, which can swap the pair straddling rank
    topk when their true scores agree to ~1e-8 (measured). Adjudicate each
    differing row by recomputing the disputed scores in fp64: the swap is
    accepted only if every disputed score sits within 1e-5 (rel) of their
    median — a real selection bug fails this immediately.

    Returns (equivalent, number of adjudicated rows).
    """
    # Cheap GPU paths first: sorted-tensor equality proves set equality in
    # milliseconds even at 1M context; only a mismatch pays for the host-side
    # set walk and fp64 adjudication below.
    if torch.equal(out, reference) or torch.equal(
        out.sort(dim=-1).values, reference.sort(dim=-1).values
    ):
        return True, 0
    A, B = _index_sets(out), _index_sets(reference)
    if A == B:
        return True, 0
    n_heads, total_q, _ = reference.shape
    cu = inp.cu_seqlens.tolist()
    swaps = 0
    for i, (a, b) in enumerate(zip(A, B)):
        if a == b:
            continue
        missing, extra = a - b, b - a
        if len(a) != len(b) or not missing:
            return False, swaps
        _, g_row = divmod(i, total_q)
        req = next(j for j in range(len(cu) - 1) if cu[j] <= g_row < cu[j + 1])
        h = i // total_q
        q = inp.idx_q[g_row, h].double()
        sid = int(inp.slot_ids[req])
        scores = []
        for pos in (*missing, *extra):
            slot = int(inp.req_to_token[sid, pos])
            scores.append(float(q @ inp.idx_k_cache[slot, 0].double()))
        med = sorted(scores)[len(scores) // 2]
        if any(abs(s - med) > 1e-5 * (1.0 + abs(med)) for s in scores):
            return False, swaps
        swaps += 1
    return True, swaps


def _prefill_kwargs(cfg, inp) -> dict:
    return dict(
        idx_q=inp.idx_q,
        idx_k_cache=inp.idx_k_cache,
        req_to_token=inp.req_to_token,
        slot_ids=inp.slot_ids,
        cu_seqlens=inp.cu_seqlens,
        seq_lens=inp.seq_lens,
        prefix_lens=inp.prefix_lens,
        max_seqlen_q=inp.max_seqlen_q,
        max_seqlen_k=inp.max_seqlen_k,
        topk=cfg.effective_topk_tokens,
        init_tokens=cfg.init_tokens,
        local_tokens=cfg.local_tokens,
        seqlens_cpu=inp.seqlens_cpu,
        prefix_lens_cpu=inp.prefix_lens_cpu,
    )


def _equivalent(out: torch.Tensor, reference: torch.Tensor, inp) -> tuple[bool, int]:
    """Selection equivalence, cheap path first.

    Sorted-tensor equality on the GPU proves set equality in milliseconds even
    at 1M context; only a mismatch pays for the fp64 tie adjudication of
    ``_sets_equivalent`` (wmma vs wgmma round the same bf16 dot apart by
    ~1 ulp, which can swap a rank-topk boundary pair).
    """
    if torch.equal(out, reference):
        return True, 0
    if torch.equal(out.sort(dim=-1).values, reference.sort(dim=-1).values):
        return True, 0
    return _sets_equivalent(out, reference, inp)


def _time_point(run: Callable[[], object], iters: int, profile_iters: int) -> dict:
    timing = bench_cuda(run, warmup=max(3, iters // 4), iters=iters)
    row = {
        "latency_median_ms": round(timing.median_ms, 6),
        "latency_mean_ms": round(timing.mean_ms, 6),
        "latency_min_ms": round(timing.min_ms, 6),
    }
    if profile_iters > 0:
        stages, _ = profile_breakdown(run, iters=profile_iters, warmup=2)
        for stage in STAGES:
            row[f"stage_{stage}_ms"] = round(stages.get(stage, 0.0), 6)
        extra = sum(ms for s, ms in stages.items() if s not in STAGES)
        row["stage_other_ms"] = round(row.get("stage_other_ms", 0.0) + extra, 6)
        row["stage_sum_ms"] = round(sum(stages.values()), 6)
    return row


def _stage_note(row: dict) -> str:
    if "stage_sum_ms" not in row:
        return ""
    return (f"  (score {row['stage_indexer_score_ms']:8.3f} | "
            f"select {row['stage_topk_select_ms']:8.3f} | "
            f"other {row['stage_sum_ms'] - row['stage_indexer_score_ms'] - row['stage_topk_select_ms']:6.3f})")


def _run_prefill(ctx: int, chunk: int, impls: list[str], args) -> list[dict]:
    rows: list[dict] = []
    cfg_t = m3_config(granularity="token")
    inp_t = build_prefill_inputs(
        cfg_t, batch_size=1, context_len=ctx, chunk_len=chunk,
        device=torch.device("cuda"),
    )
    kwargs = _prefill_kwargs(cfg_t, inp_t)
    reference = token_select_prefill(**kwargs) if not args.skip_check else None

    for name in impls:
        row = {"phase": "prefill", "impl": name, "context_len": ctx,
               "batch_size": 1, "chunk_len": chunk}
        try:
            if name == "block":
                cfg_b = m3_config(granularity="block")
                inp_b = build_prefill_inputs(
                    cfg_b, batch_size=1, context_len=ctx, chunk_len=chunk,
                    device=torch.device("cuda"),
                )
                run = BUILDERS[("block", "prefill")](cfg_b, inp_b)
                row["equiv"] = "ref"  # block ids, different contract
            else:
                fn = PREFILL_IMPLS[name]
                if reference is not None and name != "current":
                    ok, swaps = _equivalent(fn(**kwargs), reference, inp_t)
                    row["equiv"] = "yes" if ok else "NO"
                    row["tie_rows"] = swaps
                else:
                    row["equiv"] = "yes" if name == "current" else "skipped"
                run = lambda: fn(**kwargs)  # noqa: B023
            row.update(_time_point(run, args.prefill_iters, args.profile_iters))
            row["status"] = "ok"
            print(f"  prefill {name:<10} ctx={ctx:<8} "
                  f"{row['latency_median_ms']:9.3f} ms  equiv={row['equiv']}"
                  f"{_stage_note(row)}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row["status"] = "oom"
            print(f"  prefill {name:<10} ctx={ctx:<8} OOM")
        except Exception as err:  # keep the sweep going; record what failed
            traceback.print_exc()
            row["status"] = "error"
            row["error"] = repr(err)
            print(f"  prefill {name:<10} ctx={ctx:<8} ERROR {type(err).__name__}")
        rows.append(row)
        if name == "block":
            del inp_b
        torch.cuda.empty_cache()

    del inp_t, reference
    torch.cuda.empty_cache()
    return rows


def _run_decode(ctx: int, impls: list[str], args) -> list[dict]:
    rows: list[dict] = []
    for name in impls:
        row = {"phase": "decode", "impl": name, "context_len": ctx, "batch_size": 1}
        try:
            gran = "block" if name == "block" else "token"
            cfg = m3_config(granularity=gran)
            inp = build_decode_inputs(
                cfg, batch_size=1, context_len=ctx, device=torch.device("cuda")
            )
            run = BUILDERS[(gran, "decode")](cfg, inp)
            row.update(_time_point(run, args.decode_iters, args.profile_iters))
            row["status"] = "ok"
            print(f"  decode  {name:<10} ctx={ctx:<8} "
                  f"{row['latency_median_ms']:9.3f} ms{_stage_note(row)}")
            del inp
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row["status"] = "oom"
            print(f"  decode  {name:<10} ctx={ctx:<8} OOM")
        except Exception as err:
            traceback.print_exc()
            row["status"] = "error"
            row["error"] = repr(err)
            print(f"  decode  {name:<10} ctx={ctx:<8} ERROR {type(err).__name__}")
        rows.append(row)
        torch.cuda.empty_cache()
    return rows


def _print_breakdown(rows: list[dict], phase: str, impls: list[str]) -> None:
    """Per-point stage split: where each implementation's time actually goes."""
    ok = [r for r in rows if r["phase"] == phase and r.get("status") == "ok"
          and "stage_sum_ms" in r]
    if not ok:
        return
    print(f"\n=== {phase} stage breakdown (ms, kernel self-time) ===")
    print(f"{'context':>9} {'impl':<11}{'score':>10}{'select':>10}"
          f"{'overhead':>10}{'init':>8}{'other':>8}{'sum':>10}")
    for ctx in sorted({r["context_len"] for r in ok}):
        for n in impls:
            r = next((x for x in ok if x["context_len"] == ctx and x["impl"] == n), None)
            if r is None:
                continue
            print(f"{ctx:>9} {n:<11}"
                  f"{r['stage_indexer_score_ms']:>10.3f}"
                  f"{r['stage_topk_select_ms']:>10.3f}"
                  f"{r['stage_select_overhead_ms']:>10.3f}"
                  f"{r['stage_buffer_init_ms']:>8.3f}"
                  f"{r['stage_other_ms']:>8.3f}"
                  f"{r['stage_sum_ms']:>10.3f}")


def _print_pivot(rows: list[dict], phase: str, impls: list[str]) -> None:
    ok = [r for r in rows if r["phase"] == phase]
    ctxs = sorted({r["context_len"] for r in ok})
    if not ctxs:
        return
    print(f"\n=== {phase} — median ms (bs=1) ===")
    print(f"{'context':>9}" + "".join(f"{n:>12}" for n in impls))
    for ctx in ctxs:
        cells = []
        for n in impls:
            r = next((x for x in ok if x["context_len"] == ctx and x["impl"] == n), None)
            if r is None or r.get("status") != "ok":
                cells.append(f"{r['status'] if r else '-':>12}")
            else:
                mark = "" if r.get("equiv") in ("yes", "ref", "skipped", None) else "!"
                cells.append(f"{r['latency_median_ms']:>11.3f}{mark or ' '}")
        print(f"{ctx:>9}" + "".join(cells))


def _int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context-lens", type=_int_list,
                   default=[4096, 8192, 16384, 65536, 131072, 524288, 1048576])
    p.add_argument("--phases", default="prefill,decode")
    p.add_argument("--prefill-impls",
                   default="block,current,fused,fused_cuda,seg,onepass,tau_emit")
    p.add_argument("--decode-impls", default="block,token")
    p.add_argument("--prefill-chunk", type=int, default=8192)
    p.add_argument("--prefill-iters", type=int, default=10)
    p.add_argument("--decode-iters", type=int, default=50)
    p.add_argument("--profile-iters", type=int, default=8,
                   help="iterations per torch.profiler pass for the stage "
                        "breakdown; 0 disables it")
    p.add_argument("--skip-check", action="store_true",
                   help="skip the selection-equivalence verification")
    p.add_argument("-o", "--out", type=Path,
                   default=Path(__file__).resolve().parent / "results")
    p.add_argument("--no-plots", action="store_true",
                   help="skip the plots this benchmark emits when it finishes")
    p.add_argument("--tag", default="indexer_topk")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    phases = [s.strip() for s in args.phases.split(",") if s.strip()]
    prefill_impls = [s.strip() for s in args.prefill_impls.split(",") if s.strip()]
    decode_impls = [s.strip() for s in args.decode_impls.split(",") if s.strip()]
    for n in prefill_impls:
        if n != "block" and n not in PREFILL_IMPLS:
            raise SystemExit(f"unknown prefill impl {n!r}")

    torch.cuda.init()
    warn_if_contended()
    cfg = m3_config()
    print("MiniMax-M3 indexer + top-k — unified selection sweep")
    print(f"  device : {torch.cuda.get_device_name(0)}")
    print(f"  config : {cfg.shape_tag()}  budget {cfg.effective_topk_tokens} tokens")

    rows: list[dict] = []
    for ctx in args.context_lens:
        print(f"\n--- ctx={ctx} ---")
        if "prefill" in phases:
            chunk = min(args.prefill_chunk, ctx)
            rows.extend(_run_prefill(ctx, chunk, prefill_impls, args))
        if "decode" in phases:
            rows.extend(_run_decode(ctx, decode_impls, args))

    json_path, csv_path = write_results(rows, args.out, args.tag)
    print(f"\nwrote {json_path}\n      {csv_path}")
    if not args.no_plots:
        import plot_indexer_topk
        emit_plots(plot_indexer_topk.main,
                   ["--results", str(json_path), "--out", str(args.out / "plots")],
                   label="plots")
        print(f"      {args.out / 'plots'}/*.png")
    if "prefill" in phases:
        _print_pivot(rows, "prefill", prefill_impls)
        _print_breakdown(rows, "prefill", prefill_impls)
    if "decode" in phases:
        _print_pivot(rows, "decode", decode_impls)
        _print_breakdown(rows, "decode", decode_impls)
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
