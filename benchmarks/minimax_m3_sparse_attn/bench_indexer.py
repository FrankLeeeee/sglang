#!/usr/bin/env python3
"""Indexer-only benchmark: token-granularity vs block-granularity selection.

``bench_kernels.py`` measures the whole sparse attention op and attributes the
indexer as two of its stages. This isolates just the selection pipeline, so the
two granularities can be compared without the sparse-attention kernel — which
reads the same ``topk`` tokens either way — diluting the difference.

What each granularity does for one query row over a context of L keys:

  block   score every key against the index heads, pool each ``block_size``
          keys to one score, select ``topk_blocks`` blocks
          -> score buffer [idx_heads, rows, L / block_size]

  token   score every key, no pooling, select ``topk_blocks * block_size``
          individual positions
          -> score buffer [idx_heads, rows, L]        (block_size x larger)

Both read the *same* index KV cache, so the q.k work is identical and the
difference is entirely in what each writes and then selects over. That is what
this measures.

    CUDA_VISIBLE_DEVICES=0 python bench_indexer.py
    python bench_indexer.py --context-lens 65536,262144 --decode-batch-sizes 32
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
    measure_transient_bytes,
    profile_breakdown,
    wait_for_idle,
    warn_if_contended,
    write_results,
)
from m3_config import SparseAttnConfig, m3_config  # noqa: E402

from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (  # noqa: E402
    flash_prefill_with_topk_index,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (  # noqa: E402
    flash_decode_with_topk_idx,
)
from sglang.kernels.ops.attention.minimax_sparse.token.index_score import (  # noqa: E402
    plan_query_chunk,
    token_select_decode,
    token_select_prefill,
)

# Stages the indexer is made of. `indexer_score` is the q.k scan over the index
# KV cache; `topk_select` is the selection over whatever that produced.
INDEXER_STAGES = ("indexer_score", "topk_select", "select_overhead", "other")


def _block_prefill(cfg: SparseAttnConfig, inp) -> Callable[[], object]:
    """Block indexer: fused score + block-pool + select, value path disabled."""

    def run():
        return flash_prefill_with_topk_index(
            q=inp.idx_q,
            k_cache=inp.idx_k_cache,
            v_cache=None,
            sink=None,
            req_to_token=inp.req_to_token,
            slot_ids=inp.slot_ids,
            cu_seqlens=inp.cu_seqlens,
            seq_lens=inp.seq_lens,
            prefix_lens=inp.prefix_lens,
            max_seqlen_q=inp.max_seqlen_q,
            max_seqlen_k=inp.max_seqlen_k,
            block_size_q=1,
            block_size_k=cfg.block_size,
            topk=cfg.topk_blocks,
            init_blocks=cfg.init_blocks,
            local_blocks=cfg.local_blocks,
            disable_index_value=True,
            cu_seqblocks_q=inp.cu_seqblocks_q,
            max_seqblock_q=inp.max_seqblock_q,
            all_seqblock_q=inp.all_seqblock_q,
        )

    return run


def _block_decode(cfg: SparseAttnConfig, inp) -> Callable[[], object]:
    def run():
        return flash_decode_with_topk_idx(
            q=inp.idx_q,
            k_cache=inp.idx_k_cache,
            v_cache=None,
            sink=None,
            req_to_token=inp.req_to_token,
            slot_ids=inp.slot_ids,
            seq_lens=inp.seq_lens,
            max_seqlen=inp.max_seqlen,
            block_size=cfg.block_size,
            topk=cfg.topk_blocks,
            init_blocks=cfg.init_blocks,
            local_blocks=cfg.local_blocks,
            disable_index_value=True,
        )

    return run


def _token_prefill(
    cfg: SparseAttnConfig, inp, *, score_writeback: bool = True
) -> Callable[[], object]:
    def run():
        return token_select_prefill(
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
            score_writeback=score_writeback,
        )

    return run


def _token_decode(
    cfg: SparseAttnConfig, inp, *, score_writeback: bool = True
) -> Callable[[], object]:
    def run():
        return token_select_decode(
            idx_q=inp.idx_q,
            idx_k_cache=inp.idx_k_cache,
            req_to_token=inp.req_to_token,
            slot_ids=inp.slot_ids,
            seq_lens=inp.seq_lens,
            max_seqlen=inp.max_seqlen,
            topk=cfg.effective_topk_tokens,
            init_tokens=cfg.init_tokens,
            local_tokens=cfg.local_tokens,
            score_writeback=score_writeback,
        )

    return run


BUILDERS = {
    ("block", "prefill"): _block_prefill,
    ("block", "decode"): _block_decode,
    ("token", "prefill"): _token_prefill,
    ("token", "decode"): _token_decode,
}

# Same scoring kernels with the score-matrix HBM store disabled (and no
# selection). Timing these against the full runs is what splits the score stage
# into compute vs write-back. Only the token kernels expose the hook in
# production; the block kernels are shared upstream code, so their differential
# runs on the bench-local copies in inner_profile.py instead (with the
# streaming-bandwidth estimate as a fallback).
COMPUTE_ONLY_BUILDERS = {
    ("token", "prefill"): lambda cfg, inp: _token_prefill(
        cfg, inp, score_writeback=False
    ),
    ("token", "decode"): lambda cfg, inp: _token_decode(
        cfg, inp, score_writeback=False
    ),
}


def _score_matrix_bytes(cfg: SparseAttnConfig, *, rows: int, context_len: int) -> int:
    """Full [idx_heads, rows, width] float32 score matrix, in bytes.

    This is the *whole* matrix, not what is resident: prefill chunks the query
    axis to a byte budget and decode windows the key axis, so the live slice is
    ``transient_bytes``. The gap between the two is exactly what the chunking
    buys, which is why both are recorded.
    """
    width = context_len if cfg.granularity == "token" else -(-context_len // cfg.block_size)
    return cfg.num_idx_heads * rows * width * 4


def _score_written_bytes(
    cfg: SparseAttnConfig,
    *,
    phase: str,
    batch_size: int,
    context_len: int,
    chunk_len: int,
    full_matrix_bytes: int,
) -> int:
    """Bytes the score kernel actually stores to HBM at this point.

    Token prefill writes less than the full matrix: the query axis is chunked
    and each chunk's row is only as wide as its queries can see, so this mirrors
    ``token_select_prefill``'s chunk loop. Block prefill is causal per query
    tile (approximated per row here — at most one query tile of error). The
    decode paths write their full matrix: token decode windows the key axis but
    still stores every column exactly once.
    """
    heads = cfg.num_idx_heads
    if phase == "decode":
        return full_matrix_bytes
    prefix = context_len - chunk_len
    if cfg.granularity == "token":
        batch_cap, chunk_cap = plan_query_chunk(
            batch_size=batch_size,
            max_seqlen_q=chunk_len,
            max_seqlen_k=context_len,
            num_idx_heads=heads,
        )
        total = 0
        for b_start in range(0, batch_size, batch_cap):
            b_count = min(batch_cap, batch_size - b_start)
            for q_start in range(0, chunk_len, chunk_cap):
                kv_width = min(context_len, prefix + q_start + chunk_cap)
                rows = min(chunk_len, q_start + chunk_cap) - q_start
                total += heads * b_count * rows * kv_width * 4
        return total
    width_blocks = sum(
        -(-min(context_len, prefix + i + 1) // cfg.block_size)
        for i in range(chunk_len)
    )
    return heads * batch_size * width_blocks * 4


def measure_write_bandwidth_bytes_s(nbytes: int = 1 << 30) -> float:
    """Streaming fp32-store bandwidth of this GPU, from a plain tensor fill.

    Fallback basis for the block write-back split when the bench-local kernel
    copies in inner_profile.py cannot run. ``min`` over iterations because this
    stands in for the kernel's best-case store rate.
    """
    buf = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
    timing = bench_cuda(lambda: buf.fill_(1.0), warmup=5, iters=20, flush_l2=False)
    del buf
    return nbytes / (timing.min_ms / 1e3)


def run_point(
    cfg: SparseAttnConfig,
    *,
    granularity: str,
    phase: str,
    batch_size: int,
    context_len: int,
    chunk_len: int,
    args,
    write_bw_bytes_s: float = 0.0,
) -> Optional[dict]:
    label = f"{granularity:<5} {phase:<7} bs={batch_size:<4} ctx={context_len:<8}"
    try:
        if phase == "prefill":
            inp = build_prefill_inputs(
                cfg,
                batch_size=batch_size,
                context_len=context_len,
                chunk_len=chunk_len,
                device=torch.device("cuda"),
            )
            rows = inp.num_query_tokens
        else:
            inp = build_decode_inputs(
                cfg,
                batch_size=batch_size,
                context_len=context_len,
                device=torch.device("cuda"),
            )
            rows = batch_size

        run = BUILDERS[(granularity, phase)](cfg, inp)
        iters = args.prefill_iters if phase == "prefill" else args.decode_iters
        timing = bench_cuda(run, warmup=max(3, iters // 4), iters=iters)
        stages, _ = profile_breakdown(run, iters=args.profile_iters, warmup=3)

        # Index-K bytes the scan must read: one shared head, idx_head_dim wide.
        idx_k_bytes = batch_size * context_len * cfg.idx_head_dim * 2
        score_bytes = _score_matrix_bytes(cfg, rows=rows, context_len=context_len)
        stage_sum = sum(stages.values())

        # Split the score stage into compute vs HBM write-back. Token kernels:
        # measured, by re-profiling with the score store disabled. Block
        # kernels: estimated from analytic written bytes at streaming-store
        # bandwidth (their matrix is 128x smaller, so the stakes are too).
        score_total_ms = stages.get("indexer_score", 0.0)
        written_bytes = _score_written_bytes(
            cfg,
            phase=phase,
            batch_size=batch_size,
            context_len=context_len,
            chunk_len=chunk_len,
            full_matrix_bytes=score_bytes,
        )
        compute_builder = COMPUTE_ONLY_BUILDERS.get((granularity, phase))
        if compute_builder is not None:
            # Token: the production kernels expose the store-disable hook.
            run_compute = compute_builder(cfg, inp)
            stages_compute, _ = profile_breakdown(
                run_compute, iters=args.profile_iters, warmup=3
            )
            score_compute_ms = stages_compute.get("indexer_score", 0.0)
            score_write_ms = max(0.0, score_total_ms - score_compute_ms)
            write_method = "measured"
            del run_compute
        else:
            # Block: the same differential, from the bench-local kernel copies
            # in inner_profile.py; the bandwidth-based estimate is the fallback.
            try:
                from inner_profile import block_score_write_split

                with_store_ms, without_store_ms = block_score_write_split(
                    cfg, inp, phase=phase, iters=args.profile_iters
                )
                # The copy must behave like the production kernel for the
                # differential to transfer; a large gap means transcription or
                # config drift, so treat it as a failure.
                if score_total_ms > 0.01 and not (
                    0.8 <= with_store_ms / score_total_ms <= 1.25
                ):
                    raise RuntimeError(
                        f"bench copy at {with_store_ms:.4f} ms vs production "
                        f"score stage {score_total_ms:.4f} ms"
                    )
                score_write_ms = max(0.0, with_store_ms - without_store_ms)
                score_compute_ms = max(0.0, score_total_ms - score_write_ms)
                write_method = "measured"
            except Exception as err:
                print(f"  [block write split estimated: {type(err).__name__}: {err}]")
                score_write_ms = (
                    min(score_total_ms, written_bytes / write_bw_bytes_s * 1e3)
                    if write_bw_bytes_s > 0
                    else 0.0
                )
                score_compute_ms = max(0.0, score_total_ms - score_write_ms)
                write_method = "estimated"

        # Intra-kernel view: warp-cycle shares of the score kernel's inner ops
        # from a Proton-instrumented copy, scaled onto the uninstrumented
        # kernel time. See inner_profile.py for the caveats.
        inner_shares: dict[str, float] = {}
        if args.inner_iters > 0:
            try:
                from inner_profile import profile_score_inner

                cycles = profile_score_inner(
                    cfg,
                    inp,
                    granularity=granularity,
                    phase=phase,
                    iters=args.inner_iters,
                    out_dir=args.out / "inner",
                    tag=f"{granularity}_{phase}_bs{batch_size}_ctx{context_len}",
                )
                total_cycles = sum(cycles.values())
                if total_cycles > 0:
                    inner_shares = {k: v / total_cycles for k, v in cycles.items()}
            except Exception as err:
                print(f"  [inner profile skipped: {type(err).__name__}: {err}]")

        row = {
            "phase": phase,
            "sweep": "context",
            "granularity": granularity,
            "level": "indexer",
            "context_len": context_len,
            "batch_size": batch_size,
            "num_query_tokens": rows,
            "num_idx_heads": cfg.num_idx_heads,
            "idx_head_dim": cfg.idx_head_dim,
            "block_size": cfg.block_size,
            "topk_blocks": cfg.topk_blocks,
            "topk_tokens": cfg.effective_topk_tokens,
            "page_size": cfg.page_size,
            "dtype": cfg.dtype,
            "latency_mean_ms": round(timing.mean_ms, 6),
            "latency_median_ms": round(timing.median_ms, 6),
            "latency_min_ms": round(timing.min_ms, 6),
            "latency_p90_ms": round(timing.p90_ms, 6),
        }
        for stage in INDEXER_STAGES:
            row[f"stage_{stage}_ms"] = round(stages.get(stage, 0.0), 6)
        row["stage_sum_ms"] = round(stage_sum, 6)
        row["score_compute_ms"] = round(score_compute_ms, 6)
        row["score_write_ms"] = round(score_write_ms, 6)
        row["score_write_method"] = write_method
        row["score_written_bytes"] = written_bytes
        row["score_write_tb_s"] = (
            round(written_bytes / (score_write_ms / 1e3) / 1024**4, 4)
            if score_write_ms > 1e-6
            else 0.0
        )
        # The selection reads back what the score kernel wrote; how close this
        # sits to DRAM read bandwidth says whether select is traffic-bound.
        select_ms = stages.get("topk_select", 0.0)
        row["topk_read_tb_s"] = (
            round(written_bytes / (select_ms / 1e3) / 1024**4, 4)
            if select_ms > 1e-6
            else 0.0
        )
        if inner_shares:
            from inner_profile import SCOPES

            for scope in SCOPES:
                share = inner_shares.get(scope, 0.0)
                row[f"inner_{scope}_pct"] = round(100 * share, 2)
                row[f"inner_{scope}_ms"] = round(score_total_ms * share, 6)
        row["transient_bytes"] = measure_transient_bytes(run)
        row["score_matrix_bytes"] = score_bytes
        row["idx_k_read_bytes"] = idx_k_bytes
        # Effective bandwidth on the one read both granularities share.
        row["idx_k_read_tb_s"] = (
            round(idx_k_bytes / (stage_sum / 1e3) / 1024**4, 4) if stage_sum else 0.0
        )
        row["status"] = "ok"

        est_mark = "~" if write_method == "estimated" else ""
        print(
            f"  {label} {timing.median_ms:9.4f} ms "
            f"(score {row['stage_indexer_score_ms']:8.4f} = "
            f"cmp {score_compute_ms:8.4f} + wb{est_mark} {score_write_ms:7.4f} | "
            f"select {row['stage_topk_select_ms']:8.4f})  "
            f"scores={score_bytes / 1024**2:8.1f} MiB  "
            f"ws={row['transient_bytes'] / 1024**2:7.1f} MiB"
        )
        if inner_shares:
            from inner_profile import SCOPES

            parts = [
                f"{scope} {100 * inner_shares.get(scope, 0.0):4.1f}%"
                f" ({score_total_ms * inner_shares.get(scope, 0.0):8.4f} ms)"
                for scope in SCOPES
                if inner_shares.get(scope, 0.0) >= 0.005
            ]
            print(f"    inner scopes: {' | '.join(parts)}")
        del inp, run
        torch.cuda.empty_cache()
        return row
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  {label} OOM")
        return {
            "phase": phase, "granularity": granularity, "context_len": context_len,
            "batch_size": batch_size, "status": "oom",
        }
    except Exception as err:  # keep the sweep going; record what failed
        traceback.print_exc()
        print(f"  {label} ERROR {type(err).__name__}")
        return {
            "phase": phase, "granularity": granularity, "context_len": context_len,
            "batch_size": batch_size, "status": "error", "error": repr(err),
        }


def _int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context-lens", type=_int_list,
                   default=[4096, 16384, 65536, 262144, 1048576])
    p.add_argument("--decode-batch-sizes", type=_int_list, default=[1, 32])
    p.add_argument("--prefill-batch-size", type=int, default=1)
    p.add_argument("--prefill-chunk", type=int, default=8192,
                   help="extend-chunk size; matches sglang's resolved "
                        "chunked_prefill_size on a >=90 GiB GPU")
    p.add_argument("--granularity", default="block,token",
                   help="comma-separated subset of: block, token")
    p.add_argument("--prefill-iters", type=int, default=10)
    p.add_argument("--decode-iters", type=int, default=50)
    p.add_argument("--profile-iters", type=int, default=10)
    p.add_argument("--inner-iters", type=int, default=5,
                   help="iterations for the Proton intra-kernel breakdown of "
                        "the score kernels; 0 disables it")
    p.add_argument("--wait-for-idle", type=float, default=0.0)
    p.add_argument("-o", "--out", type=Path, default=Path(__file__).resolve().parent / "results")
    p.add_argument("--no-plots", action="store_true",
                   help="skip the plots this benchmark emits when it finishes")
    p.add_argument("--tag", default="indexer")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    grans = [g.strip() for g in args.granularity.split(",") if g.strip()]
    for g in grans:
        if g not in ("block", "token"):
            raise SystemExit(f"unsupported granularity {g!r} (expected block or token)")

    torch.cuda.init()
    if args.wait_for_idle > 0:
        wait_for_idle(args.wait_for_idle)
    util = warn_if_contended()

    base = m3_config()
    write_bw = measure_write_bandwidth_bytes_s()
    print("MiniMax-M3 indexer benchmark — selection pipeline only")
    print(f"  device : {torch.cuda.get_device_name(0)}")
    print(f"  config : {base.shape_tag()}")
    print(f"  budget : {base.effective_topk_tokens} tokens "
          f"({base.topk_blocks} blocks x {base.block_size})")
    print(f"  wr bw  : {write_bw / 1024**4:.2f} TiB/s streaming store "
          f"(basis for the estimated 'wb~' splits)")

    rows: list[dict] = []
    for gran in grans:
        cfg = m3_config(granularity=gran)
        print(f"\n=== granularity = {gran} ===")
        print("--- prefill ---")
        for ctx in args.context_lens:
            chunk = min(args.prefill_chunk, ctx) if args.prefill_chunk else ctx
            r = run_point(cfg, granularity=gran, phase="prefill",
                          batch_size=args.prefill_batch_size, context_len=ctx,
                          chunk_len=chunk, args=args, write_bw_bytes_s=write_bw)
            if r:
                r["gpu"] = torch.cuda.get_device_name(0)
                r["gpu_util_before_pct"] = util
                rows.append(r)
        print("--- decode ---")
        for bs in args.decode_batch_sizes:
            for ctx in args.context_lens:
                r = run_point(cfg, granularity=gran, phase="decode",
                              batch_size=bs, context_len=ctx, chunk_len=1,
                              args=args, write_bw_bytes_s=write_bw)
                if r:
                    r["gpu"] = torch.cuda.get_device_name(0)
                    r["gpu_util_before_pct"] = util
                    rows.append(r)

    json_path, csv_path = write_results(rows, args.out, args.tag)
    print(f"\nwrote {json_path}\n      {csv_path}")
    if not args.no_plots:
        import plot_results
        emit_plots(plot_results.main,
                   ["--results", str(args.out), "--out", str(args.out / "plots")],
                   label="plots")
        print(f"      {args.out / 'plots'}/*.png")

    _print_comparison(rows)
    _print_score_split(rows)
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


def _print_comparison(rows: list[dict]) -> None:
    """token / block ratio at each measured point — the headline of this bench."""
    ok = [r for r in rows if r.get("status") == "ok"]
    keys = sorted({(r["phase"], r["batch_size"], r["context_len"]) for r in ok})
    if not keys:
        return
    print("\n=== token vs block indexer (stage_sum_ms) ===")
    print(f"{'phase':<8}{'bs':>5}{'context':>10}{'block':>10}{'token':>10}{'ratio':>8}"
          f"{'block MiB':>11}{'token MiB':>11}")
    for phase, bs, ctx in keys:
        got = {r["granularity"]: r for r in ok
               if (r["phase"], r["batch_size"], r["context_len"]) == (phase, bs, ctx)}
        if len(got) < 2:
            continue
        b, t = got["block"], got["token"]
        print(f"{phase:<8}{bs:>5}{ctx:>10}"
              f"{b['stage_sum_ms']:>10.4f}{t['stage_sum_ms']:>10.4f}"
              f"{t['stage_sum_ms'] / b['stage_sum_ms']:>7.2f}x"
              f"{b['score_matrix_bytes'] / 1024**2:>11.1f}"
              f"{t['score_matrix_bytes'] / 1024**2:>11.1f}")


def _print_score_split(rows: list[dict]) -> None:
    """Compute vs HBM write-back inside the score kernel, per measured point.

    token rows are measured (store-disabled kernel differential); block rows
    are estimated (~) from analytic written bytes at streaming-store bandwidth.
    """
    ok = [r for r in rows if r.get("status") == "ok" and "score_compute_ms" in r]
    if not ok:
        return
    print("\n=== score kernel: compute vs HBM write-back ===")
    print(f"{'phase':<8}{'gran':<7}{'bs':>5}{'context':>10}{'score ms':>10}"
          f"{'compute':>10}{'writeback':>11}{'wb TiB/s':>10}{'wb MiB':>10}")
    for r in ok:
        mark = "~" if r["score_write_method"] == "estimated" else " "
        print(f"{r['phase']:<8}{r['granularity']:<7}{r['batch_size']:>5}"
              f"{r['context_len']:>10}{r['stage_indexer_score_ms']:>10.4f}"
              f"{r['score_compute_ms']:>10.4f}{r['score_write_ms']:>10.4f}{mark}"
              f"{r['score_write_tb_s']:>10.3f}"
              f"{r['score_written_bytes'] / 1024**2:>10.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
