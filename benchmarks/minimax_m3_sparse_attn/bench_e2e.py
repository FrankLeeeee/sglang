#!/usr/bin/env python3
"""Serving-level benchmark: TTFT, TPOT, ITL and throughput through real sglang.

``bench_kernels.py`` measures kernels and ``bench_indexer.py`` measures selection.
Neither sees the scheduler. This drives an actual ``sglang.Engine`` — real
continuous batching, real paged KV cache, real CUDA graphs, real attention
backend — so the numbers include everything a served request pays except the
weights being meaningful.

The model is MiniMax-M3's own architecture cut to ``--num-layers`` layers and
loaded with ``load_format=dummy``. That is the whole trick: the 435B checkpoint
is not needed to measure system behaviour, because latency depends on shapes and
access patterns, not on what the weights contain. One layer of the real config
is ~14.5 GiB of MoE plus ~5 GiB of embedding/LM head, which fits on one H200 and
leaves the rest for KV cache.

**These are one-layer numbers.** ``--project-layers`` reports what the same
measurement implies for the full 60-layer model, but that is arithmetic, not a
measurement: it assumes layers are uniform and ignores anything that does not
scale with depth (sampling, detokenization, scheduling overhead per step).

    CUDA_VISIBLE_DEVICES=0 python bench_e2e.py
    python bench_e2e.py --granularity token --input-lens 32768 --batch-sizes 1,8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import wait_for_idle, warn_if_contended, write_results  # noqa: E402
from m3_config import (  # noqa: E402
    DEFAULT_KV_PAGE_SIZE,
    M3_NUM_LAYERS,
    m3_config,
)

# The Hub model id; bench_e2e needs the real config (see _load_full_config).

from bench_layer import HF_MODEL_ID  # noqa: E402


def _load_full_config(allow_download: bool) -> dict:
    """The released config.json, whole.

    The benchmark keeps the file structurally intact rather than synthesising a
    text-only config: the release is a VL wrapper (``model_type`` is
    ``minimax_m3_vl``) around the sparse text stack, and that is the shape both
    ``AutoConfig`` and sglang's registered ``MiniMaxM3VLConfig`` know how to
    parse. Text-only requests never enter the vision tower, so its dummy weights
    cost one small allocation and nothing per step.
    """
    from huggingface_hub import hf_hub_download

    for local_only in (True, False):
        if not local_only and not allow_download:
            break
        try:
            path = hf_hub_download(
                HF_MODEL_ID, "config.json", local_files_only=local_only
            )
            return json.loads(Path(path).read_text())
        except Exception as err:
            where = "cache" if local_only else "hub"
            print(f"  [config] no {where} copy of {HF_MODEL_ID} ({type(err).__name__})")

    raise SystemExit(
        f"bench_e2e needs the real {HF_MODEL_ID} config.json — it stands up an "
        f"actual engine, so it needs the released architecture, not the partial "
        f"pinned copy the shape-level benchmarks use. Fetch it once with:\n"
        f"    huggingface-cli download {HF_MODEL_ID} config.json\n"
        f"(weights are not needed: --load-format dummy)."
    )


def build_model_dir(
    *,
    num_layers: int,
    granularity: str,
    num_experts: Optional[int],
    allow_download: bool,
    dest: Path,
) -> Path:
    """Write a config.json for an ``num_layers``-deep MiniMax-M3.

    Everything about the attention layer is left exactly as released — head
    counts, index heads, block size, top-k, RoPE, MoE width. Only depth changes,
    plus the per-layer sparse pattern, which has to be re-expressed for it.
    """
    cfg = _load_full_config(allow_download)
    # `auto_map` points at modeling code that lives in the Hub repo, which we do
    # not have (config only). Dropping it makes transformers resolve the type
    # through its registry instead, where sglang has registered
    # MiniMaxM3VLConfig for `minimax_m3_vl` — so no remote code is needed.
    cfg.pop("auto_map", None)
    text = cfg["text_config"]
    original_depth = text["num_hidden_layers"]
    text["num_hidden_layers"] = num_layers

    # The config carries several per-layer lists (`mlp_layer_types`,
    # `sparse_attention_freq`, ...) and the model validates that each is exactly
    # `num_hidden_layers` long. Truncate anything of the original depth rather
    # than naming them individually, so a config that grows a new per-layer list
    # keeps working.
    for key, value in list(text.items()):
        if isinstance(value, list) and len(value) == original_depth:
            text[key] = value[:num_layers]

    # Speculative-decoding modules would add weights and a second model; this
    # measures the base decode loop.
    text["num_mtp_modules"] = 0
    text["num_nextn_predict_layers"] = 0

    if num_experts is not None:
        text["num_local_experts"] = num_experts
        text["num_experts_per_tok"] = min(text.get("num_experts_per_tok", 4), num_experts)

    sparse = dict(text["sparse_attention_config"])
    # `sparse_attention_freq` is per layer: 0 = dense, non-zero = sparse. The
    # released 60-layer model is 57 sparse / 3 dense; at reduced depth every
    # layer matches the granularity under test, so the measurement is not
    # diluted by whichever layers happened to be dense.
    sparse["sparse_attention_freq"] = [0 if granularity == "dense" else 1] * num_layers
    if "sparse_disable_index_value" in sparse:
        sparse["sparse_disable_index_value"] = [1] * num_layers
    sparse["use_sparse_attention"] = granularity != "dense"
    text["sparse_attention_config"] = sparse

    # Emit the *text* stack alone. Serving the released VL wrapper would make
    # sglang build a multimodal processor, which needs a preprocessor_config.json
    # this config-only checkout does not have — and the vision tower is dead
    # weight for a text benchmark. `minimax_m3_vl_text` is a registered
    # transformers type, so this resolves without any remote code.
    text["model_type"] = "minimax_m3_vl_text"
    text["architectures"] = ["MiniMaxM3SparseForCausalLM"]
    text.setdefault("torch_dtype", cfg.get("torch_dtype", "bfloat16"))

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(text, indent=2))
    return dest


def _percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}
    s = sorted(xs)
    def at(q: float) -> float:
        return s[min(len(s) - 1, int(q * len(s)))]
    return {
        "mean": sum(s) / len(s),
        "median": at(0.5),
        "p90": at(0.90),
        "p99": at(0.99),
    }


def run_point(engine, *, batch_size: int, input_len: int, output_len: int,
              vocab_size: int, rng: random.Random) -> dict:
    """One steady-state batch: TTFT from the first token, ITL from the rest."""
    import asyncio

    prompts = [
        [rng.randrange(1, vocab_size - 1) for _ in range(input_len)]
        for _ in range(batch_size)
    ]
    sampling = {"max_new_tokens": output_len, "ignore_eos": True, "temperature": 0.0}

    async def one(ids):
        first: Optional[float] = None
        prev: Optional[float] = None
        itls: list[float] = []
        start = time.perf_counter()
        n = 0
        gen = await engine.async_generate(
            input_ids=ids, sampling_params=sampling, stream=True
        )
        async for chunk in gen:
            now = time.perf_counter()
            produced = chunk["meta_info"].get("completion_tokens", 0)
            if produced <= n:
                continue
            if first is None:
                first = now - start
            else:
                itls.append((now - prev) * 1000.0 / max(1, produced - n))
            prev, n = now, produced
        return {
            "ttft_ms": (first or 0.0) * 1000.0,
            "itls_ms": itls,
            "e2e_ms": (time.perf_counter() - start) * 1000.0,
            "output_tokens": n,
        }

    async def all_of():
        return await asyncio.gather(*(one(p) for p in prompts))

    t0 = time.perf_counter()
    results = asyncio.run(all_of())
    wall_s = time.perf_counter() - t0

    ttfts = [r["ttft_ms"] for r in results]
    itls = [x for r in results for x in r["itls_ms"]]
    # TPOT: per-request decode time averaged over its own output tokens.
    tpots = [
        (r["e2e_ms"] - r["ttft_ms"]) / max(1, r["output_tokens"] - 1)
        for r in results if r["output_tokens"] > 1
    ]
    total_out = sum(r["output_tokens"] for r in results)
    return {
        "ttft": _percentiles(ttfts),
        "tpot": _percentiles(tpots),
        "itl": _percentiles(itls),
        "e2e": _percentiles([r["e2e_ms"] for r in results]),
        "output_tokens": total_out,
        "wall_s": wall_s,
        "output_tok_per_s": total_out / wall_s if wall_s else 0.0,
        "request_per_s": len(results) / wall_s if wall_s else 0.0,
    }


def _int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-layers", type=int, default=1,
                   help="depth of the dummy model (default 1)")
    p.add_argument("--granularity", choices=["block", "token", "dense"], default="block")
    p.add_argument("--input-lens", type=_int_list, default=[4096, 32768, 131072])
    p.add_argument("--batch-sizes", type=_int_list, default=[1, 8, 32])
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--num-experts", type=int, default=None,
                   help="shrink the MoE (default: the released 128) if memory is tight")
    p.add_argument("--page-size", type=int, default=DEFAULT_KV_PAGE_SIZE)
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
    p.add_argument("--attention-backend", default=None)
    p.add_argument("--project-layers", type=int, default=M3_NUM_LAYERS,
                   help="also report the figure scaled to this depth (0 = off)")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wait-for-idle", type=float, default=0.0)
    p.add_argument("-o", "--out", type=Path,
                   default=Path(__file__).resolve().parent / "results")
    p.add_argument("--tag", default="e2e")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.granularity == "token":
        os.environ.setdefault("SGLANG_USE_MINIMAX_TOKEN_SPARSE", "1")

    import torch

    torch.cuda.init()
    if args.wait_for_idle > 0:
        wait_for_idle(args.wait_for_idle)
    util = warn_if_contended()

    base = m3_config(granularity=args.granularity)
    rng = random.Random(args.seed)

    with tempfile.TemporaryDirectory(prefix="m3_e2e_") as tmp:
        model_dir = build_model_dir(
            num_layers=args.num_layers,
            granularity=args.granularity,
            num_experts=args.num_experts,
            allow_download=not args.no_download,
            dest=Path(tmp) / "model",
        )
        cfg = json.loads((model_dir / "config.json").read_text())
        vocab_size = cfg["vocab_size"]

        print("MiniMax-M3 end-to-end serving benchmark (dummy weights)")
        print(f"  device      : {torch.cuda.get_device_name(0)}")
        print(f"  layers      : {args.num_layers} of {M3_NUM_LAYERS}")
        print(f"  granularity : {args.granularity}  |  {base.shape_tag()}")
        print(f"  experts     : {cfg['num_local_experts']} local, "
              f"{cfg['num_experts_per_tok']} per token")

        import sglang as sgl

        engine_kwargs = dict(
            model_path=str(model_dir),
            load_format="dummy",
            skip_tokenizer_init=True,
            page_size=args.page_size,
            mem_fraction_static=args.mem_fraction_static,
            trust_remote_code=False,
            disable_radix_cache=True,  # prefix reuse would hide prefill cost
            random_seed=args.seed,
            # MiniMaxM3VLTextConfig is a strict dataclass and silently drops
            # `sparse_attention_config`, which is a MiniMax-M3 extension rather
            # than an upstream field — without it sglang cannot tell the model
            # is sparse. Overrides are applied after parsing, so this puts it
            # back on the config object.
            json_model_override_args=json.dumps(
                {"sparse_attention_config": cfg["sparse_attention_config"]}
            ),
        )
        if args.attention_backend:
            engine_kwargs["attention_backend"] = args.attention_backend

        engine = sgl.Engine(**engine_kwargs)
        rows: list[dict] = []
        try:
            for bs in args.batch_sizes:
                for ilen in args.input_lens:
                    label = f"  bs={bs:<4} in={ilen:<8} out={args.output_len:<4}"
                    try:
                        for _ in range(args.warmup):
                            run_point(engine, batch_size=bs, input_len=ilen,
                                      output_len=min(4, args.output_len),
                                      vocab_size=vocab_size, rng=rng)
                        m = run_point(engine, batch_size=bs, input_len=ilen,
                                      output_len=args.output_len,
                                      vocab_size=vocab_size, rng=rng)
                    except Exception as err:
                        traceback.print_exc()
                        print(f"{label} ERROR {type(err).__name__}")
                        rows.append({"batch_size": bs, "input_len": ilen,
                                     "status": "error", "error": repr(err)})
                        continue

                    row = {
                        "granularity": args.granularity,
                        "num_layers": args.num_layers,
                        "batch_size": bs,
                        "input_len": ilen,
                        "output_len": args.output_len,
                        "page_size": args.page_size,
                        "num_local_experts": cfg["num_local_experts"],
                        "ttft_mean_ms": round(m["ttft"]["mean"], 4),
                        "ttft_median_ms": round(m["ttft"]["median"], 4),
                        "ttft_p99_ms": round(m["ttft"]["p99"], 4),
                        "tpot_mean_ms": round(m["tpot"]["mean"], 4),
                        "tpot_median_ms": round(m["tpot"]["median"], 4),
                        "tpot_p99_ms": round(m["tpot"]["p99"], 4),
                        "itl_median_ms": round(m["itl"]["median"], 4),
                        "itl_p99_ms": round(m["itl"]["p99"], 4),
                        "e2e_median_ms": round(m["e2e"]["median"], 4),
                        "output_tok_per_s": round(m["output_tok_per_s"], 2),
                        "request_per_s": round(m["request_per_s"], 4),
                        "gpu": torch.cuda.get_device_name(0),
                        "gpu_util_before_pct": util,
                        "status": "ok",
                    }
                    if args.project_layers:
                        scale = args.project_layers / args.num_layers
                        row["projected_layers"] = args.project_layers
                        row["ttft_median_ms_x_layers"] = round(
                            m["ttft"]["median"] * scale, 2)
                        row["tpot_median_ms_x_layers"] = round(
                            m["tpot"]["median"] * scale, 4)
                    rows.append(row)
                    print(f"{label} TTFT {row['ttft_median_ms']:9.2f} ms | "
                          f"TPOT {row['tpot_median_ms']:7.3f} ms | "
                          f"ITL {row['itl_median_ms']:7.3f} ms | "
                          f"{row['output_tok_per_s']:8.1f} tok/s")
        finally:
            engine.shutdown()

    json_path, csv_path = write_results(rows, args.out, args.tag)
    print(f"\nwrote {json_path}\n      {csv_path}")
    if args.project_layers and any(r.get("status") == "ok" for r in rows):
        print(f"\nnote: *_x_layers columns scale the {args.num_layers}-layer measurement "
              f"to {args.project_layers} layers. That is arithmetic on the attention/MoE "
              f"stack only — per-step scheduling, sampling and detokenization do not "
              f"scale with depth, so a real {args.project_layers}-layer server is slower "
              f"than the projection.")
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
