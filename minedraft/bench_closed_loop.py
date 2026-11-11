"""Closed-loop benchmark against a running SGLang server (native /generate).

Keeps exactly C requests in flight, greedy, fixed max_new_tokens, cycles over the
90 stage-02 prompts. Reports output tok/s, accept length (completion / verify ct),
and per-request outputs (for exact-match checks vs AR).
"""
import argparse, asyncio, json, time

import aiohttp
from transformers import AutoTokenizer


async def worker(session, url, queue, results, max_new, tok_name):
    while True:
        try:
            idx, text = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        payload = {
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new},
        }
        t0 = time.perf_counter()
        async with session.post(url + "/generate", json=payload) as r:
            out = await r.json()
        t1 = time.perf_counter()
        mi = out.get("meta_info", {})
        results.append({
            "idx": idx,
            "text": out.get("text", ""),
            "completion_tokens": mi.get("completion_tokens", 0),
            "spec_verify_ct": mi.get("spec_verify_ct", 0),
            "latency": t1 - t0,
        })


async def run(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts = [json.loads(l)["prompt"] for l in open(args.prompts)]
    texts = []
    for p in prompts:
        kw = {"enable_thinking": False} if "qwen3" in args.tokenizer.lower() else {}
        texts.append(tok.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True, **kw))
    out_all = {}
    timeout = aiohttp.ClientTimeout(total=3600)
    # aiohttp caps a session at 100 connections by default, which would cap C at 100.
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # warmup
        q = asyncio.Queue()
        for i in range(min(4, len(texts))):
            q.put_nowait((i, texts[i]))
        await asyncio.gather(*[worker(session, args.url, q, [], 32, args.tokenizer) for _ in range(4)])
        for c in args.conc:
            n = max(args.min_requests, c * args.rounds)
            q = asyncio.Queue()
            for i in range(n):
                q.put_nowait((i % len(texts), texts[i % len(texts)]))
            res = []
            t0 = time.perf_counter()
            await asyncio.gather(*[worker(session, args.url, q, res, args.max_new, args.tokenizer) for _ in range(c)])
            dt = time.perf_counter() - t0
            toks = sum(r["completion_tokens"] for r in res)
            ver = sum(r["spec_verify_ct"] for r in res)
            rec = {
                "conc": c, "num_requests": n, "wall_s": dt, "out_tokens": toks,
                "tok_s": toks / dt,
                "accept_len": (toks / ver) if ver else 1.0,
                "mean_latency_s": sum(r["latency"] for r in res) / len(res),
            }
            print(json.dumps(rec), flush=True)
            out_all[c] = {"summary": rec,
                          "outputs": {str(r["idx"]): r["text"] for r in sorted(res, key=lambda r: r["idx"])}}
    json.dump(out_all, open(args.out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--conc", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 16, 64])
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--min-requests", type=int, default=16)
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))
