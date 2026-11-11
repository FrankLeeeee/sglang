"""MineDraft (batch-parallel speculative decoding) on SGLang: target engine on GPU0, draft engine on GPU1.

Port of the scheduling loop of the official vLLM plugin (electron-shaders/MineDraft@272c855,
ParallelSpecDecodeWorker._run_speculative_decoding_step + Scheduler batch_flag logic):
  - running requests carry a cohort flag; the engine keeps 2m requests in flight;
  - each step verifies one cohort with the proposals drafted for it in the previous step, while the
    other cohort is drafted concurrently on the draft GPU; then the cohorts swap;
  - new requests join the cohort that is drafted next (the smaller one in steady state);
  - with a single cohort it falls back to plain sequential SD (draft, then verify).
Both models are standalone LMs served by unmodified SGLang servers. Verification is one target
extend of the committed prefix + k draft tokens (prefix KV comes from the radix cache) with the
target's top-1 at each of the k+1 positions; greedy acceptance on the client (the official code
accepts on the target rank). Mode "sd" runs the same harness sequentially (draft all, verify all).
"""
import argparse, asyncio, json, time

import aiohttp
from transformers import AutoTokenizer


class Req:
    __slots__ = ("idx", "ids", "n_prompt", "props", "done", "steps", "accepted")

    def __init__(self, idx, ids):
        self.idx, self.ids, self.n_prompt = idx, list(ids), len(ids)
        self.props, self.done, self.steps, self.accepted = None, False, 0, 0


class Engines:
    def __init__(self, session, target_url, draft_url, k, max_new, eos_ids):
        self.s, self.t, self.d = session, target_url, draft_url
        self.k, self.max_new, self.eos = k, max_new, set(eos_ids)

    async def _post(self, url, payload):
        async with self.s.post(url + "/generate", json=payload) as r:
            out = await r.json()
        if isinstance(out, dict) and "output_ids" not in out:
            raise RuntimeError(out)
        return out

    async def draft(self, reqs):
        """Greedy k-token proposals for each request from the draft engine."""
        if not reqs:
            return
        ks = [max(1, min(self.k, self.max_new - (len(r.ids) - r.n_prompt) - 1)) for r in reqs]
        payload = {
            "input_ids": [r.ids for r in reqs],
            "sampling_params": [{"temperature": 0, "max_new_tokens": kk, "ignore_eos": True} for kk in ks],
        }
        outs = await self._post(self.d, payload)
        for r, o in zip(reqs, outs):
            r.props = list(o["output_ids"])

    async def verify(self, reqs):
        """Score prefix + proposals on the target in one extend; greedy-accept; append tokens."""
        if not reqs:
            return 0
        payload = {
            "input_ids": [r.ids + r.props for r in reqs],
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": [len(r.ids) - 1 for r in reqs],
            "top_logprobs_num": 1,
        }
        outs = await self._post(self.t, payload)
        new_total = 0
        for r, o in zip(reqs, outs):
            # With logprob_start_len = len(ids) - 1, entry j is the target's top-1 for the token at
            # position len(ids) - 1 + j (entry 0 is None); the generated token follows the last draft.
            top = o["meta_info"]["input_top_logprobs"]
            k = len(r.props)
            preds = [top[j][0][1] for j in range(1, k + 1)] + [o["meta_info"]["output_token_logprobs"][0][1]]
            n = 0
            while n < k and r.props[n] == preds[n]:
                n += 1
            new = r.props[:n] + [preds[n]]
            # stop at EOS / budget
            budget = self.max_new - (len(r.ids) - r.n_prompt)
            new = new[:budget]
            for j, t in enumerate(new):
                if t in self.eos:
                    new = new[: j + 1]
                    break
            r.ids += new
            r.steps += 1
            r.accepted += len(new)
            r.props = None
            new_total += len(new)
            if (len(r.ids) - r.n_prompt) >= self.max_new or (new and new[-1] in self.eos):
                r.done = True
        return new_total


async def run_point(eng, prompts, C, n_requests, mode):
    queue = list(range(n_requests))
    finished = []

    def admit(n):
        out = []
        for _ in range(n):
            if queue:
                i = queue.pop(0)
                out.append(Req(i, prompts[i % len(prompts)]))
        return out

    t0 = time.perf_counter()
    steps = 0
    tim = {"draft": 0.0, "verify": 0.0, "md_step": 0.0}
    if mode == "ar":  # target alone, same client, plain greedy decode
        batch = admit(C)
        async def one(r):
            o = await eng._post(eng.t, {"input_ids": r.ids, "sampling_params": {
                "temperature": 0, "max_new_tokens": eng.max_new, "stop_token_ids": sorted(eng.eos)}})
            out = list(o["output_ids"])
            r.ids += out; r.steps += len(out); r.done = True
        async def lane(r):
            while r is not None:
                await one(r)
                finished.append(r)
                nxt = admit(1)
                r = nxt[0] if nxt else None
        await asyncio.gather(*[lane(r) for r in batch])
    elif mode == "sd":
        batch = admit(C)
        while batch:
            ta = time.perf_counter()
            await eng.draft(batch)
            tb = time.perf_counter()
            await eng.verify(batch)
            tc = time.perf_counter()
            tim["draft"] += tb - ta; tim["verify"] += tc - tb
            steps += 1
            keep = [r for r in batch if not r.done]
            finished += [r for r in batch if r.done]
            batch = keep + admit(C - len(keep))
    else:  # minedraft
        A = admit((C + 1) // 2)  # verified next
        B = admit(C // 2)        # drafted next
        await eng.draft(A)       # first step has no proposals: draft A serially (reference: plain SD step)
        while A or B:
            if not B:  # one cohort left: plain SD (reference falls back when a batch is empty)
                await eng.verify(A)
                steps += 1
                done = [r for r in A if r.done]
                finished += done
                A = [r for r in A if not r.done] + admit(len(done))
                if A:
                    await eng.draft(A)
                A, B = A, B
                continue
            ta = time.perf_counter()
            await asyncio.gather(eng.verify(A), eng.draft(B))
            tim["md_step"] += time.perf_counter() - ta
            steps += 1
            done = [r for r in A if r.done]
            finished += done
            A = [r for r in A if not r.done] + admit(len(done))
            # new requests in A have no proposals yet; A is drafted next step (after the swap)
            for r in A:
                r.props = None
            A, B = B, A
            # B (old A) is drafted next while A (old B, already drafted) is verified
    dt = time.perf_counter() - t0
    toks = sum(len(r.ids) - r.n_prompt for r in finished)
    verifies = sum(r.steps for r in finished)
    return {"mode": mode, "conc": C, "num_requests": len(finished), "wall_s": dt, "out_tokens": toks,
            "tok_s": toks / dt, "accept_len": toks / max(1, verifies), "engine_steps": steps,
            "mean_ms": {k: 1e3 * v / max(1, steps) for k, v in tim.items()}}, finished


async def main(a):
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    kw = {"enable_thinking": False} if "qwen3" in a.tokenizer.lower() else {}
    texts = [tok.apply_chat_template([{"role": "user", "content": json.loads(l)["prompt"]}],
                                     tokenize=False, add_generation_prompt=True, **kw)
             for l in open(a.prompts)]
    prompts = [tok.encode(t, add_special_tokens=False) for t in texts]
    eos = [tok.eos_token_id] + list(a.extra_eos)
    res = {}
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600), connector=conn) as s:
        eng = Engines(s, a.target_url, a.draft_url, a.k, a.max_new, eos)
        await run_point(eng, prompts, 2, 4, "sd")  # warmup
        for mode in a.modes:
            for C in a.conc:
                n = max(a.min_requests, a.rounds * C)
                rec, fin = await run_point(eng, prompts, C, n, mode)
                print(json.dumps(rec), flush=True)
                res[f"{mode}|{C}"] = {"summary": rec, "outputs": {str(r.idx): tok.decode(r.ids[r.n_prompt:]) for r in fin}}
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-url", default="http://127.0.0.1:30000")
    ap.add_argument("--draft-url", default="http://127.0.0.1:30001")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--conc", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 16, 64, 128])
    ap.add_argument("--modes", type=lambda s: s.split(","), default=["sd", "minedraft"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--min-requests", type=int, default=16)
    ap.add_argument("--extra-eos", type=lambda s: [int(x) for x in s.split(",") if x], default=[])
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
