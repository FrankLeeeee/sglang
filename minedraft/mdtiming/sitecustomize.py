# Profiling-only instrumentation, enabled with MD_TIMING_OUT=<path prefix>.
# Records GPU time (CUDA events on the current stream) of
#   - the target forward (TpModelWorker.forward_batch_generation) = verify / AR step
#   - the whole spec step (<SpecWorker>.forward_batch_generation) = draft + verify + glue
# for decode/verify batches only, keyed by batch size. Dumped as JSON every N steps.
import json, os, sys, threading

_OUT = os.environ.get("MD_TIMING_OUT")


def _install():
    import torch

    stats = {}  # (kind, bs) -> [n, sum_ms]
    pending = []
    lock = threading.Lock()
    counter = [0]

    def _bs_and_mode(batch):
        fm = getattr(batch, "forward_mode", None)
        if fm is None:
            return None, None
        try:
            if fm.is_idle():
                return None, None
            if fm.is_extend() and not fm.is_target_verify():
                return None, None
        except Exception:
            return None, None
        if getattr(batch, "is_extend_in_batch", False) and not fm.is_target_verify():
            return None, None
        sl = getattr(batch, "seq_lens", None)
        bs = int(sl.shape[0]) if sl is not None else None
        return bs, str(fm)

    def _flush(force=False):
        keep = []
        for kind, bs, e0, e1 in pending:
            if force:
                e1.synchronize()
            if e1.query():
                ms = e0.elapsed_time(e1)
                s = stats.setdefault(f"{kind}|{bs}", [0, 0.0])
                s[0] += 1
                s[1] += ms
            else:
                keep.append((kind, bs, e0, e1))
        pending[:] = keep

    def _dump():
        path = f"{_OUT}.{os.getpid()}.json"
        with open(path, "w") as f:
            json.dump({k: {"n": v[0], "mean_ms": v[1] / v[0]} for k, v in stats.items() if v[0]}, f)

    def wrap(cls, kind):
        orig = cls.forward_batch_generation

        def fbg(self, batch=None, *a, **kw):
            fb = kw.get("forward_batch")
            bs, _ = _bs_and_mode(batch if batch is not None else fb)
            if bs is None:
                return orig(self, batch, *a, **kw)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            out = orig(self, batch, *a, **kw)
            e1.record()
            with lock:
                pending.append((kind, bs, e0, e1))
                counter[0] += 1
                if counter[0] % 50 == 0:
                    _flush()
                    _dump()
            return out

        cls.forward_batch_generation = fbg

    from sglang.srt.managers.tp_worker import TpModelWorker
    wrap(TpModelWorker, "target")
    for mod, name in [
        ("sglang.srt.speculative.eagle_worker_v2", "EAGLEWorkerV2"),
        ("sglang.srt.speculative.dflash_worker_v2", "DFlashWorkerV2"),
        ("sglang.srt.speculative.dspark_components.dspark_worker_v2", "DSparkWorkerV2"),
    ]:
        try:
            m = __import__(mod, fromlist=[name])
            wrap(getattr(m, name), "spec")
        except Exception as e:  # class name differs in this fork
            print(f"[mdtiming] skip {mod}.{name}: {e}", flush=True)


if _OUT:
    import builtins
    _orig_import = builtins.__import__
    _done = [False]

    def _hook(name, *a, **kw):
        mod = _orig_import(name, *a, **kw)
        if not _done[0]:
            sm = sys.modules.get("sglang.srt.managers.scheduler")
            if sm is not None and hasattr(sm, "Scheduler") and hasattr(sm, "run_scheduler_process"):
                _done[0] = True
                try:
                    _install()
                    print("[mdtiming] installed", flush=True)
                except Exception as e:
                    print(f"[mdtiming] install failed: {e}", flush=True)
        return mod

    builtins.__import__ = _hook
