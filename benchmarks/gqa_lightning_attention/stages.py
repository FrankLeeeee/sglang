"""Named GPU-time stage windows; a no-op unless a recorder is active.

``stage(name)`` brackets the work launched in a block with CUDA events. Run
under ``record_stages`` the forward is preceded by a spin kernel so the CPU is
ahead of the GPU, which makes each window pure GPU time without launch gaps.
"""

from contextlib import contextmanager, nullcontext

import torch

_spans = None

# About 20 ms on H200: longer than the CPU needs to enqueue a whole forward.
_SPIN_CYCLES = 40_000_000


def stage(name):
    if _spans is None:
        return nullcontext()
    return _span(name)


@contextmanager
def _span(name):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    yield
    end.record()
    _spans.append((name, start, end))


def record_stages(fn):
    """Run ``fn`` once and return {stage name: GPU milliseconds}."""
    global _spans
    torch.cuda.synchronize()
    _spans = []
    try:
        torch.cuda._sleep(_SPIN_CYCLES)
        fn()
        torch.cuda.synchronize()
        totals = {}
        for name, start, end in _spans:
            totals[name] = totals.get(name, 0.0) + start.elapsed_time(end)
        return totals
    finally:
        _spans = None
