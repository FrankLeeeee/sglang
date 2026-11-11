"""Device boundary for running an EAGLE draft worker on its own GPU (MineDraft prototype).

The target owns the schedule batch on its GPU. Each draft entry point gets shallow proxies of
its arguments whose tensors live on the draft GPU; after the call, attributes the draft wrote
on a proxy are copied back to the original object on the target GPU.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any

import torch

# Heavy or shared objects that must never be copied or walked.
_SKIP_ATTRS = frozenset(
    {
        "reqs",
        "tree_cache",
        "req_to_token_pool",
        "token_to_kv_pool_allocator",
        "model_config",
        "model_runner",
        "attn_backend",
        "decoding_reqs",
        "chunked_req",
        "grammars",
    }
)
_MAX_DEPTH = 3


def _fields(obj: Any):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [f.name for f in dataclasses.fields(obj)]
    struct_fields = getattr(type(obj), "__struct_fields__", None)
    if struct_fields is not None:
        return list(struct_fields)
    if hasattr(obj, "__dict__"):
        return list(vars(obj).keys())
    return []


def to_device(value: Any, device: torch.device, depth: int = 0) -> Any:
    """Copy of value with every CUDA tensor on device; plain objects are shallow-copied."""
    if isinstance(value, torch.Tensor):
        if value.is_cuda and value.device != device:
            return value.to(device, non_blocking=True)
        return value
    if depth >= _MAX_DEPTH or value is None or isinstance(value, (int, float, str, bool, bytes)):
        return value
    if isinstance(value, list):
        return [to_device(v, device, depth + 1) for v in value]
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return tuple(to_device(v, device, depth + 1) for v in value)
    if isinstance(value, dict):
        return {k: to_device(v, device, depth + 1) for k, v in value.items()}
    names = _fields(value)
    if not names or isinstance(value, torch.nn.Module):
        return value
    proxy = copy.copy(value)
    for name in names:
        if name in _SKIP_ATTRS:
            continue
        try:
            current = getattr(value, name)
        except AttributeError:
            continue
        moved = to_device(current, device, depth + 1)
        if moved is not current:
            try:
                object.__setattr__(proxy, name, moved)
            except (AttributeError, TypeError):
                pass
    return proxy


def _is_plain_object(value: Any) -> bool:
    return (
        value is not None
        and not isinstance(value, (torch.Tensor, int, float, str, bool, bytes, list, tuple, dict, torch.nn.Module))
        and bool(_fields(value))
    )


def snapshot(proxy: Any, depth: int = 0) -> dict:
    """Attribute identities of proxy (and of its nested plain objects) before the callee runs."""
    snap = {}
    for name in _fields(proxy):
        if name in _SKIP_ATTRS:
            continue
        value = getattr(proxy, name, None)
        nested = snapshot(value, depth + 1) if depth < _MAX_DEPTH and _is_plain_object(value) else None
        snap[name] = (value, nested)
    return snap


def write_back(original: Any, proxy: Any, before: dict, device: torch.device, depth: int = 0) -> None:
    """Copy attributes the callee rebound on proxy, or on its nested objects, back onto original."""
    if original is None or proxy is None or original is proxy:
        return
    for name in _fields(proxy):
        if name in _SKIP_ATTRS:
            continue
        new = getattr(proxy, name, None)
        old, nested = before.get(name, (None, None))
        if new is old:
            orig_value = getattr(original, name, None)
            if nested is not None and orig_value is not new:
                write_back(orig_value, new, nested, device, depth + 1)
            continue
        try:
            object.__setattr__(original, name, to_device(new, device))
        except (AttributeError, TypeError):
            pass


class DraftDeviceAllocator:
    """Shared KV-slot allocator seen from the draft GPU: slot ids move across devices."""

    def __init__(self, base, *, draft_device: torch.device, target_device: torch.device):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_draft_device", draft_device)
        object.__setattr__(self, "_target_device", target_device)

    def __getattr__(self, name):
        attr = getattr(self._base, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            args = to_device(list(args), self._target_device)
            kwargs = to_device(kwargs, self._target_device)
            out = attr(*args, **kwargs)
            return to_device(out, self._draft_device)

        return call

    def __setattr__(self, name, value):
        setattr(self._base, name, value)
