"""Fused per-head RMSNorm/cache writes and grouped Lightning scores."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, Y, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + d, d < D, 0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    tl.store(Y + row * D + d, y, d < D)


def rms_norm_per_head(x, eps=1e-6):
    y = torch.empty_like(x)
    _norm[(x.numel() // x.shape[-1],)](
        x, y, x.shape[-1], eps, triton.next_power_of_2(x.shape[-1])
    )
    return y


@triton.jit
def _cache_write(X, CACHE, SLOTS, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row).to(tl.int64)
    value = tl.load(X + row * WIDTH + col, col < WIDTH, 0)
    tl.store(CACHE + slot * WIDTH + col, value, col < WIDTH)


def cache_write(x, cache, slots):
    width = x.shape[-1] * x.shape[-2]
    _cache_write[(x.shape[0], triton.cdiv(width, 256))](x, cache, slots, width, 256)


@triton.jit
def _score(
    Q,
    K,
    W,
    R2T,
    REQ,
    POS,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    STRIDE_R: tl.constexpr,
    G: tl.constexpr,
    J: tl.constexpr,
    IK: tl.constexpr,
    MQ: tl.constexpr,
    H: tl.constexpr,
    BN: tl.constexpr,
):
    tile, group, kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # A query tile belongs to one request. Decode uses MQ=1.
    rh = tl.arange(0, MQ * H)
    row = tile * MQ + rh // H
    head = rh % H
    d = tl.arange(0, 64)
    n = kb * BN + tl.arange(0, BN)
    request = tl.load(REQ + tile * MQ)
    slots = tl.load(R2T + request * STRIDE_R + n, n < N, 0).to(tl.int64)
    kg = 0 if IK == 1 else group
    q = tl.load(
        Q + (row[:, None] * G * J + group * J + head[:, None]) * 64 + d[None, :],
        (row[:, None] < M) & (head[:, None] < J),
        0,
    )
    k = tl.load(K + (slots[None, :] * IK + kg) * 64 + d[:, None], n[None, :] < N, 0)
    dot = tl.maximum(tl.dot(q, k), 0.0)
    w = tl.load(W + row * G * J + group * J + head, (row < M) & (head < J), 0).to(
        tl.float32
    )
    weighted = (dot * w[:, None]).reshape(MQ, H, BN)
    scores = tl.sum(weighted, axis=1)
    qr = tile * MQ + tl.arange(0, MQ)
    pos = tl.load(POS + qr, qr < M, -1)
    scores = tl.where(n[None, :] <= pos[:, None], scores, -float("inf"))
    tl.store(
        OUT + (group * M + qr[:, None]) * N + n[None, :],
        scores,
        (qr[:, None] < M) & (n[None, :] < N),
    )


def lightning_scores(
    q,
    k_cache,
    w,
    req_to_token,
    request_ids,
    positions,
    context_len,
    groups,
    query_tile=4,
):
    """Return [G, M, N], fusing ReLU and head reduction into tensor-core GEMM.

    No RoPE or score scaling. All rows in a query tile must share a request.
    Keys can be shared (IK=1) or per-group (IK=G).
    """
    m, heads, dim = q.shape
    assert dim == 64 and heads % groups == 0
    assert k_cache.shape[1] in (1, groups)
    j = heads // groups
    scores = torch.empty((groups, m, context_len), device=q.device, dtype=torch.float32)
    _score[(triton.cdiv(m, query_tile), groups, triton.cdiv(context_len, 128))](
        q,
        k_cache,
        w,
        req_to_token,
        request_ids,
        positions,
        scores,
        m,
        context_len,
        req_to_token.shape[1],
        groups,
        j,
        k_cache.shape[1],
        query_tile,
        max(16 // query_tile, triton.next_power_of_2(j)),
        128,
        num_warps=4,
    )
    return scores
