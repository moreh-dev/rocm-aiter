# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Accuracy (and timing) of the GLM-5.3-Flash indexer kpool cache writer
``kpool_compress_insert``.

Every case runs a pure-torch reference of the vLLM prefill insert and compares
the whole kv_cache (dequantized fp8 keys and fp32 scales).
The reference rounds at the same two bf16 points as the kernel; the only
sources of difference are 1-ulp fp32 choices (exp, division, log2), so the
dequantized keys agree to one fp8 ulp and the bytes agree exactly for all but
a handful of tokens. With ``--vllm-ref <path to vllm kpool_compress.py>`` the
Triton production kernel is run too and compared byte for byte.

    python3 test_kpool_compress_insert.py
    python3 test_kpool_compress_insert.py --quick
"""

import argparse
import importlib.util
import math
import sys

import pandas as pd
import torch

import aiter
from aiter import get_gfx
from aiter.ops.kpool_compress_insert import kpool_pick_lanes
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

HEAD_DIM = 128
FP8_MAX = 448.0
HAD_SCALE = 0.08838834764831845  # 1/sqrt(128), the fp32 constant of the kernels
FP8 = torch.float8_e4m3fn
TILE = 16

# --------------------------------------------------------------------------
# torch reference of the pool math and the cache layout
# --------------------------------------------------------------------------


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard, stages of increasing stride (same order as the kernels)."""
    n = x.shape[-1]
    h = 1
    while h < n:
        v = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = v[..., 0, :], v[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape)
        h *= 2
    return x


def compress_pools(k_rows, g_rows, ape, round_scale=True):
    """k_rows, g_rows [n, pool, 128] bf16; ape [pool, 128] fp32 ->
    fp8 bytes [n, 128] uint8 and fp32 scale [n]."""
    u = g_rows.float() + ape.float()[None]
    m = u.amax(dim=1, keepdim=True)
    e = torch.exp(u - m)
    z = (e * k_rows.float()).sum(1) / e.sum(1)
    x = z.to(torch.bfloat16).float()
    y = (hadamard128(x) * HAD_SCALE).to(torch.bfloat16).float()
    a = y.abs().amax(1).clamp_min(1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(a / FP8_MAX)))
    else:
        scale = a / FP8_MAX
    q = (y / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.view(torch.uint8), scale


def _k_offsets(tok, page):
    """Byte offsets of the 128 keys of token(s) ``tok`` inside a page."""
    d = torch.arange(HEAD_DIM)
    if page > 1:
        return (
            (tok // TILE)[..., None] * (TILE * HEAD_DIM)
            + (d // TILE)[None] * (TILE * TILE)
            + (tok % TILE)[..., None] * TILE
            + (d % TILE)[None]
        )
    return d[None].expand(*tok.shape, HEAD_DIM)


def write_cache(kv, loc, q_u8, scale):
    """Store fp8 rows and scales at paged slots ``loc`` (int64, all >= 0)."""
    nb, page, cs = kv.shape
    flat = kv.view(nb, page * cs)
    pg, tok = loc // page, loc % page
    flat[pg[:, None], _k_offsets(tok, page)] = q_u8
    soff = page * HEAD_DIM + tok[:, None] * 4 + torch.arange(4)[None]
    flat[pg[:, None], soff] = scale.contiguous().view(torch.uint8).view(-1, 4)


def read_cache(kv):
    """kv_cache -> dequantized keys [nb, page, 128] fp32 and scales [nb, page]."""
    nb, page, cs = kv.shape
    flat = kv.view(nb, page * cs)
    kbytes = flat[:, : page * HEAD_DIM]
    k = kbytes[:, _k_offsets(torch.arange(page), page)]  # [nb, page, 128]
    scale = flat[:, page * HEAD_DIM :].contiguous().view(torch.float32).view(nb, page)
    return k.view(FP8).float() * scale[..., None], scale


def ref_insert(kv, k, gate, ape, slot, pool, round_scale=True):
    n = k.shape[0]
    t = torch.arange(n)
    closing = (slot >= 0) & (t >= pool - 1)
    tc = t[closing]
    if tc.numel() == 0:
        return
    idx = tc[:, None] - (pool - 1) + torch.arange(pool)[None]
    q, s = compress_pools(k[idx], gate[idx], ape, round_scale)
    write_cache(kv, slot[closing].long(), q, s)


# --------------------------------------------------------------------------
# optional byte-exact reference: the vLLM Triton kernels
# --------------------------------------------------------------------------

_VLLM = None


def load_vllm_ref(path):
    global _VLLM
    spec = importlib.util.spec_from_file_location("kpool_compress_vllm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kpool_compress_vllm"] = mod
    spec.loader.exec_module(mod)
    _VLLM = mod


def vllm_insert(kv, k, gate, ape, slot, pool, round_scale=True):
    n = slot.shape[0]
    if n < pool:
        return
    t = torch.arange(n)
    write_mask = (slot >= 0) & (t >= pool - 1)
    idx = (t - (pool - 1)).clamp_min(0)[:, None] + torch.arange(pool)[None]
    _VLLM.kpool_compress_and_write_cache(
        kv,
        k[idx],
        gate[idx],
        ape,
        slot.to(torch.int64),
        pool_size=pool,
        head_dim=HEAD_DIM,
        write_mask=write_mask,
        round_scale=round_scale,
        write_cache=True,
        return_compressed=False,
    )


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def make_prefill(lens, pool, page, slot_dtype=torch.int64, seed=0):
    """Varlen prefill batch, pools aligned to each request start, random paged
    slots."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    slots = []
    n_pools = sum(length // pool for length in lens)
    num_blocks = max(1, math.ceil(n_pools / page)) + 2
    perm = torch.randperm(num_blocks * page, device="cuda", generator=g)[
        :n_pools
    ].tolist()
    for r, length in enumerate(lens):
        for t in range(length):
            slots.append(perm.pop() if t % pool == pool - 1 else -1)
    n = len(slots)
    k = torch.randn(n, HEAD_DIM, generator=g, device="cuda").bfloat16()
    gate = (torch.randn(n, HEAD_DIM, generator=g, device="cuda") * 3).bfloat16()
    ape = torch.randn(pool, HEAD_DIM, generator=g, device="cuda")
    slot = torch.tensor(slots, dtype=slot_dtype)
    kv = torch.zeros(num_blocks, page, HEAD_DIM + 4, dtype=torch.uint8)
    return k, gate, ape, slot, kv


def compare_cache(kv_ref, kv_got, msg):
    y_ref, s_ref = read_cache(kv_ref)
    y_got, s_got = read_cache(kv_got)
    err = checkAllclose(
        y_ref, y_got, rtol=0.13, atol=2e-3, msg=f"{msg} keys (dequantized)"
    )
    err_s = checkAllclose(s_ref, s_got, rtol=1e-6, atol=0, msg=f"{msg} scales")
    diff_bytes = int((kv_ref != kv_got).sum())
    return err, err_s, diff_bytes


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


@benchmark()
def test_insert(lens, pool, page, slot_dtype=torch.int64, lanes=0, round_scale=True):
    k, gate, ape, slot, kv = make_prefill(lens, pool, page, slot_dtype)
    n = k.shape[0]
    ref = kv.clone()
    ref_insert(ref, k, gate, ape, slot, pool, round_scale)
    got = kv.clone()
    _, us = run_perftest(
        aiter.kpool_compress_insert,
        got,
        k,
        gate,
        ape,
        slot,
        pool,
        round_scale,
        lanes,
        num_iters=20,
        num_warmup=2,
    )
    err, err_s, diff_bytes = compare_cache(ref, got, "insert")
    ret = {
        "tokens": n,
        "lanes": lanes or kpool_pick_lanes(n, pool),
        "err": err,
        "scale err": err_s,
        "diff bytes": diff_bytes,
        "us": us,
        "TB/s (K+gate)": n * 2 * HEAD_DIM * 2 / us / 1e6,
    }
    if _VLLM is not None:
        exact = kv.clone()
        vllm_insert(exact, k, gate, ape, slot, pool, round_scale)
        ret["vllm diff bytes"] = int((exact != got).sum())
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test the GLM-5.3-Flash kpool prefill cache writer.",
)
parser.add_argument("--quick", action="store_true", help="fewer and smaller cases")
parser.add_argument(
    "--pool", type=int, nargs="*", default=[4, 2, 3, 8, 16], help="pool sizes"
)
parser.add_argument(
    "--page", type=int, nargs="*", default=[64, 1], help="cache page sizes"
)
parser.add_argument(
    "--vllm-ref",
    type=str,
    default=None,
    help="path to vllm's kpool_compress.py: also compare byte for byte",
)
args = parser.parse_args()

if get_gfx() != "gfx950":
    aiter.logger.info(
        "kpool writers target gfx950 (fp8 e4m3fn, permlane swaps); skipping"
    )
    sys.exit(0)
if args.vllm_ref:
    load_vllm_ref(args.vllm_ref)

tables = {}
rows = []
for pool in args.pool:
    for page in args.page:
        rows.append(test_insert([1024], pool, page))
for lanes in (16, 32, 64):
    rows.append(test_insert([2048], 4, 64, lanes=lanes))
rows.append(test_insert([1, 7, 130, 1000, 33, 4, 5], 4, 32, slot_dtype=torch.int32))
rows.append(test_insert([4099, 17, 2048, 3], 3, 32))
rows.append(test_insert([300, 77, 2048], 4, 64, round_scale=False))
if not args.quick:
    rows.append(test_insert([8192] * 8, 4, 64))
    rows.append(test_insert([131072] * 2, 4, 64))
tables["insert"] = pd.DataFrame(rows)
for name, df in tables.items():
    aiter.logger.info(
        "kpool %s summary (markdown):\n%s", name, df.to_markdown(index=False)
    )
