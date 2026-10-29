# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Accuracy (and timing) of the GLM-5.3-Flash indexer kpool mixed-step writer
``kpool_mixed_write`` (prefill insert + tail seeding + decode in one launch).

Every case runs a pure-torch reference of the vLLM semantics (prefill insert,
``kpool_seed_tail_cache``, the ordered decode walk) and compares the whole
kv_cache (dequantized fp8 keys and fp32 scales) and the whole tail cache.
The reference rounds at the same two bf16 points as the kernel; the only
sources of difference are 1-ulp fp32 choices (exp, division, log2), so the
dequantized keys agree to one fp8 ulp and the bytes agree exactly for all but
a handful of tokens (the standalone writers' tests compare byte for byte
against the vLLM Triton kernels).

    python3 test_kpool_mixed_write.py
    python3 test_kpool_mixed_write.py --quick
"""

import argparse
import math
import sys

import pandas as pd
import torch

import aiter
from aiter import get_gfx
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


def ref_seed(tail, k, gate, tail_slot, pool):
    """kpool_seed_tail_cache: a token is among the last ``pool`` tokens of its
    request when the token ``pool`` ahead has no tail slot or another block."""
    n = k.shape[0]
    ring = tail.shape[2]
    ts = tail_slot.long()
    ahead = torch.full_like(ts, -1)
    if n > pool:
        ahead[: n - pool] = ts[pool:]
    write = (ts >= 0) & ((ahead < 0) | (ahead // ring != ts // ring))
    idx = write.nonzero().flatten()
    tail[ts[idx] // ring, 0, ts[idx] % ring] = k[idx]
    tail[ts[idx] // ring, 1, ts[idx] % ring] = gate[idx]


def ref_decode(
    kv, tail, tail_slot, key, score, ape, slot_map, pos, pool, round_scale=True
):
    """The ordered walk: per request, token by token, read ring then stash."""
    B, next_n = key.shape[:2]
    ring = tail.shape[2]
    ts_c, sm_c, pos_c = tail_slot.cpu(), slot_map.cpu(), pos.cpu()
    for b in range(B):
        for i in range(next_n):
            p, ts, loc = int(pos_c[b, i]), int(ts_c[b, i]), int(sm_c[b, i])
            if p >= 0 and loc >= 0 and p % pool == pool - 1:
                blk, slot = max(ts, 0) // ring, p % pool
                k_rows = torch.empty(pool, HEAD_DIM, dtype=torch.bfloat16)
                g_rows = torch.empty_like(k_rows)
                for s in range(pool):
                    if s == slot:
                        k_rows[s], g_rows[s] = key[b, i], score[b, i]
                    else:
                        phys = (p - slot + s) % ring
                        k_rows[s], g_rows[s] = tail[blk, 0, phys], tail[blk, 1, phys]
                q, sc = compress_pools(k_rows[None], g_rows[None], ape, round_scale)
                write_cache(kv, torch.tensor([loc]), q, sc)
            if p >= 0 and ts >= 0:
                tail[ts // ring, 0, p % ring] = key[b, i]
                tail[ts // ring, 1, p % ring] = score[b, i]


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def make_prefill(
    lens, pool, page, slot_dtype=torch.int64, seed=0, ring=None, nblock_offset=0
):
    """Varlen prefill batch, pools aligned to each request start, random paged
    slots. With ``ring`` also returns the tail slot mapping for seeding."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    slots, tslots = [], []
    n_pools = sum(length // pool for length in lens)
    num_blocks = max(1, math.ceil(n_pools / page)) + 2
    perm = torch.randperm(num_blocks * page, device="cuda", generator=g)[
        :n_pools
    ].tolist()
    for r, length in enumerate(lens):
        for t in range(length):
            slots.append(perm.pop() if t % pool == pool - 1 else -1)
            tslots.append((nblock_offset + r) * (ring or pool) + t % (ring or pool))
    n = len(slots)
    k = torch.randn(n, HEAD_DIM, generator=g, device="cuda").bfloat16()
    gate = (torch.randn(n, HEAD_DIM, generator=g, device="cuda") * 3).bfloat16()
    ape = torch.randn(pool, HEAD_DIM, generator=g, device="cuda")
    slot = torch.tensor(slots, dtype=slot_dtype)
    tslot = torch.tensor(
        tslots, dtype=slot_dtype
    )  # the seed slots share the slot_mapping dtype
    kv = torch.zeros(num_blocks, page, HEAD_DIM + 4, dtype=torch.uint8)
    return k, gate, ape, slot, tslot, kv


def make_decode(
    B,
    next_n,
    pool,
    page=64,
    seed=0,
    ring=None,
    nblocks=None,
    idx_dtype=torch.int32,
    tail_pad=1,
):
    """B requests x next_n consecutive tokens from random context lengths.
    ``idx_dtype`` (int32 / int64) is the dtype of the three metadata tensors
    (vLLM keeps its slot mappings and positions in int64). ``tail_pad`` > 1
    pads every tail page to ``tail_pad`` rings (vLLM pads the tail page to the
    KV-cache page, so stride(0) >> 2 * ring * 128 and the tail can exceed 2^31
    elements): the tail is then a strided view and the requests are spread
    over all blocks."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ring = ring or pool
    nblocks = nblocks or B + 2
    tail_base = torch.randn(
        nblocks, 2, ring * tail_pad, HEAD_DIM, generator=g, device="cuda"
    ).bfloat16()
    tail = tail_base[:, :, :ring] if tail_pad > 1 else tail_base
    ctx = torch.randint(pool, 100000, (B,), generator=g, device="cuda")
    pos = (ctx[:, None] + torch.arange(next_n)[None]).to(torch.int32)
    spread = max(1, (nblocks - 1) // B) if tail_pad > 1 else 1
    blocks = (torch.arange(B, dtype=torch.int32) * spread)[:, None]
    tail_slot = (blocks * ring + pos % ring).to(torch.int32)
    n_pages = math.ceil(B * next_n / page) + 2
    locs = torch.randperm(n_pages * page, generator=g, device="cuda")[: B * next_n]
    locs = locs.view(B, next_n).to(torch.int32)
    completes = pos % pool == pool - 1
    slot_map = torch.where(completes, locs, torch.full_like(locs, -1))
    key = torch.randn(B, next_n, HEAD_DIM, generator=g, device="cuda").bfloat16()
    score = torch.randn(B, next_n, HEAD_DIM, generator=g, device="cuda").bfloat16()
    ape = torch.randn(pool, HEAD_DIM, generator=g, device="cuda")
    kv = torch.zeros(n_pages, page, HEAD_DIM + 4, dtype=torch.uint8)
    return {
        "kv": kv,
        "tail": tail,
        "tail_base": tail_base,
        "tail_slot": tail_slot.to(idx_dtype),
        "key": key,
        "score": score,
        "ape": ape,
        "slot_map": slot_map.to(idx_dtype),
        "pos": pos.to(idx_dtype),
        "pool": pool,
    }


def tail_clone(d):
    """Copy of the tail that keeps a padded tail's strides (clone() would compact the view)."""
    base = d["tail_base"].clone()
    return (
        base[:, :, : d["tail"].shape[2]]
        if d["tail"].shape[2] != base.shape[2]
        else base
    )


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
def test_mixed(
    lens, B, next_n, pool, page=64, round_scale=True, idx_dtype=torch.int64, tail_pad=1
):
    ring = pool
    # one index dtype per launch: prefill slot_mapping, seed slots and decode metadata
    d = make_decode(
        B,
        next_n,
        pool,
        page,
        seed=7,
        ring=ring,
        nblocks=B + len(lens) + 2,
        idx_dtype=idx_dtype,
        tail_pad=tail_pad,
    )
    k, gate, ape, slot, tslot, kv_p = make_prefill(
        lens, pool, page, idx_dtype, seed=11, ring=ring, nblock_offset=B
    )
    # prefill cache slots live in their own pages, after the decode pages
    slot = torch.where(slot >= 0, slot + d["kv"].shape[0] * page, slot)
    kv = torch.cat([d["kv"], kv_p], 0)
    ape = d["ape"]
    kv_ref, tail_ref = kv.clone(), tail_clone(d)
    ref_insert(kv_ref, k, gate, ape, slot, pool, round_scale)
    ref_seed(tail_ref, k, gate, tslot, pool)
    ref_decode(
        kv_ref,
        tail_ref,
        d["tail_slot"],
        d["key"],
        d["score"],
        ape,
        d["slot_map"],
        d["pos"],
        pool,
        round_scale,
    )
    kv_got, tail_got = kv.clone(), tail_clone(d)
    aiter.kpool_mixed_write(
        kv_got,
        ape,
        pool,
        round_scale,
        0,
        k,
        gate,
        slot,
        tslot,
        tail_got,
        d["tail_slot"],
        d["key"],
        d["score"],
        d["slot_map"],
        d["pos"],
    )
    torch.cuda.synchronize()
    err, err_s, diff_bytes = compare_cache(kv_ref, kv_got, "mixed")
    _, us = run_perftest(
        aiter.kpool_mixed_write,
        kv.clone(),
        ape,
        pool,
        round_scale,
        0,
        k,
        gate,
        slot,
        tslot,
        tail_clone(d),
        d["tail_slot"],
        d["key"],
        d["score"],
        d["slot_map"],
        d["pos"],
        num_iters=20,
        num_warmup=2,
    )
    tail_ok = torch.equal(tail_ref, tail_got)
    assert tail_ok, "tail ring mismatch"
    return {
        "prefill tokens": k.shape[0],
        "err": err,
        "scale err": err_s,
        "diff bytes": diff_bytes,
        "tail exact": tail_ok,
        "us": us,
    }


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test the GLM-5.3-Flash kpool mixed-step cache writer.",
)
parser.add_argument("--quick", action="store_true", help="fewer and smaller cases")
args = parser.parse_args()

if get_gfx() != "gfx950":
    aiter.logger.info(
        "kpool writers target gfx950 (fp8 e4m3fn, permlane swaps); skipping"
    )
    sys.exit(0)

tables = {}
rows = [
    test_mixed([64, 9], 1, 1, 4),
    test_mixed([4096, 17, 300], 8, 4, 4),
    test_mixed([64, 9], 1, 1, 4, idx_dtype=torch.int32),
    test_mixed([300, 17], 8, 4, 4, idx_dtype=torch.int32),
]
if not args.quick:
    rows.append(test_mixed([2048] * 4, 32, 2, 4))
    # vLLM layout: padded tail pages (strided view) for the seed + decode roles
    rows.append(test_mixed([4096, 17, 300], 8, 1, 4, tail_pad=16))
    rows.append(test_mixed([300, 77], 256, 1, 4, tail_pad=512))
tables["mixed"] = pd.DataFrame(rows)
for name, df in tables.items():
    aiter.logger.info(
        "kpool %s summary (markdown):\n%s", name, df.to_markdown(index=False)
    )
