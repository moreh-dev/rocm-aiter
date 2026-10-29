#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the GLM-5.3-Flash indexer kpool mixed-step cache writer (HIP):
prefill tokens + decode requests in one launch.

    python bench_kpool_mixed_write.py                         # default sweep
    python bench_kpool_mixed_write.py --mixed-tokens 32768 --mixed-batch 256 --decode-next-n 4
    python bench_kpool_mixed_write.py -o results.csv

Timings are triton.testing.do_bench medians (L2 flushed).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import triton

import aiter
from aiter.ops.kpool_compress_insert import kpool_pick_lanes

DEVICE = "cuda"
HEAD_DIM = 128
DECODE_NEXT_N = [1, 4]


def make_prefill(n_tokens, pool, page, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    n_pools = n_tokens // pool
    num_blocks = math.ceil(n_pools / page) + 2
    slot = torch.full((n_tokens,), -1, dtype=torch.int64, device=DEVICE)
    pools = torch.randperm(num_blocks * page, generator=g, device=DEVICE)[:n_pools]
    slot[torch.arange(n_pools, device=DEVICE) * pool + pool - 1] = pools
    k = torch.randn(n_tokens, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    gate = (torch.randn(n_tokens, HEAD_DIM, generator=g, device=DEVICE) * 2).bfloat16()
    ape = torch.randn(pool, HEAD_DIM, generator=g, device=DEVICE)
    kv = torch.zeros(num_blocks, page, HEAD_DIM + 4, dtype=torch.uint8, device=DEVICE)
    return k, gate, ape, slot, kv


def make_decode(B, next_n, pool, page, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # production tail ring (vLLM GLM-5.3 MTP): pool * pow2(ceil((pool + next_n - 1) / pool)), == pool for next_n = 1
    ring = pool * (1 << (-(-(pool + next_n - 1) // pool) - 1).bit_length())
    tail = torch.randn(B + 2, 2, ring, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    ctx = torch.randint(pool, 100000, (B,), generator=g, device=DEVICE)
    pos = (ctx[:, None] + torch.arange(next_n, device=DEVICE)[None]).to(torch.int32)
    blocks = torch.arange(B, dtype=torch.int32, device=DEVICE)[:, None]
    tail_slot = (blocks * ring + pos % ring).to(torch.int32)
    n_pages = math.ceil(B * next_n / page) + 2
    locs = torch.randperm(n_pages * page, generator=g, device=DEVICE)[: B * next_n]
    locs = locs.view(B, next_n).to(torch.int32)
    slot_map = torch.where(pos % pool == pool - 1, locs, torch.full_like(locs, -1))
    key = torch.randn(B, next_n, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    score = torch.randn(B, next_n, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    ape = torch.randn(pool, HEAD_DIM, generator=g, device=DEVICE)
    kv = torch.zeros(n_pages, page, HEAD_DIM + 4, dtype=torch.uint8, device=DEVICE)
    return kv, tail, tail_slot, key, score, ape, slot_map, pos


def bench(fn, warmup=5, rep=50):
    ms, p20, p80 = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
    return ms * 1e3, p20 * 1e3, p80 * 1e3


def run_mixed(args):
    rows = []
    print(f"\nmixed (pool {args.pool}, page {args.page}): prefill tokens + decode requests in one launch")
    print(f"{'tokens':>9} {'B':>6} {'next_n':>6} {'us':>9} {'p20':>9} {'p80':>9}")
    for n in args.mixed_tokens:
        for B in args.mixed_batch:
            next_n = args.decode_next_n[0]
            kv_d, tail, tail_slot, key, score, ape, slot_map, pos = make_decode(B, next_n, args.pool, args.page)
            k, gate, _, slot, kv_p = make_prefill(n, args.pool, args.page, seed=1)
            slot = torch.where(slot >= 0, slot + kv_d.shape[0] * args.page, slot)
            kv = torch.cat([kv_d, kv_p], 0)
            # the prefill rows seed tail block B; every index tensor of one launch shares the prefill slot dtype
            ring = tail.shape[2]
            tslot = (torch.full((n,), B, dtype=slot.dtype, device=DEVICE) * ring
                     + torch.arange(n, device=DEVICE, dtype=slot.dtype) % ring)
            tail_slot, slot_map, pos = (t.to(slot.dtype) for t in (tail_slot, slot_map, pos))
            tail = torch.cat([tail, torch.zeros(1, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)], 0)
            us, p20, p80 = bench(lambda: aiter.kpool_mixed_write(
                kv, ape, args.pool, True, 0, k, gate, slot, tslot, tail, tail_slot, key, score, slot_map, pos))
            print(f"{n:>9} {B:>6} {next_n:>6} {us:9.2f} {p20:9.2f} {p80:9.2f}")
            rows.append(("mixed", n, B, next_n, kpool_pick_lanes(n, args.pool), us, p20, p80, 0.0))
            del kv, tail, k, gate
            torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser(description="Benchmark the kpool mixed-step cache writer.")
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--page", type=int, default=64)
    p.add_argument("--decode-next-n", type=int, nargs="*", default=DECODE_NEXT_N, help="next_n of the decode requests (first value)")
    p.add_argument("--mixed-tokens", type=int, nargs="*", default=[2048, 32768])
    p.add_argument("--mixed-batch", type=int, nargs="*", default=[64, 256])
    p.add_argument("-o", type=str, default=None, help="save results to CSV")
    args = p.parse_args()

    rows = run_mixed(args)
    if args.o:
        import csv

        with Path(args.o).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["op", "tokens", "batch", "next_n", "lanes", "us", "p20_us", "p80_us", "TB_s"])
            w.writerows(rows)
        print(f"\nsaved {len(rows)} rows to {args.o}")


if __name__ == "__main__":
    main()
