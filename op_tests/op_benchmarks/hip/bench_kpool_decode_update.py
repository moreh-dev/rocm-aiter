#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the GLM-5.3-Flash indexer kpool decode cache writer (HIP).

    python bench_kpool_decode_update.py                         # default sweep
    python bench_kpool_decode_update.py --decode-batch 256 --decode-next-n 1 4
    python bench_kpool_decode_update.py -o results.csv

Timings are triton.testing.do_bench medians (L2 flushed). Bandwidth counts
the compulsory bytes only: per decode request 512 B K/gate + 512 B stash
(+ 1536 B ring + 132 B pooled key when a pool closes).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import triton

import aiter

DEVICE = "cuda"
HEAD_DIM = 128
DECODE_BATCH = [1, 64, 256, 1024, 4096]
DECODE_NEXT_N = [1, 4]


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


def run_decode(args):
    rows = []
    print(f"\ndecode (pool {args.pool}, page {args.page})")
    print(f"{'B':>6} {'next_n':>6} {'us':>9} {'p20':>9} {'p80':>9} {'TB/s':>6}")
    for B in args.decode_batch:
        for next_n in args.decode_next_n:
            kv, tail, tail_slot, key, score, ape, slot_map, pos = make_decode(B, next_n, args.pool, args.page)
            us, p20, p80 = bench(lambda: aiter.kpool_decode_update(kv, tail, tail_slot, key, score, ape, slot_map, pos, args.pool))
            closes = int((slot_map >= 0).sum())
            nbytes = B * next_n * 1024 + closes * (1536 + 132)
            tbs = nbytes / us / 1e6
            print(f"{B:>6} {next_n:>6} {us:9.2f} {p20:9.2f} {p80:9.2f} {tbs:6.2f}")
            rows.append(("decode", 0, B, next_n, 64, us, p20, p80, tbs))
    return rows


def main():
    p = argparse.ArgumentParser(description="Benchmark the kpool decode cache writer.")
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--page", type=int, default=64)
    p.add_argument("--decode-batch", type=int, nargs="*", default=DECODE_BATCH)
    p.add_argument("--decode-next-n", type=int, nargs="*", default=DECODE_NEXT_N)
    p.add_argument("-o", type=str, default=None, help="save results to CSV")
    args = p.parse_args()

    rows = run_decode(args)
    if args.o:
        import csv

        with Path(args.o).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["op", "tokens", "batch", "next_n", "lanes", "us", "p20_us", "p80_us", "TB_s"])
            w.writerows(rows)
        print(f"\nsaved {len(rows)} rows to {args.o}")


if __name__ == "__main__":
    main()
