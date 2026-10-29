# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the GLM-5.3-Flash indexer kpool prefill cache writer (HIP):
the pool insert, and with --seed-tail the whole prefill step (insert + tail
ring seeding in one launch).

    python bench_kpool_compress_insert.py                         # default sweep
    python bench_kpool_compress_insert.py --insert-tokens 32768 1048576 --seed-tail
    python bench_kpool_compress_insert.py -o results.csv

Timings are triton.testing.do_bench medians (L2 flushed). Bandwidth counts
the compulsory bytes only: 512 B read + 33 B written per prefill token
(APE and slots hit L2).
"""

from __future__ import annotations

import argparse
import functools
import math
from pathlib import Path

import torch
import triton

import aiter
from aiter.ops.kpool_compress_insert import kpool_pick_lanes

DEVICE = "cuda"
HEAD_DIM = 128
INSERT_TOKENS = [2048, 8192, 32768, 131072, 1048576]


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


def bench(fn, warmup=5, rep=50):
    ms, p20, p80 = triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]
    )
    return ms * 1e3, p20 * 1e3, p80 * 1e3


def run_insert(args):
    rows = []
    op = "insert+seed" if args.seed_tail else "insert"
    print(f"\n{op} (pool {args.pool}, page {args.page}, ring {args.ring or args.pool})")
    print(f"{'tokens':>9} {'lanes':>5} {'us':>9} {'p20':>9} {'p80':>9} {'TB/s':>6}")
    for n in args.insert_tokens:
        k, gate, ape, slot, kv = make_prefill(n, args.pool, args.page)
        lanes = args.lanes or kpool_pick_lanes(n, args.pool)
        tail = tslot = None
        if args.seed_tail:
            # one request in tail block 1: its last pool_size rows are seeded
            ring = args.ring or args.pool
            tail = torch.zeros(
                3, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
            )
            tslot = ring + torch.arange(n, device=DEVICE, dtype=slot.dtype) % ring
        us, p20, p80 = bench(
            functools.partial(
                aiter.kpool_compress_insert,
                kv,
                k,
                gate,
                ape,
                slot,
                args.pool,
                True,
                lanes,
                tail,
                tslot,
            )
        )
        tbs = n * (2 * HEAD_DIM * 2 + 33) / us / 1e6
        print(f"{n:>9} {lanes:>5} {us:9.2f} {p20:9.2f} {p80:9.2f} {tbs:6.2f}")
        rows.append((op, n, 0, 0, lanes, us, p20, p80, tbs))
        del k, gate, ape, slot, kv
        torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser(description="Benchmark the kpool prefill cache writer.")
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--page", type=int, default=64)
    p.add_argument(
        "--lanes",
        type=int,
        default=0,
        help="force lanes per pool for insert (16/32/64)",
    )
    p.add_argument("--insert-tokens", type=int, nargs="*", default=INSERT_TOKENS)
    p.add_argument(
        "--seed-tail",
        action="store_true",
        help="also seed the tail ring (the full prefill step)",
    )
    p.add_argument(
        "--ring",
        type=int,
        default=0,
        help="tail ring size for --seed-tail (default pool_size)",
    )
    p.add_argument("-o", type=str, default=None, help="save results to CSV")
    args = p.parse_args()

    rows = run_insert(args)
    if args.o:
        import csv

        with Path(args.o).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "op",
                    "tokens",
                    "batch",
                    "next_n",
                    "lanes",
                    "us",
                    "p20_us",
                    "p80_us",
                    "TB_s",
                ]
            )
            w.writerows(rows)
        print(f"\nsaved {len(rows)} rows to {args.o}")


if __name__ == "__main__":
    main()
