# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf sweep for `aiter.ops.fp8_mqa_logits.fp8_mqa_logits_topk`.

Graded against a top-K of the fp32 logits `fp8_mqa_logits` writes for the same
inputs: per row, the fp32 logits at the picked indices must be exactly the K largest
of the row (any member of an exact fp32 tie may be picked), and a row with len <= K
must read 0..len-1 then -1. Timed against what it replaces: `fp8_mqa_logits` + aiter's
`top_k_per_row_prefill`.

gfx950 only (the logits kernel is), nh=32 / hd=128.
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.fp8_mqa_logits import fp8_mqa_logits, fp8_mqa_logits_topk
from aiter.test_common import run_perftest
from op_tests.test_fp8_mqa_logits import _num_requests, make_windows

torch.set_default_device("cuda")

NUM_HEADS, HEAD_DIM = 32, 128


def check_topk(idx, logits, ks, ke, k):
    """Number of rows that break the contract (0 = pass)."""
    s_k = logits.shape[1]
    lens = (ke.clamp(max=s_k) - ks.clamp(min=0)).long()
    bad = 0
    short = lens <= k
    if short.any():
        ar = torch.arange(k)
        want = torch.where(ar[None] < lens[short, None], ar[None], -1)
        bad += int((idx[short].long() != want).any(1).sum())
    long_ = ~short
    if long_.any():
        rows = logits[long_]
        want = rows.topk(k, dim=1).values.sort(dim=1).values
        cols = idx[long_].long() + ks[long_, None].long()
        in_win = (cols >= ks[long_, None]) & (cols < ke[long_, None])
        got = rows.gather(1, cols.clamp(0, s_k - 1)).sort(dim=1).values
        distinct = (cols.sort(dim=1).values.diff(dim=1) > 0).all(1)
        bad += int((~(in_win.all(1) & distinct & (got == want).all(1))).sum())
    return bad


def test_fp8_mqa_logits_topk(s_q, s_k, k, window):
    torch.manual_seed(0)
    q_fp8 = torch.randn(s_q, NUM_HEADS, HEAD_DIM).to(dtypes.fp8)
    kv_fp8 = torch.randn(s_k, HEAD_DIM).to(dtypes.fp8)
    scales = torch.rand(s_k) + 0.5
    weights = torch.randn(s_q, NUM_HEADS)
    ks, ke = make_windows(s_q, s_k, window)

    logits = fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke)
    idx = torch.full((s_q, k), -7, dtype=dtypes.i32)
    fp8_mqa_logits_topk(q_fp8, kv_fp8, scales, weights, ks, ke, idx)
    bad = check_topk(idx, logits, ks, ke, k)
    assert bad == 0, f"{bad} of {s_q} rows differ from the fp32 top-{k}"
    del logits

    def _base():
        lg = fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits=False)
        aiter.top_k_per_row_prefill(lg, ks, ke, idx, None, s_q, lg.stride(0), lg.stride(1), k)

    def _fused():
        fp8_mqa_logits_topk(q_fp8, kv_fp8, scales, weights, ks, ke, idx)

    _, base_us = run_perftest(_base)
    _, fused_us = run_perftest(_fused)
    return {
        "s_q": s_q, "s_k": s_k, "k": k, "window": window, "rows ok": True,
        "logits+topk us": base_us, "fused us": fused_us, "speedup": base_us / fused_us,
    }


def main():
    if get_gfx() != "gfx950":
        aiter.logger.warning("fp8_mqa_logits_topk unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(description="fp8_mqa_logits_topk sweep")
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (31, 1000),
            (100, 4096),  # <= 4096 keys: one wave per row holds the whole row
            (300, 4160),  # past it: filter pass first
            (2048, 8192),
            (8192, 16384),
            (4096, 32768),
            (512, 40000),  # past the filter's 32k resident keys: re-read per pass
            (1024, 131072),
        ],
        help="(s_q, s_k) pairs",
    )
    parser.add_argument("-k", type=int, nargs="*", default=[64, 512])
    parser.add_argument(
        "-w",
        "--window",
        type=str,
        nargs="*",
        default=["causal", "misaligned", "empty", "past_end", "multi_req"],
        choices=["causal", "misaligned", "empty", "past_end", "multi_req"],
    )
    args = parser.parse_args()
    df = []
    for (s_q, s_k), k, win in itertools.product(args.shapes, args.k, args.window):
        if win == "multi_req" and _num_requests(s_q, s_k) < 2:
            continue
        df.append(test_fp8_mqa_logits_topk(s_q, s_k, k, win))
    df = pd.DataFrame(df)
    aiter.logger.info("fp8_mqa_logits_topk summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
