#!/usr/bin/env python3
"""Shipped aiter fp8_mqa_logits (v35wes), clean_logits=False, repeated on fixed inputs: out pre-filled with a sentinel
before every call. Inside [ks, ke) values must equal the first run bit-for-bit; outside, the sentinel must survive
(a changed sentinel = a store that should have been dropped landed there).
  python3 repro_shipped.py --shape 1000x3000 --iters 1000 [--ours]"""
import argparse
import torch
import fmt_harness as F
import fmt_steps as S

ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="1000x3000")
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--kind", default="rand")
ap.add_argument("--noncausal", action="store_true")
ap.add_argument("--ours", action="store_true", help="our MODE_DENSE copy (fmt_dense) instead of the shipped op")
a = ap.parse_args()
M, N = map(int, a.shape.split("x"))
q8, k8, sc, w, ks, ke = F.gen(M, N, a.kind, 1, not a.noncausal)
Nt = k8.shape[0]
SENT = 0x7FC0DEAD
out = torch.empty(M, Nt, dtype=torch.float32, device="cuda")
col = torch.arange(Nt, device="cuda")[None, :]
win = (col >= ks.long().clamp(min=0)[:, None]) & (col < ke.long().clamp(max=Nt)[:, None])
ref = None
bad_in = bad_out = 0
for it in range(a.iters):
    out.view(torch.int32).fill_(SENT)
    if a.ours:
        S.dense(q8, k8, sc, w, ks, ke, out)
    else:
        F.dense_logits(q8, k8, sc, w, ks, ke, clean=False, out=out)
    torch.cuda.synchronize()
    bits = out.view(torch.int32)
    if ref is None:
        ref = bits.clone()
        print(f"first run: written outside windows {int(((bits != SENT) & ~win).sum())} (expect 0)")
        continue
    din = int(((bits != ref) & win).sum())
    dout = int(((bits != SENT) & ~win).sum())
    if din or dout:
        bad_in += din > 0; bad_out += dout > 0
        pts = torch.nonzero(((bits != ref) & win) | ((bits != SENT) & ~win)).tolist()
        print(f"iter {it}: changed inside windows {din}, stray writes outside {dout}; e.g. {pts[:6]}")
print(f"== {a.shape} {'ours' if a.ours else 'shipped'}: {bad_in} runs with in-window differences, {bad_out} runs with stray writes, of {a.iters - 1}")
