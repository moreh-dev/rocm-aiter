#!/usr/bin/env python3
"""Reproduce an intermittent wrong-row failure and dump per-split candidate diagnostics.
  python3 repro_flaky.py --step 2 --shape 1000x3000 --iters 300"""
import argparse
import torch
import fmt_harness as F
import fmt_steps as S
from test_exact import make_step, poison

ap = argparse.ArgumentParser()
ap.add_argument("--step", type=int, default=2)
ap.add_argument("--shape", default="1000x3000")
ap.add_argument("--kind", default="rand")
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--nopoison", action="store_true")
ap.add_argument("--sync", action="store_true", help="synchronize after the workspace poisoning, before the launch")
a = ap.parse_args()
M, N = map(int, a.shape.split("x"))
q8, k8, sc, w, ks, ke = F.gen(M, N, a.kind, 1, True)
Nt = k8.shape[0]
lg = F.dense_logits(q8, k8, sc, w, ks, ke, clean=True)   # [M, Nt], -inf outside the window
nfail = 0
for it in range(a.iters):
    st = make_step(a.step)
    out = torch.full((M, F.TOPK), -7, dtype=torch.int32, device="cuda")
    st.prep(M, Nt)
    if not a.nopoison:
        poison(st)
    if a.sync:
        torch.cuda.synchronize()
    st(q8, k8, sc, w, ks, ke, out, count_fb=True)
    torch.cuda.synchronize()
    wrong = F.check(out, lg, ks, ke)
    nw = int(wrong.sum())
    if nw == 0:
        continue
    nfail += 1
    rows = torch.nonzero(wrong).flatten().tolist()
    sp = st.split
    Mp = M + (M & 1)
    cnt = st.cnt.view(sp, M)
    cand = st.cand.view(sp, Mp // 2, st.cap, 2)
    print(f"iter {it}: {nw} wrong rows {rows[:8]} split {sp} rb {(M + 31) // 32}")
    for m in rows[:4]:
        tau = float(st.tau[m])
        row = lg[m]
        nge = int((row >= tau).sum())
        kth = float(torch.topk(row[torch.isfinite(row)], F.TOPK).values[-1])
        print(f"  row {m} (block {m // 32}, wave {(m % 32) // 4}): tau {tau:.6g}  keys>=tau {nge}  Kth {kth:.6g}")
        for s in range(sp):
            c = int(cnt[s, m])
            recs = cand[s, m // 2, :min(c, st.cap), m & 1]
            vals = (recs & 0xFFFFFFFF).to(torch.int32).view(torch.float32) if c > 0 else torch.empty(0, device="cuda")
            idx = (recs >> 32).to(torch.int64) if c > 0 else torch.empty(0, dtype=torch.int64, device="cuda")
            vmin = float(vals.min()) if c > 0 else float("nan")
            # keys of this split's range >= tau according to the dense logits
            if c > 0:
                lo, hi = int(idx.min()), int(idx.max())
            print(f"    split {s}: cnt {c:5d}  min cand {vmin:.6g}  idx range {lo if c else '-'}..{hi if c else '-'}")
            if c > 0 and vmin < tau:
                bad_slots = torch.nonzero(recs == 0x3C3C3C3C3C3C3C3C).flatten().tolist()
                good = set(idx[recs != 0x3C3C3C3C3C3C3C3C].tolist())
                b0, b1 = int(idx[recs != 0x3C3C3C3C3C3C3C3C].min()), int(idx[recs != 0x3C3C3C3C3C3C3C3C].max())
                # expected keys of this split's key range: every key n in [2688.., split end) with logit >= tau
                exp = torch.nonzero(row >= tau).flatten().tolist()
                ks7 = 2688 if a.shape == "1000x3000" else b0
                exp_s = [n for n in exp if n >= ks7]
                missing = [n for n in exp_s if n not in good]
                extra = sorted(good - set(exp_s))
                pres = idx[recs != 0x3C3C3C3C3C3C3C3C].tolist()
                print(f"      expected {len(exp_s)} present {len(good)} (dups {len(pres) - len(good)}) extra {extra[:8]}; last present slots idx {pres[-6:]}")
                print(f"      poison slots {len(bad_slots)}: {bad_slots[:12]}{'...' if len(bad_slots) > 12 else ''}")
                print(f"      missing keys {len(missing)}: {missing[:16]}  (key % 64: {[n % 64 for n in missing[:16]]}, stage: {sorted(set((n - 2688) // 64 for n in missing))})")
print(f"== {nfail}/{a.iters} iterations with wrong rows")
