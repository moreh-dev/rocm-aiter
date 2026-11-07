#!/usr/bin/env python3
"""Fused kernel only, tau = -inf (dbg 6): every valid key must be emitted exactly once per (split, row) with the dense
logit bits. Repeats the launch and reports every mismatch (lost / wrong / extra records) with its position.
  python3 repro_values.py --step 2 --shape 1000x3000 --iters 300"""
import argparse, collections
import torch
import fmt_harness as F
import fmt_steps as S
from test_exact import make_step

ap = argparse.ArgumentParser()
ap.add_argument("--step", type=int, default=2)
ap.add_argument("--shape", default="1000x3000")
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--dbg", type=int, default=2)
ap.add_argument("--full", action="store_true", help="run select + fallback too (dbg without bit 1)")
a = ap.parse_args()
M, N = map(int, a.shape.split("x"))
q8, k8, sc, w, ks, ke = F.gen(M, N, "rand", 1, True)
Nt = k8.shape[0]
lg = F.dense_logits(q8, k8, sc, w, ks, ke, clean=True)
st = make_step(a.step)
st.prep(M, Nt)
assert st.cap >= Nt
import sys
st.force_fallback = int(sys.argv[sys.argv.index("--dbg") + 1]) if "--dbg" in sys.argv else 2   # 2: fused only (finite tau)
out = torch.empty(M, F.TOPK, dtype=torch.int32, device="cuda")
sp, Mp = st.split, M + (M & 1)
valid = torch.isfinite(lg)
nfail = 0
hist = collections.Counter()
for it in range(a.iters):
    st.cand.fill_(0x3C3C3C3C3C3C3C3C); st.cnt.fill_(-5)
    st(q8, k8, sc, w, ks, ke, out)
    torch.cuda.synchronize()
    cand = st.cand.view(sp, Mp // 2, st.cap, 2).permute(0, 1, 3, 2).reshape(sp, Mp, st.cap)[:, :M]   # [s][m][i]
    cnt = st.cnt.view(sp, M)
    # rebuild a dense [M, Nt] image from all splits' records
    i_ar = torch.arange(st.cap, device="cuda")
    live = i_ar[None, None, :] < cnt[:, :, None].clamp(min=0)
    vals = (cand & 0xFFFFFFFF).to(torch.int32)
    idx = (cand >> 32).to(torch.int64)
    img = torch.full((M, Nt), 0x7FC00001, dtype=torch.int32, device="cuda")   # NaN marker = not emitted
    hits = torch.zeros((M, Nt), dtype=torch.int32, device="cuda")
    s_i, m_i, k_i = torch.nonzero(live, as_tuple=True)
    n_i = idx[s_i, m_i, k_i]
    ok_idx = (n_i >= 0) & (n_i < Nt)
    bad_idx = int((~ok_idx).sum())
    m_v, n_v, v_v = m_i[ok_idx], n_i[ok_idx], vals[s_i[ok_idx], m_i[ok_idx], k_i[ok_idx]]
    img[m_v, n_v] = v_v
    hits.index_put_((m_v, n_v), torch.ones_like(v_v), accumulate=True)
    ref = lg.view(torch.int32)
    tau = st.tau.view(-1)[:M]
    want = valid & (lg >= tau[:, None])          # expected candidates: valid keys >= the kernel's own row tau
    lost = want & (hits == 0)
    dup = hits > 1
    wrong = valid & (hits == 1) & (img != ref)
    extra = (~want) & (hits > 0)
    tot = int(lost.sum()) + int(dup.sum()) + int(wrong.sum()) + int(extra.sum()) + bad_idx
    if tot == 0:
        continue
    nfail += 1
    print(f"iter {it}: lost {int(lost.sum())} dup {int(dup.sum())} wrong {int(wrong.sum())} extra {int(extra.sum())} bad_idx {bad_idx}")
    for name, msk in (("lost", lost), ("wrong", wrong), ("dup", dup), ("extra", extra)):
        pts = torch.nonzero(msk).tolist()
        if not pts: continue
        rows = sorted(set(p[0] for p in pts))
        keys = [p[1] for p in pts]
        print(f"   {name}: rows {rows[:8]} keys {keys[:12]} (key%64 {[k % 64 for k in keys[:12]]})")
        if name == "wrong":
            for m_, n_ in pts[:4]:
                print(f"      row {m_} key {n_}: got {img[m_, n_].view(torch.float32).item():.6g} want {lg[m_, n_].item():.6g}")
        for m_, n_ in pts[:200]:
            hist[(name, m_ // 32, (m_ % 32) // 4, m_ % 4)] += 1
print(f"== {nfail}/{a.iters} launches with bad records; by (kind, block, wave, row-in-wave): {dict(hist.most_common(12))}")
