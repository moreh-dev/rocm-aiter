#!/usr/bin/env python3
"""Distributed tau (forced SplitN) vs single-split tau on the same data (fused kernel only), per row.
  python3 cmp_tau.py --shape 4096x24000 --split 8"""
import argparse
import torch
import fmt_harness as F
import fmt_steps as S

ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="4096x24000")
ap.add_argument("--split", type=int, default=8)
ap.add_argument("--kind", default="rand")
a = ap.parse_args()
M, N = map(int, a.shape.split("x"))
q8, k8, sc, w, ks, ke = F.gen(M, N, a.kind, 1, True)
Nt = k8.shape[0]
out = torch.empty(M, F.TOPK, dtype=torch.int32, device="cuda")
res = {}
for sp in (1, a.split):
    st = S.Step2(split=sp); st.prep(M, Nt); st.force_fallback = 2
    st.tau.fill_(1.2345e30)
    st(q8, k8, sc, w, ks, ke, out); torch.cuda.synchronize()
    res[sp] = (st.tau.clone(), st)
t1, td = res[1][0], res[a.split][0]
sd = res[a.split][1]
print(f"split used: 1 -> {res[1][1].split}, forced -> {sd.split}")
fin = torch.isfinite(t1) & torch.isfinite(td)
print("rows", M, "tau=-inf single", int(torch.isinf(t1).sum()), "dist", int(torch.isinf(td).sum()),
      "poison left (1.2345e30) dist", int((td == 1.2345e30).sum()))
d = (td - t1)[fin]
print("finite rows", int(fin.sum()), " dist-single: mean", float(d.mean()) if d.numel() else 0, "min", float(d.min()) if d.numel() else 0,
      "max", float(d.max()) if d.numel() else 0)
bad = torch.nonzero(fin & ((td - t1).abs() > 1e-3 * t1.abs().clamp(min=1))).flatten()
print("rows differing", bad.numel(), bad[:10].tolist())
if bad.numel():
    m = int(bad[0]); Mp = M + (M & 1)
    cnt = sd.scnt.view(sd.split, M)[:, m].tolist()
    print(f"row {m}: tau single {float(t1[m]):.4f} dist {float(td[m]):.4f}; dist per-chunk counts {cnt} (sum {sum(cnt)})")
    seg = sd.smp[:sd.split * Mp * sd.nsbc].view(sd.split, Mp // 2, sd.nsbc, 2)
    for c in range(min(3, sd.split)):
        recs = seg[c, m // 2, :cnt[c], m & 1]
        vals = (recs & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
        idx = (recs >> 32)
        print(f"  chunk {c}: n {cnt[c]} vals[min,max] {float(vals.min()) if cnt[c] else 0:.3f},{float(vals.max()) if cnt[c] else 0:.3f} "
              f"idx[min,max] {int(idx.min()) if cnt[c] else 0},{int(idx.max()) if cnt[c] else 0}")
ninf = torch.isinf(td) & ~torch.isinf(t1)
rows = torch.nonzero(ninf).flatten().cpu()
import collections
byr = collections.Counter((rows % 32).tolist()); byb = collections.Counter((rows // 32).tolist())
print("dist -inf rows by row%32:", dict(sorted(byr.items())))
print("dist -inf rows: blocks with any", len(byb), "of", (M + 31) // 32, "; per-block counts (first 12):", dict(sorted(byb.items())[:12]))
