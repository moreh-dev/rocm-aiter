#!/usr/bin/env python3
"""Per-row diagnostics of the distributed tau (build: ./build.sh -DFMT_TS -DFMT_DBGTAU)."""
import argparse, ctypes, collections
import torch
import fmt_harness as F
import fmt_steps as S
ap = argparse.ArgumentParser(); ap.add_argument("--shape", default="4096x24000"); ap.add_argument("--split", type=int, default=8)
a = ap.parse_args()
M, N = map(int, a.shape.split("x"))
q8, k8, sc, w, ks, ke = F.gen(M, N, "rand", 1, True)
L = S.lib(); L.fmt_set_ts.argtypes = [ctypes.c_void_p]
st = S.Step2(split=a.split); st.prep(M, k8.shape[0]); st.force_fallback = 2
dbg = torch.full((M * 24,), -9, dtype=torch.int64, device="cuda")
out = torch.empty(M, F.TOPK, dtype=torch.int32, device="cuda")
L.fmt_set_ts(F.ptr(dbg)); st(q8, k8, sc, w, ks, ke, out); torch.cuda.synchronize(); L.fmt_set_ts(None)
d = dbg.view(M, 24).cpu()
tau = st.tau.cpu()
bad = [m for m in range(M) if torch.isinf(tau[m])]
good = [m for m in range(M) if not torch.isinf(tau[m])]
print("rows -inf", len(bad), "finite", len(good))
for name, rows in (("good", good[:3]), ("bad", bad[:6])):
    for m in rows:
        r = d[m].tolist()
        print(f"{name} row {m} (r%32={m%32}): claimer s={r[0]} warp={r[1]} tgt={r[2]} act={r[3]} kept={r[4]} NR={r[5]} ns_table={r[7]} table={r[8:8+a.split]}")
c = collections.Counter((int(d[m,0]), int(d[m,1])) for m in bad)
print("bad rows by claimer (s, warp):", dict(c.most_common(10)))
c2 = collections.Counter((int(d[m,0]), int(d[m,1])) for m in good)
print("good rows by claimer (s, warp):", dict(c2.most_common(10)))
