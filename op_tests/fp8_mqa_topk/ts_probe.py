#!/usr/bin/env python3
"""Phase timestamps of the fused kernel (build with ./build.sh -DFMT_TS). Per WG (bx, s), wave 0, s_memrealtime (100 MHz):
0 item start, 1 split 0 sampling done, 2 tau published (s 0) / tau received (s > 0), 3 EMIT done, 4 item end.
  python3 ts_probe.py --shape 576x250000 --step 2"""
import argparse, ctypes
import torch
import fmt_harness as F
import fmt_steps as S

ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="576x250000")
ap.add_argument("--step", type=int, default=2)
ap.add_argument("--kind", default="rand")
a = ap.parse_args()
M, N = map(int, a.shape.split("x"))
q8, k8, sc, w, ks, ke = F.gen(M, N, a.kind, 1, True)
st = {2: S.Step2, 3: S.Step3}[a.step]()
st.prep(M, k8.shape[0])
L = S.lib()
L.fmt_set_ts.argtypes = [ctypes.c_void_p]
rb = (M + 31) // 32
ts = torch.zeros(rb * st.split * 8, dtype=torch.int64, device="cuda")
out = torch.empty(M, F.TOPK, dtype=torch.int32, device="cuda")
for it in range(4):
    ts.zero_()
    L.fmt_set_ts(F.ptr(ts) if it == 3 else None)
    st(q8, k8, sc, w, ks, ke, out)
    torch.cuda.synchronize()
L.fmt_set_ts(None)
t = ts.view(rb, st.split, 8).cpu().double()
t0 = t[:, :, 0][t[:, :, 0] > 0].min()
us = lambda x: (x - t0) / 100.0   # 100 MHz -> us
print(f"{a.shape} step {a.step}: split {st.split}, rb {rb}")
s0 = t[:, 0, :]
print(f"  split 0: start {us(s0[:,0]).mean():7.1f} | sampled {us(s0[:,1]).mean():7.1f} | tau {us(s0[:,2]).mean():7.1f} | "
      f"emit {us(s0[:,3]).mean():7.1f} | end {us(s0[:,4]).mean():7.1f}  (means over row blocks, us from first start)")
if st.split > 1:
    so = t[:, 1:, :]
    print(f"  split>0: start {us(so[:,:,0]).mean():7.1f} | tau rx {us(so[:,:,2]).mean():7.1f} | emit {us(so[:,:,3]).mean():7.1f} | "
          f"end {us(so[:,:,4]).mean():7.1f} | max end {us(so[:,:,4]).max():7.1f}")
    print(f"  per-split EMIT duration mean {((so[:,:,3]-so[:,:,2])/100).mean():.1f} us; split-0 sample {((s0[:,1]-s0[:,0])/100).mean():.1f} us, "
          f"tau {((s0[:,2]-s0[:,1])/100).mean():.1f} us")
else:
    print(f"  sample {((s0[:,1]-s0[:,0])/100).mean():.1f} us, tau {((s0[:,2]-s0[:,1])/100).mean():.1f} us, "
          f"emit {((s0[:,3]-s0[:,2])/100).mean():.1f} us, select {((s0[:,4]-s0[:,3])/100).mean():.1f} us per block")
