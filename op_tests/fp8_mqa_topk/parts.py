#!/usr/bin/env python3
"""Per-part timing of the fused pipelines on one chunk (each part timed alone, cache flushed before each call)."""
import argparse, statistics
import torch
import fmt_harness as F, fmt_steps as S

ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="16384x28672")
ap.add_argument("--iters", type=int, default=15)
a = ap.parse_args()
M, N = map(int, a.shape.split("x"))
q8, k8, sc, w, ks, ke = F.gen(M, N)
L, p, st = S.lib(), F.ptr, F.stream
s0 = S.Step0(); out = torch.empty(M, F.TOPK, dtype=torch.int32, device="cuda")
s0(q8, k8, sc, w, ks, ke, out); torch.cuda.synchronize()
lg = torch.empty(M, N, device="cuda")
SS = s0.S
ksmp = k8[::SS].contiguous(); scsmp = sc[::SS].contiguous(); ks_s = (ks + SS - 1) // SS; ke_s = (ke + SS - 1) // SS
done = torch.zeros(64, dtype=torch.int32, device="cuda")
print("sample-pass grid choice (M, Ns):", S.choice(M, s0.Ns))
def pre():
    k_s = k8[::SS].contiguous(); sc_s = sc[::SS].contiguous(); ks_s = (ks + SS - 1) // SS; ke_s = (ke + SS - 1) // SS
    L.fmt_dense(p(q8), p(k_s), p(sc_s), p(w), p(ks_s), p(ke_s), p(s0.ls), M, s0.Ns, s0.Ns, 0, -1, st())
    L.fmt_tau(p(s0.ls), s0.Ns, SS, p(ks), p(ke), M, N, 512, s0.cap, p(s0.tau), p(s0.fbc), st())
parts = {
    "aiter_dense(clean=0)": lambda: F.dense_logits(q8, k8, sc, w, ks, ke, clean=False, out=lg),
    "our_dense": lambda: S.dense(q8, k8, sc, w, ks, ke, lg),
    "pre(torch+sample+tau)": pre,
    "emit": lambda: L.fmt_emit(p(q8), p(k8), p(sc), p(w), p(ks), p(ke), p(s0.tau), p(s0.cand), p(s0.cnt), s0.cap, M, N, s0.split, s0.rev, None, st()),
    "select0(4 kernels)": lambda: L.fmt_select0(p(s0.cand), p(s0.cnt), s0.split, s0.cap, p(ks), p(ke), M, N, 512, p(out), p(s0.flag), p(s0.fbc), p(s0.scratch), p(q8), p(k8), p(sc), p(w), None, st()),
    "step0_total": lambda: s0(q8, k8, sc, w, ks, ke, out),
    "s1_sample_tau(1 kernel)": lambda: L.fmt_sample_tau(p(q8), p(k8), p(sc), p(w), p(ks), p(ke), p(s0.ls), M, N, SS, 512, s0.cap, p(s0.tau), p(s0.fbc), st()),
    "s0_sample_dense_only": lambda: L.fmt_dense(p(q8), p(ksmp), p(scsmp), p(w), p(ks_s), p(ke_s), p(s0.ls), M, s0.Ns, s0.Ns, 0, -1, st()),
    "s0_sample_dense_split1": lambda: L.fmt_dense(p(q8), p(ksmp), p(scsmp), p(w), p(ks_s), p(ke_s), p(s0.ls), M, s0.Ns, s0.Ns, 1, 1, st()),
    "s0_tau_only": lambda: L.fmt_tau(p(s0.ls), s0.Ns, SS, p(ks), p(ke), M, N, 512, s0.cap, p(s0.tau), p(s0.fbc), st()),
    "s1_select1(2 kernels)": lambda: L.fmt_select1(p(s0.cand), p(s0.cnt), s0.split, s0.cap, p(ks), p(ke), M, N, 512, p(out), p(s0.flag), p(s0.fbc), p(done), p(s0.scratch), p(q8), p(k8), p(sc), p(w), None, st()),
}
for r in range(2):
    res = {k: F.timeit(f, a.iters) for k, f in parts.items()}
print(f"{a.shape} split {s0.split}: " + " | ".join(f"{k} {v:.4f}" for k, v in res.items()))
