#!/usr/bin/env python3
"""Where step 2's time goes per chunk: dense v35wes scan (floor), step-2 fused kernel alone (sample + tau + EMIT),
full step 2 (+ select + fallback kernels). ms, median of iters, caches flushed.
  python3 breakdown.py --isl 128000"""
import argparse
import torch
import fmt_harness as F
import fmt_steps as S

ap = argparse.ArgumentParser()
ap.add_argument("--isl", type=int, default=128000)
ap.add_argument("--iters", type=int, default=10)
a = ap.parse_args()
tot = dict(dense=0.0, fused=0.0, step2=0.0)
print(f"{'chunk':>14} | {'dense':>7} | {'fused':>7} | {'step2':>7} | fused-dense | select+fb")
for M, N in F.chunks(a.isl):
    q8, k8, sc, w, ks, ke = F.gen(M, N)
    Nt = k8.shape[0]
    lg = torch.empty(M, Nt, device="cuda")
    out = torch.empty(M, F.TOPK, dtype=torch.int32, device="cuda")
    t_d = F.timeit(lambda: F.dense_logits(q8, k8, sc, w, ks, ke, clean=False, out=lg), a.iters)
    st = S.Step2(); st.prep(M, Nt)
    st.force_fallback = 2                      # dbg bit 1: fused kernel only
    t_f = F.timeit(lambda: st(q8, k8, sc, w, ks, ke, out), a.iters)
    st.force_fallback = 0
    t_s = F.timeit(lambda: st(q8, k8, sc, w, ks, ke, out), a.iters)
    tot["dense"] += t_d; tot["fused"] += t_f; tot["step2"] += t_s
    print(f"{M:>6}x{Nt:<7} | {t_d:7.4f} | {t_f:7.4f} | {t_s:7.4f} | {t_f - t_d:+8.4f}    | {t_s - t_f:+8.4f}")
    del lg, st
    torch.cuda.empty_cache()
d, f, s = tot["dense"], tot["fused"], tot["step2"]
print(f"ISL {a.isl}: dense {d:.3f} | fused {f:.3f} ({(f - d) / s * 100:.1f} % of step 2 above dense) | step2 {s:.3f} "
      f"(select + fallback {(s - f):.3f} = {(s - f) / s * 100:.1f} %)")
