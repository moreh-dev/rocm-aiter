#!/usr/bin/env python3
"""Run a few pipeline calls for a kernel trace (rocprofv3 --kernel-trace): python3 trace_run.py SHAPE PATH[,PATH] S"""
import sys, torch
import fmt_harness as F, fmt_steps as S
M, N = map(int, sys.argv[1].split("x"))
names = sys.argv[2].split(",")
SS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
q8, k8, sc, w, ks, ke = F.gen(M, N)
mk = {"step0": lambda: S.Step0(S=SS), "step1": lambda: S.Step1(S=SS), "baseu": F.BaseU, "hi16u": F.Hi16U}
for n in names:
    p = mk[n](); out = torch.empty(M, F.TOPK, dtype=torch.int32, device="cuda")
    for i in range(6):
        F.flush_caches(); torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"{n}_{i}") if hasattr(torch.cuda, "nvtx") else None
        p(q8, k8, sc, w, ks, ke, out); torch.cuda.synchronize()
