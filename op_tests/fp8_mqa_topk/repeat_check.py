#!/usr/bin/env python3
"""Self-resetting counters: run a step many times back to back (alternating shapes) and check exactness at the end."""
import sys, torch
import fmt_harness as F, fmt_steps as S
step = {"2": S.Step2, "3": S.Step3, "4": S.Step4}[sys.argv[1]]
shapes = [(4096, 24000), (2048, 20000), (1000, 9000)]
data = [F.gen(M, N, "rand", 2 if i == 1 else 1) for i, (M, N) in enumerate(shapes)]
st = {sh: step(S=32) for sh in shapes}
outs = {sh: torch.full((sh[0], F.TOPK), -7, dtype=torch.int32, device="cuda") for sh in shapes}
for it in range(30):
    for sh, d in zip(shapes, data):
        st[sh](*d, outs[sh])
    torch.cuda.synchronize()
bad = 0
for sh, d in zip(shapes, data):
    lg = F.dense_logits(*d, clean=True)
    w = int(F.check(outs[sh], lg, d[4], d[5]).sum()); bad += w
    print(sh, "split", st[sh].split, "wrong rows after 30 launches:", w)
print("REPEAT", "OK" if bad == 0 else "BAD")
