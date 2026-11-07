#!/usr/bin/env python3
"""Exactness tests of the fused pipelines against the dense v35wes logits + exact top-K (fmt_harness.check).

  python3 test_exact.py --step 0 [--quick]
Also checks that our DENSE scan is bit-exact with the aiter v35wes op on the valid windows.
"""
import argparse, json, sys, time
import torch
import fmt_harness as F
import fmt_steps as S

QUICK = [  # (M, N, kind, reqs, causal)
    (16384, 32000, "rand", 1, True), (4096, 8192, "rand", 1, True), (1000, 3000, "rand", 1, True),
    (16384, 4096, "rand", 1, True), (2048, 20000, "rand", 2, True), (777, 9000, "rand", 1, False),
    (4096, 24000, "offset", 1, True), (4096, 24000, "ties", 1, True), (4096, 64000, "periodic", 1, True),
    (4096, 24000, "increasing", 1, True), (576, 250000, "rand", 1, True),
]
FULL = QUICK + [
    (16384, 16384, "rand", 1, True), (13312, 32000, "rand", 1, True), (8192, 2048, "rand", 1, True),
    (16384, 8192, "rand", 4, True), (64, 40000, "rand", 1, True), (33, 70000, "offset", 1, True),
    (16384, 32000, "offset", 1, True), (16384, 32000, "ties", 1, True), (8192, 128000, "periodic", 1, True),
    (31, 513, "rand", 1, False), (100, 100, "rand", 1, False),
]


def make_step(step, **kw):
    return {0: S.Step0}[step](**kw)


def run_case(step, M, N, kind, reqs, causal, force_fb=False, split=None):
    q8, k8, sc, w, ks, ke = F.gen(M, N, kind, reqs, causal)
    Nt = k8.shape[0]
    lg = F.dense_logits(q8, k8, sc, w, ks, ke, clean=True)
    res = dict(M=M, N=N, kind=kind, reqs=reqs, causal=causal, force_fb=force_fb)
    # DENSE bit-exactness vs the aiter op (valid window)
    mine = torch.full((M, Nt), float("nan"), device="cuda")
    S.dense(q8, k8, sc, w, ks, ke, mine)
    win = torch.isfinite(lg)
    res["dense_bitexact"] = bool((mine[win].view(torch.int32) == lg[win].view(torch.int32)).all())
    st = make_step(step)
    if force_fb:
        st.force_fallback = True
    out = torch.full((M, F.TOPK), -7, dtype=torch.int32, device="cuda")
    st(q8, k8, sc, w, ks, ke, out, count_fb=True)
    torch.cuda.synchronize()
    wrong = F.check(out, lg, ks, ke)
    res["wrong_rows"] = int(wrong.sum())
    res.update(st.stats(M))
    del lg, mine
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cases = QUICK if a.quick else FULL
    allres, bad = [], 0
    for c in cases:
        t0 = time.time()
        r = run_case(a.step, *c)
        allres.append(r)
        ok = r["wrong_rows"] == 0 and r["dense_bitexact"]
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {c}: wrong {r['wrong_rows']} dense_bitexact {r['dense_bitexact']} split {r['split']} "
              f"cand {r['cand_mean']:.0f}/{r['cand_max']} tau=-inf {r['tau_ninf']} fallback {r['fallback_rows']} ({time.time()-t0:.1f}s)", flush=True)
    # forced fallback on a few shapes (every row through the exact recompute path)
    for c in [(512, 12000, "rand", 1, True), (300, 5000, "ties", 1, True), (130, 70000, "rand", 2, True)]:
        r = run_case(a.step, *c, force_fb=True)
        allres.append(r)
        ok = r["wrong_rows"] == 0
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} forced-fallback {c}: wrong {r['wrong_rows']} fallback {r['fallback_rows']}", flush=True)
    print(f"== step {a.step}: {len(allres) - bad}/{len(allres)} cases OK")
    if a.out:
        json.dump(allres, open(a.out, "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
