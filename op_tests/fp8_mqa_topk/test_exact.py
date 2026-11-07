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
    (1500, 9000, "odd", 1, False), (1024, 196608, "increasing", 1, True),
]
FULL = QUICK + [
    (16384, 16384, "rand", 1, True), (13312, 32000, "rand", 1, True), (8192, 2048, "rand", 1, True),
    (16384, 8192, "rand", 4, True), (64, 40000, "rand", 1, True), (33, 70000, "offset", 1, True),
    (16384, 32000, "offset", 1, True), (16384, 32000, "ties", 1, True), (8192, 128000, "periodic", 1, True),
    (31, 513, "rand", 1, False), (100, 100, "rand", 1, False), (2048, 131072, "ties", 1, True),
]


def make_step(step, **kw):
    return {0: S.Step0, 1: S.Step1, 2: S.Step2, 3: S.Step3, 4: S.Step4}[step](**kw)


def poison(st):
    """Fill every workspace buffer of a step with garbage before a call (stale-state bugs must not pass)."""
    for name in ("ls", "tau", "cand", "cnt", "flag", "scratch", "smp", "fscratch"):
        t = getattr(st, name, None)
        if t is not None:
            if t.dtype.is_floating_point:
                t.fill_(1.2345e30)
            else:
                t.fill_(0x3C3C3C3C if t.dtype == torch.int32 else 0x3C3C3C3C3C3C3C3C)


def gate_emit_complete(M, N, kind="rand", reqs=1, causal=True, split=0):
    """EMIT with tau = -inf and cap >= N: every valid (m, n) emitted exactly once across splits, bits == dense."""
    q8, k8, sc, w, ks, ke = F.gen(M, N, kind, reqs, causal)
    Nt = k8.shape[0]
    lg = F.dense_logits(q8, k8, sc, w, ks, ke, clean=True)
    L, p = S.lib(), F.ptr
    sp, rv = S.choice(M, Nt)
    sp = split or sp
    cap = Nt
    Mp = M + (M & 1)
    cand = torch.full((sp * Mp * cap,), -1, dtype=torch.int64, device="cuda")
    cnt = torch.full((sp * M,), -1, dtype=torch.int32, device="cuda")
    tau = torch.full((M,), float("-inf"), device="cuda")
    r = L.fmt_emit(p(q8), p(k8), p(sc), p(w), p(ks), p(ke), p(tau), p(cand), p(cnt), cap, M, Nt, sp, rv, None, F.stream())
    assert r == 0
    torch.cuda.synchronize()
    cand = cand.view(sp, Mp // 2, cap, 2)     # [split][pair][i][row parity]
    cnt = cnt.view(sp, M)
    bad = 0
    for m in range(M):
        cols, bits = [], []
        for s_ in range(sp):
            c = int(cnt[s_, m])
            d = cand[s_, m // 2, :c, m & 1]
            cols.append((d >> 32).to(torch.int64)); bits.append((d & 0xFFFFFFFF).to(torch.int64))
        cols = torch.cat(cols); bits = torch.cat(bits)
        valid = torch.isfinite(lg[m]).nonzero().flatten()
        if cols.numel() != valid.numel() or torch.unique(cols).numel() != cols.numel():
            bad += 1; continue
        order = cols.argsort()
        if not torch.equal(cols[order], valid):
            bad += 1; continue
        ref_bits = lg[m, valid].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        if not torch.equal(bits[order], ref_bits):
            bad += 1
    return bad, sp


def gate_sample_bits(M, N, S_=32):
    """The strided sample pass stores exactly the dense logits at keys j*S."""
    q8, k8, sc, w, ks, ke = F.gen(M, N)
    lg = F.dense_logits(q8, k8, sc, w, ks, ke, clean=True)
    st = S.Step1(S=S_); st.prep(M, N)
    st.ls.fill_(float("nan"))
    L, p = S.lib(), F.ptr
    assert L.fmt_sample_tau(p(q8), p(k8), p(sc), p(w), p(ks), p(ke), p(st.ls), M, N, S_, 512, st.cap, p(st.tau), p(st.fbc), F.stream()) == 0
    torch.cuda.synchronize()
    ref = lg[:, ::S_][:, :st.Ns]
    win = torch.isfinite(ref)
    return int((st.ls[win].view(torch.int32) != ref[win].view(torch.int32)).sum())


def odd_windows(M, N, seed=1):
    """Rows with ks < 0, ke > N, ke < ks, empty rows and full rows mixed (same K/Q generator)."""
    q8, k8, sc, w, ks, ke = F.gen(M, N, "rand", 1, False, seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randint(-200, N + 200, (M,), generator=g)
    b = torch.randint(-200, N + 400, (M,), generator=g)
    ks = a.to(torch.int32).cuda(); ke = b.to(torch.int32).cuda()   # ke < ks in about half of the rows
    return q8, k8, sc, w, ks, ke


def run_case(step, M, N, kind, reqs, causal, force_fb=False, split=None):
    if kind == "odd":
        q8, k8, sc, w, ks, ke = odd_windows(M, N)
    else:
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
    st.prep(M, Nt)
    poison(st)
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
    # gates (independent of the step)
    for args in [(300, 3000, "rand", 1, True), (257, 5000, "rand", 2, True), (200, 7000, "rand", 1, False, 5)]:
        nb, sp = gate_emit_complete(*args)
        bad += nb > 0
        print(f"{'OK ' if nb == 0 else 'BAD'} gate EMIT(tau=-inf) complete+bit-exact {args}: bad rows {nb} split {sp}", flush=True)
    for SS in (16, 32):
        nb = gate_sample_bits(2048, 30000, SS)
        bad += nb > 0
        print(f"{'OK ' if nb == 0 else 'BAD'} gate strided samples == dense at j*S (S={SS}): mismatches {nb}", flush=True)
    print(f"== step {a.step}: {len(allres) - bad}/{len(allres)} cases OK (+ gates)")
    if a.out:
        json.dump(allres, open(a.out, "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
