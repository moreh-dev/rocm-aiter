#!/usr/bin/env python3
"""Benchmark the fused pipelines against the references, 1 layer, top-512, per prefill chunk (default 16,384 tokens).

  python3 bench.py --isl 8192,60000,128000 --paths baseu,hi16u,fuse48,step0 [--shapes 16384x32000,...]

Every path is checked for exactness on every chunk (wrong-row count vs the dense v35wes logits + exact top-K) before
timing. Timing: L2/MALL flush before every call, synchronize after every call, median of `iters`, interleaved
`rounds`, median over rounds. ISL totals = sum of the chunk medians.
"""
import argparse, json, statistics, time
import torch
import fmt_harness as F
import fmt_steps as S

PATHS = {
    "baseu": lambda: F.BaseU(),
    "hi16u": lambda: F.Hi16U(),
    "pr16": lambda: F.PR16(),        # moreh-dev/rocm-aiter PR #16 (fp8_mqa_logits_topk, 16-bit logits + top-K)
    "fuse48": lambda: F.Fuse48(),
    "step0": lambda: S.Step0(),
    "step1": lambda: S.Step1(),
    "step0s32": lambda: S.Step0(S=32),
    "step0ks32": lambda: S.Step0(S=32, krule=True),   # step 0 with the step 2-4 tau = -inf rule (nvalid <= K)
    "step1s32": lambda: S.Step1(S=32),
    "step2": lambda: S.Step2(S=32),
    "step0ks32sp1": lambda: S.Step0(S=32, krule=True, split=1),   # grid-model diagnostics: SplitN forced to 1
    "step2sp1": lambda: S.Step2(S=32, split=1),
    "step3": lambda: S.Step3(S=32),
    "step4": lambda: S.Step4(S=32),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isl", default="8192,60000,128000")
    ap.add_argument("--shapes", default="", help="explicit MxN chunks instead of --isl")
    ap.add_argument("--chunk", type=int, default=F.CHUNK, help="prefill chunk (tokens) for --isl")
    ap.add_argument("--paths", default="baseu,hi16u,fuse48,step0")
    ap.add_argument("--kind", default="rand")
    ap.add_argument("--reqs", type=int, default=1)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    names = a.paths.split(",")
    paths = {n: PATHS[n]() for n in names}
    if a.shapes:
        plan = [(s, [tuple(map(int, s.split("x")))]) for s in a.shapes.split(",")]
    else:
        plan = [(str(L), F.chunks(L, chunk=a.chunk)) for L in map(int, a.isl.split(","))]
    cache, res = {}, {"args": vars(a), "chunks": {}, "isl": {}}
    for name, cl in plan:
        for (M, N) in cl:
            if (M, N) in cache:
                continue
            q8, k8, sc, w, ks, ke = F.gen(M, N, a.kind, a.reqs)
            Nt = k8.shape[0]
            outs = {n: torch.full((M, F.TOPK), -7, dtype=torch.int32, device="cuda") for n in names}
            info = {}
            if not a.no_check:
                lg = F.dense_logits(q8, k8, sc, w, ks, ke, clean=True)
                for n in names:
                    p = paths[n]
                    if hasattr(p, "stats"):
                        p(q8, k8, sc, w, ks, ke, outs[n], count_fb=True)
                    else:
                        p(q8, k8, sc, w, ks, ke, outs[n])
                    torch.cuda.synchronize()
                    info[n] = {"wrong": int(F.check(outs[n], lg, ks, ke).sum())}
                    if hasattr(p, "stats"):
                        info[n].update(p.stats(M))
                del lg
                torch.cuda.empty_cache()
            T = {n: [] for n in names}
            for _ in range(a.rounds):
                for n in names:
                    p, o = paths[n], outs[n]
                    T[n].append(F.timeit(lambda: p(q8, k8, sc, w, ks, ke, o), a.iters))
            med = {n: statistics.median(v) for n, v in T.items()}
            cache[(M, N)] = c = dict(M=M, N=Nt, ms=med, info=info)
            res["chunks"][f"{M}x{Nt}"] = c
            ref = med[names[0]]
            line = " | ".join(f"{n} {med[n]:.4f}" + (f" ({100 * (med[n] / ref - 1):+.1f}%)" if n != names[0] else "") for n in names)
            wr = {n: info[n]["wrong"] for n in info}
            extra = ""
            for n in names:
                if n in info and "cand_mean" in info[n]:
                    i = info[n]
                    extra += f" [{n}: split {i['split']} cand {i['cand_mean']:.0f}/{i['cand_max']} fb {i['fallback_rows']}]"
            print(f"  chunk {M}x{Nt}: {line} | wrong {wr}{extra}", flush=True)
            del q8, k8
            torch.cuda.empty_cache()
        tot = {n: sum(cache[c]["ms"][n] for c in cl) for n in names}
        res["isl"][name] = dict(chunks=cl, ms=tot)
        ref = tot[names[0]]
        print(f"ISL {name}: " + " | ".join(f"{n} {tot[n]:.3f}" + (f" ({100 * (tot[n] / ref - 1):+.1f}%)" if n != names[0] else "") for n in names), flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
