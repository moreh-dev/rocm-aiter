#!/usr/bin/env python3
"""Python drivers of the fused pipelines (steps 0..4) on top of build/libfmt.so (see DESIGN.md section 4.3)."""
import ctypes
import math

import torch

import fmt_harness as F

c_int, c_vp = ctypes.c_int, ctypes.c_void_p
_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        L = F.load_lib()
        sig = {
            "fmt_choice": [c_int, c_int, c_vp, c_vp],
            "fmt_dense": [c_vp] * 7 + [c_int] * 5 + [c_vp],
            "fmt_tau": [c_vp, c_int, c_int, c_vp, c_vp, c_int, c_int, c_int, c_int, c_vp, c_vp, c_vp],
            "fmt_emit": [c_vp] * 9 + [c_int] * 5 + [c_vp, c_vp],
            "fmt_select0": [c_vp, c_vp, c_int, c_int, c_vp, c_vp, c_int, c_int, c_int] + [c_vp] * 10,
        }
        for name, args in sig.items():
            fn = getattr(L, name)
            fn.argtypes = args
            fn.restype = c_int
        _LIB = L
    return _LIB


def _ck(r, what):
    if r != 0:
        raise RuntimeError(f"{what} returned {r}")


def choice(M, N):
    s, r = c_int(0), c_int(0)
    lib().fmt_choice(M, N, ctypes.byref(s), ctypes.byref(r))
    return s.value, r.value


def dense(q8, k8, sc, w, ks, ke, out, split=0, rev=-1):
    """Our DENSE scan (== v35wes, valid window only). out: [M, out_cols] f32."""
    M, N = q8.shape[0], k8.shape[0]
    p = F.ptr
    _ck(lib().fmt_dense(p(q8), p(k8), p(sc), p(w), p(ks), p(ke), p(out), M, N, out.shape[1], split, rev, F.stream()), "fmt_dense")
    return out


class Step0:
    """fuse48 structure on v35wes: torch K[::S] / scale[::S] / ks_s / ke_s, DENSE scan on the sample (fp32), tau kernel,
    EMIT scan (row-wide tau, v35wes split), select + 3 fallback kernels. 7 kernels + 6 torch ops per call."""
    name = "step0"

    def __init__(self, S=16, cap=4096, K=F.TOPK):
        self.S, self.cap, self.K = S, cap, K
        self.shape = None
        self.force_fallback = False   # debug: tau = +inf -> no candidates -> every row takes the exact fallback

    def prep(self, M, N):
        d = "cuda"
        self.split, self.rev = choice(M, N)
        self.Ns = (N + self.S - 1) // self.S
        self.ls = torch.empty(M, self.Ns, device=d)
        self.tau = torch.empty(M, device=d)
        self.cand = torch.empty(self.split * M * (self.cap + 1), dtype=torch.int64, device=d)
        self.cnt = torch.empty(self.split * M, dtype=torch.int32, device=d)
        self.flag = torch.empty(M, dtype=torch.int32, device=d)
        self.fbc = torch.zeros(1, dtype=torch.int32, device=d)
        self.nfb = torch.zeros(1, dtype=torch.int32, device=d)
        self.scratch = torch.empty(128 * N, device=d)
        self.shape = (M, N)

    def __call__(self, q8, k8, sc, w, ks, ke, out, count_fb=False):
        M, N, S = q8.shape[0], k8.shape[0], self.S
        if self.shape != (M, N):
            self.prep(M, N)
        L, p, st = lib(), F.ptr, F.stream()
        k_s = k8[::S].contiguous()
        sc_s = sc[::S].contiguous()
        ks_s = (ks + (S - 1)) // S
        ke_s = (ke + (S - 1)) // S
        _ck(L.fmt_dense(p(q8), p(k_s), p(sc_s), p(w), p(ks_s), p(ke_s), p(self.ls), M, self.Ns, self.Ns, 0, -1, st), "dense(sample)")
        _ck(L.fmt_tau(p(self.ls), self.Ns, S, p(ks), p(ke), M, N, self.K, self.cap, p(self.tau), p(self.fbc), st), "tau")
        if self.force_fallback:
            self.tau.fill_(float("inf"))
        _ck(L.fmt_emit(p(q8), p(k8), p(sc), p(w), p(ks), p(ke), p(self.tau), p(self.cand), p(self.cnt), self.cap, M, N,
                       self.split, self.rev, None, st), "emit")
        _ck(L.fmt_select0(p(self.cand), p(self.cnt), self.split, self.cap, p(ks), p(ke), M, N, self.K, p(out), p(self.flag),
                          p(self.fbc), p(self.scratch), p(q8), p(k8), p(sc), p(w), p(self.nfb) if count_fb else None, st), "select0")

    def stats(self, M):
        cn = self.cnt.view(self.split, M).sum(0).float()
        return dict(split=self.split, cand_mean=float(cn.mean()), cand_max=int(cn.max()),
                    tau_ninf=int(torch.isinf(self.tau).sum()), fallback_rows=int(self.nfb.item()))
