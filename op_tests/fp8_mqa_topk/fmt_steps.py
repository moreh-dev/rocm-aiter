#!/usr/bin/env python3
"""Python drivers of the fused pipelines (steps 0..4) on top of build/libfmt.so (see DESIGN.md section 4.3)."""
import ctypes
import math

import torch

import fmt_harness as F

c_int, c_vp = ctypes.c_int, ctypes.c_void_p
_LIB = None
SMP_SLACK = 8192   # records: the tau loads read up to B * 64 records past a row's count (masked, not clamped)


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
            "fmt_sample_tau": [c_vp] * 7 + [c_int] * 5 + [c_vp, c_vp, c_vp],
            "fmt_select1": [c_vp, c_vp, c_int, c_int, c_vp, c_vp, c_int, c_int, c_int] + [c_vp] * 11,
            "fmt_fused_nsb": [c_int, c_int, c_int],
            "fmt_fused_nsbc": [c_int, c_int, c_int],
            "fmt_step2": [c_vp] * 6 + [c_int] * 8 + [c_vp] * 5 + [c_int] + [c_vp] * 6 + [c_int, c_int] + [c_vp] * 4,
            "fmt_step3": [c_vp] * 6 + [c_int] * 8 + [c_vp] * 5 + [c_int] + [c_vp] * 8 + [c_int, c_int] + [c_vp] * 4,
            "fmt_step4": [c_vp] * 6 + [c_int] * 8 + [c_vp] * 5 + [c_int] + [c_vp] * 7 + [c_int, c_int] + [c_vp] * 4,
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

    def __init__(self, S=16, cap=4096, K=F.TOPK, krule=False):
        self.S, self.cap, self.K = S, cap, K
        # tau = -inf rule: rows with nvalid <= rule get every key as a candidate. Steps 0/1 use CAP; krule=True uses K
        # (the step 2-4 rule) for a like-for-like comparison.
        self.rule = K if krule else cap
        self.shape = None
        self.force_fallback = False   # debug: tau = +inf -> no candidates -> every row takes the exact fallback

    def prep(self, M, N):
        d = "cuda"
        self.split, self.rev = choice(M, N)
        self.Ns = (N + self.S - 1) // self.S
        self.ls = torch.empty(M, self.Ns, device=d)
        self.tau = torch.empty(M, device=d)
        self.cand = torch.empty(self.split * (M + (M & 1)) * self.cap, dtype=torch.int64, device=d)
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
        _ck(L.fmt_tau(p(self.ls), self.Ns, S, p(ks), p(ke), M, N, self.K, self.rule, p(self.tau), p(self.fbc), st), "tau")
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


class Step1(Step0):
    """Step 1: ONE sample kernel (strided DENSE scan over keys j*S + tau in its tail, resets fbc), EMIT scan (row-wide
    tau, v35wes split), select + ONE fallback kernel. 4 launches, no torch ops."""
    name = "step1"

    def prep(self, M, N):
        super().prep(M, N)
        self.done = torch.zeros(64, dtype=torch.int32, device="cuda")   # FBR completion counters (self-resetting)

    def __call__(self, q8, k8, sc, w, ks, ke, out, count_fb=False):
        M, N, S = q8.shape[0], k8.shape[0], self.S
        if self.shape != (M, N):
            self.prep(M, N)
        L, p, st = lib(), F.ptr, F.stream()
        _ck(L.fmt_sample_tau(p(q8), p(k8), p(sc), p(w), p(ks), p(ke), p(self.ls), M, N, S, self.K, self.rule, p(self.tau),
                             p(self.fbc), st), "sample_tau")
        if self.force_fallback:
            self.tau.fill_(float("inf"))
        _ck(L.fmt_emit(p(q8), p(k8), p(sc), p(w), p(ks), p(ke), p(self.tau), p(self.cand), p(self.cnt), self.cap, M, N,
                       self.split, self.rev, None, st), "emit")
        _ck(L.fmt_select1(p(self.cand), p(self.cnt), self.split, self.cap, p(ks), p(ke), M, N, self.K, p(out), p(self.flag),
                          p(self.fbc), p(self.done), p(self.scratch), p(q8), p(k8), p(sc), p(w),
                          p(self.nfb) if count_fb else None, st), "select1")


class Step2:
    """Step 2: fused scan kernel (per split: strided sample pass + tau + EMIT; per-split exactness counters) ->
    select (per-split need) -> fallback1. 3 launches."""
    name = "step2"
    NAME = "fmt_step2"

    def __init__(self, S=32, cap=4096, K=F.TOPK, split=None):
        self.S, self.cap, self.K = S, cap, K
        self.split_override = split   # test hook: force SplitN (grids larger than the GPU exercise chunk stealing)
        self.dbg_bits = 0             # extra kernel dbg bits (8: DIST variant for any SplitN > 1, 16: chunk per split)
        self.shape = None
        self.force_fallback = False

    def prep(self, M, N):
        d = "cuda"
        self.split, self.rev = choice(M, N)
        if self.split_override:
            self.split = self.split_override
        self.nsb = lib().fmt_fused_nsb(N, self.split, self.S)
        self.rb = (M + 31) // 32
        Mp = M + (M & 1)
        # SplitN > 1: distributed sampling, one [Mp][nsbc] segment per chunk (= split); else rows of nsb records
        self.nsbc = lib().fmt_fused_nsbc(N, self.split, self.S) if self.split > 1 else 0
        # (SplitN > 1 uses one whole-range chunk of nsb records when the grid does not fit on the GPU)
        nrec = max(self.split * Mp * self.nsbc, Mp * self.nsb) if self.split > 1 else Mp * self.nsb
        self.smp = torch.empty(nrec + SMP_SLACK, dtype=torch.int64, device=d)
        self.sflag = torch.zeros(2 * self.rb * self.split, dtype=torch.int32, device=d)   # epoch-valued, zeroed once
        self.scnt = torch.empty(self.split * M, dtype=torch.int32, device=d)
        self.tflag = torch.zeros(2 * self.rb * 32, dtype=torch.int32, device=d)
        self.tau = torch.empty(M, device=d)
        self.tau_ready = torch.zeros(self.rb, dtype=torch.int32, device=d)
        self.epoch = 0
        self.cand = torch.empty(self.split * Mp * self.cap, dtype=torch.int64, device=d)
        self.cnt = torch.empty(self.split * M, dtype=torch.int32, device=d)
        self.flag = torch.empty(M, dtype=torch.int32, device=d)
        self.fbc = torch.zeros(1, dtype=torch.int32, device=d)
        self.done = torch.zeros(64, dtype=torch.int32, device=d)
        self.block_done = torch.zeros(self.rb, dtype=torch.int32, device=d)
        self.fb_exit = torch.zeros(1, dtype=torch.int32, device=d)
        self.scratch = torch.empty(128 * N, device=d)
        self.nfb = torch.zeros(1, dtype=torch.int32, device=d)
        self.shape = (M, N)

    def _launch(self, q8, k8, sc, w, ks, ke, out, nfb):
        M, N = q8.shape[0], k8.shape[0]
        L, p, st = lib(), F.ptr, F.stream()
        _ck(L.fmt_step2(p(q8), p(k8), p(sc), p(w), p(ks), p(ke), M, N, self.split, self.rev, self.S, self.K, self.cap,
                        self.nsb, p(self.smp), p(self.cand), p(self.cnt), p(self.tau), p(self.tau_ready), self._ep(), p(out),
                        p(self.flag), p(self.fbc), p(self.done), p(self.scratch), nfb, int(self.force_fallback) | self.dbg_bits, self.nsbc,
                        p(self.sflag), p(self.scnt), p(self.tflag), st), self.NAME)

    def _ep(self):
        if self.epoch >= 2000000000:   # wrap: flags of old launches must not match a reused epoch
            self.epoch = 0
            for t in (self.sflag, self.tflag, self.tau_ready):
                t.zero_()
        self.epoch += 1
        return self.epoch

    def __call__(self, q8, k8, sc, w, ks, ke, out, count_fb=False):
        M, N = q8.shape[0], k8.shape[0]
        if self.shape != (M, N):
            self.prep(M, N)
        self._launch(q8, k8, sc, w, ks, ke, out, F.ptr(self.nfb) if count_fb else None)

    def stats(self, M):
        cn = self.cnt.view(self.split, M).sum(0).float()
        return dict(split=self.split, cand_mean=float(cn.mean()), cand_max=int(cn.max()), tau_ninf=int(torch.isinf(self.tau).sum()),
                    fallback_rows=int(self.nfb.item()))


class Step3(Step2):
    """Step 3: fused scan kernel + in-kernel select by the last split of each row block -> fallback1. 2 launches."""
    name = "step3"
    NAME = "fmt_step3"

    def _launch(self, q8, k8, sc, w, ks, ke, out, nfb):
        M, N = q8.shape[0], k8.shape[0]
        L, p, st = lib(), F.ptr, F.stream()
        _ck(L.fmt_step3(p(q8), p(k8), p(sc), p(w), p(ks), p(ke), M, N, self.split, self.rev, self.S, self.K, self.cap,
                        self.nsb, p(self.smp), p(self.cand), p(self.cnt), p(self.tau), p(self.tau_ready), self._ep(), p(out),
                        p(self.flag), p(self.fbc), p(self.block_done), p(self.done), p(self.fb_exit), p(self.scratch), nfb,
                        int(self.force_fallback) | self.dbg_bits, self.nsbc, p(self.sflag), p(self.scnt), p(self.tflag), st), self.NAME)


class Step4(Step2):
    """Step 4: ONE persistent launch (1 WG per CU claims (row block, split) items heaviest first; sample + row-wide tau
    + EMIT + in-kernel select of the block by its last split; in-kernel exact fallback drain; self-resetting counters)."""
    name = "step4"
    NAME = "fmt_step4"
    G = 256

    def prep(self, M, N):
        super().prep(M, N)
        d = "cuda"
        # the persistent kernel always runs the split-0 (non-distributed) variant: per-WG sample slots of nsb records
        self.smp = torch.empty(self.G * 32 * self.nsb + SMP_SLACK, dtype=torch.int64, device=d)
        self.ctr = torch.zeros(4, dtype=torch.int32, device=d)
        self.fscratch = torch.empty(self.G * N, device=d)

    def _launch(self, q8, k8, sc, w, ks, ke, out, nfb):
        M, N = q8.shape[0], k8.shape[0]
        L, p, st = lib(), F.ptr, F.stream()
        _ck(L.fmt_step4(p(q8), p(k8), p(sc), p(w), p(ks), p(ke), M, N, self.split, self.S, self.K, self.cap, self.nsb,
                        self.G, p(self.smp), p(self.cand), p(self.cnt), p(self.tau), p(self.tau_ready), self._ep(), p(out),
                        p(self.flag), p(self.fbc), p(self.block_done), p(self.ctr), p(self.fscratch), nfb,
                        int(self.force_fallback) | self.dbg_bits, self.nsbc, p(self.sflag), p(self.scnt), p(self.tflag), st), self.NAME)
