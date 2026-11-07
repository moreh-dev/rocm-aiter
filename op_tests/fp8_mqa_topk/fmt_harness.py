#!/usr/bin/env python3
"""fmt_harness: shared harness for the fused FP8 MQA logits + top-k dev work (see DESIGN.md / PROGRESS.md).

Data generators (incl. stress sets), the exactness check, timing (L2/MALL flush, synchronize after every call),
the reference paths (baseu = aiter v35wes logits + vLLM top-k, hi16u = Nam's v35hw + topk3, fuse48 = Nam's v35a-based
fused pipeline) and a ctypes loader for our own library (build/libfmt.so).

Run only through gpu7.sh (amd-smi GPU 7 = HIP_VISIBLE_DEVICES=4, serialized).
"""
import ctypes
import math
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.environ.get("FMT_REF_DIR", "/workspace/experiments/fuse_topk_ref")  # Nam's builds (container path)
FP8 = torch.float8_e4m3fn
H, D = 32, 128
TOPK = 512
CHUNK = 16384
NEG_INF = float("-inf")


# ----------------------------------------------------------------------------------------------- data
def ue8m0(amax):
    return torch.exp2(torch.ceil(torch.log2(amax.clamp(min=1e-4) / 448.0)))


def quant_q(q):
    return (q / q.abs().amax(-1, keepdim=True).clamp(min=1e-6) * 448.0).to(FP8)


def quant_k(k):
    sc = ue8m0(k.abs().amax(-1)).contiguous()
    return (k / sc[:, None]).to(FP8).contiguous(), sc


def layout(M, N, kpool=4, causal=True, reqs=1, device="cuda"):
    """Nam's chunk layout. reqs == 1: one causal request, row t of the chunk is token pos = N*kpool - M + t and sees
    pools [0, (pos+1)//kpool). reqs > 1: prefix-cache layout, request r owns its own K copy [r*N, (r+1)*N) and
    M/reqs rows; Nt = reqs * N. causal=False: every row sees [0, N)."""
    if reqs > 1:
        ks = torch.empty(M, dtype=torch.int32)
        ke = torch.empty(M, dtype=torch.int32)
        m_per = M // reqs
        for r in range(reqs):
            a, b = r * m_per, (r + 1) * m_per if r < reqs - 1 else M
            pos = torch.arange(N * kpool - (b - a), N * kpool)
            ks[a:b] = r * N
            ke[a:b] = r * N + ((pos + 1) // kpool).clamp(max=N)
        return ks.to(device), ke.to(device), reqs * N
    ks = torch.zeros(M, dtype=torch.int32, device=device)
    if causal:
        pos = torch.arange(N * kpool - M, N * kpool, device=device)
        ke = ((pos + 1) // kpool).clamp(min=0).to(torch.int32)
    else:
        ke = torch.full((M,), N, dtype=torch.int32, device=device)
    return ks, ke, N


def gen(M, N, kind="rand", reqs=1, causal=True, seed=0, kpool=4):
    """Returns q8 [M,32,128] fp8, k8 [Nt,128] fp8, sc [Nt] f32, w [M,32] f32, ks, ke [M] int32.

    kind: rand      Nam's generator (random Q/K, weights (rand-0.2)*0.1, some negative)
          offset    every score shares a large common offset, narrow top tail (GLM-5.2 layer-1 like)
          ties      only 61 distinct K vectors -> massive exact ties in every row
          periodic  keys at n % 16 == 8 get a large boost (adversarial for a stride-16 sample at offset 0)
          increasing scores grow with key position n
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    ks, ke, Nt = layout(M, N, kpool, causal, reqs)
    Nk = N  # distinct K rows before the reqs copy
    q = torch.randn(M, H, D, device="cuda", generator=g)
    w = (torch.rand(M, H, device="cuda", generator=g) - 0.2) * 0.1
    if kind == "rand":
        k = torch.randn(Nk, D, device="cuda", generator=g)
    elif kind == "offset":
        u = torch.randn(D, device="cuda", generator=g)
        u = u / u.norm()
        k = 0.08 * torch.randn(Nk, D, device="cuda", generator=g) + 1.0 * u
        q = 0.08 * q + 1.0 * u
        w = 0.02 + 0.08 * torch.rand(M, H, device="cuda", generator=g)          # positive -> one-sided offset
    elif kind == "ties":
        base = torch.randn(61, D, device="cuda", generator=g)
        k = base[torch.arange(Nk, device="cuda") % 61]
    elif kind == "periodic":
        u = torch.randn(D, device="cuda", generator=g)
        u = u / u.norm()
        k = torch.randn(Nk, D, device="cuda", generator=g)
        n = torch.arange(Nk, device="cuda")
        k = k + (n % 16 == 8).float()[:, None] * 6.0 * u
        q = q + 3.0 * u
    elif kind == "increasing":
        u = torch.randn(D, device="cuda", generator=g)
        u = u / u.norm()
        k = 0.5 * torch.randn(Nk, D, device="cuda", generator=g)
        n = torch.arange(Nk, device="cuda", dtype=torch.float32)
        k = k + (3.0 * n / max(Nk - 1, 1))[:, None] * u
        q = q + 3.0 * u
    else:
        raise ValueError(kind)
    q8 = quant_q(q)
    k8, sc = quant_k(k)
    if reqs > 1:
        k8 = k8.repeat(reqs, 1).contiguous()
        sc = sc.repeat(reqs).contiguous()
    return q8.contiguous(), k8, sc, w.contiguous(), ks, ke


def chunks(isl, chunk=CHUNK, kpool=4):
    """vLLM prefill chunks of one prompt: (rows, pools seen by the last row)."""
    out, s = [], 0
    while s < isl:
        e = min(isl, s + chunk)
        out.append((e - s, e // kpool))
        s = e
    return out


# ----------------------------------------------------------------------------------------------- reference
_aiter_logits = None


def dense_logits(q8, k8, sc, w, ks, ke, clean=True, out=None):
    """The shipped v35wes kernel (aiter.ops.fp8_mqa_logits): fp32 [M, Nt], -inf outside [ks, ke) when clean."""
    global _aiter_logits
    if _aiter_logits is None:
        from aiter.ops.fp8_mqa_logits import fp8_mqa_logits as f
        _aiter_logits = f
    return _aiter_logits(q8, k8, sc, w, ks, ke, clean_logits=clean, out=out)


def check(out, logits, ks, ke, K=TOPK):
    """Exactness check. out [M, K] int32 indices relative to ks, -1 padded; logits = dense fp32 (-inf outside the
    window). Returns a bool tensor [M]: True = wrong row (value multiset differs from the exact top-K, wrong -1 count,
    index outside the window, or duplicate index)."""
    M, N = logits.shape
    ksl = ks.long().clamp(0, N)
    kel = ke.long().clamp(0, N)
    nvalid = (kel - ksl).clamp(min=0)
    idx = out.long()
    valid = idx >= 0
    absidx = torch.where(valid, idx + ksl[:, None], torch.zeros_like(idx))
    bad_range = valid & ((absidx < ksl[:, None]) | (absidx >= kel[:, None]))
    vals = torch.gather(logits, 1, absidx.clamp(0, N - 1))
    vals = torch.where(valid & ~bad_range, vals, torch.full_like(vals, NEG_INF))
    got = vals.sort(1, descending=True).values
    kk = min(K, N)
    ref = torch.topk(logits, kk, dim=1).values
    if kk < K:
        ref = torch.cat([ref, torch.full((M, K - kk), NEG_INF, device=ref.device)], 1)
    cnt_bad = valid.sum(1) != nvalid.clamp(max=K)
    s_idx = torch.where(valid, absidx, torch.full_like(absidx, -1 - N)).sort(1).values
    dup = ((s_idx[:, 1:] == s_idx[:, :-1]) & (s_idx[:, 1:] >= 0)).any(1)
    return (got != ref).any(1) | cnt_bad | bad_range.any(1) | dup


# ----------------------------------------------------------------------------------------------- timing
_FLUSH = None


def flush_caches():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    _FLUSH.fill_(1)


def timeit(fn, iters=10, flush=True):
    """Median ms; flushes L2/MALL before every call and synchronizes after every call (bounded GPU queue)."""
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if flush:
            flush_caches()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


# ----------------------------------------------------------------------------------------------- reference paths
class BaseU:
    """Production path on the branch: aiter fp8_mqa_logits (v35wes, clean_logits=False) + vLLM top_k_per_row_prefill."""
    name = "baseu"

    def __init__(self):
        import vllm._custom_ops as vops
        self.vops = vops
        self.bufs = {}

    def __call__(self, q8, k8, sc, w, ks, ke, out):
        M, N = q8.shape[0], k8.shape[0]
        key = (M, N)
        if key not in self.bufs:
            self.bufs = {key: torch.empty(M, N, device="cuda")}
        lg = self.bufs[key]
        dense_logits(q8, k8, sc, w, ks, ke, clean=False, out=lg)
        self.vops.top_k_per_row_prefill(lg, ks, ke, out, M, N, 1, out.shape[1])


def _ref_import():
    if REF_DIR not in sys.path:
        sys.path.insert(0, REF_DIR)


class Hi16U:
    """Nam's strongest unfused path: v35hw 16-bit logits + topk3 (exact, boundary recompute)."""
    name = "hi16u"

    def __init__(self):
        _ref_import()
        import harness as NH          # Nam's harness (REF_DIR); ours is fmt_harness to avoid the name clash
        import hi16_bench as NB
        self.v = NH.Variant("v35hw")
        self.tk = NB.TopK()
        self.bufs = {}

    def __call__(self, q8, k8, sc, w, ks, ke, out):
        M, N = q8.shape[0], k8.shape[0]
        key = (M, N)
        if key not in self.bufs:
            self.bufs = {key: torch.empty(M, N, dtype=torch.int16, device="cuda")}
        lh = self.bufs[key]
        self.v(q8, k8, sc, w, ks, ke, lh)
        self.tk.hi16(lh, ks, ke, out, q8, k8, sc, w)


class Fuse48:
    """Nam's fuse48 (v35a based): torch sample prep + v35h + tau + v48 + select + 3 fallback kernels."""
    name = "fuse48"

    def __init__(self, S=16, cap=4096):
        _ref_import()
        import fuse_bench as FB
        self.f = FB.F48(S, cap)
        self.shape = None

    def __call__(self, q8, k8, sc, w, ks, ke, out):
        M, N = q8.shape[0], k8.shape[0]
        if self.shape != (M, N):
            self.f.prep(M, N)
            self.shape = (M, N)
        self.f(q8, k8, sc, w, ks, ke, out)


# ----------------------------------------------------------------------------------------------- our library
def load_lib(path=None):
    path = path or os.path.join(HERE, "build", "libfmt.so")
    return ctypes.CDLL(path)


def ptr(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p(0)


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
