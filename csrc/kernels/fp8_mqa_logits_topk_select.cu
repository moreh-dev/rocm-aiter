// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Prefill indexer top-k (K <= 512) over 16-bit logits: the high half of each fp32 logit, as stored by the HI16
// mode of fp8_mqa_logits_topk.cu. Two passes:
//   filter (one workgroup per row, the row held in VGPRs as order keys): threshold L from a 1/8 sample histogram
//     (11 coarse + 5 fine bits, ~1.5x the expected k-th rank), exact count of keys >= L (retried if < K or > CAPG),
//     block prefix scan, candidates (key, col) written contiguously to a per-row slot list (~800 of 32k for K 512).
//   select (one wave per row): 2 x 8-bit radix over the candidates, emission. The elements whose 16-bit key EQUALS
//     the k-th key and do not all fit (typ. ~30 per 32k row) get their exact fp32 logit recomputed in the wave with
//     the logits kernel's own MFMA, epilogue and summation order (same bits), and are ranked exactly. A boundary
//     bucket larger than TC elements (e.g. many identical logits) flags the row for select_big, an exact radix over
//     the recomputed low halves.
//   Rows with N <= 4096 skip the filter: select takes the whole row from registers (DIRECT).
// Output per row: indices relative to ks of the top-K logits in [ks, ke), unordered; 0..len-1 then -1 when len <= K
// (as vLLM's top_k_per_row_prefill).
#include "fp8_mqa_logits_topk.h"
#include <hip/hip_fp8.h>
#include <cstdint>

namespace topk_hi16 {
using fp8 = __hip_fp8_storage_t;
constexpr int WARP = 64, NUM_HEADS = 32, HEAD_SIZE = 128;
using u32x4 = __attribute__((__vector_size__(16))) unsigned int;
using f8x16 = __attribute__((__vector_size__(16))) fp8;
using f8x32 = __attribute__((__vector_size__(32))) fp8;
using f32x4 = __attribute__((__vector_size__(16))) float;
using f32x2 = __attribute__((__vector_size__(8))) float;
using VecOut = __attribute__((__vector_size__(16 * sizeof(float)))) float;
#define BUF_RSRC_WORD3 0x00020000

constexpr int EPC = 8;          // keys per 16-B chunk
constexpr int SH0 = 5;          // filter histogram: coarse bins = key >> 5 (11 bits), fine = next 5 bits
constexpr uint32_t SENT = 0;    // order key 0 <=> bits 0xFFFF (a NaN): never a real logit, marks "no element"
__device__ __forceinline__ uint32_t order_key(uint32_t h) { return (h & 0x8000u) ? (~h & 0xFFFFu) : (h | 0x8000u); }

__device__ __forceinline__ f8x32 pack(f8x16 lo, f8x16 hi) {
    f8x32 v; *reinterpret_cast<f8x16*>(&v) = lo; *(reinterpret_cast<f8x16*>(&v) + 1) = hi; return v;
}

// Exact logit of (row m, token tok) for the 32 tokens held by lanes l % 32 (all 64 lanes take part), valid in lanes
// 0..31. Operand layout, MFMA, epilogue and reduction order are fp8_mqa_logits_topk.cu's -> the same bits.
struct QW { f8x32 q0, q1; f32x2 w2[8]; };
__device__ __forceinline__ QW load_qw(int m, const fp8* __restrict__ Q, const float* __restrict__ W) {
    const int l = threadIdx.x % WARP, nA = l % 32, dhalf = l / 32, hset = l / 32;
    const int64_t qb = ((int64_t)m * NUM_HEADS + nA) * HEAD_SIZE;
    QW r;
    r.q0 = pack(*reinterpret_cast<const f8x16*>(&Q[qb + dhalf * 16]), *reinterpret_cast<const f8x16*>(&Q[qb + (2 + dhalf) * 16]));
    r.q1 = pack(*reinterpret_cast<const f8x16*>(&Q[qb + (4 + dhalf) * 16]), *reinterpret_cast<const f8x16*>(&Q[qb + (6 + dhalf) * 16]));
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const f32x4 wv = *reinterpret_cast<const f32x4*>(&W[(int64_t)m * NUM_HEADS + 8 * j + 4 * hset]);
        r.w2[2 * j] = f32x2{wv[0], wv[1]};
        r.w2[2 * j + 1] = f32x2{wv[2], wv[3]};
    }
    return r;
}
__device__ __forceinline__ float recompute32(const QW& qw, int tok, const fp8* __restrict__ Kc, const float* __restrict__ kv_scale) {
    const int l = threadIdx.x % WARP, dhalf = l / 32;
    const int64_t kb = (int64_t)tok * HEAD_SIZE;
    const f8x32 b0 = pack(*reinterpret_cast<const f8x16*>(&Kc[kb + dhalf * 16]), *reinterpret_cast<const f8x16*>(&Kc[kb + (2 + dhalf) * 16]));
    const f8x32 b1 = pack(*reinterpret_cast<const f8x16*>(&Kc[kb + (4 + dhalf) * 16]), *reinterpret_cast<const f8x16*>(&Kc[kb + (6 + dhalf) * 16]));
    const float scale = kv_scale[tok];
    VecOut c = {};
    c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(qw.q0, b0, c, 0, 0, 0, 0, 0, 0);
    c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(qw.q1, b1, c, 0, 0, 0, 0, 0, 0);
    float ae, ao;
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        const float ye = __builtin_fmaxf(c[2 * v], 0.f), yo = __builtin_fmaxf(c[2 * v + 1], 0.f);
        if (v == 0) { ae = ye * qw.w2[0][0]; ao = yo * qw.w2[0][1]; }
        else { ae = __builtin_fmaf(ye, qw.w2[v][0], ae); ao = __builtin_fmaf(yo, qw.w2[v][1], ao); }
    }
    float p0 = ae + ao, p1 = p0;
    asm("s_nop 1\n\tv_permlane32_swap_b32 %0, %1" : "+v"(p0), "+v"(p1));   // lanes 0-31: (own half, upper half)
    return (p0 + p1) * scale;
}

template <int NT, int CH>
__global__ __launch_bounds__(NT) void filter_kernel(const uint16_t* __restrict__ logits, const int* __restrict__ ks_p,
                                                    const int* __restrict__ ke_p, int M, int N, int K, int CAPG,
                                                    uint32_t* __restrict__ ckey, int* __restrict__ ccol, int* __restrict__ ccount) {
    constexpr int NW = NT / WARP, HB = 2048;
    __shared__ uint32_t hist[HB];
    __shared__ uint32_t hfine[32];
    __shared__ int scan_w[NW], s_wc[NW];
    __shared__ int s_bin, s_above, s_total;
    const int tid = threadIdx.x, lane = tid % WARP, warp = tid / WARP;
    const int m = blockIdx.x;
    const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N), nvalid = ke - ks;
    if (nvalid <= K) { if (tid == 0) ccount[m] = -1; return; }
    const int64_t rowg = (int64_t)m * N;
    const int64_t ga = (rowg + ks) & ~(int64_t)(EPC - 1);
    const int64_t gend = rowg + ke;
    const int nch = (int)((gend - ga + EPC - 1) / EPC);
    const int col0 = (int)(ga - rowg);
    const __amdgpu_buffer_rsrc_t rs = __builtin_amdgcn_make_buffer_rsrc((void*)(logits + ga), 0,
                                                                        (int)((((gend - ga) * 2) + 3) & ~3), BUF_RSRC_WORD3);
    auto raw = [&](int c) -> u32x4 { return __builtin_bit_cast(u32x4, __builtin_amdgcn_raw_buffer_load_b128(rs, c * 16, 0, 0)); };
    // bits -> order keys; cells outside [ks, ke) become 0xFFFF bits -> key SENT
    auto fix = [&](int c, u32x4 v) -> u32x4 {
        if (c == 0 || c >= nch - 1) {
            #pragma unroll
            for (int e = 0; e < EPC; ++e) {
                const int col = col0 + c * EPC + e;
                if (col < ks || col >= ke) v[e >> 1] |= 0xFFFFu << (16 * (e & 1));
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] ^= 0x80008000u | (((v[i] >> 15) & 0x00010001u) * 0x7FFFu);
        return v;
    };
    auto kelem = [](const u32x4& v, int e) -> uint32_t { return (v[e >> 1] >> (16 * (e & 1))) & 0xFFFFu; };
    u32x4 rv[CH];
    #pragma unroll
    for (int j = 0; j < CH; ++j) rv[j] = raw(tid + j * NT);
    #pragma unroll
    for (int j = 0; j < CH; ++j) rv[j] = fix(tid + j * NT, rv[j]);
    // Unrolled 64-key passes over the resident row: the empty asm per pass stops the compiler from keeping every
    // key's derived value live across passes (VGPR blow-up), sched_barrier per chunk bounds the code it hoists.
    auto for_all = [&](auto&& f) {
        #pragma unroll
        for (int j = 0; j < CH; ++j) asm volatile("" : "+v"(rv[j]));
        #pragma unroll
        for (int j = 0; j < CH; ++j) {
            const int cb = col0 + (tid + j * NT) * EPC;
            #pragma unroll
            for (int e = 0; e < EPC; ++e) f(kelem(rv[j], e), cb + e);
            __builtin_amdgcn_sched_barrier(0);
        }
        for (int c = NT * CH + tid; c < nch; c += NT) {   // rows longer than NT * CH * 8: re-read from global
            const u32x4 v = fix(c, raw(c));
            const int cb = col0 + c * EPC;
            #pragma unroll
            for (int e = 0; e < EPC; ++e) f(kelem(v, e), cb + e);
        }
    };
    auto for_samples = [&](auto&& f) {
        #pragma unroll
        for (int j = 0; j < CH; ++j) f(kelem(rv[j], 0));
        for (int c = NT * CH + tid; c < nch; c += NT) f(kelem(fix(c, raw(c)), 0));
    };
    // block-wide: bin b with (count above b) < target <= (count above b) + hst[b] -> s_bin, s_above, s_total
    auto find_bin = [&](const uint32_t* hst, int nbins, int target) {
        const int bpt = (nbins + NT - 1) / NT;
        const int lo = min(tid * bpt, nbins), hi = min(lo + bpt, nbins);
        int s = 0;
        for (int b = lo; b < hi; ++b) s += hst[b];
        int incl = s;
        #pragma unroll
        for (int d = 1; d < WARP; d <<= 1) { const int o = __shfl_up(incl, d); if (lane >= d) incl += o; }
        if (lane == WARP - 1) scan_w[warp] = incl;
        __syncthreads();
        int before = 0, total = 0;
        #pragma unroll
        for (int w = 0; w < NW; ++w) { const int x = scan_w[w]; total += x; if (w < warp) before += x; }
        int cum = total - (before + incl);
        if (tid == 0) s_total = total;
        for (int b = hi - 1; b >= lo; --b) {
            const int h = hst[b];
            if (cum < target && cum + h >= target) { s_bin = b; s_above = cum; }
            cum += h;
        }
        __syncthreads();
    };
    for (int b = tid; b < HB; b += NT) hist[b] = 0;
    __syncthreads();
    for_samples([&](uint32_t k) { if (k != SENT) atomicAdd(&hist[k >> SH0], 1u); });
    __syncthreads();
    const int expct = (int)((int64_t)K * (nvalid / EPC) / nvalid);
    int target = expct + expct / 2 + 8;
    bool ok = false;
    uint32_t L = 1u;
    int cnt = 0, C = 0;
    for (int it = 0; it < 3 && !ok; ++it) {
        find_bin(hist, HB, target);
        L = 1u;
        if (s_total >= target) {
            const uint32_t cb = (uint32_t)s_bin;
            const int tf = target - s_above;
            if (tid < 32) hfine[tid] = 0;
            __syncthreads();
            for_samples([&](uint32_t k) { if ((k >> SH0) == cb) atomicAdd(&hfine[(k >> (SH0 - 5)) & 31], 1u); });
            __syncthreads();
            find_bin(hfine, 32, tf);
            L = max((cb << SH0) | ((uint32_t)s_bin << (SH0 - 5)), 1u);   // bin 0 holds only NaN / SENT
        }
        cnt = 0;
        for_all([&](uint32_t k, int) { cnt += (k >= L); });
        int ws = cnt;
        #pragma unroll
        for (int d = 32; d >= 1; d >>= 1) ws += __shfl_xor(ws, d);
        if (lane == 0) s_wc[warp] = ws;
        __syncthreads();
        C = 0;
        #pragma unroll
        for (int w = 0; w < NW; ++w) C += s_wc[w];
        ok = C >= K && C <= CAPG;
        target = (C < K) ? target * 2 : max(target / 2, 1);
    }
    if (!ok) { if (tid == 0) ccount[m] = -2; return; }   // select falls back to the whole row
    // exclusive block prefix of the per-lane counts, then contiguous writes
    int incl = cnt;
    #pragma unroll
    for (int d = 1; d < WARP; d <<= 1) { const int o = __shfl_up(incl, d); if (lane >= d) incl += o; }
    int off = incl - cnt;
    #pragma unroll
    for (int w = 0; w < NW; ++w) off += (w < warp) ? s_wc[w] : 0;
    uint32_t* kr = ckey + (int64_t)m * CAPG;
    int* cr = ccol + (int64_t)m * CAPG;
    for_all([&](uint32_t k, int col) { if (k >= L) { kr[off] = k; cr[off] = col; ++off; } });
    if (tid == 0) ccount[m] = C;
}

// one wave per row: radix over the candidate list (or, if the filter gave up, over the whole row), emission, ties
template <int TC, int DIRECT, int BIG>
__device__ __forceinline__ void select_row(const int m, const uint16_t* __restrict__ logits, const int* __restrict__ ks_p,
                                           const int* __restrict__ ke_p, int* __restrict__ out, int N, int K, int CAPG,
                                           const uint32_t* __restrict__ ckey, const int* __restrict__ ccol,
                                           const int* __restrict__ ccount, const fp8* __restrict__ Q, const fp8* __restrict__ Kc,
                                           const float* __restrict__ kv_scale, const float* __restrict__ W,
                                           uint32_t* __restrict__ hist, float* __restrict__ tv, int* __restrict__ tcol,
                                           int* __restrict__ bigflag) {
    constexpr int HB = 256, NRD = 2;                      // 2 rounds of 8-bit digits
    const int lane = threadIdx.x % WARP;
    if (!BIG && lane == 0) bigflag[m] = 0;
    int* orow = out + (int64_t)m * K;
    const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
    // DIRECT (short rows, N <= DIRECT * 64): no filter kernel, the whole row is the candidate list
    const int C = DIRECT ? (ke - ks <= K ? -1 : -3) : ccount[m];
    if (C == -1) { for (int i = lane; i < K; i += WARP) orow[i] = i < ke - ks ? i : -1; return; }
    const uint32_t* kr = ckey + (int64_t)m * CAPG;
    const int* cr = ccol + (int64_t)m * CAPG;
    // candidates 0 .. RC*64-1 in registers (one global read), the rest (C > RC*64, rare) re-read per pass
    constexpr int RC = DIRECT ? DIRECT : 16;
    uint32_t rk[RC]; int rc[RC];
    if constexpr (DIRECT) {
        #pragma unroll
        for (int j = 0; j < RC; ++j) {
            const int col = ks + j * WARP + lane;
            rk[j] = col < ke ? order_key(logits[(int64_t)m * N + col]) : SENT; rc[j] = col;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < RC; ++j) { const int i = j * WARP + lane; const bool v = i < C; rk[j] = v ? kr[i] : SENT; rc[j] = v ? cr[i] : 0; }
    }
    const QW qw = load_qw(m, Q, W);    // prefetch for the (likely) boundary recompute
    auto row_key = [&](int col) { return order_key(logits[(int64_t)m * N + col]); };
    auto each = [&](auto&& f) {
        if (DIRECT) {
            #pragma unroll
            for (int j = 0; j < RC; ++j) if (rk[j] != SENT) f(rk[j], rc[j]);
        } else if (C >= 0) {
            #pragma unroll
            for (int j = 0; j < RC; ++j) if (j * WARP + lane < C) f(rk[j], rc[j]);
            for (int i = RC * WARP + lane; i < C; i += WARP) f(kr[i], cr[i]);
        } else {   // filter gave up (pathological row): whole row from global, element by element
            for (int col = ks + lane; col < ke; col += WARP) f(row_key(col), col);
        }
    };
    // calls g(key, col) with all 64 lanes active (ballots inside), SENT for lanes without an element
    auto each_uniform = [&](auto&& g) {
        if (DIRECT) {
            #pragma unroll
            for (int j = 0; j < RC; ++j) g(rk[j], rc[j]);
        } else if (C >= 0) {
            #pragma unroll
            for (int j = 0; j < RC; ++j) if (j * WARP < C) g(j * WARP + lane < C ? rk[j] : SENT, rc[j]);
            for (int i0 = RC * WARP; i0 < C; i0 += WARP) { const int i = i0 + lane; const bool v = i < C; g(v ? kr[i] : SENT, v ? cr[i] : 0); }
        } else {
            for (int c0 = ks; c0 < ke; c0 += WARP) { const int col = c0 + lane; g(col < ke ? row_key(col) : SENT, col); }
        }
    };
    auto mem = [] { asm volatile("" ::: "memory"); };
    uint32_t prefix = 0;
    int above = 0, cnt_eq = 0;
    #pragma unroll
    for (int r = 0; r < NRD; ++r) {
        const int sh = 16 - 8 * (r + 1), hsh = sh + 8;
        for (int b = lane; b < HB; b += WARP) hist[b] = 0;
        mem();
        const uint32_t pf = prefix;
        each([&](uint32_t k, int) { if (r == 0 ? (k != SENT) : ((k >> hsh) == pf)) atomicAdd(&hist[(k >> sh) & (HB - 1)], 1u); });
        mem();
        // wave-level find: lane owns bins [lane*bpl, (lane+1)*bpl)
        constexpr int bpl = HB / WARP;
        const int lo = lane * bpl, hi = lo + bpl;
        int s = 0;
        for (int b = lo; b < hi; ++b) s += hist[b];
        int incl = s;
        #pragma unroll
        for (int d = 1; d < WARP; d <<= 1) { const int o = __shfl_up(incl, d); if (lane >= d) incl += o; }
        const int total = __shfl(incl, WARP - 1);
        int cum = total - incl;
        const int target = K - above;
        int fb = -1, fa = 0, fc = 0;
        for (int b = hi - 1; b >= lo; --b) {
            const int h = hist[b];
            if (cum < target && cum + h >= target) { fb = b; fa = cum; fc = h; }
            cum += h;
        }
        const uint64_t fm = __ballot(fb >= 0);
        const int src = __builtin_ctzll(fm);
        prefix = (prefix << 8) | (uint32_t)__shfl(fb, src);
        above += __shfl(fa, src);
        cnt_eq = __shfl(fc, src);
        mem();
    }
    const uint32_t Tk = prefix;          // 16-bit key of the K-th element
    const int need = K - above;          // how many of the cnt_eq elements with key Tk are taken
    const bool direct = cnt_eq == need;  // all of them: no fp32 needed
    int pdef = 0, ptie = 0;
    each_uniform([&](uint32_t k, int col) {
        const bool gt = k > Tk, eq = k == Tk;
        const uint64_t mg = __ballot(gt), me = __ballot(eq);
        const int og = pdef + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(mg >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)mg, 0));
        const int oe = ptie + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(me >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)me, 0));
        if (gt && og < K) orow[og] = col - ks;   // og < K always, unless NaN logits were dropped as SENT
        if (eq) {
            if (direct) { if (oe < need) orow[above + oe] = col - ks; }
            else if (oe < TC) tcol[oe] = col;
        }
        pdef += __popcll(mg); ptie += __popcll(me);
    });
    if (direct) return;
    mem();
    const int n = min(cnt_eq, TC);
    const uint32_t hT = (Tk & 0x8000u) ? (Tk & 0x7FFFu) : (~Tk & 0xFFFFu);   // high-half bits of the boundary bucket
    if (cnt_eq > TC) {
      if constexpr (!BIG) { if (lane == 0) bigflag[m] = 1; return; } else {
        // big boundary bucket: exact 2 x 8-bit radix on the LOW half, recomputed in every pass, ties gathered 32 at a
        // time (ballot into tcol, which holds >= 96 entries)
        static_assert(TC >= 96, "gather buffer");
        const uint32_t lflip = (hT & 0x8000u) ? 0xFFFFu : 0u;     // order key of the low half (sign of the bucket)
        auto stream = [&](auto&& g) {                              // g(valid, lkey, col), called by all 64 lanes
            int cur = 0;
            auto flush = [&](int nn) {
                const int i = lane % 32;
                const int col = tcol[min(i, nn - 1)];
                const float v = recompute32(qw, col, Kc, kv_scale);
                g(lane < 32 && i < nn, (__builtin_bit_cast(uint32_t, v) & 0xFFFFu) ^ lflip, col);
            };
            each_uniform([&](uint32_t k, int col) {
                const bool eq = k == Tk;
                const uint64_t me = __ballot(eq);
                const int pos = cur + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(me >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)me, 0));
                if (eq) tcol[pos] = col;
                cur += __popcll(me);
                mem();
                while (cur >= 32) {
                    flush(32);
                    const int rem = cur - 32;
                    const int x = lane < rem ? tcol[32 + lane] : 0;
                    mem();
                    if (lane < rem) tcol[lane] = x;
                    mem();
                    cur = rem;
                }
            });
            if (cur > 0) flush(cur);
        };
        auto find256 = [&](int target, int& fbin, int& fab) {      // wave-level: bin b with above < target <= above + h
            const int lo = lane * 4;
            int s4 = 0;
            for (int b = lo; b < lo + 4; ++b) s4 += hist[b];
            int incl = s4;
            #pragma unroll
            for (int d = 1; d < WARP; d <<= 1) { const int o = __shfl_up(incl, d); if (lane >= d) incl += o; }
            int cum = __shfl(incl, WARP - 1) - incl, fb = -1, fa = 0;
            for (int b = lo + 3; b >= lo; --b) { const int h = hist[b]; if (cum < target && cum + h >= target) { fb = b; fa = cum; } cum += h; }
            const int src = __builtin_ctzll(__ballot(fb >= 0));
            fbin = __shfl(fb, src); fab = __shfl(fa, src);
        };
        int b0, a0, b1, a1;
        for (int b = lane; b < 256; b += WARP) hist[b] = 0;
        mem();
        stream([&](bool ok, uint32_t lk, int) { if (ok) atomicAdd(&hist[lk >> 8], 1u); });
        mem();
        find256(need, b0, a0);
        mem();
        for (int b = lane; b < 256; b += WARP) hist[b] = 0;
        mem();
        stream([&](bool ok, uint32_t lk, int) { if (ok && (lk >> 8) == (uint32_t)b0) atomicAdd(&hist[lk & 255], 1u); });
        mem();
        find256(need - a0, b1, a1);
        const uint32_t TL = ((uint32_t)b0 << 8) | (uint32_t)b1;
        const int ngt = a0 + a1, neq = need - ngt;                 // > TL: all; == TL: any neq of them (equal fp32)
        int pg = 0, pe = 0;
        stream([&](bool ok, uint32_t lk, int col) {
            const bool gt = ok && lk > TL, eq = ok && lk == TL;
            const uint64_t mg = __ballot(gt), me = __ballot(eq);
            const int og = pg + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(mg >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)mg, 0));
            const int oe = pe + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(me >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)me, 0));
            if (gt) orow[above + og] = col - ks;
            if (eq && oe < neq) orow[above + ngt + oe] = col - ks;
            pg += __popcll(mg); pe += __popcll(me);
        });
        return;
      }
    }
    for (int b = 0; b * 32 < n; ++b) {
        const int i = min(b * 32 + lane % 32, n - 1);
        const float v = recompute32(qw, tcol[i], Kc, kv_scale);
        if (lane < 32 && b * 32 + lane < n) tv[b * 32 + lane] = v;
    }
    mem();
    for (int i = lane; i < n; i += WARP) {
        const float v = tv[i];
        int r = 0;
        for (int j = 0; j < n; ++j) { const float u = tv[j]; r += (u > v) || (u == v && j < i); }
        if (r < need) orow[above + r] = tcol[i] - ks;
    }
}

template <int TC, int DIRECT = 0>
__global__ __launch_bounds__(256) void select_kernel(const uint16_t* __restrict__ logits, const int* __restrict__ ks_p,
                                                     const int* __restrict__ ke_p, int* __restrict__ out, int M, int N, int K,
                                                     int CAPG, const uint32_t* __restrict__ ckey, const int* __restrict__ ccol,
                                                     const int* __restrict__ ccount, const fp8* __restrict__ Q,
                                                     const fp8* __restrict__ Kc, const float* __restrict__ kv_scale,
                                                     const float* __restrict__ W, int* __restrict__ bigflag) {
    __shared__ uint32_t hist_all[4][256];
    __shared__ float tv_all[4][TC];
    __shared__ int tc_all[4][TC];
    const int w = threadIdx.x / WARP, m = blockIdx.x * 4 + w;
    if (m >= M) return;
    select_row<TC, DIRECT, 0>(m, logits, ks_p, ke_p, out, N, K, CAPG, ckey, ccol, ccount, Q, Kc, kv_scale, W,
                              hist_all[w], tv_all[w], tc_all[w], bigflag);
}
// rows whose boundary bucket exceeded TC (flagged by select_kernel): exact multi-pass path, one wave per flagged row
template <int TC, int DIRECT = 0>
__global__ __launch_bounds__(256) void select_big_kernel(const uint16_t* __restrict__ logits, const int* __restrict__ ks_p,
                                                         const int* __restrict__ ke_p, int* __restrict__ out, int M, int N, int K,
                                                         int CAPG, const uint32_t* __restrict__ ckey, const int* __restrict__ ccol,
                                                         const int* __restrict__ ccount, const fp8* __restrict__ Q,
                                                         const fp8* __restrict__ Kc, const float* __restrict__ kv_scale,
                                                         const float* __restrict__ W, int* __restrict__ bigflag) {
    __shared__ uint32_t hist_all[4][256];
    __shared__ float tv_all[4][TC];
    __shared__ int tc_all[4][TC];
    const int w = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    for (int b0 = (blockIdx.x * 4 + w) * WARP; b0 < M; b0 += gridDim.x * 4 * WARP) {   // 64 flags per wave per step
        uint64_t fm = __ballot(b0 + lane < M && bigflag[b0 + lane] != 0);
        while (fm) {
            const int m = b0 + __builtin_ctzll(fm);
            fm &= fm - 1;
            select_row<TC, DIRECT, 1>(m, logits, ks_p, ke_p, out, N, K, CAPG, ckey, ccol, ccount, Q, Kc, kv_scale, W,
                                      hist_all[w], tv_all[w], tc_all[w], bigflag);
        }
    }
}
}  // namespace topk_hi16

namespace {
constexpr int DIRECT_MAX_N = 4096;   // select alone holds a whole row of up to 64 keys per lane
// Candidate slots per row (~1.5 K + noise are used). <= N / 4 keeps the scratch (8 B per slot) plus the 16-bit
// logits within the fp32 logits' bytes, which the indexer's prefill chunking budgets for; >= 2 K for K <= 512.
int capg(int N) { return N / 4 < 2048 ? N / 4 : 2048; }
}  // namespace

// [ckey | ccol] M x CAPG each, then ccount and the big-bucket flags (M each)
int64_t topk_hi16_workspace_ints(int M, int N) {
    return N <= DIRECT_MAX_N ? (int64_t)M : 2 * (int64_t)M * capg(N) + 2 * (int64_t)M;
}

int launch_topk_hi16(const void* logits16, const int* ks, const int* ke, int* out, int M, int N, int K,
                     const void* q, const void* k, const float* kv_scale, const float* w, int* ws, hipStream_t s) {
    using namespace topk_hi16;
    if (K > TOPK_HI16_MAX_K) return -2;
    const auto* lg = static_cast<const uint16_t*>(logits16);
    const auto* Q = static_cast<const fp8*>(q);
    const auto* Kc = static_cast<const fp8*>(k);
    constexpr int TC = 256;
#define SEL(D_) \
    select_kernel<TC, D_><<<(M + 3) / 4, 256, 0, s>>>(lg, ks, ke, out, M, N, K, cg, ckey, ccol, ccnt, Q, Kc, kv_scale, w, bigf); \
    select_big_kernel<TC, D_><<<64, 256, 0, s>>>(lg, ks, ke, out, M, N, K, cg, ckey, ccol, ccnt, Q, Kc, kv_scale, w, bigf)
    if (N <= DIRECT_MAX_N) {
        const int cg = 0;
        uint32_t* ckey = nullptr; int* ccol = nullptr; int* ccnt = nullptr;
        int* bigf = ws;
        if (N <= 2048) { SEL(32); } else { SEL(64); }
    } else {
        // tuned on MI355X (9/10): the row stays in VGPRs up to NT * CH * 8 keys
        const int cg = capg(N);
        auto* ckey = reinterpret_cast<uint32_t*>(ws);
        int* ccol = ws + (int64_t)M * cg;
        int* ccnt = ccol + (int64_t)M * cg;
        int* bigf = ccnt + M;
        if (N <= 16384) filter_kernel<256, 8><<<M, 256, 0, s>>>(lg, ks, ke, M, N, K, cg, ckey, ccol, ccnt);
        else filter_kernel<512, 8><<<M, 512, 0, s>>>(lg, ks, ke, M, N, K, cg, ckey, ccol, ccnt);
        SEL(0);
    }
#undef SEL
    return (int)hipGetLastError();
}
