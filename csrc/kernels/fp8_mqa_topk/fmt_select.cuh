// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused FP8 MQA logits + top-k (gfx950): threshold, select and exact fallback (port of Nam's fuse48.hip, MV-5041
// section 18) with two changes:
//   * tau is the exact target-th largest fp32 SAMPLE logit (Nam: lower edge of a 16-bit bucket), 32-bit keys;
//   * the fallback recompute is bit-identical to v35wes (plain MFMA, scalar even/odd fma chains), not to v35/v35a.
//
//   tau  (one wave per row): e = K * ns / nvalid samples expected above the K-th logit, target = ceil(e + 5 sqrt(e)) + 8,
//        tau = target-th largest sample; -inf when nvalid <= CAP or ns < target (every valid key is a candidate).
//   select (one wave per row): exact top-K over the fp32 candidates (<= 2048 in registers: bitwise binary search;
//        else 4 x 8-bit radix re-reading global); a row whose segment overflowed or that has fewer than
//        min(K, nvalid) candidates is appended to the fallback list.
//   fallback: flagged rows only; exact logits of the whole row recomputed into scratch, then the same radix select.
// Output = column - ks (vLLM convention), -1 padding.
#pragma once
#include "fmt_common.cuh"

namespace fmt {

// ------------------------------------------------------------------------------------------------- wave helpers
// wave-level: bin b of hist[256] with above(b) < target <= above(b) + h[b]
__device__ __forceinline__ void wave_find256(const uint32_t* hist, int target, int lane, int& fbin, int& fab, int& fcnt) {
    const int lo = lane * 4;
    int s4 = 0;
    for (int b = lo; b < lo + 4; ++b) s4 += hist[b];
    const int incl = wave_incl_scan(s4, lane);
    int cum = __shfl(incl, WARP - 1) - incl, fb = -1, fa = 0, fc = 0;
    for (int b = lo + 3; b >= lo; --b) { const int h = hist[b]; if (cum < target && cum + h >= target) { fb = b; fa = cum; fc = h; } cum += h; }
    const uint64_t bm = __ballot(fb >= 0);
    const int src = bm ? __builtin_ctzll(bm) : 0;
    fbin = __shfl(fb, src); fab = __shfl(fa, src); fcnt = __shfl(fc, src);
}

// radix over 32-bit keys delivered by each(f) (f(key, col) called per element, non-uniformly) -> key T of rank K
template <int NRD, typename Each>
__device__ __forceinline__ void wave_radix(Each&& each, uint32_t* hist, int K, int lane, uint32_t& T, int& above, int& cnt_eq) {
    uint32_t prefix = 0; above = 0; cnt_eq = 0;
    #pragma unroll
    for (int r = 0; r < NRD; ++r) {
        const int sh = 8 * (NRD - 1 - r);
        for (int b = lane; b < 256; b += WARP) hist[b] = 0;
        mem_fence_wave();
        const uint32_t pf = prefix;
        each([&](uint32_t k, int) { if (r == 0 || (uint32_t)((uint64_t)k >> (sh + 8)) == pf) atomicAdd(&hist[(k >> sh) & 255], 1u); });
        mem_fence_wave();
        int fb, fa, fc;
        wave_find256(hist, K - above, lane, fb, fa, fc);
        prefix = (prefix << 8) | (uint32_t)fb; above += fa; cnt_eq = fc;
        mem_fence_wave();
    }
    T = prefix;
}
// wave sum of a per-lane int with DPP (VALU only, lane 63 ends with the total; gfx9 row_bcast), as a scalar
__device__ __forceinline__ int wave_sum_dpp(int x) {
    x += __builtin_amdgcn_update_dpp(0, x, 0x111, 0xf, 0xf, false);   // row_shr:1
    x += __builtin_amdgcn_update_dpp(0, x, 0x112, 0xf, 0xf, false);   // row_shr:2
    x += __builtin_amdgcn_update_dpp(0, x, 0x114, 0xf, 0xf, false);   // row_shr:4
    x += __builtin_amdgcn_update_dpp(0, x, 0x118, 0xf, 0xf, false);   // row_shr:8  -> lane 15 of each row = row sum
    x += __builtin_amdgcn_update_dpp(0, x, 0x142, 0xa, 0xf, false);   // row_bcast:15 into rows 1, 3
    x += __builtin_amdgcn_update_dpp(0, x, 0x143, 0xc, 0xf, false);   // row_bcast:31 into rows 2, 3
    return __builtin_amdgcn_readlane(x, 63);
}
// count of keys >= x over the wave (keys in registers rk[0..R), 0 = empty slot never counted since x >= 1).
// Per-lane VALU count (v_cmp + v_addc on VCC: VALU->VALU forwarding) + one DPP reduction. A ballot + s_bcnt1 per
// register serializes every compare through VCC -> SALU transfers (measured ~10x slower in the tau search).
template <int R>
__device__ __forceinline__ int wave_count_ge(const uint32_t (&rk)[R], uint32_t x) {
    int c = 0;
    #pragma unroll
    for (int j = 0; j < R; ++j) c += (rk[j] >= x) ? 1 : 0;
    return wave_sum_dpp(c);
}
// T = largest key with count(>= T) >= target (bitwise binary search, 32 iterations of compare + wave reduce)
template <int R>
__device__ __forceinline__ void wave_bsearch32(const uint32_t (&rk)[R], int target, uint32_t& T, int& above, int& cnt_eq) {
    uint32_t t = 0;
    #pragma unroll 1
    for (int b = 31; b >= 0; --b) { const uint32_t c = t | (1u << b); if (wave_count_ge(rk, c) >= target) t = c; }
    T = t;
    const int ge = wave_count_ge(rk, t);
    above = (t == 0xFFFFFFFFu) ? 0 : wave_count_ge(rk, t + 1);
    cnt_eq = ge - above;
}
// prefix-only search: largest T with low LOWBIT bits zero and count(>= T) >= target (32 - LOWBIT iterations).
// The float of key T is <= the target-th largest key's float: a conservative threshold.
template <int R, int LOWBIT>
__device__ __forceinline__ uint32_t wave_bsearch_prefix(const uint32_t (&rk)[R], int target) {
    uint32_t t = 0;
    #pragma unroll 1
    for (int b = 31; b >= LOWBIT; --b) { const uint32_t c = t | (1u << b); if (wave_count_ge(rk, c) >= target) t = c; }
    return t;
}

// emission over a uniform stream: each_u(g) must call g(valid, key, col) with ALL lanes, same trip count
template <typename EachU>
__device__ __forceinline__ void wave_emit(EachU&& each_u, uint32_t T, int above, int K, int ks, int lane, int* __restrict__ orow) {
    const int need = K - above;
    int pdef = 0, ptie = 0;
    each_u([&](bool v, uint32_t k, int col) {
        const bool gt = v && k > T, eq = v && k == T;
        const uint64_t mg = __ballot(gt), me = __ballot(eq);
        const int og = pdef + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(mg >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)mg, 0));
        const int oe = ptie + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(me >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)me, 0));
        if (gt) orow[og] = col - ks;
        if (eq && oe < need) orow[above + oe] = col - ks;
        pdef += __popcll(mg); ptie += __popcll(me);
    });
}

// wave min / max of a per-lane u32 with DPP (gfx9 row_bcast), result as a scalar
__device__ __forceinline__ uint32_t wave_max_u32(uint32_t x) {
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x142, 0xa, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x143, 0xc, 0xf, false));
    return (uint32_t)__builtin_amdgcn_readlane((int)x, 63);
}
__device__ __forceinline__ uint32_t wave_min_u32(uint32_t x) { return ~wave_max_u32(~x); }

// Relative two-pass radix threshold of one row (one wave): keys of the valid samples in rk (0 = empty slot).
// Pass 1 histograms the 8 bits just below the common prefix of [kmin, kmax], pass 2 the next 8 bits inside the bin
// holding the target-th largest key, so the bucket resolution is ~(kmax - kmin) / 2^16 whatever the score offset.
// Returns the lowest float of the final bucket (<= the target-th largest sample): conservative.
template <int R>
__device__ __forceinline__ float wave_tau_relradix(const uint32_t (&rk)[R], int target, uint32_t* hist, int lane) {
    uint32_t vmax = 0u, vmin = 0xFFFFFFFFu;
    #pragma unroll
    for (int j = 0; j < R; ++j) { vmax = max(vmax, rk[j]); vmin = min(vmin, rk[j] ? rk[j] : 0xFFFFFFFFu); }
    const uint32_t kmax = wave_max_u32(vmax), kmin = wave_min_u32(vmin);
    const uint32_t diff = kmax ^ kmin;
    if (diff == 0u) return __builtin_bit_cast(float, okey32_inv(kmax));          // every sample equal
    const int hb = 31 - __builtin_clz(diff);                                      // highest differing bit
    const int sh1 = max(hb - 7, 0), sh2 = max(sh1 - 8, 0);
    const uint32_t pre1 = (sh1 + 8 >= 32) ? 0u : (kmax >> (sh1 + 8));             // common prefix above pass-1 bits
    // pass 1
    for (int b = lane; b < 256; b += WARP) hist[b] = 0u;
    mem_fence_wave();
    #pragma unroll
    for (int j = 0; j < R; ++j) if (rk[j]) atomicAdd(&hist[(rk[j] >> sh1) & 255u], 1u);
    mem_fence_wave();
    int b1, a1, c1;
    wave_find256(hist, target, lane, b1, a1, c1);
    uint32_t T = (pre1 << 8 | (uint32_t)b1) << sh1;
    if (sh1 > 0) {
        // pass 2: keys whose bits above sh1 equal T's
        mem_fence_wave();
        for (int b = lane; b < 256; b += WARP) hist[b] = 0u;
        mem_fence_wave();
        const uint32_t hiT = T >> sh1;
        const int w2 = sh1 - sh2;                                                 // 1..8 bits
        #pragma unroll
        for (int j = 0; j < R; ++j)
            if (rk[j] && (rk[j] >> sh1) == hiT) atomicAdd(&hist[(rk[j] >> sh2) & ((1u << w2) - 1u)], 1u);
        mem_fence_wave();
        int b2, a2, c2;
        wave_find256(hist, target - a1, lane, b2, a2, c2);
        T |= (uint32_t)b2 << sh2;
    }
    return __builtin_bit_cast(float, okey32_inv(T));
}

// ------------------------------------------------------------------------------------------------- tau
// Threshold of one row from fp32 sample logits srow[a, b) (sample j = key j*S). One wave. Returns -inf when every
// valid key must be a candidate. `hist` = 256 u32 of LDS owned by this wave. The register count follows the sample
// count (wave-uniform branch), and only the top 20 bits of the order key are searched (tau = lowest float of that
// 2^-11-relative bucket, still <= the target-th largest sample).
constexpr int TAU_LOWBIT = 12;
template <int R, bool COHERENT>
__device__ __forceinline__ float wave_tau_regs(const float* __restrict__ srow, int a, int b, int target, uint32_t* hist, int lane) {
    uint32_t rk[R];
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    #pragma unroll
    for (int j = 0; j < R; ++j) {               // unconditional loads (clamped index), masked after: no branch + wait per load
        const int c = a + j * WARP + lane;
        const float v = srow[min(c, b - 1)];
        rk[j] = c < b ? okey32(__builtin_bit_cast(uint32_t, v)) : 0u;
    }
    return wave_tau_relradix<R>(rk, target, hist, lane);
}
template <bool COHERENT>
__device__ __forceinline__ float wave_tau(const float* __restrict__ srow, int a, int b, int nvalid, int K, int CAP,
                                          uint32_t* hist, int lane) {
    const int ns = max(b - a, 0);
    const float e = (float)K * (float)ns / (float)max(nvalid, 1);
    const int target = (int)ceilf(e + 5.f * sqrtf(e)) + 8;
    if (nvalid <= CAP || ns < target) return -__builtin_huge_valf();
    if (ns <= 8 * WARP) return wave_tau_regs<8, COHERENT>(srow, a, b, target, hist, lane);
    if (ns <= 16 * WARP) return wave_tau_regs<16, COHERENT>(srow, a, b, target, hist, lane);
    if (ns <= 32 * WARP) return wave_tau_regs<32, COHERENT>(srow, a, b, target, hist, lane);
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    constexpr int RT = 32;                          // first RT * 64 samples held in VGPRs (one read), rest re-read
    uint32_t rk[RT];
    #pragma unroll
    for (int j = 0; j < RT; ++j) { const int c = a + j * WARP + lane; const float v = srow[min(c, b - 1)]; rk[j] = c < b ? okey32(__builtin_bit_cast(uint32_t, v)) : 0u; }
    auto each = [&](auto&& f) {
        #pragma unroll
        for (int j = 0; j < RT; ++j) if (rk[j]) f(rk[j], 0);
        for (int c0 = a + RT * WARP; c0 < b; c0 += WARP) {     // uniform trip count, unconditional load
            const int c = c0 + lane; const float v = srow[min(c, b - 1)];
            if (c < b) f(okey32(__builtin_bit_cast(uint32_t, v)), c);
        }
    };
    uint32_t T; int above, ce;
    wave_radix<4>(each, hist, target, lane, T, above, ce);
    return __builtin_bit_cast(float, okey32_inv(T));
}

// Threshold of NR rows at once (one wave): the NR prefix searches run in lockstep, so every step issues NR
// independent compare/popcount chains (the single-row search is a serial chain and latency-bound at low occupancy).
// Row i: samples srow_i[a_i, b_i), wanted rank target_i; act_i false -> tau_i = -inf. All rows must have
// b_i - a_i <= R * 64. The caller handles rows needing more samples with wave_tau().
template <int R, int NR, bool COHERENT>
__device__ __forceinline__ void wave_tau_rows(const float* const (&srow)[NR], const int (&a)[NR], const int (&b)[NR],
                                              const int (&target)[NR], const bool (&act)[NR], float (&tau)[NR], int lane) {
    uint32_t rk[NR][R];
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    #pragma unroll
    for (int i = 0; i < NR; ++i) {
        const int last = act[i] ? b[i] - 1 : 0;   // inactive rows read element 0 of a valid row (value unused)
        #pragma unroll
        for (int j = 0; j < R; ++j) {
            const int c = a[i] + j * WARP + lane;
            const float v = srow[i][act[i] ? min(c, last) : 0];
            rk[i][j] = (act[i] && c < b[i]) ? okey32(__builtin_bit_cast(uint32_t, v)) : 0u;
        }
    }
    uint32_t t[NR];
    #pragma unroll
    for (int i = 0; i < NR; ++i) t[i] = 0;
    #pragma unroll 1
    for (int bit = 31; bit >= TAU_LOWBIT; --bit) {
        #pragma unroll
        for (int i = 0; i < NR; ++i) {
            const uint32_t c = t[i] | (1u << bit);
            if (wave_count_ge(rk[i], c) >= target[i]) t[i] = c;
        }
    }
    #pragma unroll
    for (int i = 0; i < NR; ++i) tau[i] = act[i] ? __builtin_bit_cast(float, okey32_inv(t[i])) : -__builtin_huge_valf();
}

// tau of the 4 rows m0..m0+3 of one wave from samples ls[m][*] (row stride ls_cols; sample j = key j*S).
template <bool COHERENT>
__device__ __forceinline__ void wave_tau4(const float* __restrict__ ls, int ls_cols, int Ns, int S, int m0, int M, int N,
                                          const int* __restrict__ ks_p, const int* __restrict__ ke_p, int K, int CAP,
                                          uint32_t* hist, int lane, float (&tau)[4]) {
    const float* srow[4]; int a[4], b[4], target[4], nv[4]; bool act[4];
    int nsmax = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + i;
        const bool ok = m < M;
        const int ksm = ok ? max(ks_p[m], 0) : 0, kem = ok ? min(ke_p[m], N) : 0;
        nv[i] = kem - ksm;
        a[i] = FMT_CDIV(ksm, S);
        b[i] = kem > 0 ? min(FMT_CDIV(kem, S), Ns) : 0;
        const int ns = max(b[i] - a[i], 0);
        const float e = (float)K * (float)ns / (float)max(nv[i], 1);
        target[i] = (int)ceilf(e + 5.f * sqrtf(e)) + 8;
        act[i] = ok && nv[i] > CAP && ns >= target[i];
        srow[i] = ls + (int64_t)(ok ? m : 0) * ls_cols;
        if (act[i]) nsmax = max(nsmax, ns);
    }
    (void)nsmax;
    #pragma unroll 1
    for (int i = 0; i < 4; ++i)
        tau[i] = act[i] ? wave_tau<COHERENT>(srow[i], a[i], b[i], nv[i], K, CAP, hist, lane) : -__builtin_huge_valf();
}

// standalone tau kernel (steps 0): fp32 sample logits ls[M, Ns] of keys j*S; 4 rows (waves) per block.
// Also resets the fallback counter of this call (select appends, fallback kernels read).
__global__ __launch_bounds__(256) void tau_kernel(const float* __restrict__ ls, int Ns, int S, const int* __restrict__ ks_p,
                                                  const int* __restrict__ ke_p, int M, int N, int K, int CAP,
                                                  float* __restrict__ tau, int* __restrict__ fbc) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *fbc = 0;
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, m = blockIdx.x * 4 + w;
    if (m >= M) return;
    const int ksm = max(ks_p[m], 0), kem = min(ke_p[m], N);
    const int nvalid = kem - ksm;
    const int a = FMT_CDIV(ksm, S), b = kem > 0 ? min(FMT_CDIV(kem, S), Ns) : 0;
    const float t = wave_tau<false>(ls + (int64_t)m * Ns, a, b, nvalid, K, CAP, hist_all[w], lane);
    if (lane == 0) tau[m] = t;
}

// ------------------------------------------------------------------------------------------------- select
// One row: candidates of nsplit segments cand[((s * M + m) * SEG + i], counts cnt[s * M + m]. Returns false if the row
// must go to the fallback (overflow, or fewer than min(K, nvalid) candidates).
template <int RC, bool COHERENT>
__device__ __forceinline__ bool wave_select_row(const unsigned long long* __restrict__ cand, const int* __restrict__ cnt,
                                                int nsplit, int CAP, int M, int m, int ks, int nvalid, int K,
                                                int* __restrict__ orow, uint32_t* hist, int lane) {
    const int SEG = CAP + 1;
    if (nvalid <= 0) { for (int i = lane; i < K; i += WARP) orow[i] = -1; return true; }
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    int c_l = cnt[(int64_t)min(lane, nsplit - 1) * M + m];
    if (lane >= nsplit) c_l = 0;
    const bool ovf = __ballot(c_l > CAP) != 0;
    c_l = min(c_l, CAP);
    const int c_incl = wave_incl_scan(c_l, lane);
    const int C = __shfl(c_incl, WARP - 1);
    if (ovf || C < min(K, nvalid)) return false;
    int pe_s[16];                                   // segment prefix ends, read with all lanes active (uniform)
    #pragma unroll
    for (int t = 0; t < 16; ++t) pe_s[t] = __shfl(c_incl, t);
    auto addr = [&](int i) -> int64_t {
        int s = 0, base = 0;
        #pragma unroll
        for (int t = 0; t < 16; ++t) if (t < nsplit && i >= pe_s[t]) { s = t + 1; base = pe_s[t]; }
        return ((int64_t)s * M + m) * SEG + (i - base);
    };
    // unconditional loads of a clamped index (masked after): no exec branch + vmcnt(0) per load
    auto ldc = [&](int i) -> unsigned long long {
        const int ic = min(i, C - 1);
        const int64_t o = nsplit == 1 ? (int64_t)m * SEG + ic : addr(ic);
        return cand[o];
    };
    uint32_t rk[RC]; int rc[RC];
    #pragma unroll
    for (int j = 0; j < RC; ++j) {
        const int i = j * WARP + lane;
        const unsigned long long d = ldc(i);
        rk[j] = i < C ? okey32((uint32_t)d) : 0u; rc[j] = (int)(d >> 32);
    }
    if (C <= K) {     // every candidate is selected (rows with < K valid keys): emit all, pad -1
        #pragma unroll
        for (int j = 0; j < RC; ++j) if (j * WARP + lane < C) orow[j * WARP + lane] = rc[j] - ks;
        for (int i0 = RC * WARP; i0 < C; i0 += WARP) { const int i = i0 + lane; const unsigned long long d = ldc(i); if (i < C) orow[i] = (int)(d >> 32) - ks; }
        for (int i = C + lane; i < K; i += WARP) orow[i] = -1;
        return true;
    }
    auto each = [&](auto&& f) {
        #pragma unroll
        for (int j = 0; j < RC; ++j) if (j * WARP + lane < C) f(rk[j], rc[j]);
        for (int i0 = RC * WARP; i0 < C; i0 += WARP) { const int i = i0 + lane; const unsigned long long d = ldc(i); if (i < C) f(okey32((uint32_t)d), (int)(d >> 32)); }
    };
    uint32_t T; int above, ce;
    if (C <= RC * WARP) wave_bsearch32(rk, K, T, above, ce);
    else wave_radix<4>(each, hist, K, lane, T, above, ce);
    wave_emit([&](auto&& g) {
        #pragma unroll
        for (int j = 0; j < RC; ++j) if (j * WARP < C) g(j * WARP + lane < C, rk[j], rc[j]);
        for (int i0 = RC * WARP; i0 < C; i0 += WARP) {
            const int i = i0 + lane; const bool v = i < C;
            const unsigned long long d = ldc(i);
            g(v, okey32((uint32_t)d), (int)(d >> 32));
        }
    }, T, above, K, ks, lane, orow);
    return true;
}

// RC from the candidate count of the row (wave-uniform): fewer registers to compare in every search step
template <bool COHERENT>
__device__ __forceinline__ bool wave_select_row_auto(const unsigned long long* __restrict__ cand, const int* __restrict__ cnt,
                                                     int nsplit, int CAP, int M, int m, int ks, int nvalid, int K,
                                                     int* __restrict__ orow, uint32_t* hist, int lane) {
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    int c_l = cnt[(int64_t)min(lane, nsplit - 1) * M + m];
    if (lane >= nsplit) c_l = 0;
    int C = min(c_l, CAP);
    #pragma unroll
    for (int d = 32; d >= 1; d >>= 1) C += __shfl_xor(C, d);
    if (C <= 16 * WARP) return wave_select_row<16, COHERENT>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist, lane);
    if (C <= 24 * WARP) return wave_select_row<24, COHERENT>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist, lane);
    return wave_select_row<32, COHERENT>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist, lane);
}

template <int RC>
__global__ __launch_bounds__(256) void select_kernel(const unsigned long long* __restrict__ cand, const int* __restrict__ cnt, int nsplit, int CAP,
                                                     const int* __restrict__ ks_p, const int* __restrict__ ke_p, int M, int N, int K,
                                                     int* __restrict__ out, int* __restrict__ flag, int* __restrict__ fbc) {
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, m = blockIdx.x * 4 + w;
    if (m >= M) return;
    const int ks = max(ks_p[m], 0), nvalid = min(ke_p[m], N) - ks;
    if (!wave_select_row_auto<false>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, out + (int64_t)m * K, hist_all[w], lane))
        if (lane == 0) flag[atomicAdd(fbc, 1)] = m;
}

// ------------------------------------------------------------------------------------------------- exact recompute
// Exact v35wes logit of (row m, token tok) for the 32 tokens held by lanes l % 32 (all 64 lanes take part); result
// valid in lanes 0..31. Same operand layout, plain MFMA, scalar even/odd fma chains, permlane pairing (hset 0 half +
// hset 1 half) and kv_scale multiply as mqa_v35wes::kernel -> bit-identical.
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
__device__ __forceinline__ float recompute_wes(const QW& qw, int tok, const fp8* __restrict__ Kc, const float* __restrict__ kv_scale) {
    const int l = threadIdx.x % WARP, dhalf = l / 32;
    const int64_t kb = (int64_t)tok * HEAD_SIZE;
    const f8x32 b0 = pack(*reinterpret_cast<const f8x16*>(&Kc[kb + dhalf * 16]), *reinterpret_cast<const f8x16*>(&Kc[kb + (2 + dhalf) * 16]));
    const f8x32 b1 = pack(*reinterpret_cast<const f8x16*>(&Kc[kb + (4 + dhalf) * 16]), *reinterpret_cast<const f8x16*>(&Kc[kb + (6 + dhalf) * 16]));
    const float scale = kv_scale[tok];
    VecOut z = {};
    VecOut c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(qw.q0, b0, z, 0, 0, 0, 0, 0, 0);
    c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(qw.q1, b1, c, 0, 0, 0, 0, 0, 0);
    float ae = 0.f, ao = 0.f;
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        const float ye = __builtin_fmaxf(c[2 * v], 0.f), yo = __builtin_fmaxf(c[2 * v + 1], 0.f);
        if (v == 0) { ae = ye * qw.w2[0][0]; ao = yo * qw.w2[0][1]; }
        else { ae = __builtin_fmaf(ye, qw.w2[v][0], ae); ao = __builtin_fmaf(yo, qw.w2[v][1], ao); }
    }
    float p0 = ae + ao, p1 = p0;
    swap32_reduce2<2>(p0, p1);      // lanes 0-31: (own hset-0 half, hset-1 half of lane + 32)
    return (p0 + p1) * scale;
}

// exact select of one row from a dense fp32 scratch row srow[ks, ke) (one wave)
template <bool COHERENT>
__device__ __forceinline__ void wave_select_dense(const float* __restrict__ srow, int ks, int ke, int K, int* __restrict__ orow,
                                                  uint32_t* hist, int lane) {
    if (ke - ks <= K) { for (int i = lane; i < K; i += WARP) orow[i] = i < ke - ks ? i : -1; return; }
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    auto ld = [&](int c) -> uint32_t { return okey32(__builtin_bit_cast(uint32_t, srow[min(c, ke - 1)])); };
    auto each = [&](auto&& f) { for (int c0 = ks; c0 < ke; c0 += WARP) { const int c = c0 + lane; const uint32_t k = ld(c); if (c < ke) f(k, c); } };
    uint32_t T; int above, ce;
    wave_radix<4>(each, hist, K, lane, T, above, ce);
    wave_emit([&](auto&& g) {
        for (int c0 = ks; c0 < ke; c0 += WARP) { const int c = c0 + lane; const bool v = c < ke; const uint32_t k = ld(c); g(v, v ? k : 0u, c); }
    }, T, above, K, ks, lane, orow);
}

// ------------------------------------------------------------------------------------------------- fallback (step 0: 3 kernels)
// The first FBR flagged rows (in list order) are recomputed by every wave of the grid in parallel (fb_compute) and
// selected one wave per row (fb_select); further flagged rows fall back to one wave per row (fallback_kernel).
constexpr int FBR = 64;
__global__ __launch_bounds__(256) void fb_compute(const int* __restrict__ flag, const int* __restrict__ fbc, const int* __restrict__ ks_p, const int* __restrict__ ke_p,
                                                  int M, int N, float* __restrict__ scratch, const fp8* __restrict__ Q, const fp8* __restrict__ Kc,
                                                  const float* __restrict__ kv_scale, const float* __restrict__ W) {
    const int lane = threadIdx.x % WARP, gw = blockIdx.x * 4 + threadIdx.x / WARP, G = gridDim.x * 4;
    const int nf = min(*fbc, FBR);
    for (int ord = 0; ord < nf; ++ord) {
        const int m = flag[ord];
        const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
        const int nb = (ke - ks + 31) / 32;
        if (gw >= nb) continue;
        const QW qw = load_qw(m, Q, W);
        for (int b = gw; b < nb; b += G) {
            const int c0 = ks + b * 32, tok = min(c0 + lane % 32, ke - 1);
            const float v = recompute_wes(qw, tok, Kc, kv_scale);
            if (lane < 32 && c0 + lane < ke) scratch[(int64_t)ord * N + c0 + lane] = v;
        }
    }
}
__global__ __launch_bounds__(256) void fb_select(const int* __restrict__ flag, const int* __restrict__ fbc, const int* __restrict__ ks_p, const int* __restrict__ ke_p,
                                                 int M, int N, int K, int* __restrict__ out, const float* __restrict__ scratch, int* __restrict__ nfb) {
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, ord = blockIdx.x * 4 + w;
    if (ord >= min(*fbc, FBR)) return;
    const int m = flag[ord];
    if (nfb && lane == 0) atomicAdd(nfb, 1);
    const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
    wave_select_dense<false>(scratch + (int64_t)ord * N, ks, ke, K, out + (int64_t)m * K, hist_all[w], lane);
}
// beyond FBR flagged rows: one wave per row, own scratch row
__global__ __launch_bounds__(256) void fallback_kernel(const int* __restrict__ flag, const int* __restrict__ fbc, const int* __restrict__ ks_p,
                                                       const int* __restrict__ ke_p, int M, int N, int K, int* __restrict__ out, float* __restrict__ scratch,
                                                       const fp8* __restrict__ Q, const fp8* __restrict__ Kc, const float* __restrict__ kv_scale,
                                                       const float* __restrict__ W, int* __restrict__ nfb) {
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, gw = blockIdx.x * 4 + w;
    uint32_t* hist = hist_all[w];
    float* srow = scratch + (int64_t)(FBR + gw) * N;
    const int nf = *fbc;
    for (int ord = FBR + gw; ord < nf; ord += gridDim.x * 4) {
        const int m = flag[ord];
        if (nfb && lane == 0) atomicAdd(nfb, 1);
        const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
        const QW qw = load_qw(m, Q, W);
        for (int c0 = ks; c0 < ke; c0 += 32) {
            const int tok = min(c0 + lane % 32, ke - 1);
            const float v = recompute_wes(qw, tok, Kc, kv_scale);
            if (lane < 32 && c0 + lane < ke) srow[c0 + lane] = v;
        }
        __threadfence_block();
        wave_select_dense<true>(srow, ks, ke, K, out + (int64_t)m * K, hist, lane);
    }
}

// ------------------------------------------------------------------------------------------------- fallback (step 1+: 1 kernel)
// The first FBR flagged rows: every wave of the grid recomputes 32-token blocks of each row (strided over the grid)
// into scratch[ord][*]; a per-row completion counter done[ord] (acq_rel at agent scope - the slices of one row may be
// computed on different XCDs, whose L2s are not coherent) lets the wave that completes the row's last block run the
// exact select; that wave resets done[ord] = 0 for the next call (done[] is zeroed once at allocation). Flagged rows
// beyond FBR: one wave per row with its own scratch row FBR + wave (first 64 waves). *fbc is reset by the next call's sample pass.
__global__ __launch_bounds__(256) void fallback1_kernel(const int* __restrict__ flag, const int* __restrict__ fbc, int* __restrict__ done,
                                                        const int* __restrict__ ks_p, const int* __restrict__ ke_p, int M, int N, int K,
                                                        int* __restrict__ out, float* __restrict__ scratch, const fp8* __restrict__ Q,
                                                        const fp8* __restrict__ Kc, const float* __restrict__ kv_scale,
                                                        const float* __restrict__ W, int* __restrict__ nfb) {
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, gw = blockIdx.x * 4 + w, G = gridDim.x * 4;
    uint32_t* hist = hist_all[w];
    const int nf = *fbc;
    const int nfa = min(nf, FBR);
    for (int ord = 0; ord < nfa; ++ord) {
        const int m = flag[ord];
        const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
        const int nb = (ke - ks + 31) / 32;
        if (gw >= nb) continue;
        int mine = 0;
        {
            const QW qw = load_qw(m, Q, W);
            for (int b = gw; b < nb; b += G, ++mine) {
                const int c0 = ks + b * 32, tok = min(c0 + lane % 32, ke - 1);
                const float v = recompute_wes(qw, tok, Kc, kv_scale);
                if (lane < 32 && c0 + lane < ke) scratch[(int64_t)ord * N + c0 + lane] = v;
            }
        }
        int old = 0;
        if (lane == 0) old = __hip_atomic_fetch_add(&done[ord], mine, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        old = __shfl(old, 0);
        if (old + mine == nb) {                 // this wave completed the row: every slice is visible (acquire)
            if (nfb && lane == 0) atomicAdd(nfb, 1);
            wave_select_dense<true>(scratch + (int64_t)ord * N, ks, ke, K, out + (int64_t)m * K, hist, lane);
            if (lane == 0) done[ord] = 0;
        }
    }
    // rows beyond FBR (rare): the first FBR2 waves only, so the scratch stays (FBR + FBR2) rows
    constexpr int FBR2 = 64;
    if (gw >= FBR2) return;
    float* srow = scratch + (int64_t)(FBR + gw) * N;
    for (int ord = FBR + gw; ord < nf; ord += FBR2) {
        const int m = flag[ord];
        if (nfb && lane == 0) atomicAdd(nfb, 1);
        const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
        const QW qw = load_qw(m, Q, W);
        for (int c0 = ks; c0 < ke; c0 += 32) {
            const int tok = min(c0 + lane % 32, ke - 1);
            const float v = recompute_wes(qw, tok, Kc, kv_scale);
            if (lane < 32 && c0 + lane < ke) srow[c0 + lane] = v;
        }
        __threadfence_block();
        wave_select_dense<true>(srow, ks, ke, K, out + (int64_t)m * K, hist, lane);
    }
}

}  // namespace fmt
