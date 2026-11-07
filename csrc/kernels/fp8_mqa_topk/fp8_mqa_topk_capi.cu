// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused FP8 MQA logits + top-k (gfx950): extern "C" launchers for the ctypes dev harness
// (op_tests/fp8_mqa_topk). Every function returns hipGetLastError() (0 = ok) or a negative argument error.
#include "fmt_scan.cuh"
#include "fmt_select.cuh"

#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <queue>
#include <tuple>
#include <vector>

namespace {
using namespace fmt;

// Makespan model of a scan launch (copy of the v35wes launcher model): WG cost = GA + GBM * masked stages +
// GBC * clean stages (us, fit on MI355X), WGs dispatched in blockIdx order round-robin over 8 XCDs, greedy onto
// the CUs of each XCD (1 WG per CU). Assumes the single-request causal layout of a prefill chunk.
constexpr double GA = 9.87, GBM = 2.962, GBC = 1.975;
double model_makespan(int M, int N, int split, int rev, int cus_per_xcd) {
    const int rb = FMT_CDIV(M, 32);
    std::vector<std::priority_queue<double, std::vector<double>, std::greater<double>>> xcd(8);
    for (auto& q : xcd) for (int i = 0; i < cus_per_xcd; ++i) q.push(0.0);
    double mk = 0.0; int idx = 0;
    for (int y = 0; y < split; ++y)
        for (int x = 0; x < rb; ++x, ++idx) {
            const int bx = rev ? rb - 1 - x : x;
            const int r0 = bx * 32, r1 = std::min(M, r0 + 32);
            auto ke = [&](int m) { return std::min((4 * N - M + m + 1) / 4, N); };
            const int hi = ke(r1 - 1), mn = ke(r0);
            const int n = FMT_CDIV(hi, 64);
            const int tc1 = (r0 + 32 <= M) ? std::min(mn / 64, n) : 0;
            auto share = [&](int n_) { const int per = FMT_CDIV(n_, split); const int b = y * per; return std::max(0, std::min(n_, b + per) - b); };
            auto& q = xcd[idx % 8];
            const double t = q.top() + GA + GBM * share(n - tc1) + GBC * share(tc1);
            q.pop(); q.push(t); mk = std::max(mk, t);
        }
    return mk;
}
void choose_grid(int M, int N, int total_cu, int& split, int& rev) {
    static std::mutex mu;
    static std::map<std::tuple<int, int, int>, std::pair<int, int>> cache;
    std::lock_guard<std::mutex> g(mu);
    const auto key = std::make_tuple(M, N, total_cu);
    auto it = cache.find(key);
    if (it != cache.end()) { split = it->second.first; rev = it->second.second; return; }
    const int cus_per_xcd = std::max(1, total_cu / 8);
    double best = 1e300; split = 1; rev = 1;
    for (int s = 1; s <= 16; ++s) for (int r = 0; r < 2; ++r) {
        const double t = model_makespan(M, N, s, r, cus_per_xcd);
        if (t < best * 0.999) { best = t; split = s; rev = r; }
    }
    cache[key] = {split, rev};
}
inline void resolve_grid(int M, int N, int& split, int& rev) {
    if (split <= 0) { int s, r; choose_grid(M, N, 256, s, r); split = s; if (rev < 0) rev = r; }
    if (rev < 0) rev = (N > 2048);
}
using fp8p = const fmt::fp8*;
}  // namespace

extern "C" {

int fmt_choice(int M, int N, int* split, int* rev) {
    int s = 0, r = -1; resolve_grid(M, N, s, r); *split = s; *rev = r; return 0;
}

// DENSE scan (== v35wes fp32 logits, valid window only, no -inf fill). out row stride = out_cols.
int fmt_dense(const void* q, const void* k, const float* sc, const float* w, const int* ks, const int* ke, float* out,
              int M, int N, int out_cols, int split, int rev, hipStream_t st) {
    if (M <= 0 || N <= 0) return 0;
    resolve_grid(M, N, split, rev);
    dim3 grid(FMT_CDIV(M, 32), split);
    if (rev) scan_kernel<8, 2, 4, MODE_DENSE, false, true><<<grid, 512, 0, st>>>((fp8p)q, (fp8p)k, sc, w, ks, ke, M, N, split, out, out_cols, 1, nullptr, nullptr, nullptr, 0);
    else     scan_kernel<8, 2, 4, MODE_DENSE, false, false><<<grid, 512, 0, st>>>((fp8p)q, (fp8p)k, sc, w, ks, ke, M, N, split, out, out_cols, 1, nullptr, nullptr, nullptr, 0);
    return (int)hipGetLastError();
}

// tau from fp32 sample logits ls[M, Ns] (sample j = key j*S); resets *fbc.
int fmt_tau(const float* ls, int Ns, int S, const int* ks, const int* ke, int M, int N, int K, int cap, float* tau, int* fbc, hipStream_t st) {
    tau_kernel<<<FMT_CDIV(M, 4), 256, 0, st>>>(ls, Ns, S, ks, ke, M, N, K, cap, tau, fbc);
    return (int)hipGetLastError();
}

// EMIT scan: candidates >= tau[m] into cand[(s*M + m)*(cap+1) + i], counts cnt[s*M + m]. split_used <- SplitN.
int fmt_emit(const void* q, const void* k, const float* sc, const float* w, const int* ks, const int* ke, const float* tau,
             unsigned long long* cand, int* cnt, int cap, int M, int N, int split, int rev, int* split_used, hipStream_t st) {
    if (M <= 0 || N <= 0) return 0;
    resolve_grid(M, N, split, rev);
    if (split_used) *split_used = split;
    dim3 grid(FMT_CDIV(M, 32), split);
    if (rev) scan_kernel<8, 2, 4, MODE_EMIT, false, true><<<grid, 512, 0, st>>>((fp8p)q, (fp8p)k, sc, w, ks, ke, M, N, split, nullptr, 0, 1, tau, cand, cnt, cap);
    else     scan_kernel<8, 2, 4, MODE_EMIT, false, false><<<grid, 512, 0, st>>>((fp8p)q, (fp8p)k, sc, w, ks, ke, M, N, split, nullptr, 0, 1, tau, cand, cnt, cap);
    return (int)hipGetLastError();
}

// step-0 select: select_kernel + fb_compute + fb_select + fallback_kernel (4 launches).
// scratch: float[(FBR + 64) * N]; flag: int[M]; fbc reset by fmt_tau of the same call; nfb (optional) counts fallback rows.
int fmt_select0(const unsigned long long* cand, const int* cnt, int nsplit, int cap, const int* ks, const int* ke, int M, int N, int K,
                int* out, int* flag, int* fbc, float* scratch, const void* q, const void* kc, const float* sc, const float* w,
                int* nfb, hipStream_t st) {
    if (nsplit > 16 || nsplit < 1) return -5;
    select_kernel<32><<<FMT_CDIV(M, 4), 256, 0, st>>>(cand, cnt, nsplit, cap, ks, ke, M, N, K, out, flag, fbc);
    fb_compute<<<64, 256, 0, st>>>(flag, fbc, ks, ke, M, N, scratch, (fp8p)q, (fp8p)kc, sc, w);
    fb_select<<<FBR / 4, 256, 0, st>>>(flag, fbc, ks, ke, M, N, K, out, scratch, nfb);
    fallback_kernel<<<16, 256, 0, st>>>(flag, fbc, ks, ke, M, N, K, out, scratch, (fp8p)q, (fp8p)kc, sc, w, nfb);
    return (int)hipGetLastError();
}

}  // extern "C"
