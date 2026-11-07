// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused FP8 MQA logits + top-k (gfx950), steps 2-4: ONE scan kernel per (row block, split) item that samples its own
// key range, derives the threshold, emits candidates and (FUSE_SELECT) selects the block's top-k; PERSIST turns it into
// a persistent kernel (1 WG per CU claiming items heaviest first, in-kernel exact fallback). See DESIGN.md section 4.3.
//
// Item (bx, s) = row block bx (32 rows) x split s:
//   * split s covers the CONTIGUOUS stage range [s*per, (s+1)*per) of the block's [0, num_tiles) (keys
//     [tok_b, tok_e)); its masked part runs as one MASKED pipeline call, its clean part as one CLEAN call;
//   * sample pass: STRIDED dense pipeline (all stages masked) over the sample keys j*S in [tok_b, tok_e), fp32 samples
//     to a scratch region of the row; tau_s per row = wave_tau over those samples (same-wave read-after-write);
//   * EMIT pipelines (v35wes + EV4 emission) with tau_s; per (split, row): cnt = candidates seen, need =
//     min(K, nvalid_s). Exactness: every split targets the full K inside its own range, so the union of the splits'
//     candidates contains the row's top-K whenever cnt_s <= CAP and cnt_s >= need_s for every split (DESIGN.md 2);
//   * FUSE_SELECT: completion protocol (every wave vmcnt(0) -> barrier -> one acq_rel agent atomic on
//     block_done[bx] -> LDS broadcast -> acquire fence in every reading wave); the last split selects the block's
//     rows; failing rows go to the fallback list. Empty blocks: split 0 writes -1 rows. Zero-work splits still
//     write cnt / need and take part in the protocol.
// Geometry is v35wes's (8 waves x 4 rows, 2 x 32-token sub-tiles per stage, 2-stage LDS ring), PF = 2 (same speed as
// PF 4, 10 VGPRs less). Build flags: -fno-honor-nans -fno-slp-vectorize -mllvm -disable-machine-sink.
#pragma once
#include "fmt_common.cuh"
#include "fmt_select.cuh"

namespace fmt {

struct FusedArgs {
    const fp8* Q; const fp8* K; const float* kv_scale; const float* W; const int* ks; const int* ke;
    int M, N, SplitN, S, Ktop, CAP, NSB, rb;
    unsigned long long* smp;        // sample records (fp32 | idx << 32), interleaved pair segments of 2*NSB: rows
                                    // m (grid, split 0 only) or slot*32 + (m - m0wg) (persistent, per-WG slot)
    unsigned long long* cand;       // [SplitN][Mp][CAP] u64, interleaved pair segments (cand_index)
    int* cnt;                       // [SplitN][M] candidates seen
    float* tau;                     // [M] row-wide tau (written by split 0 of the block)
    int* tau_ready;                 // [rb] = epoch once split 0 published the block's tau (SplitN > 1)
    int epoch;                      // launch epoch (host-incremented, > 0)
    int* out;                       // [M][Ktop] (FUSE_SELECT)
    int* flag; int* fbc;            // fallback list (FUSE_SELECT appends) / count
    int* block_done;                // [rb] split-completion counters (FUSE_SELECT, SplitN > 1), self-resetting
    int* ctr;                       // PERSIST: [0] claim, [1] items-done barrier, [2] fallback claim, [3] exit
    float* fscratch;                // PERSIST: [G][N] per-WG fallback row
    int* nfb;                       // optional: fallback rows counter (stats)
    int dbg;                        // bit 0: tau = +inf (forces every non-empty row through the exact fallback)
#ifdef FMT_TS
    long long* ts;                  // debug probe: [bx][s][8] s_memrealtime at phase ends (wave 0)
#endif
};
#ifdef FMT_TS
#define FMT_TSP(k) do { if (A.ts && tid == 0) A.ts[((int64_t)bx * SplitN + s) * 8 + (k)] = (long long)__builtin_amdgcn_s_memrealtime(); } while (0)
#else
#define FMT_TSP(k) do {} while (0)
#endif

constexpr int FUSED_THREADS = 512;
// Spin-waits poll with RELAXED agent-scope loads and acquire once (fence after the WG barrier): an acquire load per
// iteration adds a buffer_inv sc1 (L2 invalidate) per poll, which slowed the producing WGs of a 576 x 250000 chunk
// (14 splits) from 352 + 251 us (sample + tau) to 175 + 66 us when removed.
#ifndef FMT_SPIN_SLEEP
#define FMT_SPIN_SLEEP 2
#endif
#ifndef FMT_FUSED_SEL_RC
#define FMT_FUSED_SEL_RC 48   // in-kernel select: candidates held in registers (rest through the radix path)
#endif

// In-kernel select of one row behind a call boundary: its register arrays then do not add to the scan's register
// allocation (inlined, the 256-VGPR kernel spilled). Called only after the pipelines have drained (vmcnt(0)), so the
// call's register saves cannot race an in-flight inline-asm load. (One call per row measured faster than one call for
// the wave's 4 rows: the bigger callee frame cost more than the 3 extra calls.)
__device__ __noinline__ bool fused_select_row(const unsigned long long* cand, const int* cnt, int nsplit, int CAP, int M,
                                              int m, int ks, int nvalid, int K, int* orow, uint32_t* hist) {
    return wave_select_row<FMT_FUSED_SEL_RC, false>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist,
                                                     threadIdx.x % WARP);
}

// One (row block, split) item: Q/W load -> (split 0) row-wide sample + tau -> EMIT -> counts -> (FUSE_SELECT)
// completion + select. Inlined into the grid kernels (steps 2-3); behind a call in the persistent kernel (step 4),
// where the claim loop's live state would otherwise push the 256-VGPR body into spills inside the pipelines.
template <bool FUSE_SELECT, bool PERSIST>
__device__ __forceinline__ void fused_item_body(const FusedArgs& A, const int bx_in, const int s_in, const int slot_in)
{
    // behind a call the arguments (and everything loaded through A) arrive in VGPRs: make the scalars uniform (SGPR)
    // again, the inline-asm loads take them as "s" operands. No-ops when inlined.
    const int bx = __builtin_amdgcn_readfirstlane(bx_in), s = __builtin_amdgcn_readfirstlane(s_in);
    const int slot = __builtin_amdgcn_readfirstlane(slot_in);
    constexpr int NUM_WARPS  = 8, SUB = 2, PF = 2;
    constexpr int BLOCK_M    = 4;
    constexpr int ROWS_WG    = BLOCK_M * NUM_WARPS;
    constexpr int TILE_N     = 32;
    constexpr int TILE_BYTES = TILE_N * HEAD_SIZE;
    constexpr int STAGE_TOK  = TILE_N * SUB;
    constexpr int STAGE_BYTES= TILE_BYTES * SUB;
    constexpr int CHUNKS     = STAGE_BYTES / 16;
    constexpr int NTHREADS   = NUM_WARPS * WARP;
    constexpr int STAGES     = 2;
    constexpr int NACC       = 16;
    using VecOutMFMA = __attribute__((__vector_size__(NACC * sizeof(float)))) float;
    using u32x2 = __attribute__((__vector_size__(8))) unsigned int;

    __shared__ __attribute__((aligned(16))) fp8   k_lds[STAGES][STAGE_BYTES];
    __shared__ __attribute__((aligned(16))) float sc_lds[STAGES][STAGE_TOK];
    __shared__ uint32_t hist_lds[NUM_WARPS][256];   // tau / select histograms: own LDS (no reuse of the K ring)
    __shared__ int bcast;
    constexpr int SC_WAVES = STAGE_TOK / WARP;
    static_assert(STAGE_TOK % WARP == 0 && SC_WAVES <= NUM_WARPS, "scale producer mapping");

    const int tid  = threadIdx.x;
    const int l    = tid % WARP;
    const int warp = __builtin_amdgcn_readfirstlane(tid / WARP);
    const int M = __builtin_amdgcn_readfirstlane(A.M), N = __builtin_amdgcn_readfirstlane(A.N);
    const int SplitN = __builtin_amdgcn_readfirstlane(A.SplitN), S = __builtin_amdgcn_readfirstlane(A.S);
    const int Ktop = __builtin_amdgcn_readfirstlane(A.Ktop), CAP = __builtin_amdgcn_readfirstlane(A.CAP);
    const int NSB = __builtin_amdgcn_readfirstlane(A.NSB);
    const int Mp = M + (M & 1);

    const int rw    = l / 16;
    const int col   = l % 16;
    const int nA    = col + 16 * (rw & 1);
    const int dhalf = rw / 2;
    const int jtok  = l % 32;
    const int hset  = l / 32;
    const int hmask = -hset;
    uint32_t* hist = hist_lds[warp];

    // ---- producer mapping (dense and strided) and B-operand LDS offsets: item independent ----
    constexpr int CPL = FMT_CDIV(CHUNKS, NTHREADS);
    static_assert(CHUNKS % NTHREADS == 0 && CPL == 1, "one 16-B K load per lane per stage");
    constexpr int ST_PER_STAGE = 2 * SUB;
    constexpr int OPS_STAGE    = CPL + ST_PER_STAGE;
    constexpr int W_PRO        = (PF - 1) * CPL;
    constexpr int W_STEADY     = ST_PER_STAGE + (PF - 1) * OPS_STAGE;
    const int pc_t = tid / 8, pc_p = tid % 8;
    const int pf_voff_d = tid * 16;                                       // dense: chunk c = tid
    const int pf_voff_s = pc_t * S * HEAD_SIZE + pc_p * 16;               // strided sample keys
    const int pf_lds_off = pc_t * HEAD_SIZE + ((pc_p ^ ((pc_t >> 1) & 7)) * 16);
    const int rd_base = nA * HEAD_SIZE;
    const int sw = (nA >> 1) & 7;
    const int rd0 = rd_base + (((0 + dhalf) ^ sw) * 16);
    const int rd1 = rd_base + (((2 + dhalf) ^ sw) * 16);
    const int rd2 = rd_base + (((4 + dhalf) ^ sw) * 16);
    const int rd3 = rd_base + (((6 + dhalf) ^ sw) * 16);
    const i32x4 k_v = raw_rsrc(A.K, N * HEAD_SIZE);
    const i32x4 s_v = raw_rsrc(A.kv_scale, N * 4);

    const int m0wg = bx * ROWS_WG;
    const int m0   = m0wg + warp * BLOCK_M;

    // ---- my rows: Q (A operand layout) and W to registers. Lane offsets from an opaque per-item thread id, so the
    // persistent loop cannot hoist them (they would stay live through the EMIT pipelines) ----
    int tq = tid;
    asm volatile("" : "+v"(tq));
    const int nA_q = (tq % 16) + 16 * ((tq % WARP / 16) & 1), dh_q = (tq % WARP) / 32;
    f8x32 q_reg[BLOCK_M][2];
    f32x2 w2[BLOCK_M][NACC / 2];
    #pragma unroll
    for (int r = 0; r < BLOCK_M; ++r) {
        const int m = m0 + r;
        if (m >= M) {
            #pragma unroll
            for (int v = 0; v < NACC / 2; ++v) w2[r][v] = f32x2{0.f, 0.f};
            q_reg[r][0] = q_reg[r][1] = f8x32{}; continue; }
        const int64_t qb = ((int64_t)m * NUM_HEADS + nA_q) * HEAD_SIZE;
        q_reg[r][0] = pack(*reinterpret_cast<const f8x16*>(&A.Q[qb + dh_q * 16]),
                           *reinterpret_cast<const f8x16*>(&A.Q[qb + (2 + dh_q) * 16]));
        q_reg[r][1] = pack(*reinterpret_cast<const f8x16*>(&A.Q[qb + (4 + dh_q) * 16]),
                           *reinterpret_cast<const f8x16*>(&A.Q[qb + (6 + dh_q) * 16]));
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const f32x4 wv = *reinterpret_cast<const f32x4*>(&A.W[(int64_t)m * NUM_HEADS + 8 * j + 4 * dh_q]);
            w2[r][2 * j]     = f32x2{wv[0], wv[1]};
            w2[r][2 * j + 1] = f32x2{wv[2], wv[3]};
        }
    }
    FMT_TSP(0);
    // ---- WG-wide tile range (uniform across waves) ----
    int lo = INT_MAX, hi = 0, max_ks = 0, min_ke = INT_MAX;
    for (int r = 0; r < ROWS_WG; ++r) {
        const int m = m0wg + r;
        if (m < M) { const int a = A.ks[m], b = A.ke[m]; lo = min(lo, a); hi = max(hi, b); max_ks = max(max_ks, a); min_ke = min(min_ke, b); }
    }
    lo = __builtin_amdgcn_readfirstlane(max(lo, 0)); hi = __builtin_amdgcn_readfirstlane(min(hi, N));
    max_ks = __builtin_amdgcn_readfirstlane(max_ks); min_ke = __builtin_amdgcn_readfirstlane(min(min_ke, N));
    const bool empty = (lo >= hi);
    if (empty) {     // every row of the block is empty: -1 rows (split 0) / zero counts; no completion protocol
        if constexpr (FUSE_SELECT) {
            if (s == 0)
                for (int r = 0; r < BLOCK_M; ++r)
                    if (m0 + r < M) for (int i = l; i < Ktop; i += WARP) A.out[(int64_t)(m0 + r) * Ktop + i] = -1;
        } else {
            if (l == 0)
                for (int r = 0; r < BLOCK_M; ++r)
                    if (m0 + r < M) A.cnt[(int64_t)s * M + m0 + r] = 0;
        }
        return;
    }
    lo = lo & ~(TILE_N - 1);
    const int num_tiles = FMT_CDIV(hi - lo, STAGE_TOK);
    const bool all_rows = (m0wg + ROWS_WG) <= M;
    int tc0 = min(FMT_CDIV(max(max_ks - lo, 0), STAGE_TOK), num_tiles);
    int tc1 = all_rows ? min(max((min_ke - lo) / STAGE_TOK, 0), num_tiles) : 0;
    tc1 = max(tc1, tc0);
    // ---- contiguous split: stages [sb, se), keys [tok_b, tok_e) ----
    const int per = FMT_CDIV(num_tiles, SplitN);
    const int sb = min(s * per, num_tiles), se = min(sb + per, num_tiles);
    const int tok_b = min(lo + sb * STAGE_TOK, N), tok_e = min(lo + se * STAGE_TOK, N);
    // masked index range of the split (index space [0,tc0) U [tc1,num_tiles) -> [0, n_masked)) and clean range
    const int mi_b = (sb < tc0) ? sb : (sb >= tc1 ? sb - tc1 + tc0 : tc0);
    const int mi_e = (se <= tc0) ? se : (se >= tc1 ? se - tc1 + tc0 : tc0);
    const int cl_b = max(sb, tc0), cl_e = min(se, tc1);
    // per-row windows are re-derived from scalar reloads of ks / ke right before each phase (no long-lived
    // per-row SGPR arrays: they overflowed into VGPR spill lanes)
    auto row_win = [&](int r, int& ksc, int& kec) -> bool {
        const int m = m0 + r;
        const bool ok = m < M;
        ksc = ok ? max(A.ks[m], 0) : 0; kec = ok ? min(A.ke[m], N) : 0;
        return ok;
    };
    // ---- operand / epilogue helpers (v35wes) ----
    struct Bop { f8x32 b0, b1; float scale; };
    auto read_lds = [&](Bop& B, auto stage_c, auto sub_c) {
        constexpr int stage = decltype(stage_c)::value, sub = decltype(sub_c)::value;
        constexpr int so = sub * TILE_BYTES;
        B.b0 = pack(*reinterpret_cast<const f8x16*>(&k_lds[stage][so + rd0]), *reinterpret_cast<const f8x16*>(&k_lds[stage][so + rd1]));
        B.b1 = pack(*reinterpret_cast<const f8x16*>(&k_lds[stage][so + rd2]), *reinterpret_cast<const f8x16*>(&k_lds[stage][so + rd3]));
        B.scale = sc_lds[stage][sub * TILE_N + jtok];
    };
    auto mfma_a = [&](int r, const Bop& B, VecOutMFMA& c) {
        VecOutMFMA z = {};
        c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(q_reg[r][0], B.b0, z, 0, 0, 0, 0, 0, 0);
    };
    auto mfma_b = [&](int r, const Bop& B, VecOutMFMA& c) {
        c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(q_reg[r][1], B.b1, c, 0, 0, 0, 0, 0, 0);
    };
    auto epi_half = [&](int r, const VecOutMFMA& vC, auto h_c, float& ae, float& ao) {
        constexpr int H_ = decltype(h_c)::value;
        #pragma unroll
        for (int v = 4 * H_; v < 4 * H_ + 4; ++v) {
            const float ye = __builtin_fmaxf(vC[2 * v], 0.f), yo = __builtin_fmaxf(vC[2 * v + 1], 0.f);
            if (v == 0) { ae = ye * w2[r][0][0]; ao = yo * w2[r][0][1]; }
            else { ae = __builtin_fmaf(ye, w2[r][v][0], ae); ao = __builtin_fmaf(yo, w2[r][v][1], ao); }
        }
    };

    // ---- ONE emission pipeline used by both passes (one masked + one clean instantiation in the kernel). Per-pass
    // parameters (set before each call, wave-uniform): key stride kvm (1 = EMIT, S = sample pass), output segment
    // base / capacity (pairs of 2*ocap records, row index row0 + r), per-row bounds lb/hb (masked) and tau tw.
    int cnt_row[BLOCK_M] = {0, 0, 0, 0};
    int p_lo[BLOCK_M / 2], p_hi[BLOCK_M / 2];     // per-pair (lane half = row) bounds and tau of the current pass
    float p_tau[BLOCK_M / 2];
    unsigned long long* ob_cur = A.cand;
    int64_t orow0 = 0;
    int ocap_cur = CAP, kvm = 1;
    auto run_pipeline = [&](auto masked_c, const int tb, const int te, const int lo_, const int tc0_, const int tc1_) {
        constexpr bool MASKED = decltype(masked_c)::value;
        if (tb >= te) return;
        auto tokbase = [&](int t) -> int {
            int st = t;
            if constexpr (MASKED) st = (t < tc0_) ? t : (t - tc0_ + tc1_);
            return lo_ + st * STAGE_TOK;
        };
        __amdgpu_buffer_rsrc_t o_rsrc[BLOCK_M / 2];
        #pragma unroll
        for (int r = 0; r < BLOCK_M; r += 2)
            o_rsrc[r / 2] = __builtin_amdgcn_make_buffer_rsrc((void*)(ob_cur + (orow0 + r) * ocap_cur), 0, 2 * ocap_cur * 8, FMT_BUF_RSRC_WORD3);
        const int* sel_lo = p_lo; const int* sel_hi = p_hi; const float* sel_tau = p_tau;
        auto pair_finish = [&](int r, float p0, float p1, float scale, int token_base, auto mk_c) {
            constexpr bool MK_ = decltype(mk_c)::value;
            const float res = (p0 + p1) * scale;
            const int pos = token_base + jtok;
            bool cand = res >= sel_tau[r / 2];
            if constexpr (MK_) cand = cand & (pos >= sel_lo[r / 2]) & (pos < sel_hi[r / 2]);
            const uint64_t msk = __ballot(cand);
            const unsigned mlo = (unsigned)msk, mhi = (unsigned)(msk >> 32);
            const unsigned mb = __builtin_amdgcn_mbcnt_hi(mhi, __builtin_amdgcn_mbcnt_lo(mlo, 0u));
            const int c0 = cnt_row[r], c1 = cnt_row[r + 1] - __builtin_popcount(mlo);
            const int base = c0 + ((c1 - c0) & hmask);
            const unsigned vt = ((mb + (unsigned)base) * 2u + (unsigned)hset) << 3;
            const unsigned voff = cand ? vt : 0x40800000u;
            cnt_row[r] += __builtin_popcount(mlo);
            cnt_row[r + 1] += __builtin_popcount(mhi);
            __builtin_amdgcn_raw_buffer_store_b64(u32x2{__builtin_bit_cast(unsigned, res), (unsigned)pos}, o_rsrc[r / 2], voff, 0, 0);
        };
        auto pair_store = [&](int r, float p0, float p1, float scale, int token_base, auto mk_c, auto nops_c) {
            swap32_reduce2<decltype(nops_c)::value>(p0, p1);
            pair_finish(r, p0, p1, scale, token_base, mk_c);
        };
        u32x4 pf_reg0, pf_reg1;
        float pf_sc0 = 0.f, pf_sc1 = 0.f;
        auto issue_loads = [&](int t, auto slot_c) {
            constexpr int sl = decltype(slot_c)::value;
            u32x4& pf = sl == 0 ? pf_reg0 : pf_reg1;
            float& ps = sl == 0 ? pf_sc0 : pf_sc1;
            const int token_base = __builtin_amdgcn_readfirstlane(tokbase(t));
            const int soff = token_base * kvm * HEAD_SIZE;
            const int pvo = pc_t * kvm * HEAD_SIZE + pc_p * 16;   // computed here (asm operand capture)
            // s_nop 4: the descriptor / soffset SGPRs may have just been written by a VALU (v_readlane of an SGPR
            // spill slot); VALU-SGPR-write -> VMEM-read needs 5 wait states and the hazard recognizer cannot see
            // inline asm (tools/sgprhaz.py)
            asm volatile("s_nop 4\n\tbuffer_load_dwordx4 %0, %1, %2, %3 offen" : "=v"(pf) : "v"(pvo), "s"(k_v), "s"(soff) : "memory");
            if (warp < SC_WAVES) {
                const int sc_soff = token_base * kvm * 4;
                const int sc_vo = tid * kvm * 4;
                const i32x4 sv = s_v;
                asm volatile("s_nop 4\n\tbuffer_load_dword %0, %1, %2, %3 offen" : "=v"(ps) : "v"(sc_vo), "s"(sv), "s"(sc_soff) : "memory");
            }
        };
        auto write_lds = [&](auto slot_c, auto stage_c, auto wait_c) {
            constexpr int sl = decltype(slot_c)::value, stage = decltype(stage_c)::value;
            wait_vmcnt<decltype(wait_c)::value>();
            const u32x4& pf = sl == 0 ? pf_reg0 : pf_reg1;
            const float ps = sl == 0 ? pf_sc0 : pf_sc1;
            *reinterpret_cast<u32x4*>(&k_lds[stage][pf_lds_off]) = pf;
            if (warp < SC_WAVES) sc_lds[stage][tid] = ps;
        };
        Bop Bo[2];
        Bo[1].scale = 0.f;
        VecOutMFMA acc[2];
        float pd_P0 = 0.f, pd_P1 = 0.f, pd_p2 = 0.f, pd_e3 = 0.f, pd_o3 = 0.f;
        int pd_tok = 0;
        auto dummy_store = [&]() { __builtin_amdgcn_raw_buffer_store_b32(0u, o_rsrc[0], 0x40800000u, 0, 0); };
#define FLUSH_PEND() { pair_finish(0, pd_P0, pd_P1, Bo[1].scale, pd_tok, MK{}); \
                   pair_store(2, pd_p2, pd_e3 + pd_o3, Bo[1].scale, pd_tok, MK{}, IC<2>{}); }
        issue_loads(tb, IC<0>{});
        issue_loads(min(tb + 1, te - 1), IC<1>{});
        write_lds(IC<0>{}, IC<0>{}, IC<W_PRO>{});
        issue_loads(min(tb + PF, te - 1), IC<0>{});
        __syncthreads();
        read_lds(Bo[0], IC<0>{}, IC<0>{});
        mfma_a(0, Bo[0], acc[0]); mfma_b(0, Bo[0], acc[0]);
#define SUBTILE(BC, BN, TOKBASE, RD_NEXT, MK, PEND, PS, PM, RD_LATE) \
{ \
    float e0, o0, e1, o1, e2, o2; \
    if constexpr ((PS) >= 0) __builtin_amdgcn_s_setprio(PS); \
    mfma_a(1, Bo[BC], acc[1]); \
    __builtin_amdgcn_sched_barrier(0); \
    float pd_p3 = 0.f; \
    if constexpr (PEND) { pd_p3 = pd_e3 + pd_o3; pair_finish(0, pd_P0, pd_P1, Bo[BN].scale, pd_tok, MK{}); } \
    else dummy_store(); \
    __builtin_amdgcn_sched_barrier(0); \
    epi_half(0, acc[0], IC<0>{}, e0, o0); \
    __builtin_amdgcn_sched_barrier(0); \
    mfma_b(1, Bo[BC], acc[1]); \
    __builtin_amdgcn_sched_barrier(0); \
    if constexpr (PEND) swap32_reduce2<0>(pd_p2, pd_p3); \
    __builtin_amdgcn_sched_barrier(0); \
    epi_half(0, acc[0], IC<1>{}, e0, o0); \
    __builtin_amdgcn_sched_barrier(0); \
    mfma_a(2, Bo[BC], acc[0]); \
    __builtin_amdgcn_sched_barrier(0); \
    float p0 = e0 + o0; \
    if constexpr (PEND) pair_finish(2, pd_p2, pd_p3, Bo[BN].scale, pd_tok, MK{}); else dummy_store(); \
    __builtin_amdgcn_sched_barrier(0); \
    epi_half(1, acc[1], IC<0>{}, e1, o1); \
    __builtin_amdgcn_sched_barrier(0); \
    mfma_b(2, Bo[BC], acc[0]); \
    __builtin_amdgcn_sched_barrier(0); \
    epi_half(1, acc[1], IC<1>{}, e1, o1); \
    __builtin_amdgcn_sched_barrier(0); \
    mfma_a(3, Bo[BC], acc[1]); \
    __builtin_amdgcn_sched_barrier(0); \
    if constexpr ((PM) >= 0) __builtin_amdgcn_s_setprio(PM); \
    RD_NEXT; \
    epi_half(2, acc[0], IC<0>{}, e2, o2); \
    __builtin_amdgcn_sched_barrier(0); \
    mfma_b(3, Bo[BC], acc[1]); \
    __builtin_amdgcn_sched_barrier(0); \
    RD_LATE; \
    epi_half(2, acc[0], IC<1>{}, e2, o2); \
    __builtin_amdgcn_sched_barrier(0); \
    mfma_a(0, Bo[BN], acc[0]); \
    __builtin_amdgcn_sched_barrier(0); \
    float p1 = e1 + o1; \
    pd_p2 = e2 + o2; \
    __builtin_amdgcn_sched_barrier(0); \
    epi_half(3, acc[1], IC<0>{}, pd_e3, pd_o3); \
    __builtin_amdgcn_sched_barrier(0); \
    mfma_b(0, Bo[BN], acc[0]); \
    __builtin_amdgcn_sched_barrier(0); \
    swap32_reduce2<0>(p0, p1); \
    pd_P0 = p0; pd_P1 = p1; \
    __builtin_amdgcn_sched_barrier(0); \
    epi_half(3, acc[1], IC<1>{}, pd_e3, pd_o3); \
    pd_tok = TOKBASE; \
    __builtin_amdgcn_sched_barrier(0); \
}
#define NEXT_STAGE_READ(NXT) read_lds(Bo[0], IC<NXT>{}, IC<0>{})
#define STAGE_STEP(CUR, NXT, WS, TT, MK, WC, P0)                                                          \
{                                                                                                     \
    const int stage_base = __builtin_amdgcn_readfirstlane(tokbase(TT));                              \
    write_lds(IC<WS>{}, IC<NXT>{}, IC<WC>{});                                                        \
    issue_loads(min((TT) + PF + 1, te - 1), IC<WS>{});                                               \
    SUBTILE(0, 1, stage_base, read_lds(Bo[1], IC<CUR>{}, IC<1>{}), MK, P0, 1, 0, __syncthreads())    \
    SUBTILE(1, 0, stage_base + TILE_N, NEXT_STAGE_READ(NXT), MK, true, 3, 2, (void)0)                \
}
        using MK = std::remove_cv_t<std::remove_reference_t<decltype(masked_c)>>;
        constexpr int W1 = (PF - 1) * CPL;
        STAGE_STEP(0, 1, 1, tb, MK, W1, false)
        if (tb + 1 >= te) { FLUSH_PEND(); return; }
        STAGE_STEP(1, 0, 0, tb + 1, MK, OPS_STAGE, true)
        #pragma unroll 1
        for (int t = tb + 2; t < te; t += 2) {
            STAGE_STEP(0, 1, 1, t, MK, W_STEADY, true)
            if (t + 1 >= te) break;
            STAGE_STEP(1, 0, 0, t + 1, MK, W_STEADY, true)
        }
        FLUSH_PEND();
#undef STAGE_STEP
#undef NEXT_STAGE_READ
#undef FLUSH_PEND
#undef SUBTILE
    };

    // ---- 1-2. ROW-WIDE tau: split 0 samples the block's whole key range (keys j*S as records) and publishes tau
    // per row; the other splits of the block wait for it (they are dispatched / claimed after split 0, so the wait
    // cannot deadlock). Per-split taus would make every split target the full K (SplitN x the candidates).
    if (s == 0) {
        const int tokr_e = min(lo + num_tiles * STAGE_TOK, N);
        const int jr_b = FMT_CDIV(lo, S), jr_e = FMT_CDIV(tokr_e, S);
        const int64_t srow0 = PERSIST ? (int64_t)slot * ROWS_WG + (m0 - m0wg) : (int64_t)m0;
        if (jr_e > jr_b) {
            #pragma unroll
            for (int r = 0; r < BLOCK_M; r += 2) {   // sample index window of my row of the pair
                int ksa, kea, ksb, keb;
                const bool oka = row_win(r, ksa, kea), okb = row_win(r + 1, ksb, keb);
                const int la = max(ksa > 0 ? FMT_CDIV(ksa, S) : 0, jr_b), ha = oka ? min(kea > 0 ? FMT_CDIV(kea, S) : 0, jr_e) : jr_b;
                const int lb = max(ksb > 0 ? FMT_CDIV(ksb, S) : 0, jr_b), hb = okb ? min(keb > 0 ? FMT_CDIV(keb, S) : 0, jr_e) : jr_b;
                p_lo[r / 2] = hset ? lb : la; p_hi[r / 2] = hset ? hb : ha; p_tau[r / 2] = -__builtin_huge_valf();
            }
            ob_cur = A.smp; orow0 = srow0; ocap_cur = NSB; kvm = S;
            const int lo_s = jr_b & ~(TILE_N - 1);
            const int nst = FMT_CDIV(jr_e - lo_s, STAGE_TOK);
            run_pipeline(std::true_type{}, 0, nst, lo_s, nst, nst);
        }
        wait_vmcnt<0>();
        FMT_TSP(1);
        #pragma unroll 1
        for (int r = 0; r < BLOCK_M; ++r) {     // same-wave read-after-write of its own records
            const int ns = cnt_row[r];
            cnt_row[r] = 0;
            float t = -__builtin_huge_valf();
            int ksc, kec;
            const bool ok = row_win(r, ksc, kec);
            if (ok && ns > 0) {
                const unsigned long long* rec = A.smp + (srow0 + (r & ~1)) * NSB + (r & 1);   // record i at rec[2i]
                t = wave_tau_ld([&](int i) -> float { return __builtin_bit_cast(float, (unsigned)rec[2 * i]); },
                                ns, kec - ksc, Ktop, hist, l);
            }
            if (A.dbg & 1) t = __builtin_huge_valf();
            t = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, t)));
            if (ok && l == 0) A.tau[m0 + r] = t;
            if (hset == (r & 1)) p_tau[r / 2] = t;   // lanes of row r in its pair
        }
        if (SplitN > 1) {
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();
            if (tid == 0) __hip_atomic_store(&A.tau_ready[bx], A.epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        FMT_TSP(2);
    } else {
        if (tid == 0)
            while (__hip_atomic_load(&A.tau_ready[bx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != A.epoch) __builtin_amdgcn_s_sleep(FMT_SPIN_SLEEP);
        FMT_TSP(2);
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        #pragma unroll
        for (int r = 0; r < BLOCK_M; ++r) {
            const float t = (m0 + r < M) ? A.tau[m0 + r] : 0.f;
            if (hset == (r & 1)) p_tau[r / 2] = t;
        }
    }
    __syncthreads();
    // ---- 3. EMIT over the split's masked and clean stages ----
    #pragma unroll
    for (int r = 0; r < BLOCK_M; r += 2) {        // key window inside the split (rows >= M: empty)
        int ksa, kea, ksb, keb;
        const bool oka = row_win(r, ksa, kea), okb = row_win(r + 1, ksb, keb);
        const int la = max(ksa, tok_b), ha = oka ? min(kea, tok_e) : 0;
        const int lb = max(ksb, tok_b), hb = okb ? min(keb, tok_e) : 0;
        p_lo[r / 2] = hset ? lb : la; p_hi[r / 2] = hset ? hb : ha;
    }
    ob_cur = A.cand; orow0 = (int64_t)s * Mp + m0; ocap_cur = CAP; kvm = 1;
    run_pipeline(std::true_type{}, mi_b, mi_e, lo, tc0, tc1);
    wait_vmcnt<0>();
    run_pipeline(std::false_type{}, cl_b, cl_e, lo, tc0, tc1);
    wait_vmcnt<0>();                         // load-bearing: candidate stores complete (memory model)
    FMT_TSP(3);
    if (l == 0) {
        #pragma unroll
        for (int r = 0; r < BLOCK_M; ++r) {
            const int m = m0 + r;
            if (m < M) A.cnt[(int64_t)s * M + m] = cnt_row[r];
        }
    }
    if constexpr (FUSE_SELECT) {
        // ---- 4. completion: the last split of the block selects its rows ----
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        bool last = true;
        if (SplitN > 1) {
            if (tid == 0) {
                const int old = __hip_atomic_fetch_add(&A.block_done[bx], 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
                const int lst = (old == SplitN - 1);
                if (lst) A.block_done[bx] = 0;          // self-reset for the next launch
                bcast = lst;
            }
            __syncthreads();
            last = bcast != 0;
            if (last) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        }
#ifdef FMT_P4_NO_SELECT
        if (false) {
#else
        if (last) {
#endif
            #pragma unroll 1
            for (int r = 0; r < BLOCK_M; ++r) {
                const int m = m0 + r;
                if (m >= M) break;
                const int ksc = max(A.ks[m], 0), nvalid = min(A.ke[m], N) - ksc;
                const bool ok = fused_select_row(A.cand, A.cnt, SplitN, CAP, M, m, ksc, nvalid, Ktop,
                                                 A.out + (int64_t)m * Ktop, hist);
                if (!ok && l == 0) A.flag[atomicAdd(A.fbc, 1)] = m;
            }
        }
        __syncthreads();                     // histograms / bcast free before the next item
    }
    FMT_TSP(4);
}

template <bool FUSE_SELECT>
__device__ __noinline__ void fused_item_call(const FusedArgs& A, const int bx, const int s, const int slot) {
    fused_item_body<FUSE_SELECT, true>(A, bx, s, slot);
}

template <bool FUSE_SELECT, bool PERSIST, bool REVERSE_ROWS>
__global__ __launch_bounds__(FUSED_THREADS)
void fused_kernel(FusedArgs A)
{
    constexpr int NUM_WARPS = 8, ROWS_WG = 32;
    __shared__ uint32_t hist_k[NUM_WARPS][256];
    __shared__ int bcast;
    const int tid  = threadIdx.x;
    const int l    = tid % WARP;
    const int warp = __builtin_amdgcn_readfirstlane(tid / WARP);
    const int M = A.M, N = A.N, SplitN = A.SplitN, Ktop = A.Ktop;
    uint32_t* hist = hist_k[warp];
    if constexpr (!PERSIST) {
        const int bx = REVERSE_ROWS ? (gridDim.x - 1 - blockIdx.x) : blockIdx.x;
        if constexpr (!FUSE_SELECT) {
            if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) *A.fbc = 0;   // the separate select appends later
        }
        if (bx * ROWS_WG >= M) return;
        fused_item_body<FUSE_SELECT, false>(A, bx, blockIdx.y, 0);
    } else {
        // ---- persistent: claim (row block, split) items from a global counter ----
        const int G = gridDim.x, wg = blockIdx.x;
        const int n_items = A.rb * SplitN;
        #pragma unroll 1
        while (true) {
            if (tid == 0) bcast = __hip_atomic_fetch_add(&A.ctr[0], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __syncthreads();
            const int item = __builtin_amdgcn_readfirstlane(bcast);
            __syncthreads();
            if (item >= n_items) break;
            // split-major order: every block's split 0 (which samples the whole row and publishes tau) is claimed
            // before any split 1, so the other splits rarely wait; heaviest (last) row blocks first inside a split.
            fused_item_call<FUSE_SELECT>(A, A.rb - 1 - item % A.rb, item / A.rb, wg);
        }
        // ---- every item done (global barrier: all G WGs are resident, 1 per CU) -> drain the fallback list ----
        if (tid == 0) {
            __hip_atomic_fetch_add(&A.ctr[1], 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            while (__hip_atomic_load(&A.ctr[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < G) __builtin_amdgcn_s_sleep(FMT_SPIN_SLEEP);
        }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#ifndef FMT_P4_NO_DRAIN
        const int nf = *A.fbc;
        float* frow = A.fscratch + (int64_t)wg * N;
        #pragma unroll 1
        while (true) {
            if (tid == 0) bcast = __hip_atomic_fetch_add(&A.ctr[2], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __syncthreads();
            const int f = __builtin_amdgcn_readfirstlane(bcast);
            __syncthreads();
            if (f >= nf) break;
            const int m = A.flag[f];
            const int ksc = max(A.ks[m], 0), kec = min(A.ke[m], N);
            const QW qw = load_qw(m, A.Q, A.W);
            for (int c0 = ksc + warp * 32; c0 < kec; c0 += NUM_WARPS * 32) {
                const int tok = min(c0 + l % 32, kec - 1);
                const float v = recompute_wes(qw, tok, A.K, A.kv_scale);
                if (l < 32 && c0 + l < kec) frow[c0 + l] = v;
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();                     // same-CU visibility of every wave's slice
            if (warp == 0) {
                if (A.nfb && l == 0) atomicAdd(A.nfb, 1);
                wave_select_dense<false>(frow, ksc, kec, Ktop, A.out + (int64_t)m * Ktop, hist, l);
            }
            __syncthreads();
        }
#endif
        // ---- the last WG out resets the counters for the next launch ----
        if (tid == 0) {
            const int old = __hip_atomic_fetch_add(&A.ctr[3], 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            if (old == G - 1) { A.ctr[0] = 0; A.ctr[1] = 0; A.ctr[2] = 0; A.ctr[3] = 0; *A.fbc = 0; }
        }
    }
}

}  // namespace fmt
