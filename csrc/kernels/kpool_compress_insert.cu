// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// kpool cache writer for the GLM-5.3-Flash sparse-attention indexer: the
// prefill insert, plus the tail-ring seeding of the same step when the tail
// cache is given. Byte-identical to the vLLM Triton path (torch gather +
// `_kpool_softmax_rotate_write_cache_kernel`, `kpool_seed_tail_cache`).
//
// Semantics: every token t with slot_mapping[t] >= 0 and t >= POOL-1 closes
// the pool made of tokens [t-POOL+1, t]. For that pool, per head dim d:
//   score_s = f32(gate[t-POOL+1+s][d]) + ape[s][d]
//   prob_s  = exp(score_s - max_s score_s)          (exp = exp2(x * log2e))
//   x       = (sum_s k_s * prob_s) / (sum_s prob_s)   -> bf16 -> f32
//             (k_s * prob_s rounded, then added: Triton >= 3.8 does not
//             contract this into an FMA; -DKPOOL_ACC_FMA restores the FMA
//             that Triton 3.7 emitted)
//   x       = Hadamard128(x) * (1/sqrt(128))          -> bf16 -> f32
//   absmax  = max(max_d |x_d|, 1e-4)
//   scale   = round_scale ? exp2(ceil(log2(absmax/448))) : absmax/448
//   q       = clamp(x / scale, -448, 448) -> fp8 e4m3fn
// q (128 B, the MFMA 16x16 preshuffled layout of indexer_k_quant_and_cache
// when page_size > 1) and scale (fp32) go to kv_cache at slot_mapping[t]:
// kv_cache is [num_blocks, page_size, HEAD_DIM + 4] uint8, each page holding
// page_size * 128 B of keys followed by page_size fp32 scales.
//
// Mapping: one window of POOL consecutive tokens per group of LANES lanes,
// one wave (64 lanes) per workgroup. LANES = 64 (1 pool / wave, 2 dims per
// lane), 32 (2 pools / wave) or 16 (4 pools / wave, 16 B loads); picked from
// the token count unless the caller forces it. K / gate are read in place
// (no [n, POOL, 128] gather) and only pools that really close are compressed.
// Tail seeding (kpool_seed_tail_cache): with tail_kv_cache / tail_slot_mapping
// a token among its request's last POOL tokens (the token POOL ahead has
// another tail block or is past the batch) also copies its raw K / gate row
// into tail[blk, {0, 1}, slot % ring], from the registers the window already
// holds, so the next decode step can close that pool.
//
// This translation unit is torch-free (AITER_NO_TORCH_TYPES): everything
// crosses the ABI as aiter_tensor_t.

#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "kpool_compress_insert.h"

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

// global-address-space pointer (explicit, so pointers that went through an
// asm pin still produce global_* instead of flat_* accesses)
template <typename T> using gptr = T __attribute__((address_space(1)))*;
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
typedef float f32x2 __attribute__((ext_vector_type(2)));

constexpr int HEAD_DIM = 128;
constexpr float FP8_MAX = 448.0f;
// Triton: 1.0 / FP8_MAX is a Python double folded into an fp32 constant.
constexpr float FP8_MAX_INV = (float)(1.0 / 448.0);
// Triton: x * 0.08838834764831845 (python float -> fp32 constant).
constexpr float HAD_SCALE = (float)0.08838834764831845;
// Triton AMD lowers tl.exp(x) to exp2(x * log2e) with log2e as fp32.
constexpr float LOG2E = (float)1.4426950408889634;

__device__ __forceinline__ float bf16lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float bf16at(const uint32_t* w, int d) {
  return (d & 1) ? bf16hi(w[d >> 1]) : bf16lo(w[d >> 1]);
}
// Round-to-nearest-even to bf16, returned as fp32 (finite inputs).
__device__ __forceinline__ float round_bf16(float x) {
  uint32_t u = __float_as_uint(x);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return __uint_as_float(u & 0xFFFF0000u);
}
// The same for a pair: one v_cvt_pk_bf16_f32 (RNE on gfx950, denormals
// preserved) + 2 unpack ops instead of 2 x 4 integer ops. Bit-identical to
// round_bf16 for every finite input (checked byte for byte against vLLM).
#ifndef KPOOL_NO_CVT_BF16
typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ void round_bf16x2(float& a, float& b) {
  const f32x2 v = {a, b};
  const uint32_t w = __builtin_bit_cast(uint32_t, __builtin_convertvector(v, bf16x2_t));
  a = __uint_as_float(w << 16);
  b = __uint_as_float(w & 0xFFFF0000u);
}
#else
__device__ __forceinline__ void round_bf16x2(float& a, float& b) {
  a = round_bf16(a);
  b = round_bf16(b);
}
#endif
__device__ __forceinline__ float fast_exp2(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ float fast_log2(float x) { return __builtin_amdgcn_logf(x); }

// Lane exchange: v from lane (lane ^ MASK), VALU only (no LDS traffic):
//   1, 2  DPP quad_perm
//   4     DPP row_shl:4 / row_shr:4 + select on lane bit 2
//   8     DPP row_ror:8 (rotate by 8 inside a 16-lane row == xor 8)
//   16    v_permlane16_swap (gfx950)
//   32    v_permlane32_swap (gfx950)
// -DKPOOL_LDS_XCHG falls back to ds_swizzle / ds_bpermute for 4..32; selected
// automatically for every device pass that is not gfx950 (permlane*_swap
// exist on gfx950 only).
#if !defined(KPOOL_LDS_XCHG) && !defined(__gfx950__)
#define KPOOL_LDS_XCHG
#endif
template <int MASK>
__device__ __forceinline__ float xor_lane(float v, int lane) {
  int iv = __float_as_int(v);
  int r;
  if constexpr (MASK == 1) {
    r = __builtin_amdgcn_update_dpp(0, iv, 0xB1, 0xF, 0xF, true);  // quad_perm(1,0,3,2)
  } else if constexpr (MASK == 2) {
    r = __builtin_amdgcn_update_dpp(0, iv, 0x4E, 0xF, 0xF, true);  // quad_perm(2,3,0,1)
#if !defined(KPOOL_LDS_XCHG)
  } else if constexpr (MASK == 4) {
    const int up = __builtin_amdgcn_update_dpp(iv, iv, 0x104, 0xF, 0xF, false);  // row_shl:4
    const int dn = __builtin_amdgcn_update_dpp(iv, iv, 0x114, 0xF, 0xF, false);  // row_shr:4
    r = (lane & 4) ? dn : up;
  } else if constexpr (MASK == 8) {
    r = __builtin_amdgcn_update_dpp(iv, iv, 0x128, 0xF, 0xF, false);  // row_ror:8
  } else if constexpr (MASK == 16) {
    auto p = __builtin_amdgcn_permlane16_swap((unsigned)iv, (unsigned)iv, false, false);
    r = (lane & 16) ? (int)p[0] : (int)p[1];
  } else if constexpr (MASK == 32) {
    auto p = __builtin_amdgcn_permlane32_swap((unsigned)iv, (unsigned)iv, false, false);
    r = (lane & 32) ? (int)p[0] : (int)p[1];
#else
  } else if constexpr (MASK < 32) {
    r = __builtin_amdgcn_ds_swizzle(iv, (MASK << 10) | 0x1F);  // bitmode xor MASK
  } else {
    r = __shfl_xor(iv, MASK, 64);
#endif
  }
  return __int_as_float(r);
}

// v_permlane{16,32}_swap with both operands = v: returns {lo, hi} where, for
// MASK 16, lo = v of the lower half of each 32-lane pair replicated to both
// halves and hi = v of the upper half replicated (MASK 32: halves of the wave).
// So in every lane {lo, hi} = {own, partner} below the mask bit and
// {partner, own} above it: a stage can use both without keeping its own v.
#if !defined(KPOOL_LDS_XCHG)
template <int MASK>
__device__ __forceinline__ void swap_pair(float v, float& lo, float& hi) {
  static_assert(MASK == 16 || MASK == 32, "permlane swap stages");
  const unsigned iv = (unsigned)__float_as_int(v);
  if constexpr (MASK == 16) {
    auto p = __builtin_amdgcn_permlane16_swap(iv, iv, false, false);
    lo = __int_as_float((int)p[0]);
    hi = __int_as_float((int)p[1]);
  } else {
    auto p = __builtin_amdgcn_permlane32_swap(iv, iv, false, false);
    lo = __int_as_float((int)p[0]);
    hi = __int_as_float((int)p[1]);
  }
}
#endif

// One cross-lane Hadamard stage: pair (lane, lane ^ MASK); the lower lane
// keeps a + b, the upper one a - b (same op order as the Triton butterfly).
template <int MASK, int DPL>
__device__ __forceinline__ void had_stage_xlane(float (&x)[DPL], int sub) {
  const bool upper = (sub & MASK) != 0;
#pragma unroll
  for (int d = 0; d < DPL; ++d) {
#if !defined(KPOOL_LDS_XCHG)
    if constexpr (MASK >= 16) {
      // lower lane: lo = x, hi = o -> x + o; upper lane: lo = o, hi = x -> o - x
      float lo, hi;
      swap_pair<MASK>(x[d], lo, hi);
      x[d] = upper ? (lo - hi) : (lo + hi);
      continue;
    }
#endif
    float o = xor_lane<MASK>(x[d], sub);
    x[d] = upper ? (o - x[d]) : (x[d] + o);
  }
}

// max of two non-negative finite floats == unsigned max of their bit patterns;
// skips the IEEE-mode canonicalization the compiler puts in front of v_max_f32
__device__ __forceinline__ float max_nonneg(float a, float b) {
  return __uint_as_float(max(__float_as_uint(a), __float_as_uint(b)));
}
// absmax stage over the pair (lane, lane ^ MASK) for non-negative values
template <int MASK>
__device__ __forceinline__ float absmax_xlane(float am, int sub) {
#if !defined(KPOOL_LDS_XCHG)
  if constexpr (MASK >= 16) {
    float lo, hi;
    swap_pair<MASK>(am, lo, hi);
    return max_nonneg(lo, hi);
  }
#endif
  return max_nonneg(am, xor_lane<MASK>(am, sub));
}

template <int DPL>
__device__ __forceinline__ void had_stages_inlane(float (&x)[DPL]) {
#pragma unroll
  for (int s = 1; s < DPL; s <<= 1) {
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
      if ((i & s) == 0) {
        float a = x[i], b = x[i + s];
        x[i] = a + b;
        x[i + s] = a - b;
      }
    }
  }
}

// DPL consecutive bf16 of one row into DPL/2 dwords. Rows are read exactly
// once, so the vector loads are non-temporal.
template <int DPL, bool VEC>
__device__ __forceinline__ void load_row(const uint16_t* __restrict__ p, uint32_t (&w)[DPL / 2]) {
  if constexpr (VEC && DPL >= 8) {
#pragma unroll
    for (int c = 0; c < DPL / 8; ++c) {
      u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p) + c);
      w[c * 4 + 0] = v[0];
      w[c * 4 + 1] = v[1];
      w[c * 4 + 2] = v[2];
      w[c * 4 + 3] = v[3];
    }
  } else if constexpr (VEC && DPL == 4) {
    u32x2 v = __builtin_nontemporal_load(reinterpret_cast<const u32x2*>(p));
    w[0] = v[0];
    w[1] = v[1];
  } else if constexpr (VEC && DPL == 2) {
    w[0] = __builtin_nontemporal_load(reinterpret_cast<const uint32_t*>(p));
  } else {
#pragma unroll
    for (int i = 0; i < DPL / 2; ++i) {
      w[i] = (uint32_t)p[2 * i] | ((uint32_t)p[2 * i + 1] << 16);
    }
  }
}

template <int POOL, int DPL, bool VEC>
__device__ __forceinline__ void load_rows(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ gate, int64_t first,
    int64_t k_stride, int64_t gate_stride, int d0,
    uint32_t (&gw)[POOL][DPL / 2], uint32_t (&kw)[POOL][DPL / 2]) {
#pragma unroll
  for (int s = 0; s < POOL; ++s) {
    load_row<DPL, VEC>(gate + (first + s) * gate_stride + d0, gw[s]);
    load_row<DPL, VEC>(k + (first + s) * k_stride + d0, kw[s]);
  }
}

// Pool math for one pool spread over LANES lanes; this lane holds dims
// [sub*DPL, sub*DPL + DPL) of every row. Same instructions and op order as
// the Triton kernel (bit-exact).
template <int POOL, int LANES>
__device__ __forceinline__ void compress_pool(
    const uint32_t (&gw)[POOL][HEAD_DIM / LANES / 2],
    const uint32_t (&kw)[POOL][HEAD_DIM / LANES / 2],
    const float (&ape)[POOL][HEAD_DIM / LANES],
    gptr<uint8_t> pbase,          // page of the destination slot (cache + page * page_bytes)
    int tok,                      // slot inside the page
    uint32_t scale_off,           // byte offset of the fp32 scale inside the page: page_size * 128 + tok * 4
    bool preshuffle,
    bool round_scale,
    int sub) {
  constexpr int DPL = HEAD_DIM / LANES;
  const int d0 = sub * DPL;

  // --- pass 1: per-dim max of score = gate + ape ---
  float mx[DPL];
#pragma unroll
  for (int d = 0; d < DPL; ++d) mx[d] = -INFINITY;
#pragma unroll
  for (int s = 0; s < POOL; ++s) {
#pragma unroll
    for (int d = 0; d < DPL; ++d) mx[d] = fmaxf(mx[d], bf16at(gw[s], d) + ape[s][d]);
  }

  // --- pass 2: softmax-weighted sum of k (denom += p; acc += k * p) ---
  // Triton >= 3.8 (vLLM image) keeps `acc += k * prob` as a rounded multiply
  // followed by an add; Triton 3.7 contracted it into an FMA. The two differ
  // in ~11 % of the fp32 sums, which flips the bf16 rounding of x in ~1 per
  // 10^5 elements. Default: match Triton >= 3.8. -DKPOOL_ACC_FMA: Triton 3.7.
  float acc[DPL], den[DPL];
#pragma unroll
  for (int d = 0; d < DPL; ++d) { acc[d] = 0.0f; den[d] = 0.0f; }
#pragma unroll
  for (int s = 0; s < POOL; ++s) {
#pragma unroll
    for (int d = 0; d < DPL; ++d) {
      float score = bf16at(gw[s], d) + ape[s][d];
      float p = fast_exp2((score - mx[d]) * LOG2E);
      den[d] += p;
#ifdef KPOOL_ACC_FMA
      acc[d] = __builtin_fmaf(bf16at(kw[s], d), p, acc[d]);
#else
      {
#pragma clang fp contract(off)  // hipcc contracts even __fmul_rn/__fadd_rn
        acc[d] = acc[d] + bf16at(kw[s], d) * p;
      }
#endif
    }
  }

  // --- pooled vector -> bf16 -> Hadamard-128 -> bf16 ---
  float x[DPL];
#pragma unroll
  for (int d = 0; d < DPL; ++d) x[d] = acc[d] / den[d];  // exact division (rcp * mul is not bit-exact)
#pragma unroll
  for (int d = 0; d < DPL; d += 2) round_bf16x2(x[d], x[d + 1]);

  had_stages_inlane<DPL>(x);
  if constexpr (LANES >= 2) had_stage_xlane<1, DPL>(x, sub);
  if constexpr (LANES >= 4) had_stage_xlane<2, DPL>(x, sub);
  if constexpr (LANES >= 8) had_stage_xlane<4, DPL>(x, sub);
  if constexpr (LANES >= 16) had_stage_xlane<8, DPL>(x, sub);
  if constexpr (LANES >= 32) had_stage_xlane<16, DPL>(x, sub);
  if constexpr (LANES >= 64) had_stage_xlane<32, DPL>(x, sub);
#pragma unroll
  for (int d = 0; d < DPL; ++d) x[d] = x[d] * HAD_SCALE;
#pragma unroll
  for (int d = 0; d < DPL; d += 2) round_bf16x2(x[d], x[d + 1]);

  // --- per-vector absmax fp8 quant (all values finite and >= 0: unsigned
  // integer max is the exact float max, without canonicalization) ---
  float am = 0.0f;
#pragma unroll
  for (int d = 0; d < DPL; ++d) am = max_nonneg(am, fabsf(x[d]));
  if constexpr (LANES >= 2) am = absmax_xlane<1>(am, sub);
  if constexpr (LANES >= 4) am = absmax_xlane<2>(am, sub);
  if constexpr (LANES >= 8) am = absmax_xlane<4>(am, sub);
  if constexpr (LANES >= 16) am = absmax_xlane<8>(am, sub);
  if constexpr (LANES >= 32) am = absmax_xlane<16>(am, sub);
  if constexpr (LANES >= 64) am = absmax_xlane<32>(am, sub);
  am = max_nonneg(am, 1e-4f);
  // x / scale: under round_scale the scale is 2^e, so the division is the
  // exact ldexp(x, -e) (no rcp, no division sequence); otherwise exact division.
  float scale;
  float q[DPL];
  if (round_scale) {
    const float e = __builtin_ceilf(fast_log2(am * FP8_MAX_INV));
    scale = fast_exp2(e);
#ifndef KPOOL_NO_LDEXP
    const int ne = -(int)e;
#pragma unroll
    for (int d = 0; d < DPL; ++d) q[d] = __builtin_amdgcn_ldexpf(x[d], ne);
#else
#pragma unroll
    for (int d = 0; d < DPL; ++d) q[d] = x[d] / scale;
#endif
  } else {
    scale = am * FP8_MAX_INV;
#pragma unroll
    for (int d = 0; d < DPL; ++d) q[d] = x[d] / scale;
  }
#pragma unroll
  for (int d = 0; d < DPL; ++d) q[d] = __builtin_amdgcn_fmed3f(q[d], -FP8_MAX, FP8_MAX);  // == min(max(q, -448), 448)
  constexpr int NQW = DPL >= 4 ? DPL / 4 : 1;
  uint32_t qw[NQW];
  if constexpr (DPL == 2) {
    qw[0] = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(q[0], q[1], 0, false);
  }
#pragma unroll
  for (int c = 0; c < DPL / 4; ++c) {
    uint32_t w = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(q[c * 4 + 0], q[c * 4 + 1], 0, false);
    w = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(q[c * 4 + 2], q[c * 4 + 3], (int)w, true);
    qw[c] = w;
  }

  // --- store (32-bit lane offsets on the page base) ---
  if (preshuffle) {
    // byte offset of (tok, dim) = (tok/16)*16*128 + (dim/16)*256 + (tok%16)*16 + dim%16
    gptr<uint8_t> kp = pbase + (uint32_t)((tok >> 4) * (16 * HEAD_DIM) + (d0 >> 4) * 256 + (tok & 15) * 16 + (d0 & 15));
    if constexpr (DPL == 16) {
      *(gptr<u32x4>)kp = u32x4{qw[0], qw[1], qw[2], qw[3]};
    } else if constexpr (DPL == 8) {
      *(gptr<u32x2>)kp = u32x2{qw[0], qw[1]};
    } else if constexpr (DPL == 2) {
      // lane pairs (2k, 2k+1) hold dims 4k..4k+3: the even lane stores one
      // dword (2-byte stores per lane are far slower in the memory pipeline)
      const uint32_t other = __float_as_uint(xor_lane<1>(__uint_as_float(qw[0]), sub));
      if ((sub & 1) == 0) *(gptr<uint32_t>)kp = (qw[0] & 0xFFFFu) | (other << 16);
    } else {
#pragma unroll
      for (int c = 0; c < DPL / 4; ++c) ((gptr<uint32_t>)kp)[c] = qw[c];
    }
  } else {
    gptr<uint8_t> kp = pbase + (uint32_t)(tok * HEAD_DIM + d0);
    if constexpr (DPL == 2) {
      const uint32_t other = __float_as_uint(xor_lane<1>(__uint_as_float(qw[0]), sub));
      if ((sub & 1) == 0) *(gptr<uint32_t>)kp = (qw[0] & 0xFFFFu) | (other << 16);
    } else {
#pragma unroll
      for (int c = 0; c < DPL / 4; ++c) ((gptr<uint32_t>)kp)[c] = qw[c];
    }
  }
  if (sub == 0) {
    *(gptr<float>)(pbase + scale_off) = scale;
  }
}

// page / token-in-page of a cache slot (page_shift = log2(page_size), or -1
// when page_size is not a power of two)
template <typename LocT>
__device__ __forceinline__ LocT page_of(LocT loc, int page_size, int page_shift, int& tok) {
  // (the power-of-two case is the fall-through: a taken branch over the
  // division sequence costs a one-shot wave an instruction-cache miss)
  if (__builtin_expect(page_shift >= 0, 1)) {
    tok = (int)(loc & (page_size - 1));
    return loc >> page_shift;
  }
  tok = (int)(loc % page_size);
  return loc / page_size;
}

// Workgroups are dispatched round-robin over the 8 XCDs (each with its own
// L2). Remapped, each XCD gets a contiguous range of windows, so the rows a
// misaligned pool re-reads from the previous window were fetched by a
// neighbour on the same XCD (L2 hit). Bijective for any grid size.
template <int XCD>
__device__ __forceinline__ int xcd_remap(int b, int nb) {
  if constexpr (XCD <= 1) return b;
  const int q = nb / XCD, r = nb % XCD;
  const int x = b % XCD, i = b / XCD;
  return x * q + min(x, r) + i;
}

// Tail ring of the decode step: [num_blocks, 2, ring, 128] bf16, seeded by
// the prefill tokens that end a request's open pool.
struct TailArgs {
  uint16_t* tail;        // strides below in elements
  int block_elems;       // stride(0) < 2^31 (checked); vLLM pads the tail page to the KV
  int head_elems;        // stride(1)  page, so block * stride(0) is formed in 64 bits
  int ring_mask;         // ring - 1; ring = shape[2] is a power of two (checked by the entry point)
  int ring_shift;        // log2(ring)
};
__device__ __forceinline__ int ring_div(int x, const TailArgs& ta) { return x >> ta.ring_shift; }
__device__ __forceinline__ int ring_mod(int x, const TailArgs& ta) { return x & ta.ring_mask; }

// ===========================================================================
// Prefill insert window (+ tail seeding)
// ===========================================================================

template <int POOL, int LANES, bool VEC, typename SlotT>
__device__ __forceinline__ void insert_window(
    int64_t win,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ gate,
    const float* __restrict__ ape,
    const void* __restrict__ slot_mapping,
    uint8_t* __restrict__ cache,
    int64_t n_tokens,
    int64_t n_windows,
    int64_t k_stride,
    int64_t gate_stride,
    int64_t ape_stride,
    int page_size,
    int page_shift,
    int64_t page_bytes,
    int flags,  // bit0: slot int64, bit1: preshuffle, bit2: round_scale, bit3: seed tail
    const SlotT* __restrict__ seed_tslot,
    TailArgs ta,
    int grp,
    int sub) {
  constexpr int DPL = HEAD_DIM / LANES;
  static_assert(POOL <= LANES, "one lane per slot in the window");
  const bool preshuffle = flags & 2;
  const bool round_scale = flags & 4;
  const bool seed = flags & 8;
  const int d0 = sub * DPL;
  const int64_t win_first = win * POOL;

  // Slot read first, unconditional (clamped index) so it is one vector load
  // with no branch; K / gate / ape loads below are issued right behind it and
  // pinned so the compiler cannot sink them under the slot-dependent branch.
  const int64_t my_tok = win_first + (sub % POOL);
  const int64_t tok_c = my_tok < n_tokens ? my_tok : n_tokens - 1;
  const int64_t my_slot = (int64_t)reinterpret_cast<const SlotT*>(slot_mapping)[tok_c];

  float apev[POOL][DPL];
#pragma unroll
  for (int s = 0; s < POOL; ++s) {
    const float* ap = ape + s * ape_stride + d0;
    if constexpr (DPL == 2) {
      float2 a = *reinterpret_cast<const float2*>(ap);
      apev[s][0] = a.x;
      apev[s][1] = a.y;
    } else {
#pragma unroll
      for (int c = 0; c < DPL / 4; ++c) {
        float4 a = reinterpret_cast<const float4*>(ap)[c];
        apev[s][c * 4 + 0] = a.x;
        apev[s][c * 4 + 1] = a.y;
        apev[s][c * 4 + 2] = a.z;
        apev[s][c * 4 + 3] = a.w;
      }
    }
  }
  uint32_t gw[POOL][DPL / 2];
  uint32_t kw[POOL][DPL / 2];
  bool have_window = win < n_windows && win_first + POOL <= n_tokens;
  if (have_window) {
    load_rows<POOL, DPL, VEC>(k, gate, win_first, k_stride, gate_stride, d0, gw, kw);
  }
  // tail-seed metadata, independent of everything else: issued up front
  int seed_ts[POOL], seed_ahead[POOL];
  if (seed && win < n_windows) {
#pragma unroll
    for (int s = 0; s < POOL; ++s) {
      const int64_t tok = win_first + s;
      seed_ts[s] = tok < n_tokens ? (int)seed_tslot[tok] : -1;
      seed_ahead[s] = (tok + POOL < n_tokens) ? (int)seed_tslot[tok + POOL] : -1;
    }
  }

  const int closes = (win < n_windows && sub < POOL && my_tok < n_tokens && my_tok >= POOL - 1 && my_slot >= 0) ? 1 : 0;
  const uint64_t ballot = __ballot(closes);
  uint64_t bits = (ballot >> (grp * LANES)) & ((LANES == 64) ? ~0ull : ((1ull << LANES) - 1));
  // Pin the preloaded rows here (after the ballot, which must only wait for
  // the slot) so the compiler neither sinks the loads under the
  // slot-dependent branch nor waits for them before the ballot.
#pragma unroll
  for (int s2 = 0; s2 < POOL; ++s2) {
#pragma unroll
    for (int c = 0; c < DPL / 2; ++c) { asm volatile("" : "+v"(gw[s2][c])); asm volatile("" : "+v"(kw[s2][c])); }
#pragma unroll
    for (int d = 0; d < DPL; ++d) asm volatile("" : "+v"(apev[s2][d]));
  }

  while (bits) {
    const int i = __builtin_ctzll(bits);
    bits &= bits - 1;
    const int64_t last = win_first + i;
    const int64_t loc = (int64_t)reinterpret_cast<const SlotT*>(slot_mapping)[last];
    if (!(have_window && i == POOL - 1)) {
      load_rows<POOL, DPL, VEC>(k, gate, last - (POOL - 1), k_stride, gate_stride, d0, gw, kw);
      have_window = false;
    }
    int tok;
    const int64_t page = page_of<int64_t>(loc, page_size, page_shift, tok);
    compress_pool<POOL, LANES>(gw, kw, apev, (gptr<uint8_t>)(cache + page * page_bytes), tok,
                               (uint32_t)(page_size * HEAD_DIM + tok * 4), preshuffle, round_scale, sub);
  }

  // Tail seeding (kpool_seed_tail_cache): token i is among its request's last
  // POOL tokens if the token POOL ahead has another tail block (or is past
  // the batch / has slot < 0). Its raw K / gate rows go to
  // tail[blk, {0, 1}, ts % ring]. The rows are already in registers when the
  // window was preloaded and not clobbered.
  if (seed && win < n_windows) {
#pragma unroll
    for (int s = 0; s < POOL; ++s) {
      const int64_t tok = win_first + s;
      if (tok >= n_tokens) break;
      const int ts = seed_ts[s];
      if (ts < 0) continue;
      const int blk = ring_div(ts, ta);
      const int ahead = seed_ahead[s];
      if (ahead >= 0 && ring_div(ahead, ta) == blk) continue;
      uint32_t rk[DPL / 2], rs[DPL / 2];
      if (have_window) {
#pragma unroll
        for (int c = 0; c < DPL / 2; ++c) { rk[c] = kw[s][c]; rs[c] = gw[s][c]; }
      } else {
        load_row<DPL, VEC>(k + tok * k_stride + d0, rk);
        load_row<DPL, VEC>(gate + tok * gate_stride + d0, rs);
      }
      const uint64_t dst = (uint64_t)(uint32_t)blk * (uint32_t)ta.block_elems + (uint32_t)(ring_mod(ts, ta) * HEAD_DIM + d0);
      uint32_t* pk = reinterpret_cast<uint32_t*>(ta.tail + dst);
      uint32_t* ps = reinterpret_cast<uint32_t*>(ta.tail + (uint32_t)ta.head_elems + dst);
#pragma unroll
      for (int c = 0; c < DPL / 2; ++c) { pk[c] = rk[c]; ps[c] = rs[c]; }
    }
  }
}


// ---------------------------------------------------------------------------
// Prefill insert: one wave per workgroup, 64 / LANES windows per wave.
// ---------------------------------------------------------------------------
template <int POOL, int LANES, bool VEC, typename SlotT>
__global__ void __launch_bounds__(64) kpool_insert_kernel(
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ gate,
    const float* __restrict__ ape,
    const void* __restrict__ slot_mapping,
    uint8_t* __restrict__ cache,
    int64_t n_tokens,
    int64_t n_windows,
    int64_t k_stride,
    int64_t gate_stride,
    int64_t ape_stride,
    int page_size,
    int page_shift,
    int64_t page_bytes,
    const void* __restrict__ seed_tslot,  // [n_tokens] SlotT, with flags bit3
    TailArgs ta,
    int flags) {
  constexpr int GROUPS_PER_WAVE = 64 / LANES;
  const int lane = threadIdx.x;
  const int grp = GROUPS_PER_WAVE == 1 ? 0 : lane / LANES;
  const int sub = lane % LANES;
  const int64_t win = (int64_t)xcd_remap<8>(blockIdx.x, gridDim.x) * GROUPS_PER_WAVE + grp;
  insert_window<POOL, LANES, VEC, SlotT>(win, k, gate, ape, slot_mapping, cache, n_tokens, n_windows, k_stride,
                                         gate_stride, ape_stride, page_size, page_shift, page_bytes, flags,
                                         reinterpret_cast<const SlotT*>(seed_tslot), ta, grp, sub);
}

// ===========================================================================
// Launchers
// ===========================================================================

inline int page_shift_of(int page_size) {
  return (page_size > 0 && (page_size & (page_size - 1)) == 0) ? __builtin_ctz((unsigned)page_size) : -1;
}

template <int POOL, int LANES, bool VEC, typename SlotT>
int launch_insert_t(const void* k, const void* gate, const void* ape, const void* slot, void* cache,
                    int64_t n_tokens, int64_t k_stride, int64_t gate_stride, int64_t ape_stride, int page_size,
                    int64_t page_bytes, const void* seed_tslot, TailArgs ta, int flags, hipStream_t stream) {
  constexpr int GROUPS_PER_WAVE = 64 / LANES;
  const int64_t n_windows = (n_tokens + POOL - 1) / POOL;
  const int64_t grid = (n_windows + GROUPS_PER_WAVE - 1) / GROUPS_PER_WAVE;
  if (grid == 0) return 0;
  hipLaunchKernelGGL((kpool_insert_kernel<POOL, LANES, VEC, SlotT>), dim3((unsigned)grid), dim3(64), 0, stream,
                     (const uint16_t*)k, (const uint16_t*)gate, (const float*)ape, slot, (uint8_t*)cache, n_tokens,
                     n_windows, k_stride, gate_stride, ape_stride, page_size, page_shift_of(page_size), page_bytes,
                     seed_tslot, ta, flags);
  return (int)hipGetLastError();
}

template <int POOL, int LANES>
int launch_insert_l(bool vec, bool i64, const void* k, const void* gate, const void* ape, const void* slot,
                    void* cache, int64_t n_tokens, int64_t k_stride, int64_t gate_stride, int64_t ape_stride,
                    int page_size, int64_t page_bytes, const void* seed_tslot, TailArgs ta, int flags,
                    hipStream_t stream) {
#define KPOOL_I(V, S)                                                                                          \
  return launch_insert_t<POOL, LANES, V, S>(k, gate, ape, slot, cache, n_tokens, k_stride, gate_stride, ape_stride, \
                                            page_size, page_bytes, seed_tslot, ta, flags, stream);
  if (vec && i64) { KPOOL_I(true, int64_t) }
  if (vec) { KPOOL_I(true, int32_t) }
  if (i64) { KPOOL_I(false, int64_t) }
  KPOOL_I(false, int32_t)
#undef KPOOL_I
}

template <int POOL>
int launch_insert(int lanes, bool vec, bool i64, const void* k, const void* gate, const void* ape, const void* slot,
                  void* cache, int64_t n_tokens, int64_t k_stride, int64_t gate_stride, int64_t ape_stride,
                  int page_size, int64_t page_bytes, const void* seed_tslot, TailArgs ta, int flags,
                  hipStream_t stream) {
#define KPOOL_L(L)                                                                                              \
  if (lanes == L)                                                                                               \
    return launch_insert_l<POOL, L>(vec, i64, k, gate, ape, slot, cache, n_tokens, k_stride, gate_stride,       \
                                    ape_stride, page_size, page_bytes, seed_tslot, ta, flags, stream);
#ifdef KPOOL_ISA_QUICK
  KPOOL_L(64)
#else
  KPOOL_L(64) KPOOL_L(32) KPOOL_L(16)
#endif
#undef KPOOL_L
  return -2;
}

}  // namespace

// Launcher return codes: 0 ok, -2 bad lane count, -3 unsupported pool size, else hipError_t.
#ifdef KPOOL_ISA_QUICK
// -DKPOOL_ISA_QUICK: instantiate only the production shapes (pool 4, 64
// lanes) for fast ISA listings; never used for a real build.
#define KPOOL_POOL_SWITCH(EXPR)                      \
  switch (pool_size) {                               \
    case 4: { constexpr int P = 4; EXPR }            \
    default: rc = -3; break;                         \
  }
#else
#define KPOOL_POOL_SWITCH(EXPR)                      \
  switch (pool_size) {                               \
    case 2: { constexpr int P = 2; EXPR }            \
    case 3: { constexpr int P = 3; EXPR }            \
    case 4: { constexpr int P = 4; EXPR }            \
    case 8: { constexpr int P = 8; EXPR }            \
    case 16: { constexpr int P = 16; EXPR }          \
    default: rc = -3; break;                         \
  }
#endif

// ===========================================================================
// aiter entry points
// ===========================================================================

namespace aiter {

namespace {

constexpr int KPOOL_HEAD_DIM = 128;

int kpool_pick_lanes(int64_t n_tokens, int pool_size)
{
    // Chunked-prefill sizes are latency bound: one pool per wave (64 lanes,
    // 2 dims per lane) gives the most workgroups and the shortest per-lane
    // chain. Large batches are bandwidth bound: 16 lanes (8 dims, one 16 B
    // load per row per lane). Tuned on MI355X, pool_size 4.
    const int64_t n_pools = n_tokens / pool_size;
    if(n_pools <= 8192)
        return 64;
    if(n_pools <= 65536)
        return 32;
    return 16;
}

bool aligned(const void* p, int64_t stride_elems, int64_t elem_bytes, int64_t align)
{
    return (reinterpret_cast<uintptr_t>(p) % align) == 0 && ((stride_elems * elem_bytes) % align) == 0;
}

void check_rc(int rc, const char* what)
{
    AITER_CHECK(rc != -2, what, ": lanes_per_pool must be 16, 32 or 64 and >= pool_size");
    AITER_CHECK(rc != -3, what, ": pool_size must be 2, 3, 4, 8 or 16");
    AITER_CHECK(rc == 0, what, ": ", hipGetErrorString(static_cast<hipError_t>(rc)));
}

void check_cache(const aiter_tensor_t& kv_cache)
{
    AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_u8 || kv_cache.dtype() == AITER_DTYPE_fp8,
                "kv_cache must be uint8 / fp8, got ", AiterDtype_to_str(kv_cache.dtype()));
    AITER_CHECK(kv_cache.dim() == 3 && kv_cache.size(2) == KPOOL_HEAD_DIM + 4,
                "kv_cache must be [num_blocks, page_size, 132]");
    AITER_CHECK(kv_cache.is_contiguous(), "kv_cache must be contiguous");
    const int page_size = static_cast<int>(kv_cache.size(1));
    AITER_CHECK(page_size == 1 || page_size % 16 == 0,
                "page_size must be 1 or a multiple of 16 (preshuffled tiles), got ", page_size);
    AITER_CHECK(page_size <= (1 << 20), "page_size too large, got ", page_size);
}

void check_ape(const aiter_tensor_t& ape, int pool_size)
{
    AITER_CHECK(ape.dtype() == AITER_DTYPE_fp32 && ape.dim() == 2 && ape.size(0) == pool_size &&
                    ape.size(1) == KPOOL_HEAD_DIM && ape.stride(1) == 1 && ape.stride(0) == KPOOL_HEAD_DIM,
                "ape must be a contiguous fp32 [pool_size, 128] tensor");
}

void check_rows(const aiter_tensor_t& t, const char* name)
{
    AITER_CHECK(t.dtype() == AITER_DTYPE_bf16, name, " must be bf16");
    AITER_CHECK(t.dim() == 2 && t.size(1) == KPOOL_HEAD_DIM && t.stride(1) == 1,
                name, " must be [n_tokens, 128] with unit inner stride");
}

bool slot_is_i64(const aiter_tensor_t& slot_mapping)
{
    AITER_CHECK(slot_mapping.dtype() == AITER_DTYPE_i32 || slot_mapping.dtype() == AITER_DTYPE_i64,
                "slot_mapping must be int32 or int64, got ", AiterDtype_to_str(slot_mapping.dtype()));
    AITER_CHECK(slot_mapping.dim() == 1 && slot_mapping.stride(0) == 1, "slot_mapping must be a contiguous 1-D tensor");
    return slot_mapping.dtype() == AITER_DTYPE_i64;
}

TailArgs tail_args(const aiter_tensor_t& tail_kv_cache, int pool_size)
{
    AITER_CHECK(tail_kv_cache.dtype() == AITER_DTYPE_bf16 && tail_kv_cache.dim() == 4 &&
                    tail_kv_cache.size(1) == 2 && tail_kv_cache.size(3) == KPOOL_HEAD_DIM &&
                    tail_kv_cache.stride(3) == 1 && tail_kv_cache.stride(2) == KPOOL_HEAD_DIM,
                "tail_kv_cache must be bf16 [num_blocks, 2, ring, 128] with contiguous rows");
    const int ring = static_cast<int>(tail_kv_cache.size(2));
    AITER_CHECK(ring >= pool_size && ring % pool_size == 0 && (ring & (ring - 1)) == 0,
                "tail ring (", ring, ") must be a power of two and a multiple of pool_size (", pool_size, ")");
    // vLLM pads every tail page to the KV-cache page (stride(0) >> 2 * ring * 128) and the
    // whole tail can exceed 2^31 elements: block offsets are 64-bit, the strides stay 32-bit.
    AITER_CHECK(tail_kv_cache.stride(0) < (1LL << 31) && tail_kv_cache.stride(1) < (1LL << 31),
                "tail_kv_cache strides must fit in 32 bits, got ", tail_kv_cache.stride(0), ", ",
                tail_kv_cache.stride(1));
    AITER_CHECK(aligned(tail_kv_cache.data_ptr(), tail_kv_cache.stride(0), 2, 4) &&
                    aligned(tail_kv_cache.data_ptr(), tail_kv_cache.stride(1), 2, 4),
                "tail_kv_cache rows must be 4-byte aligned");
    return TailArgs{static_cast<uint16_t*>(tail_kv_cache.data_ptr()),
                    static_cast<int>(tail_kv_cache.stride(0)),
                    static_cast<int>(tail_kv_cache.stride(1)),
                    ring - 1,
                    __builtin_ctz(static_cast<unsigned>(ring))};
}

} // namespace

void kpool_compress_insert(aiter_tensor_t& kv_cache,
                           const aiter_tensor_t& k,
                           const aiter_tensor_t& gate_score,
                           const aiter_tensor_t& ape,
                           const aiter_tensor_t& slot_mapping,
                           int pool_size,
                           bool round_scale,
                           int lanes_per_pool,
                           std::optional<aiter_tensor_t> tail_kv_cache,
                           std::optional<aiter_tensor_t> tail_slot_mapping)
{
    check_cache(kv_cache);
    check_rows(k, "k");
    check_rows(gate_score, "gate_score");
    AITER_CHECK(gate_score.size(0) == k.size(0), "k and gate_score must have the same number of rows");
    check_ape(ape, pool_size);
    const bool i64 = slot_is_i64(slot_mapping);
    AITER_CHECK(slot_mapping.size(0) == k.size(0), "slot_mapping must have one entry per token");
    const int64_t n_tokens = k.size(0);
    // tail seeding: the tokens that end a request's open pool copy their raw rows
    // into the tail ring (a batch shorter than one pool still seeds)
    const bool seed = tail_kv_cache.has_value() || tail_slot_mapping.has_value();
    AITER_CHECK(!seed || (tail_kv_cache.has_value() && tail_slot_mapping.has_value()),
                "tail_kv_cache and tail_slot_mapping are required together");
    TailArgs ta{nullptr, 0, 0, 0, 0};
    const void* seed_tslot = nullptr;
    if(seed)
    {
        AITER_CHECK(slot_is_i64(*tail_slot_mapping) == i64 && tail_slot_mapping->size(0) == n_tokens,
                    "tail_slot_mapping must be a contiguous [n_tokens] tensor with the dtype of slot_mapping");
        ta         = tail_args(*tail_kv_cache, pool_size);
        seed_tslot = tail_slot_mapping->data_ptr();
    }
    if(n_tokens == 0 || (n_tokens < pool_size && !seed))
        return;
    const int lanes = lanes_per_pool > 0 ? lanes_per_pool : kpool_pick_lanes(n_tokens, pool_size);
    const int page_size = static_cast<int>(kv_cache.size(1));
    const bool vec = aligned(k.data_ptr(), k.stride(0), 2, 16) && aligned(gate_score.data_ptr(), gate_score.stride(0), 2, 16);
    const int flags = (i64 ? 1 : 0) | (page_size > 1 ? 2 : 0) | (round_scale ? 4 : 0) | (seed ? 8 : 0);

    HipDeviceGuard device_guard(kv_cache.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    int rc = -3;
    KPOOL_POOL_SWITCH(rc = launch_insert<P>(lanes, vec, i64, k.data_ptr(), gate_score.data_ptr(), ape.data_ptr(),
                                            slot_mapping.data_ptr(), kv_cache.data_ptr(), n_tokens, k.stride(0),
                                            gate_score.stride(0), ape.stride(0), page_size, kv_cache.stride(0),
                                            seed_tslot, ta, flags, stream); break;)
    check_rc(rc, "kpool_compress_insert");
}

} // namespace aiter
