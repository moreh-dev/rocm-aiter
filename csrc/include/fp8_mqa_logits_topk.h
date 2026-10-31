#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Hand-written HIP prefill-phase FP8 MQA indexer logits + top-K for gfx950.
//
// The dense half of the DeepSeek-V3.2 / GLM-5 sparse-attention lightning indexer and its top-K:
//
//   logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]   for n in [ks[m], ke[m])
//
// K has already been gathered out of the paged cache into a contiguous [N, 128]
// buffer, so there is no block table (same inputs as `aiter.ops.triton.attention.fp8_mqa_logits`).
//
// Fixed at n_heads=32, head_dim=128 -- the GLM-5.x indexer shape.

// aiter builds csrc without hipify, so the ATen CUDA headers (which pull in
// cuda_runtime_api.h) are unavailable -- use the HIP-flavoured ones, as the rest
// of csrc does.
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/util/Optional.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <cstdlib>

// The logits + top-K per row (K <= 512) without fp32 logits: 16-bit keys, exact fp32 only for the
// elements tied with the K-th key. Output as vLLM's top_k_per_row_prefill: indices relative to ks
// (unordered), or 0..len-1 then -1 for a row of len <= K.
torch::Tensor fp8_mqa_logits_topk(torch::Tensor q_fp8,
                                  torch::Tensor k_fp8,
                                  torch::Tensor kv_scale,
                                  torch::Tensor weights,
                                  torch::Tensor cu_seqlen_ks,
                                  torch::Tensor cu_seqlen_ke,
                                  torch::Tensor topk_indices, // [M, K] i32, written in place
                                  int64_t TotalCuCount);

// fp8_mqa_logits_topk_select.cu
constexpr int TOPK_HI16_MAX_K = 512;
int64_t topk_hi16_workspace_ints(int M, int N); // scratch for launch_topk_hi16
int launch_topk_hi16(const void* logits16, const int* ks, const int* ke, int* out, int M, int N, int K,
                     const void* q, const void* k, const float* kv_scale, const float* w, int* ws,
                     hipStream_t s);
