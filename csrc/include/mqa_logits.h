#pragma once
// SPDX-License-Identifier: MIT
// Hand-written HIP FP8 MQA indexer logits kernels for gfx950.
//
// Two ops, the prefill/decode halves of the DeepSeek-V3.2 / GLM-5 sparse-attention
// lightning indexer:
//
//   logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]
//
// `fp8_mqa_logits`       - K gathered into a contiguous [N, 128] buffer
// `fp8_paged_mqa_logits` - K read out of a paged cache via a block table
//
// Both are fixed at n_heads=32, head_dim=128 (the shipped GLM-5-FP8 indexer shape)
// and are the same contract as `fp8_mqa_logits` / `deepgemm_fp8_paged_mqa_logits`,
// so they drop straight into the same call sites.
// aiter builds csrc without hipify, so the ATen CUDA headers (which pull in
// cuda_runtime_api.h) are unavailable -- use the HIP-flavoured ones, as the rest
// of csrc does.
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/util/Optional.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <cstdlib>
#include <type_traits>

// Writes into `out` when given (and returns it), otherwise allocates [M, N] f32.
// Positions outside a row's [ks, ke) window read -inf when `clean_logits`, and are
// left untouched otherwise.
torch::Tensor fp8_mqa_logits(torch::Tensor q_fp8,        // [M, 32, 128] fp8
                                   torch::Tensor k_fp8,        // [N, 128]     fp8
                                   torch::Tensor kv_scale,     // [N]          f32
                                   torch::Tensor weights,      // [M, 32]      f32
                                   torch::Tensor cu_seqlen_ks, // [M]          i32
                                   torch::Tensor cu_seqlen_ke, // [M]          i32
                                   int64_t BlockM,
                                   int64_t SplitN,
                                   int64_t num_warps,
                                   int64_t TotalCuCount,
                                   bool clean_logits,
                                   int64_t Unroll2,
                                   int64_t ReverseRows,
                                   std::optional<torch::Tensor> out);

// Writes into `out` when given (and returns it), otherwise allocates
// [batch*next_n, max_model_len] f32. Only the causal window
// p <= context_lens[b] - next_n + n is written; the -inf outside it is the
// caller's, exactly as for deepgemm_fp8_paged_mqa_logits.
torch::Tensor
fp8_paged_mqa_logits(torch::Tensor q_fp8,        // [batch, next_n, 32, 128] fp8
                           torch::Tensor kv_cache_fp8, // [blocks, bsz, 1, index_dim];
                                                       // bsz=1 plain, bsz=64 preshuffled
                           torch::Tensor weights,      // [batch*next_n, 32] f32
                           torch::Tensor context_lens, // [batch]            i32
                           torch::Tensor block_tables, // [batch, max_blocks] i32
                           int64_t max_model_len,
                           int64_t ChunkK,
                           int64_t SplitKV,
                           int64_t num_warps,
                           int64_t TotalCuCount,
                           int64_t RowsPerBlock,
                           std::optional<torch::Tensor> out);
