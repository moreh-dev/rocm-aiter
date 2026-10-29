// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

namespace aiter {

// kpool cache writer for the GLM-5.3-Flash sparse-attention indexer (see
// csrc/kernels/kpool_compress_insert.cu for the semantics). Byte-exact with the vLLM
// Triton path.
//
// kv_cache: [num_blocks, page_size, 132] uint8 (128 fp8 e4m3fn keys per token
//           in the MFMA 16x16 preshuffled layout when page_size > 1, followed
//           by page_size fp32 scales); page_size must be 1 or a multiple of 16.
// ape:      [pool_size, 128] fp32 additive position embedding of the pool.
// pool_size in {2, 3, 4, 8, 16}; lanes_per_pool 0 (auto) or 16 / 32 / 64.

// Prefill insert: k, gate_score [n_tokens, 128] bf16 (rows may be strided),
// slot_mapping [n_tokens] int32 / int64 (>= 0 on the token that closes a pool).
// With tail_kv_cache [num_blocks, 2, ring, 128] bf16 (ring a power of two and
// a multiple of pool_size) and tail_slot_mapping [n_tokens] (dtype of
// slot_mapping, < 0 to skip a token) the tokens that end a request's open pool
// also seed the tail ring (kpool_seed_tail_cache): one call covers the whole
// prefill step of the indexer.
void kpool_compress_insert(aiter_tensor_t& kv_cache,
                           const aiter_tensor_t& k,
                           const aiter_tensor_t& gate_score,
                           const aiter_tensor_t& ape,
                           const aiter_tensor_t& slot_mapping,
                           int pool_size,
                           bool round_scale                                = true,
                           int lanes_per_pool                              = 0,
                           std::optional<aiter_tensor_t> tail_kv_cache     = std::nullopt,
                           std::optional<aiter_tensor_t> tail_slot_mapping = std::nullopt);

} // namespace aiter
