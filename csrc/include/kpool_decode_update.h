// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>

namespace aiter {

// kpool cache writer for the GLM-5.3-Flash sparse-attention indexer (see
// csrc/kernels/kpool_decode_update.cu for the semantics). Byte-exact with the vLLM
// Triton path.
//
// kv_cache: [num_blocks, page_size, 132] uint8 (128 fp8 e4m3fn keys per token
//           in the MFMA 16x16 preshuffled layout when page_size > 1, followed
//           by page_size fp32 scales); page_size must be 1 or a multiple of 16.
// ape:      [pool_size, 128] fp32 additive position embedding of the pool.
// pool_size in {2, 3, 4, 8, 16}.

// Decode update: each token stashes its K / gate into the request's tail ring
// tail_kv_cache [num_blocks, 2, ring, 128] bf16 (ring a multiple of pool_size);
// a token that closes a pool compresses it into kv_cache. key, slot_score
// [num_requests, next_n, 128] bf16; tail_slot_mapping, slot_mapping, positions
// [num_requests, next_n] all int32 or all int64; next_n <= 16.
void kpool_decode_update(aiter_tensor_t& kv_cache,
                         aiter_tensor_t& tail_kv_cache,
                         const aiter_tensor_t& tail_slot_mapping,
                         const aiter_tensor_t& key,
                         const aiter_tensor_t& slot_score,
                         const aiter_tensor_t& ape,
                         const aiter_tensor_t& slot_mapping,
                         const aiter_tensor_t& positions,
                         int pool_size,
                         bool round_scale = true);

} // namespace aiter
