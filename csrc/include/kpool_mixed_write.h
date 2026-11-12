// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

namespace aiter {

// kpool mixed-step cache writer for the GLM-5.3-Flash sparse-attention indexer
// (see csrc/kernels/kpool_mixed_write.cu for the semantics). Byte-exact with the
// vLLM Triton path.
//
// kv_cache: [num_blocks, page_size, 132] uint8 (128 fp8 e4m3fn keys per token
//           in the MFMA 16x16 preshuffled layout when page_size > 1, followed
//           by page_size fp32 scales); page_size must be 1 or a multiple of 16.
// ape:      [pool_size, 128] fp32 additive position embedding of the pool.
// pool_size in {2, 3, 4, 8, 16}; lanes_per_pool 0 (auto) or 16 / 32 / 64.

// Mixed step in one launch: the prefill insert on prefill_* (k, gate
// [n_tokens, 128] bf16, slot_mapping [n_tokens] int32 / int64), the tail
// seeding (kpool_seed_tail_cache) into tail_kv_cache [num_blocks, 2, ring, 128]
// bf16 when prefill_tail_slot_mapping [n_tokens] is given, and the decode
// update on decode_* (key, gate [num_requests, next_n, 128] bf16; the three
// [num_requests, next_n] metadata tensors). Either side may be absent. One
// index dtype per launch: the seed slots and the decode metadata must have the
// dtype of prefill_slot_mapping (int32 or int64). next_n <= 16.
void kpool_mixed_write(aiter_tensor_t& kv_cache,
                       const aiter_tensor_t& ape,
                       int pool_size,
                       bool round_scale,
                       int lanes_per_pool,
                       std::optional<aiter_tensor_t> prefill_k,
                       std::optional<aiter_tensor_t> prefill_gate,
                       std::optional<aiter_tensor_t> prefill_slot_mapping,
                       std::optional<aiter_tensor_t> prefill_tail_slot_mapping,
                       std::optional<aiter_tensor_t> tail_kv_cache,
                       std::optional<aiter_tensor_t> decode_tail_slot_mapping,
                       std::optional<aiter_tensor_t> decode_key,
                       std::optional<aiter_tensor_t> decode_gate,
                       std::optional<aiter_tensor_t> decode_slot_mapping,
                       std::optional<aiter_tensor_t> decode_positions);

} // namespace aiter
