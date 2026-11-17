# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""kpool cache writers for the GLM-5.3-Flash sparse-attention indexer.

The indexer caches one fp8 key per *pool* of ``pool_size`` consecutive tokens:
a softmax(gate + ape)-weighted mean of the pool's K rows, rounded to bf16,
Hadamard-128 rotated, rounded to bf16, and quantized to fp8 e4m3fn with one
ue8m0 scale. ``kv_cache`` is ``[num_blocks, page_size, 132]`` uint8: each page
holds ``page_size * 128`` key bytes (the MFMA 16x16 preshuffled layout of
``indexer_k_quant_and_cache`` when ``page_size > 1``) followed by
``page_size`` fp32 scales.

The decode writer ``kpool_decode_update`` is byte-identical to the vLLM
Triton path: each token's K / gate is stashed into the request's tail ring
and the token that closes a pool compresses it from the ring. Deterministic,
next_n <= 16.
"""

from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_kpool_decode_update"
KPOOL_HEAD_DIM = 128


@compile_ops(MD_NAME, develop=True)
def kpool_decode_update(
    kv_cache: Tensor,  # [num_blocks, page_size, 132] uint8
    tail_kv_cache: Tensor,  # [num_blocks, 2, ring, 128] bf16, ring % pool_size == 0
    tail_slot_mapping: Tensor,  # [num_requests, next_n] int32 / int64 (all three metadata tensors alike)
    key: Tensor,  # [num_requests, next_n, 128] bf16
    slot_score: Tensor,  # [num_requests, next_n, 128] bf16
    ape: Tensor,  # [pool_size, 128] fp32
    slot_mapping: Tensor,  # [num_requests, next_n] int32 / int64, >= 0 on closing tokens
    positions: Tensor,  # [num_requests, next_n] int32 / int64
    pool_size: int,
    round_scale: bool = True,
) -> None: ...
