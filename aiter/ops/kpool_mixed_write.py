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

``kpool_mixed_write`` runs the prefill insert, the tail seeding
(``kpool_seed_tail_cache``) and the decode update of one engine step in a
single launch, byte-identical to the vLLM Triton path. The standalone
writers are ``kpool_compress_insert`` and ``kpool_decode_update``.
"""

from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_kpool_mixed_write"
KPOOL_HEAD_DIM = 128


@compile_ops(MD_NAME, develop=True)
def kpool_mixed_write(
    kv_cache: Tensor,
    ape: Tensor,
    pool_size: int,
    round_scale: bool = True,
    lanes_per_pool: int = 0,
    prefill_k: Tensor | None = None,
    prefill_gate: Tensor | None = None,
    prefill_slot_mapping: Tensor | None = None,
    prefill_tail_slot_mapping: (
        Tensor | None
    ) = None,  # [n_tokens], dtype of prefill_slot_mapping: seed the tail ring
    tail_kv_cache: Tensor | None = None,
    # decode metadata: int32 or int64, the dtype of prefill_slot_mapping when both sides are given
    decode_tail_slot_mapping: Tensor | None = None,
    decode_key: Tensor | None = None,
    decode_gate: Tensor | None = None,
    decode_slot_mapping: Tensor | None = None,
    decode_positions: Tensor | None = None,
) -> None: ...
