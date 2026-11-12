# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Hand-written HIP prefill-phase FP8 MQA indexer logits + top-K for gfx950.

The dense half of the DeepSeek-V3.2 / GLM-5 sparse-attention lightning indexer and
the per-row top-K that follows it:

    logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]

computed and ranked without writing the fp32 ``[M, N]`` logits. K has already been
gathered into a contiguous ``[N, 128]`` buffer (same inputs as
``aiter.ops.triton.attention.fp8_mqa_logits``). Kernel design: see
csrc/kernels/fp8_mqa_logits_topk.cu and fp8_mqa_logits_topk_select.cu.

Fixed at n_heads=32, head_dim=128 -- the GLM-5.x indexer shape.
``is_supported()`` gates on that, so a caller that also has to serve other shapes
can route them to the Triton kernel rather than trip a TORCH_CHECK.
"""

from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx

MD_NAME = "module_fp8_mqa_logits_topk"

SUPPORTED_GFX = ("gfx950",)
NUM_HEADS = 32
HEAD_DIM = 128


def is_supported(num_heads: int, head_dim: int) -> bool:
    """True when this kernel can run this shape on this device."""
    return get_gfx() in SUPPORTED_GFX and num_heads == NUM_HEADS and head_dim == HEAD_DIM


@compile_ops(MD_NAME, fc_name="fp8_mqa_logits_topk")
def fp8_mqa_logits_topk(
    q_fp8: Tensor,
    k_fp8: Tensor,
    kv_scale: Tensor,
    weights: Tensor,
    cu_seqlen_ks: Tensor,
    cu_seqlen_ke: Tensor,
    topk_indices: Tensor,
    TotalCuCount: int = 256,
) -> Tensor:
    """Indexer logits + a per-row top-K in one call that never writes fp32 logits.

    topk_indices [M, K] i32, K <= 512, written in place (and returned): row m gets the
    indices, relative to ks[m], of its K largest logits in [ks[m], ke[m]) in no
    particular order, or 0..len-1 then -1 when the row has len <= K -- the contract of
    vLLM's top_k_per_row_prefill. The logits kernel stores only the high 16 bits of
    each logit; the elements tied with the K-th 16-bit key get their fp32 logit
    recomputed with the same MFMA and epilogue (same bits), so the selection equals
    a top-K of the fp32 logits up to exact fp32 ties.
    """
