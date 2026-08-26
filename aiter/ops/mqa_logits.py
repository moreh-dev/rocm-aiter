# SPDX-License-Identifier: MIT
"""Hand-written HIP FP8 MQA indexer logits kernels for gfx950.

The prefill/decode halves of the DeepSeek-V3.2 / GLM-5 sparse-attention lightning
indexer, both computing

    logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]

`fp8_mqa_logits` is a drop-in for `aiter.ops.triton.attention.fp8_mqa_logits`
and `fp8_paged_mqa_logits` for `deepgemm_fp8_paged_mqa_logits`, so both can be
A/B'd against the Triton/Gluon and FlyDSL kernels on identical inputs.

Both are fixed at n_heads=32, head_dim=128 -- the shipped GLM-5-FP8 indexer shape.
`is_supported()` gates on that so a sweep over other shapes can skip them rather
than trip a TORCH_CHECK.
"""

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx

MD_NAME = "module_mqa_logits"

SUPPORTED_GFX = ("gfx950",)
NUM_HEADS = 32
HEAD_DIM = 128


def is_supported(num_heads: int, head_dim: int) -> bool:
    """True when these kernels can run this shape on this device."""
    return (
        get_gfx() in SUPPORTED_GFX
        and num_heads == NUM_HEADS
        and head_dim == HEAD_DIM
    )


@compile_ops(MD_NAME, fc_name="fp8_mqa_logits")
def fp8_mqa_logits(
    q_fp8: Tensor,
    k_fp8: Tensor,
    kv_scale: Tensor,
    weights: Tensor,
    cu_seqlen_ks: Tensor,
    cu_seqlen_ke: Tensor,
    BlockM: int = 0,
    SplitN: int = 0,
    num_warps: int = 0,
    TotalCuCount: int = 256,
    clean_logits: bool = True,
    unroll2: int = -1,
    reverse_rows: int = -1,
    out: Tensor | None = None,
) -> Tensor:
    """Prefill indexer logits over a contiguous K buffer.

    q_fp8       [M, 32, 128] fp8   k_fp8    [N, 128] fp8
    kv_scale    [N] f32            weights  [M, 32] f32
    cu_seqlen_ks/ke [M] i32 -- each row m is valid on [ks[m], ke[m])

    The zero-valued tunables (BlockM, SplitN, num_warps) mean "use the host
    heuristic". Writes into `out` when given (and returns it), otherwise allocates
    [M, N] f32; outside each row's window the kernel writes -inf when clean_logits,
    and leaves the buffer untouched otherwise.
    """
    ...


@compile_ops(MD_NAME, fc_name="fp8_paged_mqa_logits")
def fp8_paged_mqa_logits(
    q_fp8: Tensor,
    kv_cache_fp8: Tensor,
    weights: Tensor,
    context_lens: Tensor,
    block_tables: Tensor,
    max_model_len: int,
    ChunkK: int = 0,
    SplitKV: int = 0,
    num_warps: int = 0,
    TotalCuCount: int = 256,
    RowsPerBlock: int = 0,
    out: Tensor | None = None,
) -> Tensor:
    """Decode indexer logits over a paged KV cache.

    q_fp8        [batch, next_n, 32, 128] fp8
    kv_cache_fp8 [num_blocks, block_size, 1, index_dim] -- block_size*128 fp8 K
                 bytes followed by block_size f32 dequant scales, so
                 index_dim == 132 for the fp8 layout.
                 block_size must be 1 or 64, and the two are tied to different
                 layouts, exactly as in production: block_size=1 takes the plain
                 cache, block_size=64 takes the `shuffle_weight(layout=(16, 16))`
                 preshuffled one.
    weights      [batch*next_n, 32] f32   context_lens [batch] i32
    block_tables [batch, max_blocks_per_seq] i32

    Writes into `out` when given (and returns it), otherwise allocates
    [batch*next_n, max_model_len] f32. Only the causal window
    p <= context_lens[b] - next_n + n is written; the -inf outside it is the
    caller's, exactly as for `deepgemm_fp8_paged_mqa_logits`.
    """
    ...
