// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "py_itfs_common.h"

struct __attribute__((packed)) TopKPrefillKernelArgs
{
    void* ptr_workspace;
    void* ptr_logits;
    void* ptr_rowStarts;
    void* ptr_rowEnds;
    void* ptr_indices;
    void* ptr_values;
    int32_t stride0;
    int32_t stride1;
};

AITER_C_ITFS void top_k_per_row_prefill_fast(
    aiter_tensor_t* logits,
    aiter_tensor_t* rowStarts,
    aiter_tensor_t* rowEnds,
    aiter_tensor_t* indices,
    aiter_tensor_t* values,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    hipStream_t stream)
{
    const HipDeviceGuard device_guard(logits->device_id);

    constexpr int kTopK = 2048;
    int64_t workspace_size = kTopK * (sizeof(float) + sizeof(int32_t)) * numRows;

    // Allocate the scratch workspace from torch's caching allocator (pooled,
    // stream-ordered) instead of a raw per-call hipMalloc/hipFree. This restores
    // the pre-refactor behavior: hipMalloc/hipFree are synchronous and dominated
    // runtime for small prefills. The tensor stays alive until this function
    // returns, i.e. after the kernel launch is enqueued on `stream`.
    auto options = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(torch::Device(torch::kCUDA, logits->device_id));
    torch::Tensor workspace_tensor = torch::empty({workspace_size}, options);
    void* workspace                = workspace_tensor.data_ptr();

    TopKPrefillKernelArgs args;
    size_t arg_size = sizeof(args);

    args.ptr_workspace = workspace;
    args.ptr_logits    = logits->data_ptr();
    args.ptr_rowStarts = rowStarts->data_ptr();
    args.ptr_rowEnds   = rowEnds->data_ptr();
    args.ptr_indices   = indices->data_ptr();
    args.ptr_values    = (values != nullptr) ? values->data_ptr() : nullptr;
    args.stride0       = static_cast<int32_t>(stride0);
    args.stride1       = static_cast<int32_t>(stride1);

    static AiterAsmKernel impl_topk_prefill(
        "_ZN5aiter11PrefillTopKL10topKPerRowILi1024ELi2048ELi2048ELi512EEEvPvPKfPKiS6_PiPfii",
        "/topk_per_row_prefill/asm_top_k_per_row_prefill.co");

    constexpr int kNumThreadsPerBlock = 1024;
    AITER_CHECK(numRows >> 31 == 0, "numRows too large: ", numRows);

    impl_topk_prefill.launch_kernel({&args,
                                     &arg_size,
                                     static_cast<int>(numRows),
                                     1, 1,
                                     kNumThreadsPerBlock,
                                     1, 1,
                                     stream});
}
