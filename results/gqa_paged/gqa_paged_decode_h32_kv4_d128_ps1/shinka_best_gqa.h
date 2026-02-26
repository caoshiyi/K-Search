#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

void launch_gqa_decode(
    __nv_bfloat16* output,
    float* lse,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int* kv_indptr,
    const int* kv_indices,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    cudaStream_t stream
);
