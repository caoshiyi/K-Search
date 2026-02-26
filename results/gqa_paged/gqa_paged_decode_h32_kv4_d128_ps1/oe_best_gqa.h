#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>

void run_gqa_decode(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float sm_scale,
    torch::Tensor output,
    torch::Tensor lse
);
