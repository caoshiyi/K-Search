#pragma once
#include <torch/extension.h>

void launch_gqa_decode(
    torch::Tensor& output,
    torch::Tensor& lse,
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& kv_indptr,
    const torch::Tensor& kv_indices,
    float sm_scale,
    torch::Tensor& tmp_buffer,
    torch::Tensor& tmp_stats,
    torch::Tensor& semaphores,
    int split_k
);
