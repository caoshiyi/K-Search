#pragma once
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

void run_dsa_attn(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    float sm_scale,
    torch::Tensor output,
    torch::Tensor lse
);