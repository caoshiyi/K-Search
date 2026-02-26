#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>

void run_mla_decode(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    torch::Tensor output,
    torch::Tensor lse,
    float sm_scale
);
