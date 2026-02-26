#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <vector>

struct TileInfo {
    int batch_idx;
    int q_start;
};

void run_mla_paged_prefill(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float sm_scale,
    torch::Tensor output,
    torch::Tensor lse,
    torch::Tensor tile_infos
);
