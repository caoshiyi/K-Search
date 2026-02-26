#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct TileInfo {
    int batch_idx;
    int q_start_in_seq;
    int q_len;
    int q_global_offset;
};

void launch_mla_paged_prefill(
    void* q_nope,
    void* q_pe,
    void* ckv_cache,
    void* kpe_cache,
    int32_t* qo_indptr,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    void* output,
    float* lse,
    float sm_scale,
    TileInfo* tile_info,
    int num_tiles,
    cudaStream_t stream
);
