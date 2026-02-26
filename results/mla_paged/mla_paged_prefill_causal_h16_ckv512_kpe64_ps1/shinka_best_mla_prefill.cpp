#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cuda_runtime.h>
#include "kernel.h"

std::vector<torch::Tensor> run(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float sm_scale
) {
    int total_q = q_nope.size(0);
    int batch_size = qo_indptr.size(0) - 1;
    
    auto output = torch::empty({total_q, 16, 512}, q_nope.options());
    auto lse = torch::empty({total_q, 16}, q_nope.options().dtype(torch::kFloat32));
    
    int block_m = 16;
    int max_tiles = (total_q + block_m - 1) / block_m + batch_size + 1; 
    
    auto opts = q_nope.options().dtype(torch::kByte);
    
    // Allocate host pinned memory
    auto tile_tensor_cpu = torch::empty(
        {(long)max_tiles * (long)sizeof(TileInfo)}, 
        torch::TensorOptions().dtype(torch::kByte).pinned_memory(true)
    );
    
    TileInfo* tiles_ptr = (TileInfo*)tile_tensor_cpu.data_ptr();
    
    auto qo_cpu = qo_indptr.to(torch::kCPU);
    auto qo_acc = qo_cpu.accessor<int32_t, 1>();
    
    int tile_count = 0;
    for (int b = 0; b < batch_size; ++b) {
        int q_start = qo_acc[b];
        int q_end = qo_acc[b+1];
        int len = q_end - q_start;
        for (int i = 0; i < len; i += block_m) {
            TileInfo& t = tiles_ptr[tile_count++];
            t.batch_idx = b;
            t.q_start_in_seq = i;
            t.q_len = std::min(block_m, len - i);
            t.q_global_offset = q_start + i;
        }
    }
    
    // Copy to device (async)
    auto tile_tensor_gpu = torch::empty(
        {(long)tile_count * (long)sizeof(TileInfo)}, 
        opts.device(q_nope.device())
    );
    
    cudaMemcpyAsync(
        tile_tensor_gpu.data_ptr(),
        tiles_ptr,
        tile_count * sizeof(TileInfo),
        cudaMemcpyHostToDevice,
        at::cuda::getCurrentCUDAStream()
    );
    
    launch_mla_paged_prefill(
        q_nope.data_ptr(),
        q_pe.data_ptr(),
        ckv_cache.data_ptr(),
        kpe_cache.data_ptr(),
        (int32_t*)qo_indptr.data_ptr(),
        (int32_t*)kv_indptr.data_ptr(),
        (int32_t*)kv_indices.data_ptr(),
        output.data_ptr(),
        (float*)lse.data_ptr(),
        sm_scale,
        (TileInfo*)tile_tensor_gpu.data_ptr(),
        tile_count,
        at::cuda::getCurrentCUDAStream().stream()
    );
    
    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "MLA Prefill");
}
