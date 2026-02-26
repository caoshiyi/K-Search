#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "kernel.h"

std::vector<torch::Tensor> run(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale
) {
    int total_q = q_nope.size(0);
    int num_heads = q_nope.size(1);
    int head_dim = 512;
    
    auto opts = q_nope.options();
    auto output = torch::empty({total_q, num_heads, head_dim}, opts);
    auto lse = torch::full({total_q, num_heads}, -std::numeric_limits<float>::infinity(), opts.dtype(torch::kFloat32));
    
    // Generate tiles on CPU
    std::vector<TileInfo> tiles;
    auto indptr_cpu = qo_indptr.to(torch::kCPU);
    const int* indptr = indptr_cpu.data_ptr<int>();
    int batch_size = indptr_cpu.size(0) - 1;
    
    for (int b = 0; b < batch_size; ++b) {
        int start = indptr[b];
        int end = indptr[b+1];
        int len = end - start;
        for (int i = 0; i < len; i += 16) { // BR=16
            tiles.push_back({b, i});
        }
    }
    
    auto tile_opts = torch::TensorOptions().dtype(torch::kByte).device(q_nope.device());
    torch::Tensor tile_tensor = torch::empty({(long)tiles.size() * (long)sizeof(TileInfo)}, tile_opts);
    cudaMemcpy(tile_tensor.data_ptr(), tiles.data(), tiles.size() * sizeof(TileInfo), cudaMemcpyHostToDevice);
    
    run_mla_paged_prefill(
        q_nope, q_pe, ckv_cache, kpe_cache,
        qo_indptr, kv_indptr, kv_indices,
        (float)sm_scale, output, lse, tile_tensor
    );
    
    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "MLA Paged Prefill");
}
