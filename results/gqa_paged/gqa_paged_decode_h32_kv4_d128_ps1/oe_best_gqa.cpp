#include <torch/extension.h>
#include "kernel.h"

std::vector<torch::Tensor> run(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale
) {
    int batch_size = q.size(0);
    int num_qo_heads = q.size(1);
    int head_dim = q.size(2);
    
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    torch::Tensor output = torch::empty({batch_size, num_qo_heads, head_dim}, options);
    
    auto options_lse = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    torch::Tensor lse = torch::empty({batch_size, num_qo_heads}, options_lse);

    run_gqa_decode(q, k_cache, v_cache, kv_indptr, kv_indices, (float)sm_scale, output, lse);

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "GQA Paged Decode Optimized");
}
