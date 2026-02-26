#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
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
    int num_kv_heads = k_cache.size(2);
    
    auto output = torch::empty({batch_size, num_qo_heads, head_dim}, q.options());
    auto lse = torch::empty({batch_size, num_qo_heads}, q.options().dtype(torch::kFloat32));
    
    launch_gqa_decode(
        (__nv_bfloat16*)output.data_ptr<at::BFloat16>(),
        lse.data_ptr<float>(),
        (const __nv_bfloat16*)q.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)k_cache.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)v_cache.data_ptr<at::BFloat16>(),
        kv_indptr.data_ptr<int32_t>(),
        kv_indices.data_ptr<int32_t>(),
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        (float)sm_scale,
        at::cuda::getCurrentCUDAStream().stream()
    );
    
    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "GQA Paged Decode");
}
