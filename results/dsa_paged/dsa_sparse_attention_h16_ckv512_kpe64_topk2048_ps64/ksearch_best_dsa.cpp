#include <torch/extension.h>
#include "kernel.h"

std::vector<torch::Tensor> run(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    double sm_scale
) {
    TORCH_CHECK(q_nope.is_cuda(), "q_nope must be a CUDA tensor");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be a CUDA tensor");
    TORCH_CHECK(ckv_cache.is_cuda(), "ckv_cache must be a CUDA tensor");
    TORCH_CHECK(kpe_cache.is_cuda(), "kpe_cache must be a CUDA tensor");
    TORCH_CHECK(sparse_indices.is_cuda(), "sparse_indices must be a CUDA tensor");
    
    TORCH_CHECK(q_nope.is_contiguous(), "q_nope must be contiguous");
    TORCH_CHECK(q_pe.is_contiguous(), "q_pe must be contiguous");
    TORCH_CHECK(ckv_cache.is_contiguous(), "ckv_cache must be contiguous");
    TORCH_CHECK(kpe_cache.is_contiguous(), "kpe_cache must be contiguous");
    TORCH_CHECK(sparse_indices.is_contiguous(), "sparse_indices must be contiguous");

    TORCH_CHECK(q_nope.size(1) == 16, "num_qo_heads must be 16");

    int num_tokens = q_nope.size(0);
    int num_qo_heads = q_nope.size(1);
    int head_dim_ckv = q_nope.size(2);
    
    auto opts_bf16 = q_nope.options();
    auto opts_fp32 = q_nope.options().dtype(torch::kFloat32);
    
    auto output = torch::empty({num_tokens, num_qo_heads, head_dim_ckv}, opts_bf16);
    auto lse = torch::empty({num_tokens, num_qo_heads}, opts_fp32);

    run_dsa_attn(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        (float)sm_scale, output, lse
    );

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "DSA Sparse Attention Run");
}