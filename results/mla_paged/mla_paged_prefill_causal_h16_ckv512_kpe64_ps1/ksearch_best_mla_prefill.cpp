#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cuda_runtime.h>
#include "ksearch_best_mla_prefill.h"

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
    q_nope = q_nope.contiguous();
    q_pe = q_pe.contiguous();
    ckv_cache = ckv_cache.contiguous();
    kpe_cache = kpe_cache.contiguous();
    qo_indptr = qo_indptr.contiguous();
    kv_indptr = kv_indptr.contiguous();
    kv_indices = kv_indices.contiguous();

    int total_q = q_nope.size(0);
    int num_heads = q_nope.size(1);
    int head_dim = q_nope.size(2);

    auto output = torch::empty({total_q, num_heads, head_dim}, q_nope.options());
    auto lse = torch::empty({total_q, num_heads}, q_nope.options().dtype(torch::kFloat32));

    launch_mla_paged(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        qo_indptr.to(torch::kInt),
        kv_indptr.to(torch::kInt),
        kv_indices.to(torch::kInt),
        output,
        lse,
        sm_scale
    );

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "MLA Paged Prefill",
          py::arg("q_nope"), py::arg("q_pe"),
          py::arg("ckv_cache"), py::arg("kpe_cache"),
          py::arg("qo_indptr"), py::arg("kv_indptr"),
          py::arg("kv_indices"), py::arg("sm_scale"));
}
