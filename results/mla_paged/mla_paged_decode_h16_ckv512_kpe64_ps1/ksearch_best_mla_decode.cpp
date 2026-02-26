#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include "ksearch_best_mla_decode.h"

std::map<std::string, torch::Tensor> run_dict(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float sm_scale
) {
    q_nope = q_nope.contiguous();
    q_pe = q_pe.contiguous();
    ckv_cache = ckv_cache.contiguous();
    kpe_cache = kpe_cache.contiguous();
    kv_indptr = kv_indptr.contiguous();
    kv_indices = kv_indices.contiguous();

    int batch_size = q_nope.size(0);
    int num_heads = q_nope.size(1);
    int head_dim = q_nope.size(2);

    auto options = q_nope.options();
    auto output = torch::zeros({batch_size, num_heads, head_dim}, options);
    auto lse = torch::empty({batch_size, num_heads}, options.dtype(torch::kFloat32));

    auto kv_indptr_i32 = kv_indptr.to(torch::kInt);
    auto kv_indices_i32 = kv_indices.to(torch::kInt);

    run_mla_decode(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        kv_indptr_i32,
        kv_indices_i32,
        output,
        lse,
        sm_scale
    );

    std::map<std::string, torch::Tensor> result;
    result["output"] = output;
    result["lse"] = lse;
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_dict, "MLA Decode Kernel",
          py::arg("q_nope"), py::arg("q_pe"),
          py::arg("ckv_cache"), py::arg("kpe_cache"),
          py::arg("kv_indptr"), py::arg("kv_indices"),
          py::arg("sm_scale"));
}
