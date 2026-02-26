#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "kernel.h"

torch::Tensor run_torch(
    torch::Tensor routing_logits,
    torch::Tensor routing_bias,
    torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale,
    torch::Tensor gemm1_weights,
    torch::Tensor gemm1_weights_scale,
    torch::Tensor gemm2_weights,
    torch::Tensor gemm2_weights_scale,
    int local_expert_offset,
    double routed_scaling_factor
) {
    int seq_len = routing_logits.size(0);
    size_t ws_size = get_workspace_size(seq_len);
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(routing_logits.device());
    auto workspace = torch::empty({(long)ws_size}, options);
    
    auto out_options = torch::TensorOptions().dtype(torch::kBFloat16).device(routing_logits.device());
    auto output = torch::zeros({seq_len, 7168}, out_options);

    gemm1_weights = gemm1_weights.contiguous();
    gemm2_weights = gemm2_weights.contiguous();
    hidden_states = hidden_states.contiguous();
    hidden_states_scale = hidden_states_scale.contiguous();
    routing_logits = routing_logits.contiguous();
    
    run_moe(
        seq_len,
        routing_logits.data_ptr<float>(),
        (const __nv_bfloat16*)routing_bias.data_ptr<at::BFloat16>(),
        (const __nv_fp8_e4m3*)hidden_states.data_ptr(),
        hidden_states_scale.data_ptr<float>(),
        (const __nv_fp8_e4m3*)gemm1_weights.data_ptr(),
        gemm1_weights_scale.data_ptr<float>(),
        (const __nv_fp8_e4m3*)gemm2_weights.data_ptr(),
        gemm2_weights_scale.data_ptr<float>(),
        local_expert_offset,
        (float)routed_scaling_factor,
        (__nv_bfloat16*)output.data_ptr<at::BFloat16>(),
        workspace.data_ptr(),
        at::cuda::getCurrentCUDAStream().stream()
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_torch, "FP8 MoE Block Scale",
        py::arg("routing_logits"),
        py::arg("routing_bias"),
        py::arg("hidden_states"),
        py::arg("hidden_states_scale"),
        py::arg("gemm1_weights"),
        py::arg("gemm1_weights_scale"),
        py::arg("gemm2_weights"),
        py::arg("gemm2_weights_scale"),
        py::arg("local_expert_offset"),
        py::arg("routed_scaling_factor")
    );
}
