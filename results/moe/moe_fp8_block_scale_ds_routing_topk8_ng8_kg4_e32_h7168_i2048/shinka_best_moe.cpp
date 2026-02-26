#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include "kernel.h"

torch::Tensor run(
    torch::Tensor routing_logits,       // [seq_len, 256] float
    torch::Tensor routing_bias,         // [256] bf16
    torch::Tensor hidden_states,        // [seq_len, 7168] fp8
    torch::Tensor hidden_states_scale,  // [56, seq_len] float
    torch::Tensor gemm1_weights,        // [32, 4096, 7168] fp8
    torch::Tensor gemm1_weights_scale,  // [32, 32, 56] float
    torch::Tensor gemm2_weights,        // [32, 7168, 2048] fp8
    torch::Tensor gemm2_weights_scale,  // [32, 56, 16] float
    int64_t local_expert_offset,
    double routed_scaling_factor
) {
    int seq_len = routing_logits.size(0);
    
    auto output = torch::zeros({seq_len, 7168}, 
        torch::TensorOptions().dtype(torch::kBFloat16).device(routing_logits.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(routing_logits.device().index()).stream();

    launch_deepseek_moe(
        seq_len,
        routing_logits.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(routing_bias.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr<at::Float8_e4m3fn>()),
        hidden_states_scale.data_ptr<float>(),
        reinterpret_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr<at::Float8_e4m3fn>()),
        gemm1_weights_scale.data_ptr<float>(),
        reinterpret_cast<const __nv_fp8_e4m3*>(gemm2_weights.data_ptr<at::Float8_e4m3fn>()),
        gemm2_weights_scale.data_ptr<float>(),
        (int)local_expert_offset,
        (float)routed_scaling_factor,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        stream
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "DeepSeek MoE Optimized");
}
