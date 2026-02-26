#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h> 
#include "kernel.h"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor run(
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
    auto seq_len = routing_logits.size(0);
    auto hidden_size = hidden_states.size(1);
    
    CHECK_CONTIGUOUS(routing_logits);
    if (routing_bias.defined()) CHECK_CONTIGUOUS(routing_bias);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_CONTIGUOUS(hidden_states_scale);
    CHECK_CONTIGUOUS(gemm1_weights);
    CHECK_CONTIGUOUS(gemm1_weights_scale);
    CHECK_CONTIGUOUS(gemm2_weights);
    CHECK_CONTIGUOUS(gemm2_weights_scale);

    auto device = routing_logits.device();
    auto output = torch::empty({seq_len, hidden_size}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    
    auto topk_ids = torch::empty({seq_len, TOP_K}, options_int);
    auto topk_weights = torch::empty({seq_len, TOP_K}, options_float);
    auto expert_counts = torch::empty({NUM_LOCAL_EXPERTS}, options_int);
    auto expert_offsets = torch::empty({NUM_LOCAL_EXPERTS}, options_int);
    
    auto sorted_token_ids = torch::empty({seq_len * TOP_K}, options_int);
    auto sorted_weights = torch::empty({seq_len * TOP_K}, options_float);
    
    auto intermediate_buffer = torch::empty({seq_len * TOP_K, INTERMEDIATE_SIZE}, options_bf16);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    const __nv_bfloat16* bias_ptr = nullptr;
    if (routing_bias.defined() && routing_bias.numel() > 0) {
        bias_ptr = reinterpret_cast<const __nv_bfloat16*>(routing_bias.data_ptr<at::BFloat16>());
    }
    
    launch_moe_pipeline(
        routing_logits.data_ptr<float>(),
        bias_ptr,
        reinterpret_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
        hidden_states_scale.data_ptr<float>(),
        reinterpret_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr()),
        gemm1_weights_scale.data_ptr<float>(),
        reinterpret_cast<const __nv_fp8_e4m3*>(gemm2_weights.data_ptr()),
        gemm2_weights_scale.data_ptr<float>(),
        (float)routed_scaling_factor,
        (int)seq_len,
        local_expert_offset,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        topk_ids.data_ptr<int>(),
        topk_weights.data_ptr<float>(),
        expert_counts.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        sorted_token_ids.data_ptr<int>(),
        sorted_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(intermediate_buffer.data_ptr<at::BFloat16>()),
        stream
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "MoE Pipeline");
}
