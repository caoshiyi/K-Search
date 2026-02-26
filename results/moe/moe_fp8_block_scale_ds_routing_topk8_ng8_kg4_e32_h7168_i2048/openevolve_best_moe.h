#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

void run_moe(
    int seq_len,
    const float* routing_logits,
    const __nv_bfloat16* routing_bias,
    const __nv_fp8_e4m3* hidden_states,
    const float* hidden_states_scale,
    const __nv_fp8_e4m3* gemm1_weights,
    const float* gemm1_weights_scale,
    const __nv_fp8_e4m3* gemm2_weights,
    const float* gemm2_weights_scale,
    int local_expert_offset,
    float routed_scaling_factor,
    __nv_bfloat16* output,
    void* workspace,
    cudaStream_t stream
);

size_t get_workspace_size(int seq_len);
