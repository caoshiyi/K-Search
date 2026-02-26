#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define NUM_EXPERTS 256
#define NUM_LOCAL_EXPERTS 32
#define TOP_K 8
#define TOP_K_GROUP 4
#define HIDDEN_SIZE 7168
#define INTERMEDIATE_SIZE 2048

// GEMM1: 32 tiles per expert * 32 experts = 1024 blocks
#define GEMM1_GRID_X 32
#define GEMM1_GRID_Y NUM_LOCAL_EXPERTS

// GEMM2: 112 tiles per expert * 32 experts = 3584 blocks
#define GEMM2_GRID_SIZE 3584

#define SMEM_PAD_K 72

void launch_moe_pipeline(
    const float* routing_logits,
    const __nv_bfloat16* routing_bias,
    const __nv_fp8_e4m3* hidden_states,
    const float* hidden_states_scale,
    const __nv_fp8_e4m3* gemm1_weights,
    const float* gemm1_weights_scale,
    const __nv_fp8_e4m3* gemm2_weights,
    const float* gemm2_weights_scale,
    float routed_scaling_factor,
    int seq_len,
    int local_expert_offset,
    __nv_bfloat16* output,
    int* topk_ids,
    float* topk_weights,
    int* expert_counts,
    int* expert_offsets,
    int* sorted_token_ids,
    float* sorted_weights,
    __nv_bfloat16* intermediate_buffer, 
    cudaStream_t stream
);
