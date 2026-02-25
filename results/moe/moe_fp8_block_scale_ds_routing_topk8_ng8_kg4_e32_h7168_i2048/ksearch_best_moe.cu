"""
World-model best kernel (gemini-3-pro-preview r90) for moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048. Same ShinkaEvolve contract: candidate_kernel() returns dict with mode, language, code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# EVOLVE-BLOCK-START
def candidate_kernel() -> Dict[str, Any]:
    
    kernel_h = r"""
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
"""

    kernel_cu = r"""
#include "kernel.h"
#include <mma.h>
#include <cmath>

using namespace nvcuda;

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_reduce_max(float val, int& src_lane) {
    int lane = threadIdx.x % 32;
    src_lane = lane;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        int other_lane = __shfl_down_sync(0xffffffff, src_lane, offset);
        if (other > val) {
            val = other;
            src_lane = other_lane;
        }
    }
    return val;
}

__device__ __forceinline__ void fast_atomic_add_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
    atomicAdd(address, val);
}

__device__ __forceinline__ int atomicAdd_int(int* address, int val) {
    return atomicAdd(address, val);
}

// -------------------------------------------------------------------------
// Routing Kernel (Optimized)
// -------------------------------------------------------------------------
__global__ void routing_kernel(
    const float* __restrict__ logits,
    const __nv_bfloat16* __restrict__ bias,
    float routed_scaling_factor,
    int seq_len,
    int local_expert_offset,
    int* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int* __restrict__ expert_counts,
    __nv_bfloat16* __restrict__ output
) {
    int token_idx = blockIdx.x;
    if (token_idx >= seq_len) return;
    int tid = threadIdx.x;
    
    // Fuse Zeroing of Output
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) {
        output[token_idx * HIDDEN_SIZE + i] = __float2bfloat16(0.0f);
    }
    
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Load Logits
    float l = logits[token_idx * NUM_EXPERTS + tid];
    float b = (bias) ? __bfloat162float(bias[tid]) : 0.0f;
    float s = sigmoid(l);
    float score_wb = s + b;

    // 1. Find Max-2 per warp (Group Max-2)
    // DeepSeek V3 Grouping: 8 groups of 32 experts. 
    // Each warp handles one group naturally (tid 0-31 -> Group 0, 32-63 -> Group 1...)
    // Since blockDim=256, we have 8 warps. Perfect.
    
    float my_score = score_wb;
    float max1 = my_score;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        max1 = fmaxf(max1, __shfl_down_sync(0xffffffff, max1, offset));
    max1 = __shfl_sync(0xffffffff, max1, 0);

    unsigned int mask = __ballot_sync(0xffffffff, my_score == max1);
    int first_winner = __ffs(mask) - 1;
    float val2 = (lane == first_winner) ? -INFINITY : my_score;

    float max2 = val2;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        max2 = fmaxf(max2, __shfl_down_sync(0xffffffff, max2, offset));
    max2 = __shfl_sync(0xffffffff, max2, 0);

    float group_sum = max1 + max2;
    
    // Share group scores
    __shared__ float smem_group_scores[8];
    if (lane == 0) smem_group_scores[warp_id] = group_sum;
    __syncthreads();

    // 2. Select Top-4 Groups
    __shared__ unsigned int group_mask;
    if (tid == 0) {
        float scores[8];
        #pragma unroll
        for(int i=0; i<8; ++i) scores[i] = smem_group_scores[i];
        unsigned int gmask = 0;
        for(int k=0; k<TOP_K_GROUP; ++k) {
            float best = -INFINITY;
            int best_idx = -1;
            for(int i=0; i<8; ++i) {
                if(scores[i] > best) { best = scores[i]; best_idx = i; }
            }
            if(best_idx != -1) {
                gmask |= (1 << best_idx);
                scores[best_idx] = -INFINITY;
            }
        }
        group_mask = gmask;
    }
    __syncthreads();

    // Mask out scores from non-selected groups
    if (!((group_mask >> warp_id) & 1)) {
        score_wb = -INFINITY;
    }

    // 3. Global Top-K Selection (Top 8 from 256)
    __shared__ int out_ids[TOP_K];
    __shared__ float out_sigs[TOP_K];

    for(int k=0; k<TOP_K; ++k) {
        int src_lane;
        float warp_max = warp_reduce_max(score_wb, src_lane);
        
        __shared__ float smem_warp_max[8];
        __shared__ int smem_warp_src[8];
        if (lane == 0) {
            smem_warp_max[warp_id] = warp_max;
            smem_warp_src[warp_id] = src_lane;
        }
        __syncthreads();

        // Warp 0 selects the global max among warps
        if (tid < 32) { // Use first warp to reduce
            if (tid == 0) {
                float global_max = -INFINITY;
                int global_warp = -1;
                for(int w=0; w<8; ++w) {
                    if (smem_warp_max[w] > global_max) {
                        global_max = smem_warp_max[w];
                        global_warp = w;
                    }
                }
                smem_warp_src[0] = global_warp * 32 + smem_warp_src[global_warp];
            }
        }
        __syncthreads();
        
        int winner_tid = smem_warp_src[0];
        if (tid == winner_tid) {
            out_ids[k] = tid;
            out_sigs[k] = s;
            score_wb = -INFINITY; // Remove from next iteration
        }
        __syncthreads();
    }

    // 4. Normalize and Store
    if (tid < TOP_K) {
        float weight = out_sigs[tid];
        float sum = 0.0f;
        for(int i=0; i<TOP_K; ++i) sum += out_sigs[i];
        if (sum > 1e-20f) weight /= sum;
        else weight = 1.0f;
        
        int expert = out_ids[tid];
        topk_ids[token_idx * TOP_K + tid] = expert;
        topk_weights[token_idx * TOP_K + tid] = weight * routed_scaling_factor;

        // Atomic count for local experts
        int local = expert - local_expert_offset;
        if (local >= 0 && local < NUM_LOCAL_EXPERTS) {
            atomicAdd_int(&expert_counts[local], 1);
        }
    }
}

// -------------------------------------------------------------------------
// Prefix Sum & Scatter Kernel
// -------------------------------------------------------------------------
__global__ void sort_scatter_kernel(
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int seq_len,
    int local_expert_offset,
    int* __restrict__ expert_counts,
    int* __restrict__ expert_offsets,
    int* __restrict__ sorted_token_ids,
    float* __restrict__ sorted_weights
) {
    __shared__ int smem_offsets[NUM_LOCAL_EXPERTS];
    int tid = threadIdx.x;
    
    // Exclusive prefix sum of counts
    if (tid < NUM_LOCAL_EXPERTS) {
        int count = expert_counts[tid];
        int sum = count;
        // Warp scan
        #pragma unroll
        for (int off = 1; off < 32; off *= 2) {
            int n = __shfl_up_sync(0xffffffff, sum, off);
            if (tid >= off) sum += n;
        }
        int offset = sum - count;
        expert_offsets[tid] = offset;
        smem_offsets[tid] = offset;
        expert_counts[tid] = 0; // Reset for atomic append
    }
    __syncthreads();

    // Scatter
    int total_elements = seq_len * TOP_K;
    for (int i = tid; i < total_elements; i += blockDim.x) {
        int expert = topk_ids[i];
        int local = expert - local_expert_offset;
        if (local >= 0 && local < NUM_LOCAL_EXPERTS) {
            int pos = atomicAdd_int(&expert_counts[local], 1);
            int dest = smem_offsets[local] + pos;
            sorted_token_ids[dest] = i / TOP_K;
            sorted_weights[dest] = topk_weights[i];
        }
    }
}

// -------------------------------------------------------------------------
// GEMM 1: FP8 -> BF16
// Grid: (32 tiles, 32 experts)
// -------------------------------------------------------------------------
__global__ __launch_bounds__(128)
void gemm1_kernel(
    const __nv_fp8_e4m3* __restrict__ hidden_states,
    const float* __restrict__ hidden_states_scale,
    const __nv_fp8_e4m3* __restrict__ gemm1_weights,
    const float* __restrict__ gemm1_weights_scale,
    const int* __restrict__ expert_counts,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    __nv_bfloat16* __restrict__ intermediate_buffer,
    int seq_len
) {
    extern __shared__ char smem_buf[];
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    // Size A: 2 buffers * 32 rows * 72 cols
    __nv_bfloat16* smem_w_g = smem_a + 2 * 32 * SMEM_PAD_K;
    // Size Wg: 2 buffers * 64 rows * 72 cols
    __nv_bfloat16* smem_w_u = smem_w_g + 2 * 64 * SMEM_PAD_K;

    int n_blk = blockIdx.x; // 0..31
    int expert_idx = blockIdx.y; // 0..31
    
    int num_tokens = expert_counts[expert_idx];
    if (num_tokens == 0) return;

    int gate_col_start = n_blk * 64;
    int up_col_start = gate_col_start + 2048;
    int scale_n_g = gate_col_start / 128;
    int scale_n_u = up_col_start / 128;
    
    int tid = threadIdx.x;
    int warp_row = (tid / 32) % 2 * 16;
    int warp_col = (tid / 32) / 2 * 32;

    for (int m_start = 0; m_start < num_tokens; m_start += 32) {
        int m_current = min(32, num_tokens - m_start);
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_g[2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_u[2];
        wmma::fill_fragment(acc_g[0], 0.0f); wmma::fill_fragment(acc_g[1], 0.0f);
        wmma::fill_fragment(acc_u[0], 0.0f); wmma::fill_fragment(acc_u[1], 0.0f);

        auto load_tile = [&](int buf_idx, int k_idx) {
            int k_base = k_idx * 64;
            int scale_idx = k_idx / 2;

            // Load A: [32, 64]
            int row = tid / 4; 
            int col = (tid % 4) * 16;
            if (row < 32) {
                float s = 0.0f;
                int4 val = make_int4(0,0,0,0);
                if (row < m_current) {
                    int token = sorted_token_ids[expert_offsets[expert_idx] + m_start + row];
                    s = hidden_states_scale[scale_idx * seq_len + token];
                    val = *reinterpret_cast<const int4*>(&hidden_states[token * HIDDEN_SIZE + k_base + col]);
                }
                __nv_fp8_e4m3 v[16]; *(int4*)v = val;
                __nv_bfloat16* d = &smem_a[buf_idx * 32 * SMEM_PAD_K + row * SMEM_PAD_K + col];
                #pragma unroll
                for(int i=0; i<16; ++i) d[i] = __float2bfloat16((float)v[i] * s);
            }

            // Load W: [64, 64]
            int t_row = tid / 4;   // 0..31
            int t_col = (tid % 4) * 16; // 0,16,32,48
            
            float sg = gemm1_weights_scale[expert_idx * 32 * 56 + scale_n_g * 56 + scale_idx];
            float su = gemm1_weights_scale[expert_idx * 32 * 56 + scale_n_u * 56 + scale_idx];
            
            #pragma unroll
            for(int r_off = 0; r_off < 64; r_off += 32) {
                int r = t_row + r_off;
                int c = t_col;
                
                long long off_g = (long long)(gate_col_start + r) * HIDDEN_SIZE + k_base + c;
                long long off_u = (long long)(up_col_start + r) * HIDDEN_SIZE + k_base + c;
                
                int4 lg = *reinterpret_cast<const int4*>(&gemm1_weights[((long long)expert_idx * 2 * INTERMEDIATE_SIZE * HIDDEN_SIZE) + off_g]);
                int4 lu = *reinterpret_cast<const int4*>(&gemm1_weights[((long long)expert_idx * 2 * INTERMEDIATE_SIZE * HIDDEN_SIZE) + off_u]);
                
                __nv_fp8_e4m3 vg[16]; *(int4*)vg = lg;
                __nv_fp8_e4m3 vu[16]; *(int4*)vu = lu;
                
                __nv_bfloat16* dg = &smem_w_g[buf_idx * 64 * SMEM_PAD_K + r * SMEM_PAD_K + c];
                __nv_bfloat16* du = &smem_w_u[buf_idx * 64 * SMEM_PAD_K + r * SMEM_PAD_K + c];
                
                #pragma unroll
                for(int i=0; i<16; ++i) {
                    dg[i] = __float2bfloat16((float)vg[i] * sg);
                    du[i] = __float2bfloat16((float)vu[i] * su);
                }
            }
        };

        load_tile(0, 0);
        __syncthreads();

        #pragma unroll 1
        for (int k_idx = 0; k_idx < 112; ++k_idx) {
            int buf_cur = k_idx % 2;
            int buf_next = (k_idx + 1) % 2;
            if (k_idx < 111) load_tile(buf_next, k_idx + 1);

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;

            #pragma unroll
            for (int ki = 0; ki < 64; ki += 16) {
                wmma::load_matrix_sync(fa, &smem_a[buf_cur * 32 * SMEM_PAD_K + warp_row * SMEM_PAD_K + ki], SMEM_PAD_K);
                
                wmma::load_matrix_sync(fb, &smem_w_g[buf_cur * 64 * SMEM_PAD_K + warp_col * SMEM_PAD_K + ki], SMEM_PAD_K);
                wmma::mma_sync(acc_g[0], fa, fb, acc_g[0]);
                wmma::load_matrix_sync(fb, &smem_w_g[buf_cur * 64 * SMEM_PAD_K + (warp_col+16) * SMEM_PAD_K + ki], SMEM_PAD_K);
                wmma::mma_sync(acc_g[1], fa, fb, acc_g[1]);
                
                wmma::load_matrix_sync(fb, &smem_w_u[buf_cur * 64 * SMEM_PAD_K + warp_col * SMEM_PAD_K + ki], SMEM_PAD_K);
                wmma::mma_sync(acc_u[0], fa, fb, acc_u[0]);
                wmma::load_matrix_sync(fb, &smem_w_u[buf_cur * 64 * SMEM_PAD_K + (warp_col+16) * SMEM_PAD_K + ki], SMEM_PAD_K);
                wmma::mma_sync(acc_u[1], fa, fb, acc_u[1]);
            }
            __syncthreads();
        }
        
        float* smem_out_g = reinterpret_cast<float*>(smem_w_g);
        float* smem_out_u = reinterpret_cast<float*>(smem_w_u);
        
        wmma::store_matrix_sync(smem_out_g + warp_row * 64 + warp_col, acc_g[0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(smem_out_g + warp_row * 64 + warp_col + 16, acc_g[1], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(smem_out_u + warp_row * 64 + warp_col, acc_u[0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(smem_out_u + warp_row * 64 + warp_col + 16, acc_u[1], 64, wmma::mem_row_major);
        __syncthreads();

        #pragma unroll
        for (int i=tid; i<2048; i+=128) {
            int r = i / 64;
            int c = i % 64;
            if (r < m_current) {
                float g = smem_out_g[i];
                float u = smem_out_u[i];
                float val = silu(u) * g;
                int dest = expert_offsets[expert_idx] + m_start + r;
                intermediate_buffer[dest * INTERMEDIATE_SIZE + gate_col_start + c] = __float2bfloat16(val);
            }
        }
        __syncthreads();
    }
}

// -------------------------------------------------------------------------
// GEMM 2: BF16 -> FP8 Weights -> BF16 Output
// Grid: (3584 tiles)
// -------------------------------------------------------------------------
__global__ __launch_bounds__(128)
void gemm2_kernel(
    const __nv_bfloat16* __restrict__ intermediate,
    const __nv_fp8_e4m3* __restrict__ gemm2_weights,
    const float* __restrict__ gemm2_weights_scale,
    const int* __restrict__ expert_counts,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    __nv_bfloat16* __restrict__ output,
    int seq_len
) {
    extern __shared__ char smem_buf[];
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* smem_w = smem_a + 2 * 32 * SMEM_PAD_K;
    
    int total_tiles = NUM_LOCAL_EXPERTS * (HIDDEN_SIZE / 64); // 3584
    int tid = threadIdx.x;
    int warp_row = (tid / 32) / 2 * 16;
    int warp_col = (tid / 32) % 2 * 32;

    for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        int expert_idx = tile_idx / 112; 
        int n_blk = tile_idx % 112;
        int num_tokens = expert_counts[expert_idx];
        if (num_tokens == 0) continue; 
        
        int col_start = n_blk * 64;
        int scale_n = col_start / 128;
        const float* scale_base = &gemm2_weights_scale[expert_idx * 56 * 16 + scale_n * 16];

        for (int m_start = 0; m_start < num_tokens; m_start += 32) {
            int m_current = min(32, num_tokens - m_start);
            
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
            wmma::fill_fragment(acc[0], 0.0f); wmma::fill_fragment(acc[1], 0.0f);

            auto load_tile = [&](int buf_idx, int k_idx) {
                int k_base = k_idx * 64;
                int scale_idx = k_idx / 2;

                // Load A: [32, 64]
                int row = tid / 4;
                int col = (tid % 4) * 16;
                if (row < 32) {
                    if (row < m_current) {
                        int off = expert_offsets[expert_idx] + m_start + row;
                        int4 v = *reinterpret_cast<const int4*>(&intermediate[off * INTERMEDIATE_SIZE + k_base + col]);
                        int4 v2 = *reinterpret_cast<const int4*>(&intermediate[off * INTERMEDIATE_SIZE + k_base + col + 8]);
                        
                        __nv_bfloat16* d = &smem_a[buf_idx * 32 * SMEM_PAD_K + row * SMEM_PAD_K + col];
                        *reinterpret_cast<int4*>(d) = v;
                        *reinterpret_cast<int4*>(d+8) = v2;
                    } else {
                        __nv_bfloat16* d = &smem_a[buf_idx * 32 * SMEM_PAD_K + row * SMEM_PAD_K + col];
                        *reinterpret_cast<int4*>(d) = make_int4(0,0,0,0);
                        *reinterpret_cast<int4*>(d+8) = make_int4(0,0,0,0);
                    }
                }

                // Load W: [64, 64]
                int t_row = tid / 4; 
                int t_col = (tid % 4) * 16;
                float s = scale_base[scale_idx];
                
                #pragma unroll
                for(int r_off = 0; r_off < 64; r_off += 32) {
                    int r = t_row + r_off;
                    int c = t_col;
                    
                    long long offset = ((long long)expert_idx * HIDDEN_SIZE + col_start + r) * INTERMEDIATE_SIZE + k_base + c;
                    int4 l = *reinterpret_cast<const int4*>(&gemm2_weights[offset]);
                    
                    __nv_fp8_e4m3 v[16]; *(int4*)v = l;
                    __nv_bfloat16* d = &smem_w[buf_idx * 64 * SMEM_PAD_K + r * SMEM_PAD_K + c];
                    
                    #pragma unroll
                    for(int i=0; i<16; ++i) d[i] = __float2bfloat16((float)v[i] * s);
                }
            };

            load_tile(0, 0);
            __syncthreads();

            #pragma unroll 1
            for (int k_idx = 0; k_idx < 32; ++k_idx) {
                int buf_cur = k_idx % 2;
                int buf_next = (k_idx + 1) % 2;
                if (k_idx < 31) load_tile(buf_next, k_idx + 1);

                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;

                #pragma unroll
                for (int ki = 0; ki < 64; ki += 16) {
                    wmma::load_matrix_sync(fa, &smem_a[buf_cur * 32 * SMEM_PAD_K + warp_row * SMEM_PAD_K + ki], SMEM_PAD_K);
                    wmma::load_matrix_sync(fb, &smem_w[buf_cur * 64 * SMEM_PAD_K + warp_col * SMEM_PAD_K + ki], SMEM_PAD_K);
                    wmma::mma_sync(acc[0], fa, fb, acc[0]);
                    wmma::load_matrix_sync(fb, &smem_w[buf_cur * 64 * SMEM_PAD_K + (warp_col+16) * SMEM_PAD_K + ki], SMEM_PAD_K);
                    wmma::mma_sync(acc[1], fa, fb, acc[1]);
                }
                __syncthreads();
            }

            float* smem_out = reinterpret_cast<float*>(smem_w);
            wmma::store_matrix_sync(smem_out + warp_row * 64 + warp_col, acc[0], 64, wmma::mem_row_major);
            wmma::store_matrix_sync(smem_out + warp_row * 64 + warp_col + 16, acc[1], 64, wmma::mem_row_major);
            __syncthreads();

            #pragma unroll
            for (int i = tid; i < 32 * 64; i += 128) {
                int r = i / 64;
                int c = i % 64;
                if (r < m_current) {
                    int slot = expert_offsets[expert_idx] + m_start + r;
                    int token = sorted_token_ids[slot];
                    float w = sorted_weights[slot];
                    float val = smem_out[i];
                    fast_atomic_add_bf16(&output[token * HIDDEN_SIZE + col_start + c], __float2bfloat16(val * w));
                }
            }
            __syncthreads();
        }
    }
}

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
) {
    if (seq_len == 0) return;
    
    cudaMemsetAsync(expert_counts, 0, NUM_LOCAL_EXPERTS * sizeof(int), stream);

    // 1. Routing
    routing_kernel<<<seq_len, 256, 0, stream>>>(
        routing_logits, routing_bias, routed_scaling_factor, seq_len, local_expert_offset,
        topk_ids, topk_weights, expert_counts, output
    );

    // 2. Sorting
    sort_scatter_kernel<<<1, 256, 0, stream>>>(
        topk_ids, topk_weights, seq_len, local_expert_offset, 
        expert_counts, expert_offsets, sorted_token_ids, sorted_weights
    );

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(gemm1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        cudaFuncSetAttribute(gemm2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
        configured = true;
    }

    // 3. GEMM1: Grid (32 tiles, 32 experts) = 1024 blocks
    // SMEM: 48000 bytes
    dim3 grid_g1(GEMM1_GRID_X, GEMM1_GRID_Y);
    gemm1_kernel<<<grid_g1, 128, 48000, stream>>>(
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        expert_counts, expert_offsets, sorted_token_ids,
        intermediate_buffer, seq_len
    );
    
    // 4. GEMM2: Grid 3584 blocks
    // SMEM: 32000 bytes
    gemm2_kernel<<<GEMM2_GRID_SIZE, 128, 32000, stream>>>(
        intermediate_buffer, gemm2_weights, gemm2_weights_scale,
        expert_counts, expert_offsets, sorted_token_ids, sorted_weights,
        output, seq_len
    );
}
"""

    main_cpp = r"""
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
"""

    return {
        "mode": "code",
        "language": "cuda",
        "code": f"""
<header_file name="kernel.h">
{kernel_h}
</header_file>
<cuda_file name="kernel.cu">
{kernel_cu}
</cuda_file>
<cpp_file name="main.cpp">
{main_cpp}
</cpp_file>
"""
    }
# EVOLVE-BLOCK-END