#include "kernel.h"
#include <stdio.h>

// -------------------------------------------------------------------------
// Helper Functions
// -------------------------------------------------------------------------

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu(float x) {
    return x * sigmoid(x);
}

__device__ __forceinline__ float fp82float(__nv_fp8_e4m3 x) {
    return float(x);
}

// -------------------------------------------------------------------------
// Routing Kernel
// -------------------------------------------------------------------------
// Computes TopK routing and populates per-expert work queues.
// No CPU sync required; fills counts and indices on device.
__global__ void routing_kernel(
    int seq_len,
    const float* __restrict__ logits,
    const __nv_bfloat16* __restrict__ bias,
    int local_expert_offset,
    float routed_scaling_factor,
    int* __restrict__ expert_counts,       // [32]
    int* __restrict__ expert_indices,      // [32, seq_len]
    float* __restrict__ expert_weights     // [32, seq_len]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seq_len) return;

    // 1. Load and Score
    float scores[256];
    float scores_biased[256];
    
    for (int i = 0; i < 256; ++i) {
        float l = logits[tid * 256 + i];
        float s = sigmoid(l);
        scores[i] = s;
        scores_biased[i] = s + __bfloat162float(bias[i]);
    }

    // 2. Group Top2
    float group_scores[8];
    for (int g = 0; g < 8; ++g) {
        int start = g * 32;
        float max1 = -1e20f, max2 = -1e20f;
        for (int i = 0; i < 32; ++i) {
            float val = scores_biased[start + i];
            if (val > max1) {
                max2 = max1;
                max1 = val;
            } else if (val > max2) {
                max2 = val;
            }
        }
        group_scores[g] = max1 + max2;
    }

    // 3. Top 4 Groups
    int top_groups[4];
    bool used[8] = {0};
    for (int i = 0; i < 4; ++i) {
        float best_val = -1e20f;
        int best_idx = -1;
        for (int g = 0; g < 8; ++g) {
            if (!used[g] && group_scores[g] > best_val) {
                best_val = group_scores[g];
                best_idx = g;
            }
        }
        top_groups[i] = best_idx;
        used[best_idx] = true;
    }

    // 4. Collect Top 8 Experts
    struct ExpertCand { int id; float val; };
    ExpertCand candidates[128]; 
    int cand_count = 0;

    for (int i = 0; i < 4; ++i) {
        int g = top_groups[i];
        int start = g * 32;
        if (g != -1) {
            for (int j = 0; j < 32; ++j) {
                candidates[cand_count++] = {start + j, scores_biased[start + j]};
            }
        }
    }

    ExpertCand top8[8];
    for(int k=0; k<8; ++k) {
        float best_val = -1e20f;
        int best_idx = -1;
        int cand_idx = -1;
        
        for(int c=0; c<cand_count; ++c) {
            if(candidates[c].val > best_val) {
                best_val = candidates[c].val;
                best_idx = candidates[c].id;
                cand_idx = c;
            }
        }
        top8[k] = {best_idx, best_val};
        if (cand_idx != -1) candidates[cand_idx].val = -1e20f; 
    }

    // 5. Normalize
    float sum_w = 0.0f;
    for(int k=0; k<8; ++k) {
        if(top8[k].id != -1) {
            float w = scores[top8[k].id]; 
            sum_w += w;
            top8[k].val = w; 
        }
    }

    // 6. Output
    float inv_sum = 1.0f / (sum_w + 1e-6f);
    for(int k=0; k<8; ++k) {
        int gid = top8[k].id;
        if(gid == -1) continue;
        
        if (gid >= local_expert_offset && gid < local_expert_offset + 32) {
            int lid = gid - local_expert_offset;
            float final_w = (top8[k].val * inv_sum) * routed_scaling_factor;
            
            int pos = atomicAdd(&expert_counts[lid], 1);
            if (pos < seq_len) {
                expert_indices[lid * seq_len + pos] = tid;
                expert_weights[lid * seq_len + pos] = final_w;
            }
        }
    }
}

// -------------------------------------------------------------------------
// GEMM1: Fused Gate+Up -> SwiGLU
// -------------------------------------------------------------------------
// Grid: (seq_len * 256, 32, 1)
// Y-dim maps to expert. X-dim maps to token job (256 blocks per token).
__global__ void gemm1_swiglu_kernel(
    int seq_len,
    const int* __restrict__ expert_counts,
    const int* __restrict__ expert_indices, // [32, seq_len]
    const __nv_fp8_e4m3* __restrict__ hidden_states,
    const float* __restrict__ hidden_states_scale,
    const __nv_fp8_e4m3* __restrict__ w1,        // [32, 4096, 7168]
    const float* __restrict__ w1_scale,          // [32, 32, 56]
    float* __restrict__ intermediate_buffer      // [32, seq_len, 2048]
) {
    int expert_idx = blockIdx.y;
    int job_idx = blockIdx.x;
    
    // Check count for this expert (avoid launch overhead/host sync)
    int count = expert_counts[expert_idx];
    
    int token_job_idx = job_idx / 256;
    if (token_job_idx >= count) return;
    
    int chunk_idx = job_idx % 256;

    // Fetch actual token index
    int token_idx = expert_indices[expert_idx * seq_len + token_job_idx];
    
    // Shared Memory for Input X and Scales
    __shared__ __nv_fp8_e4m3 s_x[7168];
    __shared__ float s_x_scale[56];

    int tid = threadIdx.x;
    
    // Load Input
    for (int i = tid; i < 7168; i += 256) {
        s_x[i] = hidden_states[token_idx * 7168 + i];
    }
    if (tid < 56) {
        s_x_scale[tid] = hidden_states_scale[tid * seq_len + token_idx];
    }
    
    __syncthreads();

    // Compute 8 pairs (16 rows)
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    int pair_base = chunk_idx * 8;
    int pair_offset = warp_id; 
    int pair_idx = pair_base + pair_offset; // 0..2047
    
    int row_g = pair_idx;
    int row_u = pair_idx + 2048;

    int row_blk_g = row_g / 128;
    int row_blk_u = row_u / 128;
    
    float sum_g = 0.0f;
    float sum_u = 0.0f;

    const __nv_fp8_e4m3* w_base = w1 + (size_t)expert_idx * 4096 * 7168;
    const __nv_fp8_e4m3* wg_ptr = w_base + row_g * 7168;
    const __nv_fp8_e4m3* wu_ptr = w_base + row_u * 7168;
    
    const float* s_base = w1_scale + expert_idx * 32 * 56;

    for (int k_base = 0; k_base < 7168; k_base += 32) {
        int k = k_base + lane_id;
        int kb = k / 128;
        
        float scale_x = s_x_scale[kb];
        float val_x = fp82float(s_x[k]) * scale_x;

        float scale_wg = s_base[row_blk_g * 56 + kb];
        float val_wg = fp82float(wg_ptr[k]) * scale_wg;
        
        float scale_wu = s_base[row_blk_u * 56 + kb];
        float val_wu = fp82float(wu_ptr[k]) * scale_wu;

        sum_g += val_x * val_wg;
        sum_u += val_x * val_wu;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum_g += __shfl_down_sync(0xffffffff, sum_g, offset);
        sum_u += __shfl_down_sync(0xffffffff, sum_u, offset);
    }

    if (lane_id == 0) {
        float res = silu(sum_u) * sum_g; 
        // Write to expert-specific buffer
        // Offset: expert * seq_len * 2048 + token_job * 2048 + pair
        float* dst = intermediate_buffer + (size_t)expert_idx * seq_len * 2048;
        dst[token_job_idx * 2048 + pair_idx] = res;
    }
}

// -------------------------------------------------------------------------
// GEMM2: Accumulate
// -------------------------------------------------------------------------
// Grid: (seq_len * 896, 32, 1)
// 896 blocks per token to cover 7168 outputs.
__global__ void gemm2_kernel(
    int seq_len,
    const int* __restrict__ expert_counts,
    const int* __restrict__ expert_indices,
    const float* __restrict__ expert_weights,
    const float* __restrict__ intermediate_buffer, 
    const __nv_fp8_e4m3* __restrict__ w2,          // [32, 7168, 2048]
    const float* __restrict__ w2_scale,            // [32, 56, 16]
    __nv_bfloat16* __restrict__ output             // [Seq, 7168]
) {
    int expert_idx = blockIdx.y;
    int job_idx = blockIdx.x;
    
    int count = expert_counts[expert_idx];
    
    int token_job_idx = job_idx / 896;
    if (token_job_idx >= count) return;
    
    int chunk_idx = job_idx % 896;
    
    int token_idx = expert_indices[expert_idx * seq_len + token_job_idx];
    float routing_w = expert_weights[expert_idx * seq_len + token_job_idx];

    __shared__ float s_in[2048];
    
    int tid = threadIdx.x;
    
    // Load Intermediate
    const float* src = intermediate_buffer + (size_t)expert_idx * seq_len * 2048 + token_job_idx * 2048;
    for (int i = tid; i < 2048; i += 256) {
        s_in[i] = src[i];
    }
    __syncthreads();

    // Compute 8 rows
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    int row_base = chunk_idx * 8;
    int row = row_base + warp_id; 

    int row_blk = row / 128;
    float sum = 0.0f;

    const __nv_fp8_e4m3* w_base = w2 + (size_t)expert_idx * 7168 * 2048;
    const __nv_fp8_e4m3* w_ptr = w_base + row * 2048;
    
    const float* s_base = w2_scale + expert_idx * 56 * 16;

    for (int k_base = 0; k_base < 2048; k_base += 32) {
        int k = k_base + lane_id;
        int kb = k / 128;
        
        float val_in = s_in[k];
        float scale = s_base[row_blk * 16 + kb];
        float val_w = fp82float(w_ptr[k]) * scale;
        
        sum += val_in * val_w;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        float final_val = sum * routing_w;
#if __CUDA_ARCH__ >= 800
        atomicAdd((__nv_bfloat16*)&output[token_idx * 7168 + row], __float2bfloat16(final_val));
#else
        // Fallback CAS loop
        unsigned short* addr = (unsigned short*)&output[token_idx * 7168 + row];
        unsigned short assumed, old = *addr;
        do {
            assumed = old;
            float current = __bfloat162float(*(__nv_bfloat16*)&assumed);
            __nv_bfloat16 next = __float2bfloat16(current + final_val);
            old = atomicCAS(addr, assumed, *(unsigned short*)&next);
        } while (assumed != old);
#endif
    }
}

// -------------------------------------------------------------------------
// Host Launcher
// -------------------------------------------------------------------------
void launch_deepseek_moe(
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
    cudaStream_t stream
) {
    int* d_counts;
    int* d_indices;
    float* d_weights;
    float* d_inter_buf; 

    // Max intermediate size: 32 experts * seq_len * 2048
    size_t inter_size = 32 * (size_t)seq_len * 2048 * sizeof(float);
    
    cudaMallocAsync(&d_counts, 32 * sizeof(int), stream);
    cudaMallocAsync(&d_indices, 32 * seq_len * sizeof(int), stream);
    cudaMallocAsync(&d_weights, 32 * seq_len * sizeof(float), stream);
    cudaMallocAsync(&d_inter_buf, inter_size, stream);

    cudaMemsetAsync(d_counts, 0, 32 * sizeof(int), stream);
    cudaMemsetAsync(output, 0, seq_len * 7168 * sizeof(__nv_bfloat16), stream);

    // 1. Routing (fills counts/indices on device)
    int r_threads = 256;
    int r_blocks = (seq_len + r_threads - 1) / r_threads;
    routing_kernel<<<r_blocks, r_threads, 0, stream>>>(
        seq_len, routing_logits, routing_bias, local_expert_offset, routed_scaling_factor,
        d_counts, d_indices, d_weights
    );

    // 2. GEMM1: Launch Global Grid covering all experts
    // Grid: (seq_len * 256, 32)
    dim3 grid1(seq_len * 256, 32);
    gemm1_swiglu_kernel<<<grid1, 256, 0, stream>>>(
        seq_len, d_counts, d_indices,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, d_inter_buf
    );
            
    // 3. GEMM2: Launch Global Grid covering all experts
    // Grid: (seq_len * 896, 32)
    dim3 grid2(seq_len * 896, 32);
    gemm2_kernel<<<grid2, 256, 0, stream>>>(
        seq_len, d_counts, d_indices, d_weights,
        d_inter_buf,
        gemm2_weights, gemm2_weights_scale, output
    );

    cudaFreeAsync(d_counts, stream);
    cudaFreeAsync(d_indices, stream);
    cudaFreeAsync(d_weights, stream);
    cudaFreeAsync(d_inter_buf, stream);
}
