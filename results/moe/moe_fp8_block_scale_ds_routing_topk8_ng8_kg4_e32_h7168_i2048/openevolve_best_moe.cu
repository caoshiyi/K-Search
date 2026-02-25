#include "kernel.h"
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define NUM_EXPERTS 256
#define NUM_LOCAL_EXPERTS 32
#define TILE_M 16
#define TILE_N 128
#define TILE_K 128
#define SMEM_PAD 8
#define SMEM_STRIDE (TILE_K + SMEM_PAD)
// Pad inter stride to avoid bank conflicts (4096 + 8 = 4104 elements)
// 4104 * 2 bytes = 8208 bytes. 8208 % 128 = 16 (Bank offset 4).
#define INTER_PAD 8
#define INTER_STRIDE (4096 + INTER_PAD)

// Helper for non-coherent loads to bypass L1 cache
__device__ __forceinline__ int4 load_int4_nc(const void* ptr) {
    int4 v;
    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];" 
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) 
        : "l"(ptr));
    return v;
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu(float x) {
    return x * sigmoid(x);
}

__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_e4m3 val, float scale) {
    return __float2bfloat16((float)val * scale);
}

// -------------------------------------------------------------------------
// Routing Kernel (Aggregated: 8 tokens per block)
// -------------------------------------------------------------------------
__global__ void routing_kernel_optimized(
    int seq_len,
    const float* __restrict__ logits,
    const __nv_bfloat16* __restrict__ bias,
    int local_expert_offset,
    float routed_scaling_factor,
    int* __restrict__ expert_counts,
    int* __restrict__ expert_indices,
    float* __restrict__ expert_weights
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int token_idx = blockIdx.x * 8 + warp_id;
    
    // Shared memory for aggregating counts within the block
    __shared__ int smem_counts[32];
    if (threadIdx.x < 32) smem_counts[threadIdx.x] = 0;
    __syncthreads();

    int selected_ids[8];
    float selected_probs[8];
    float scale = 0.0f;

    if (token_idx < seq_len) {
        float my_scores[8];
        float my_probs[8];
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            int e_idx = lane_id + g * 32;
            float l = logits[token_idx * NUM_EXPERTS + e_idx];
            float b = __bfloat162float(bias[e_idx]);
            float s = sigmoid(l);
            my_probs[g] = s;
            my_scores[g] = s + b;
        }

        float group_scores[8];
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            float val = my_scores[g];
            float max1 = val, max2 = -1e9f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                float other = __shfl_xor_sync(0xFFFFFFFF, max1, offset);
                float other2 = __shfl_xor_sync(0xFFFFFFFF, max2, offset);
                if (other > max1) { max2 = fmaxf(max1, other2); max1 = other; }
                else { max2 = fmaxf(max2, other); }
            }
            group_scores[g] = __shfl_sync(0xFFFFFFFF, max1 + max2, 0);
        }

        int top_groups[4];
        float gs[8];
        for(int i=0; i<8; ++i) gs[i] = group_scores[i];
        for(int k=0; k<4; ++k) {
            float m = -1e9f; int idx = -1;
            for(int i=0; i<8; ++i) if (gs[i] > m) { m = gs[i]; idx = i; }
            top_groups[k] = idx;
            if (idx != -1) gs[idx] = -1e9f;
        }

        float cand_scores[4], cand_probs[4];
        int cand_ids[4];
        for(int k=0; k<4; ++k) {
            int g = top_groups[k];
            if (g != -1) {
                cand_scores[k] = my_scores[g];
                cand_probs[k] = my_probs[g];
                cand_ids[k] = lane_id + g * 32;
            } else {
                cand_scores[k] = -1e9f; cand_probs[k] = 0.0f; cand_ids[k] = -1;
            }
        }

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float local_max = -1e9f; int local_idx = -1;
            for (int i = 0; i < 4; ++i) if (cand_scores[i] > local_max) { local_max = cand_scores[i]; local_idx = i; }
            
            float global_max = local_max;
            int winner_lane = lane_id;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                float other = __shfl_xor_sync(0xFFFFFFFF, global_max, offset);
                int other_winner = __shfl_xor_sync(0xFFFFFFFF, winner_lane, offset);
                if (other > global_max) { global_max = other; winner_lane = other_winner; }
                else if (other == global_max && other_winner < winner_lane) winner_lane = other_winner;
            }
            winner_lane = __shfl_sync(0xFFFFFFFF, winner_lane, 0);

            int win_id = -1; float win_prob = 0.0f;
            if (lane_id == winner_lane) {
                win_id = cand_ids[local_idx]; win_prob = cand_probs[local_idx]; cand_scores[local_idx] = -1e9f;
            }
            selected_ids[k] = __shfl_sync(0xFFFFFFFF, win_id, winner_lane);
            selected_probs[k] = __shfl_sync(0xFFFFFFFF, win_prob, winner_lane);
        }

        float sum = 0.0f;
        for(int k=0; k<8; ++k) if(selected_ids[k] != -1) sum += selected_probs[k];
        scale = routed_scaling_factor / (sum + 1e-6f);
    }

    int my_offsets[8];
    if (token_idx < seq_len && lane_id == 0) {
        for (int k = 0; k < 8; ++k) {
            int e = selected_ids[k];
            if (e >= local_expert_offset && e < local_expert_offset + 32) {
                int local_idx = e - local_expert_offset;
                my_offsets[k] = atomicAdd(&smem_counts[local_idx], 1);
            } else {
                my_offsets[k] = -1;
            }
        }
    }
    __syncthreads();

    __shared__ int global_bases[32];
    if (threadIdx.x < 32) {
        int c = smem_counts[threadIdx.x];
        if (c > 0) {
            global_bases[threadIdx.x] = atomicAdd(&expert_counts[threadIdx.x], c);
        }
    }
    __syncthreads();

    if (token_idx < seq_len && lane_id == 0) {
        for (int k = 0; k < 8; ++k) {
            if (my_offsets[k] != -1) {
                int e = selected_ids[k];
                int local_idx = e - local_expert_offset;
                int pos = global_bases[local_idx] + my_offsets[k];
                expert_indices[local_idx * seq_len + pos] = token_idx;
                expert_weights[local_idx * seq_len + pos] = selected_probs[k] * scale;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Scheduler Kernel
// -------------------------------------------------------------------------
__global__ void scheduler_kernel(int* counts, int* offsets, int* total_tiles, int* global_counter) {
    if (threadIdx.x == 0) {
        *global_counter = 0;
        int sum = 0;
        for(int i=0; i<32; ++i) {
            offsets[i] = sum;
            int c = counts[i];
            int tiles = (c + 15) / 16;
            sum += tiles;
        }
        offsets[32] = sum;
        *total_tiles = sum;
    }
}

// -------------------------------------------------------------------------
// Persistent GEMM Kernel (N=128, 256 Threads, Pipeline)
// -------------------------------------------------------------------------
__global__ void __launch_bounds__(256) fused_moe_kernel_persistent(
    const int* __restrict__ tile_offsets,
    const int* __restrict__ expert_counts,
    const int* __restrict__ total_tiles_ptr,
    int* __restrict__ global_task_counter,
    const int* __restrict__ expert_indices,
    const float* __restrict__ expert_weights,
    const __nv_fp8_e4m3* __restrict__ hidden_states,
    const float* __restrict__ hidden_scales,
    const __nv_fp8_e4m3* __restrict__ w1,
    const float* __restrict__ w1_scale,
    const __nv_fp8_e4m3* __restrict__ w2,
    const float* __restrict__ w2_scale,
    __nv_bfloat16* __restrict__ output,
    int seq_len
) {
    // Shared Memory Layout:
    // inter: [16, INTER_STRIDE] BF16 (128KB + Padding)
    // smem_a: [16, 128] BF16 (4KB)
    // smem_b: [128, 128] BF16 (32KB + Padding)
    // float_stage: [16, 128] Float (8KB)
    extern __shared__ char smem_raw[];
    __nv_bfloat16* inter = (__nv_bfloat16*)smem_raw; 
    __nv_bfloat16* smem_a = inter + 16 * INTER_STRIDE; 
    __nv_bfloat16* smem_b = smem_a + 16 * SMEM_STRIDE; 
    float* float_stage = (float*)(smem_b + 128 * SMEM_STRIDE);

    __shared__ int smem_offsets[33];
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    if (tid < 33) smem_offsets[tid] = tile_offsets[tid];
    __syncthreads();

    int total_tasks = smem_offsets[32];

    while (true) {
        __shared__ int task_id;
        if (tid == 0) task_id = atomicAdd(global_task_counter, 1);
        __syncthreads();
        
        if (task_id >= total_tasks) break;

        int expert_id = 0;
        for(int i=0; i<32; ++i) {
            if (task_id < smem_offsets[i+1]) {
                expert_id = i;
                break;
            }
        }
        int tile_idx = task_id - smem_offsets[expert_id];
        int count = expert_counts[expert_id];
        int start_token = tile_idx * TILE_M;
        int num_tokens = min(TILE_M, count - start_token);
        
        int indices[TILE_M];
        int* smem_indices_ptr = (int*)float_stage; // Reuse float stage for indices
        if (tid < num_tokens) smem_indices_ptr[tid] = expert_indices[expert_id * seq_len + start_token + tid];
        __syncthreads();
        #pragma unroll
        for(int i=0; i<TILE_M; ++i) indices[i] = (i < num_tokens) ? smem_indices_ptr[i] : 0;
        __syncthreads();

        // ----------------------------------------------------------------
        // GEMM 1: A[16, 7168] @ W1[32, 4096, 7168]^T -> [16, 4096]
        // ----------------------------------------------------------------
        for (int n = 0; n < 4096; n += 128) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            int4 a_reg;
            float a_scale;
            int4 b_regs[4];
            float b_scale;

            // Prologue: Load k=0
            {
                int k_blk = 0;
                
                // Load A (16 rows, 128 cols) - 128 threads load 1 int4 each
                if (tid < 128) {
                    int a_row = tid / 8;
                    int a_col = (tid % 8) * 16;
                    __nv_bfloat16* dest_a = smem_a + a_row * SMEM_STRIDE + a_col;
                    if (a_row < num_tokens) {
                        a_scale = hidden_scales[(k_blk/128)*seq_len + indices[a_row]];
                        a_reg = load_int4_nc(&hidden_states[indices[a_row]*7168 + k_blk + a_col]);
                        __nv_fp8_e4m3* vp = (__nv_fp8_e4m3*)&a_reg;
                        __nv_bfloat16 tmp[16];
                        #pragma unroll
                        for(int i=0; i<16; ++i) tmp[i] = fp8_to_bf16(vp[i], a_scale);
                        *(int4*)dest_a = *(int4*)&tmp[0];
                        *(int4*)(dest_a + 8) = *(int4*)&tmp[8];
                    } else {
                        *(int4*)dest_a = {0,0,0,0};
                        *(int4*)(dest_a + 8) = {0,0,0,0};
                    }
                }

                // Load B (128 rows, 128 cols) - 256 threads load 4 int4s each (64 elements)
                int b_row = tid / 2; // 0..127
                int b_col_base = (tid % 2) * 64; // 0 or 64
                b_scale = w1_scale[expert_id*32*56 + ((n+b_row)/128)*56 + (k_blk/128)];
                __nv_bfloat16* dest_b = smem_b + b_row * SMEM_STRIDE + b_col_base;
                
                #pragma unroll
                for(int i=0; i<4; ++i) {
                    b_regs[i] = load_int4_nc(&w1[expert_id*4096*7168 + (n + b_row)*7168 + k_blk + b_col_base + i*16]);
                    __nv_fp8_e4m3* vp = (__nv_fp8_e4m3*)&b_regs[i];
                    __nv_bfloat16 tmp[16];
                    #pragma unroll
                    for(int x=0; x<16; ++x) tmp[x] = fp8_to_bf16(vp[x], b_scale);
                    *(int4*)(dest_b + i*16) = *(int4*)&tmp[0];
                    *(int4*)(dest_b + i*16 + 8) = *(int4*)&tmp[8];
                }
            }
            __syncthreads();

            // Loop over K
            for (int k_blk = 0; k_blk < 7168; k_blk += 128) {
                int k_next = k_blk + 128;

                // 1. Load Next Tile to Registers
                if (k_next < 7168) {
                    // Load A
                    if (tid < 128) {
                        int a_row = tid / 8;
                        int a_col = (tid % 8) * 16;
                        if (a_row < num_tokens) {
                            a_scale = hidden_scales[(k_next/128)*seq_len + indices[a_row]];
                            a_reg = load_int4_nc(&hidden_states[indices[a_row]*7168 + k_next + a_col]);
                        }
                    }

                    // Load B
                    int b_row = tid / 2;
                    int b_col_base = (tid % 2) * 64;
                    b_scale = w1_scale[expert_id*32*56 + ((n+b_row)/128)*56 + (k_next/128)];
                    #pragma unroll
                    for(int i=0; i<4; ++i) {
                        b_regs[i] = load_int4_nc(&w1[expert_id*4096*7168 + (n + b_row)*7168 + k_next + b_col_base + i*16]);
                    }
                }

                // 2. Compute Current Tile
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;

                int n_offset = warp_id * 16;
                #pragma unroll
                for (int k = 0; k < 128; k += 16) {
                    wmma::load_matrix_sync(a_frag, smem_a + k, SMEM_STRIDE);
                    wmma::load_matrix_sync(b_frag, smem_b + k + n_offset * SMEM_STRIDE, SMEM_STRIDE);
                    wmma::mma_sync(acc, a_frag, b_frag, acc);
                }
                __syncthreads();

                // 3. Store Next Tile to Smem
                if (k_next < 7168) {
                    if (tid < 128) {
                        int a_row = tid / 8;
                        int a_col = (tid % 8) * 16;
                        __nv_bfloat16* dest_a = smem_a + a_row * SMEM_STRIDE + a_col;
                        if (a_row < num_tokens) {
                            __nv_fp8_e4m3* vp = (__nv_fp8_e4m3*)&a_reg;
                            __nv_bfloat16 tmp[16];
                            #pragma unroll
                            for(int i=0; i<16; ++i) tmp[i] = fp8_to_bf16(vp[i], a_scale);
                            *(int4*)dest_a = *(int4*)&tmp[0];
                            *(int4*)(dest_a + 8) = *(int4*)&tmp[8];
                        }
                    }

                    int b_row = tid / 2;
                    int b_col_base = (tid % 2) * 64;
                    __nv_bfloat16* dest_b = smem_b + b_row * SMEM_STRIDE + b_col_base;
                    #pragma unroll
                    for(int i=0; i<4; ++i) {
                        __nv_fp8_e4m3* vp = (__nv_fp8_e4m3*)&b_regs[i];
                        __nv_bfloat16 tmp[16];
                        #pragma unroll
                        for(int x=0; x<16; ++x) tmp[x] = fp8_to_bf16(vp[x], b_scale);
                        *(int4*)(dest_b + i*16) = *(int4*)&tmp[0];
                        *(int4*)(dest_b + i*16 + 8) = *(int4*)&tmp[8];
                    }
                    __syncthreads();
                }
            }

            // Store to float_stage
            int n_offset = warp_id * 16;
            wmma::store_matrix_sync(float_stage + n_offset, acc, 128, wmma::mem_row_major);
            __syncthreads();
            
            // Write to Inter
            for(int i = tid; i < 16 * 128; i += blockDim.x) {
                int r = i / 128; int c = i % 128;
                inter[r * INTER_STRIDE + n + c] = __float2bfloat16(float_stage[r * 128 + c]);
            }
            __syncthreads();
        }

        // ----------------------------------------------------------------
        // SwiGLU: C = silu(X2) * X1
        // ----------------------------------------------------------------
        for (int i = tid; i < 16 * 2048 / 8; i += blockDim.x) {
            int r = i / 256;
            int c = (i % 256) * 8;
            int idx_l = r * INTER_STRIDE + c;
            int idx_u = r * INTER_STRIDE + 2048 + c;
            
            float4 l_vec = *(float4*)&inter[idx_l];
            float4 u_vec = *(float4*)&inter[idx_u];
            __nv_bfloat16* l = (__nv_bfloat16*)&l_vec;
            __nv_bfloat16* u = (__nv_bfloat16*)&u_vec;
            
            #pragma unroll
            for(int k=0; k<8; ++k) {
                float lf = __bfloat162float(l[k]);
                float uf = __bfloat162float(u[k]);
                l[k] = __float2bfloat16(silu(uf) * lf);
            }
            *(float4*)&inter[idx_l] = l_vec;
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // GEMM 2: inter[16, 2048] @ W2[32, 7168, 2048]^T -> [16, 7168]
        // ----------------------------------------------------------------
        for (int n = 0; n < 7168; n += 128) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            int4 b_regs[4];
            float b_scale;

            // Prologue Load B (K=0)
            {
                int k_blk = 0;
                int b_row = tid / 2;
                int b_col_base = (tid % 2) * 64;
                b_scale = w2_scale[expert_id*56*16 + ((n+b_row)/128)*16 + (k_blk/128)];
                __nv_bfloat16* dest_b = smem_b + b_row * SMEM_STRIDE + b_col_base;
                
                #pragma unroll
                for(int i=0; i<4; ++i) {
                    b_regs[i] = load_int4_nc(&w2[expert_id*7168*2048 + (n + b_row)*2048 + k_blk + b_col_base + i*16]);
                    __nv_fp8_e4m3* vp = (__nv_fp8_e4m3*)&b_regs[i];
                    __nv_bfloat16 tmp[16];
                    #pragma unroll
                    for(int x=0; x<16; ++x) tmp[x] = fp8_to_bf16(vp[x], b_scale);
                    *(int4*)(dest_b + i*16) = *(int4*)&tmp[0];
                    *(int4*)(dest_b + i*16 + 8) = *(int4*)&tmp[8];
                }
            }
            __syncthreads();

            for (int k_blk = 0; k_blk < 2048; k_blk += 128) {
                int k_next = k_blk + 128;

                // 1. Load Next Tile to Registers
                if (k_next < 2048) {
                    int b_row = tid / 2;
                    int b_col_base = (tid % 2) * 64;
                    b_scale = w2_scale[expert_id*56*16 + ((n+b_row)/128)*16 + (k_next/128)];
                    #pragma unroll
                    for(int i=0; i<4; ++i) {
                        b_regs[i] = load_int4_nc(&w2[expert_id*7168*2048 + (n + b_row)*2048 + k_next + b_col_base + i*16]);
                    }
                }

                // 2. Compute Current Tile
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;

                int n_offset = warp_id * 16;
                #pragma unroll
                for (int k = 0; k < 128; k += 16) {
                    // Zero-Copy Load A from Inter
                    wmma::load_matrix_sync(a_frag, inter + k_blk + k, INTER_STRIDE);
                    wmma::load_matrix_sync(b_frag, smem_b + k + n_offset * SMEM_STRIDE, SMEM_STRIDE);
                    wmma::mma_sync(acc, a_frag, b_frag, acc);
                }
                __syncthreads();

                // 3. Store Next Tile to Smem
                if (k_next < 2048) {
                    int b_row = tid / 2;
                    int b_col_base = (tid % 2) * 64;
                    __nv_bfloat16* dest_b = smem_b + b_row * SMEM_STRIDE + b_col_base;
                    #pragma unroll
                    for(int i=0; i<4; ++i) {
                        __nv_fp8_e4m3* vp = (__nv_fp8_e4m3*)&b_regs[i];
                        __nv_bfloat16 tmp[16];
                        #pragma unroll
                        for(int x=0; x<16; ++x) tmp[x] = fp8_to_bf16(vp[x], b_scale);
                        *(int4*)(dest_b + i*16) = *(int4*)&tmp[0];
                        *(int4*)(dest_b + i*16 + 8) = *(int4*)&tmp[8];
                    }
                    __syncthreads();
                }
            }

            // Store weights for output scaling
            float* weights_storage = (float*)smem_a; 
            if (tid < num_tokens) weights_storage[tid] = expert_weights[expert_id * seq_len + start_token + tid];
            __syncthreads();

            int n_offset = warp_id * 16;
            wmma::store_matrix_sync(float_stage + n_offset, acc, 128, wmma::mem_row_major);
            __syncthreads();

            for(int i = tid; i < 16 * 128; i += blockDim.x) {
                int r = i / 128; int c = i % 128;
                if (r < num_tokens) {
                    float val = float_stage[r * 128 + c];
                    float w = weights_storage[r];
                    atomicAdd(&output[indices[r]*7168 + n + c], __float2bfloat16(val * w));
                }
            }
            __syncthreads();
        }
    }
}

size_t get_workspace_size(int seq_len) {
    size_t s = 0;
    s += 32 * sizeof(int); // counts
    s += 33 * sizeof(int); // offsets (32 + total)
    s += 256; // align
    s += 2 * sizeof(int); // total_tiles, global_counter
    s += 256; // align
    s += 32 * seq_len * sizeof(int); // indices
    s += 32 * seq_len * sizeof(float); // weights
    return s;
}

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
) {
    char* ptr = (char*)workspace;
    int* d_counts = (int*)ptr; ptr += 32 * sizeof(int);
    int* d_offsets = (int*)ptr; ptr += 33 * sizeof(int);
    ptr = (char*)(((size_t)ptr + 255) & ~255);
    int* d_total_tiles = (int*)ptr; ptr += sizeof(int);
    int* d_global_counter = (int*)ptr; ptr += sizeof(int);
    ptr = (char*)(((size_t)ptr + 255) & ~255);
    int* d_indices = (int*)ptr; ptr += 32 * seq_len * sizeof(int);
    float* d_weights = (float*)ptr;

    cudaMemsetAsync(d_counts, 0, 32 * sizeof(int), stream);
    
    // 1. Routing
    int routing_threads = 256;
    int routing_blocks = (seq_len + 7) / 8; // 8 tokens per block (1 per warp)
    routing_kernel_optimized<<<routing_blocks, routing_threads, 0, stream>>>(
        seq_len, routing_logits, routing_bias, local_expert_offset, routed_scaling_factor,
        d_counts, d_indices, d_weights
    );

    // 2. Scheduler
    scheduler_kernel<<<1, 1, 0, stream>>>(d_counts, d_offsets, d_total_tiles, d_global_counter);

    // 3. Persistent GEMM
    int num_persistent_blocks = 256;
    // Smem: Inter (132KB) + SmemA (4KB) + SmemB (32KB) + FloatStage (8KB) ~= 176KB
    int smem = 16 * INTER_STRIDE * 2 + 16 * SMEM_STRIDE * 2 + 128 * SMEM_STRIDE * 2 + 16 * 128 * 4 + 1024;
    cudaFuncSetAttribute(fused_moe_kernel_persistent, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    
    fused_moe_kernel_persistent<<<num_persistent_blocks, 256, smem, stream>>>(
        d_offsets, d_counts, d_total_tiles, d_global_counter, d_indices, d_weights,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        output, seq_len
    );
}
