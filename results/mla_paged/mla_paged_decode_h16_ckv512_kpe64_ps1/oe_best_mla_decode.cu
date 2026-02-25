#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cmath>
#include <algorithm>

using namespace nvcuda;

// Tuning params for H100
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;

constexpr int NUM_HEADS = 16;
constexpr int HEAD_DIM_CKV = 512;
constexpr int HEAD_DIM_KPE = 64;

constexpr int CHUNK_SIZE = 64; 

// Memory Layout Constants
// Padding to avoid bank conflicts (512 + 8 = 520 elements)
constexpr int STRIDE_CKV = HEAD_DIM_CKV + 8; 
constexpr int STRIDE_KPE = HEAD_DIM_KPE + 8; 
constexpr int S_STRIDE = 72; // 64 + 8

// Padding for s_partials between warps to avoid bank conflicts during reduction
// 16 * 72 = 1152 floats. 1152 * 4 = 4608 bytes. 4608 % 128 = 0.
// We add 4 floats (16 bytes) padding so stride % 128 != 0.
constexpr int S_BUF_PAD = 4; 
constexpr int S_BUF_SIZE = 16 * S_STRIDE + S_BUF_PAD;

struct __align__(128) SharedStorage {
    // Q matrices
    __nv_bfloat16 q_nope[NUM_HEADS * STRIDE_CKV]; // 16KB
    __nv_bfloat16 q_pe[NUM_HEADS * STRIDE_KPE];   // 2KB
    
    // Indices Buffer (Double Buffered)
    int indices_buf[2][CHUNK_SIZE]; // 512B

    // KV Buffers (Double Buffered)
    __nv_bfloat16 kc_buf[2][CHUNK_SIZE * STRIDE_CKV]; // ~133KB
    __nv_bfloat16 kp_buf[2][CHUNK_SIZE * STRIDE_KPE]; // ~18KB
    
    // P Matrix for Softmax -> Output accumulation
    // Reuse stride KPE for padding
    __nv_bfloat16 p_mat[NUM_HEADS * STRIDE_KPE]; // ~2.3KB

    // Reused scratch memory
    union {
        // S Partials: [8 warps][16 heads * 72 stride + pad]
        float s_partials[NUM_WARPS][S_BUF_SIZE]; // ~37KB
        
        // Output exchange buffer for final reduction within block
        float o_exchange[NUM_HEADS * HEAD_DIM_CKV]; // 32KB
    } scratch;

    // Softmax statistics
    float lse_max[NUM_HEADS];
    float lse_sum[NUM_HEADS];
    float lse_inv_sum[NUM_HEADS]; // Precomputed inverse sum
    float broadcast_alpha[NUM_HEADS];
};

__device__ __forceinline__ void load_q(
    SharedStorage& smem,
    const __nv_bfloat16* __restrict__ qn_g,
    const __nv_bfloat16* __restrict__ qp_g,
    int tid
) {
    // Load Q_nope: [16, 512] -> [16, 520]
    const int4* src_qn = reinterpret_cast<const int4*>(qn_g);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = i * BLOCK_SIZE + tid;
        if (idx < NUM_HEADS * 64) { // 16 * 64 int4s = 1024
            int row = idx / 64;
            int col = idx % 64;
            reinterpret_cast<int4*>(smem.q_nope + row * STRIDE_CKV)[col] = src_qn[idx];
        }
    }
    
    // Load Q_pe: [16, 64] -> [16, 72]
    const int4* src_qp = reinterpret_cast<const int4*>(qp_g);
    // 16 * 8 int4s = 128
    if (tid < 128) { 
        int row = tid / 8;
        int col = tid % 8;
        reinterpret_cast<int4*>(smem.q_pe + row * STRIDE_KPE)[col] = src_qp[tid];
    }
}

__device__ __forceinline__ void load_indices(
    SharedStorage& smem,
    int buf_idx,
    const int* __restrict__ kv_indices,
    int offset,
    int valid_rows,
    int tid
) {
    if (tid < 64) {
        int val = 0;
        if (tid < valid_rows) {
            val = kv_indices[offset + tid];
        }
        smem.indices_buf[buf_idx][tid] = val;
    }
}

__device__ __forceinline__ void load_kv_chunk(
    SharedStorage& smem,
    int buf_idx,
    int valid_rows,
    const __nv_bfloat16* __restrict__ ckv_base,
    const __nv_bfloat16* __restrict__ kpe_base,
    int tid
) {
    // Load CKV: [64, 512]
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        int job_idx = tid + k * BLOCK_SIZE;
        int row = job_idx / 64; 
        int col_chunk = job_idx % 64;
        
        if (row < valid_rows) {
            int page_idx = smem.indices_buf[buf_idx][row];
            const void* src = ckv_base + (long long)page_idx * HEAD_DIM_CKV + col_chunk * 8;
            void* dst = smem.kc_buf[buf_idx] + row * STRIDE_CKV + col_chunk * 8;
            __pipeline_memcpy_async(dst, src, 16);
        }
    }
    
    // Load KPE: [64, 64]
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        int job_idx = tid + k * BLOCK_SIZE;
        if (job_idx < 512) {
            int row = job_idx / 8;
            int col_chunk = job_idx % 8;
            if (row < valid_rows) {
                int page_idx = smem.indices_buf[buf_idx][row];
                const void* src = kpe_base + (long long)page_idx * HEAD_DIM_KPE + col_chunk * 8;
                void* dst = smem.kp_buf[buf_idx] + row * STRIDE_KPE + col_chunk * 8;
                __pipeline_memcpy_async(dst, src, 16);
            }
        }
    }
}

__device__ __forceinline__ void zero_invalid_rows(
    SharedStorage& smem,
    int buf_idx,
    int valid_rows,
    int tid
) {
    if (valid_rows >= CHUNK_SIZE) return;

    // Zero CKV
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        int job_idx = tid + k * BLOCK_SIZE;
        int row = job_idx / 64;
        int col_chunk = job_idx % 64;
        
        if (row >= valid_rows && row < CHUNK_SIZE) {
             int4* dst = reinterpret_cast<int4*>(smem.kc_buf[buf_idx] + row * STRIDE_CKV + col_chunk * 8);
             *dst = make_int4(0, 0, 0, 0);
        }
    }
    
    // Zero KPE
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        int job_idx = tid + k * BLOCK_SIZE;
        if (job_idx < 512) {
            int row = job_idx / 8;
            int col_chunk = job_idx % 8;
            if (row >= valid_rows && row < CHUNK_SIZE) {
                int4* dst = reinterpret_cast<int4*>(smem.kp_buf[buf_idx] + row * STRIDE_KPE + col_chunk * 8);
                *dst = make_int4(0, 0, 0, 0);
            }
        }
    }
}

// Grid: (num_splits, batch_size)
template<bool WriteToTemp>
__global__ __launch_bounds__(256, 1)
void mla_decode_step_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    __nv_bfloat16* __restrict__ output_tensor, 
    float* __restrict__ lse_tensor,            
    __grid_constant__ const float sm_scale,
    int num_splits
) {
    extern __shared__ char smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);
    
    int split_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    if (tid < NUM_HEADS) {
        smem.lse_max[tid] = -INFINITY;
        smem.lse_sum[tid] = 0.0f;
    }

    const __nv_bfloat16* qn_g = q_nope + batch_idx * NUM_HEADS * HEAD_DIM_CKV;
    const __nv_bfloat16* qp_g = q_pe + batch_idx * NUM_HEADS * HEAD_DIM_KPE;
    
    load_q(smem, qn_g, qp_g, tid);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[4];
    #pragma unroll
    for (int i=0; i<4; ++i) wmma::fill_fragment(acc_o[i], 0.0f);
    
    __syncthreads();

    // Cache Q fragments in registers to save SMEM bandwidth
    // Balanced Strategy: Warps 0-3 take 4 CKV tiles. Warps 4-7 take 4 CKV + 1 KPE tile.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_q[5];
    int num_tiles = (warp_id >= 4) ? 5 : 4;
    
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        if (i < num_tiles) {
            if (i < 4) {
                // CKV tile
                int k_curr = (warp_id * 4 + i) * 16;
                wmma::load_matrix_sync(frag_q[i], smem.q_nope + k_curr, STRIDE_CKV);
            } else {
                // KPE tile (only for warps 4-7, i=4)
                int k_curr = (warp_id - 4) * 16;
                wmma::load_matrix_sync(frag_q[i], smem.q_pe + k_curr, STRIDE_KPE);
            }
        }
    }
    
    int page_start = kv_indptr[batch_idx];
    int total_tokens_all = kv_indptr[batch_idx + 1] - page_start;
    
    int tokens_per_split = (total_tokens_all + num_splits - 1) / num_splits;
    int start_token = split_idx * tokens_per_split;
    int end_token = min(start_token + tokens_per_split, total_tokens_all);
    int total_tokens = max(0, end_token - start_token);
    
    if (total_tokens == 0) {
        if (tid < NUM_HEADS) {
            float val = -INFINITY;
            if constexpr (!WriteToTemp) lse_tensor[batch_idx * NUM_HEADS + tid] = val;
            else lse_tensor[(batch_idx * num_splits + split_idx) * NUM_HEADS + tid] = val;
        }
        __nv_bfloat16* out_ptr = WriteToTemp ? 
            (output_tensor + (long long)(batch_idx * num_splits + split_idx) * NUM_HEADS * HEAD_DIM_CKV) :
            (output_tensor + batch_idx * NUM_HEADS * HEAD_DIM_CKV);
            
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
             int vec_idx = tid + i * BLOCK_SIZE;
             reinterpret_cast<int4*>(out_ptr)[vec_idx] = make_int4(0,0,0,0);
        }
        return;
    }
    
    int num_chunks = (total_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int valid_rows = min(CHUNK_SIZE, total_tokens);
    
    // Prologue: Load Indices 0
    load_indices(smem, 0, kv_indices, page_start + start_token, valid_rows, tid);
    __syncthreads();
    
    // Issue KV 0
    load_kv_chunk(smem, 0, valid_rows, ckv_cache, kpe_cache, tid);
    __pipeline_commit();
    
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_k;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s[4]; 

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_p;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_v;

    for (int step = 0; step < num_chunks; ++step) {
        int buf_idx = step % 2;
        int next_step = step + 1;
        
        if (next_step < num_chunks) {
            int next_valid = min(CHUNK_SIZE, total_tokens - next_step * CHUNK_SIZE);
            load_indices(smem, next_step % 2, kv_indices, page_start + start_token + next_step * CHUNK_SIZE, next_valid, tid);
            __syncthreads();
            load_kv_chunk(smem, next_step % 2, next_valid, ckv_cache, kpe_cache, tid);
            __pipeline_commit();
        }
        
        __pipeline_wait_prior(next_step < num_chunks ? 1 : 0);
        zero_invalid_rows(smem, buf_idx, valid_rows, tid);
        __syncthreads();

        #pragma unroll
        for (int i=0; i<4; ++i) wmma::fill_fragment(acc_s[i], 0.0f);
        
        // Balanced Split-K Loop
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            if (i < num_tiles) {
                 if (i < 4) {
                     // CKV
                     int k_curr = (warp_id * 4 + i) * 16;
                     #pragma unroll
                     for (int t = 0; t < 4; ++t) {
                         wmma::load_matrix_sync(frag_k, smem.kc_buf[buf_idx] + t * 16 * STRIDE_CKV + k_curr, STRIDE_CKV);
                         wmma::mma_sync(acc_s[t], frag_q[i], frag_k, acc_s[t]);
                     }
                 } else {
                     // KPE
                     int k_curr = (warp_id - 4) * 16;
                     #pragma unroll
                     for (int t = 0; t < 4; ++t) {
                         wmma::load_matrix_sync(frag_k, smem.kp_buf[buf_idx] + t * 16 * STRIDE_KPE + k_curr, STRIDE_KPE);
                         wmma::mma_sync(acc_s[t], frag_q[i], frag_k, acc_s[t]);
                     }
                 }
            }
        }

        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            wmma::store_matrix_sync(smem.scratch.s_partials[warp_id] + t * 16, acc_s[t], S_STRIDE, wmma::mem_row_major);
        }
        __syncthreads();
        
        // Reduce S & Softmax
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * BLOCK_SIZE; 
            int h = idx >> 6;
            int c = idx & 63;
            
            float sum = 0.0f;
            int offset = h * S_STRIDE + c;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) {
                sum += smem.scratch.s_partials[w][offset];
            }
            
            if (step * CHUNK_SIZE + c >= total_tokens) {
                sum = -INFINITY;
            } else {
                sum *= sm_scale;
            }
            smem.scratch.s_partials[0][offset] = sum;
        }
        __syncthreads();
        
        int row = tid / 16; 
        int col_lane = tid % 16;
        
        if (row < 16) {
            float max_val = -INFINITY;
            for (int k = col_lane; k < 64; k += 16) {
                max_val = max(max_val, smem.scratch.s_partials[0][row * S_STRIDE + k]);
            }
            #pragma unroll
            for (int mask = 8; mask > 0; mask /= 2)
                max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));
            
            if (col_lane == 0) {
                float prev_max = smem.lse_max[row];
                float cur_max = max(prev_max, max_val);
                float alpha = (prev_max == -INFINITY) ? 0.0f : expf(prev_max - cur_max);
                if (prev_max == -INFINITY && cur_max == -INFINITY) alpha = 1.0f;
                
                smem.lse_max[row] = cur_max;
                smem.broadcast_alpha[row] = alpha;
            }
        }
        __syncthreads();

        if (row < 16) {
            float cur_max = smem.lse_max[row];
            float sum_p = 0.0f;
            
            for (int k = col_lane; k < 64; k += 16) {
                float val = smem.scratch.s_partials[0][row * S_STRIDE + k];
                float p = (cur_max == -INFINITY) ? 0.0f : expf(val - cur_max);
                sum_p += p;
                smem.p_mat[row * STRIDE_KPE + k] = __float2bfloat16(p);
            }
            #pragma unroll
            for (int mask = 8; mask > 0; mask /= 2)
                sum_p += __shfl_xor_sync(0xffffffff, sum_p, mask);
            
            if (col_lane == 0) {
                float alpha = smem.broadcast_alpha[row];
                float total_sum = smem.lse_sum[row] * alpha + sum_p;
                smem.lse_sum[row] = total_sum;
                smem.lse_inv_sum[row] = (total_sum == 0.0f) ? 0.0f : (1.0f / total_sum);
            }
        }
        __syncthreads();
        
        // Rescale Output Accumulator via SMEM roundtrip
        int warp_col_start = warp_id * 64;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
             wmma::store_matrix_sync(smem.scratch.o_exchange + warp_col_start + k * 16, acc_o[k], HEAD_DIM_CKV, wmma::mem_row_major);
        }
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int idx = tid + i * BLOCK_SIZE;
            if (idx < 16 * HEAD_DIM_CKV) {
                int r = idx / HEAD_DIM_CKV;
                float alpha = smem.broadcast_alpha[r];
                smem.scratch.o_exchange[idx] *= alpha;
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
             wmma::load_matrix_sync(acc_o[k], smem.scratch.o_exchange + warp_col_start + k * 16, HEAD_DIM_CKV, wmma::mem_row_major);
        }
        
        // Accumulate P * V (Split-N: Warp handles disjoint Output cols)
        #pragma unroll
        for (int k_step = 0; k_step < 4; ++k_step) { // K loop (Chunk=64 / 16 = 4)
            int k_offset = k_step * 16;
            wmma::load_matrix_sync(frag_p, smem.p_mat + k_offset, STRIDE_KPE); 
            
            #pragma unroll
            for (int n_step = 0; n_step < 4; ++n_step) { // N loop (Warp_col=64 / 16 = 4)
                int col_sub = warp_col_start + n_step * 16;
                wmma::load_matrix_sync(frag_v, smem.kc_buf[buf_idx] + k_offset * STRIDE_CKV + col_sub, STRIDE_CKV);
                wmma::mma_sync(acc_o[n_step], frag_p, frag_v, acc_o[n_step]);
            }
        }
        __syncthreads();
        
        if (next_step < num_chunks) {
            valid_rows = min(CHUNK_SIZE, total_tokens - next_step * CHUNK_SIZE);
        }
    }
    
    // Epilogue
    int warp_col_start = warp_id * 64;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
         wmma::store_matrix_sync(smem.scratch.o_exchange + warp_col_start + k * 16, acc_o[k], HEAD_DIM_CKV, wmma::mem_row_major);
    }
    __syncthreads();
    
    if (tid < NUM_HEADS) {
        float val = logf(smem.lse_sum[tid]) + smem.lse_max[tid];
        if (smem.lse_sum[tid] == 0.0f) val = -INFINITY;
        
        if constexpr (!WriteToTemp) {
            if (val != -INFINITY) val *= 1.44269504f;
            lse_tensor[batch_idx * NUM_HEADS + tid] = val;
        } else {
            lse_tensor[(batch_idx * num_splits + split_idx) * NUM_HEADS + tid] = val; 
        }
    }

    if constexpr (!WriteToTemp) {
        __nv_bfloat16* out_ptr = output_tensor + batch_idx * NUM_HEADS * HEAD_DIM_CKV;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int vec_idx = tid + i * BLOCK_SIZE;
            int elem_idx = vec_idx * 8; 
            
            __nv_bfloat16 vals[8];
            #pragma unroll
            for (int k=0; k<8; ++k) {
                 int idx = elem_idx + k;
                 float val = smem.scratch.o_exchange[idx];
                 int r = idx / HEAD_DIM_CKV;
                 float inv_sum = smem.lse_inv_sum[r];
                 float res = val * inv_sum;
                 vals[k] = __float2bfloat16(res);
            }
            reinterpret_cast<int4*>(out_ptr)[vec_idx] = *reinterpret_cast<int4*>(vals);
        }
    } else {
        __nv_bfloat16* out_ptr = output_tensor + (long long)batch_idx * num_splits * NUM_HEADS * HEAD_DIM_CKV + 
                                 (long long)split_idx * NUM_HEADS * HEAD_DIM_CKV;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int idx = tid + i * BLOCK_SIZE;
            if (idx < NUM_HEADS * HEAD_DIM_CKV) {
                int r = idx / HEAD_DIM_CKV;
                float val = smem.scratch.o_exchange[idx];
                float inv_sum = smem.lse_inv_sum[r];
                float res = val * inv_sum;
                out_ptr[idx] = __float2bfloat16(res);
            }
        }
    }
}

// Reduce Kernel with Parallel Reduction
__global__ void mla_decode_reduce_kernel(
    const __nv_bfloat16* __restrict__ temp_out,
    const float* __restrict__ temp_lse,
    const int* __restrict__ kv_indptr,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    int num_splits
) {
    extern __shared__ float smem_weights[]; 

    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    // 1. Parallel Reduction for LSE
    float val = -INFINITY;
    if (tid < num_splits) {
        val = temp_lse[(batch_idx * num_splits + tid) * NUM_HEADS + head_idx];
    }
    
    // Block Reduce Max
    float max_val = val;
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    
    static __shared__ float shared_max[8]; // Max 8 warps
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    
    if (tid == 0) {
        float final_max = -INFINITY;
        // Assume blockDim.x=256 -> 8 warps
        for (int i=0; i<8; ++i) final_max = max(final_max, shared_max[i]);
        shared_max[0] = final_max;
    }
    __syncthreads();
    float global_max = shared_max[0];
    
    // Compute Exp
    float w = 0.0f;
    if (tid < num_splits) {
        if (val != -INFINITY) w = expf(val - global_max);
    }
    
    // Block Reduce Sum
    float sum_val = w;
    for (int offset = 16; offset > 0; offset /= 2)
        sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
    
    static __shared__ float shared_sum[8];
    if (lane_id == 0) shared_sum[warp_id] = sum_val;
    __syncthreads();
    
    if (tid == 0) {
        float final_sum = 0.0f;
        for (int i=0; i<8; ++i) final_sum += shared_sum[i];
        shared_sum[0] = final_sum;
        
        // Write LSE
        float lse_res = (global_max == -INFINITY) ? -INFINITY : (global_max + logf(final_sum));
        if (final_sum == 0.0f) lse_res = -INFINITY;
        lse[batch_idx * NUM_HEADS + head_idx] = lse_res * 1.44269504f; 
    }
    __syncthreads();
    
    float global_sum = shared_sum[0];
    float inv_sum = (global_sum == 0.0f) ? 0.0f : (1.0f / global_sum);
    
    if (tid < num_splits) {
        smem_weights[tid] = w * inv_sum;
    }
    __syncthreads();
    
    // 2. Accumulate Output
    int i = tid * 2; 
    if (i < HEAD_DIM_CKV) {
        float2 acc = make_float2(0.0f, 0.0f);
        
        for (int s = 0; s < num_splits; ++s) {
            float weight = smem_weights[s];
            if (weight > 1e-10f) {
                long long offset = (long long)batch_idx * num_splits * NUM_HEADS * HEAD_DIM_CKV + 
                                   (long long)s * NUM_HEADS * HEAD_DIM_CKV + 
                                   (long long)head_idx * HEAD_DIM_CKV + i;
                __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(temp_out + offset);
                
                acc.x += __bfloat162float(val.x) * weight;
                acc.y += __bfloat162float(val.y) * weight;
            }
        }
        
        __nv_bfloat162 res;
        res.x = __float2bfloat16(acc.x);
        res.y = __float2bfloat16(acc.y);
        
        *reinterpret_cast<__nv_bfloat162*>(output + (batch_idx * NUM_HEADS + head_idx) * HEAD_DIM_CKV + i) = res;
    }
}

void run_mla_decode(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    torch::Tensor output,
    torch::Tensor lse,
    float sm_scale
) {
    int batch_size = q_nope.size(0);
    int total_tokens = kv_indices.size(0);
    
    int num_splits = 1;
    if (batch_size < 132) {
        int target_blocks = 132;
        int target_splits = (target_blocks + batch_size - 1) / batch_size;
        
        // Allow splitting down to 16 tokens per split to use more SMs for short sequences
        // avg_len is used as an estimate
        int avg_len = (total_tokens + batch_size - 1) / batch_size;
        int max_splits_len = std::max(1, (avg_len + 15) / 16);
        
        num_splits = std::min(target_splits, max_splits_len);
        num_splits = std::min(num_splits, 128); 
        num_splits = std::max(1, num_splits);
    }

    size_t smem_size = sizeof(SharedStorage);
    
    if (num_splits == 1) {
        dim3 grid(1, batch_size);
        dim3 block(256);
        cudaFuncSetAttribute(mla_decode_step_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        mla_decode_step_kernel<false><<<grid, block, smem_size>>>(
            reinterpret_cast<__nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            kv_indptr.data_ptr<int>(),
            kv_indices.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            sm_scale,
            1
        );
    } else {
        auto options = q_nope.options();
        auto temp_out = torch::empty({batch_size, num_splits, NUM_HEADS, HEAD_DIM_CKV}, options);
        auto temp_lse = torch::empty({batch_size, num_splits, NUM_HEADS}, options.dtype(torch::kFloat32));
        
        dim3 grid_step(num_splits, batch_size);
        dim3 block_step(256);
        
        cudaFuncSetAttribute(mla_decode_step_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        
        mla_decode_step_kernel<true><<<grid_step, block_step, smem_size>>>(
            reinterpret_cast<__nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
            kv_indptr.data_ptr<int>(),
            kv_indices.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(temp_out.data_ptr<at::BFloat16>()),
            temp_lse.data_ptr<float>(),
            sm_scale,
            num_splits
        );
        
        dim3 grid_reduce(batch_size, NUM_HEADS);
        dim3 block_reduce(256);
        size_t smem_reduce = num_splits * sizeof(float);
        
        mla_decode_reduce_kernel<<<grid_reduce, block_reduce, smem_reduce>>>(
            reinterpret_cast<__nv_bfloat16*>(temp_out.data_ptr<at::BFloat16>()),
            temp_lse.data_ptr<float>(),
            kv_indptr.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            num_splits
        );
    }
}
