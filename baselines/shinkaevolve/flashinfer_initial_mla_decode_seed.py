"""
Seed initial program for ShinkaEvolve MLA-decode evolution.

This seeds `candidate_kernel()` with the OpenEvolve advanced MMA-based MLA decode kernel:
  `examples/openevolve/initial_mla_decode_mma_advanced.txt`

ShinkaEvolve will mutate the region between `EVOLVE-BLOCK-START/END`.
"""

from __future__ import annotations

from typing import Any, Dict


# EVOLVE-BLOCK-START
def candidate_kernel() -> Dict[str, Any]:
    """
    Return the candidate kernel spec to evaluate.

    We use CUDA "KernelGenerator XML" with exactly these 3 files:
      - <header_file name="kernel.h"> ... </header_file>
      - <cuda_file name="kernel.cu"> ... </cuda_file>
      - <cpp_file name="main.cpp"> ... </cpp_file>
    """

    kernel_xml = r"""
<header_file name="kernel.h">
#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>

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
);
</header_file>

<cuda_file name="kernel.cu">
#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cmath>

using namespace nvcuda;

// Tuning params for H100
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;

constexpr int NUM_HEADS = 16;
constexpr int HEAD_DIM_CKV = 512;
constexpr int HEAD_DIM_KPE = 64;

constexpr int CHUNK_SIZE = 64; 

// Memory Layout Constants
constexpr int PAD_CKV = 8;
constexpr int STRIDE_CKV = HEAD_DIM_CKV + PAD_CKV; // 520
constexpr int PAD_KPE = 8;
constexpr int STRIDE_KPE = HEAD_DIM_KPE + PAD_KPE; // 72

// Stride for S_partials (Split-K reduction buffer)
constexpr int S_STRIDE = 72; // 64 + 8

struct __align__(128) SharedStorage {
    // Q matrices
    __nv_bfloat16 q_nope[NUM_HEADS * HEAD_DIM_CKV]; // 16KB
    __nv_bfloat16 q_pe[NUM_HEADS * HEAD_DIM_KPE];   // 2KB
    
    // KV Buffers (Double Buffered)
    __nv_bfloat16 kc_buf[2][CHUNK_SIZE * STRIDE_CKV]; // ~133KB
    __nv_bfloat16 kp_buf[2][CHUNK_SIZE * STRIDE_KPE]; // ~18KB
    
    // P Matrix for Softmax -> Output accumulation
    __nv_bfloat16 p_mat[NUM_HEADS * CHUNK_SIZE]; // 2KB

    // Reused scratch memory
    union {
        // S Partials
        float s_partials[NUM_WARPS][16 * S_STRIDE]; // ~36KB
        
        // Output exchange buffer for final reduction within block
        float o_exchange[NUM_HEADS * HEAD_DIM_CKV]; // 32KB
    } scratch;

    // Softmax statistics
    float lse_max[NUM_HEADS];
    float lse_sum[NUM_HEADS];
    float broadcast_alpha[NUM_HEADS];
};

__device__ __forceinline__ void load_q(
    SharedStorage& smem,
    const __nv_bfloat16* __restrict__ qn_g,
    const __nv_bfloat16* __restrict__ qp_g,
    int tid
) {
    const int4* src_qn = reinterpret_cast<const int4*>(qn_g);
    int4* dst_qn = reinterpret_cast<int4*>(smem.q_nope);
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = i * BLOCK_SIZE + tid;
        dst_qn[idx] = src_qn[idx];
    }
    
    const int4* src_qp = reinterpret_cast<const int4*>(qp_g);
    int4* dst_qp = reinterpret_cast<int4*>(smem.q_pe);
    if (tid < 128) { 
        dst_qp[tid] = src_qp[tid];
    }
}

__device__ __forceinline__ void load_kv_chunk(
    SharedStorage& smem,
    int buf_idx,
    const int* __restrict__ kv_indices,
    int page_start_offset,
    int valid_rows,
    const __nv_bfloat16* __restrict__ ckv_base,
    const __nv_bfloat16* __restrict__ kpe_base,
    int tid
) {
    // Load CKV: [64, 512]
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int job_idx = tid + i * BLOCK_SIZE;
        int row = job_idx >> 6; // / 64
        int col_chunk = job_idx & 63; // % 64
        
        if (row < valid_rows) {
            int page_idx = kv_indices[page_start_offset + row];
            const void* src = ckv_base + (long long)page_idx * HEAD_DIM_CKV + col_chunk * 8;
            void* dst = smem.kc_buf[buf_idx] + row * STRIDE_CKV + col_chunk * 8;
            __pipeline_memcpy_async(dst, src, 16);
        }
    }
    
    // Load KPE: [64, 64]
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int job_idx = tid + i * BLOCK_SIZE;
        if (job_idx < 512) {
            int row = job_idx >> 3; // / 8
            int col_chunk = job_idx & 7; // % 8
            if (row < valid_rows) {
                int page_idx = kv_indices[page_start_offset + row];
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
    for (int i = 0; i < 16; ++i) { 
        int job_idx = tid + i * BLOCK_SIZE;
        int row = job_idx >> 6;
        int col_chunk = job_idx & 63;
        
        if (row >= valid_rows && row < CHUNK_SIZE) {
             int4* dst = reinterpret_cast<int4*>(smem.kc_buf[buf_idx] + row * STRIDE_CKV + col_chunk * 8);
             *dst = make_int4(0, 0, 0, 0);
        }
    }
    
    // Zero KPE
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int job_idx = tid + i * BLOCK_SIZE;
        if (job_idx < 512) {
            int row = job_idx >> 3;
            int col_chunk = job_idx & 7;
            if (row >= valid_rows && row < CHUNK_SIZE) {
                int4* dst = reinterpret_cast<int4*>(smem.kp_buf[buf_idx] + row * STRIDE_KPE + col_chunk * 8);
                *dst = make_int4(0, 0, 0, 0);
            }
        }
    }
}

// Grid: (num_splits, batch_size)
__global__ __launch_bounds__(256, 1)
void mla_decode_step_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    __nv_bfloat16* __restrict__ temp_out, // [batch, num_splits, num_heads, head_dim]
    float* __restrict__ temp_lse,         // [batch, num_splits, num_heads]
    float sm_scale,
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
    
    int page_start = kv_indptr[batch_idx];
    int total_tokens_all = kv_indptr[batch_idx + 1] - page_start;
    
    // Calculate token range for this split
    int total_tokens = 0;
    int start_token = 0;
    
    if (total_tokens_all > 0) {
        // Evenly distribute
        int tokens_per_split = (total_tokens_all + num_splits - 1) / num_splits;
        start_token = split_idx * tokens_per_split;
        int end_token = min(start_token + tokens_per_split, total_tokens_all);
        if (start_token < end_token) {
            total_tokens = end_token - start_token;
        }
    }
    
    // If no work for this split, write -inf and exit
    if (total_tokens <= 0) {
        if (tid < NUM_HEADS) {
            temp_lse[(batch_idx * num_splits + split_idx) * NUM_HEADS + tid] = -INFINITY;
        }
        // Output doesn't matter if lse is -inf, but write 0 for safety
        __nv_bfloat16* t_out_ptr = temp_out + (batch_idx * num_splits + split_idx) * NUM_HEADS * HEAD_DIM_CKV;
        for (int i = tid; i < NUM_HEADS * HEAD_DIM_CKV; i += BLOCK_SIZE) {
             t_out_ptr[i] = __float2bfloat16(0.0f);
        }
        return;
    }
    
    int num_chunks = (total_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int valid_rows = min(CHUNK_SIZE, total_tokens);
    
    load_kv_chunk(smem, 0, kv_indices, page_start + start_token, valid_rows, ckv_cache, kpe_cache, tid);
    __pipeline_commit();
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_q;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_k;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s[4]; 

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_p;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_v;

    for (int step = 0; step < num_chunks; ++step) {
        __pipeline_wait_prior(0);
        
        int buf_idx = step % 2;
        int next_step = step + 1;
        
        zero_invalid_rows(smem, buf_idx, valid_rows, tid);
        __syncthreads();
        
        if (next_step < num_chunks) {
            int next_valid = min(CHUNK_SIZE, total_tokens - next_step * CHUNK_SIZE);
            load_kv_chunk(smem, next_step % 2, kv_indices, page_start + start_token + next_step * CHUNK_SIZE, next_valid, ckv_cache, kpe_cache, tid);
            __pipeline_commit();
        }

        // --- 1. Compute Scores (Split-K) ---
        #pragma unroll
        for (int i=0; i<4; ++i) wmma::fill_fragment(acc_s[i], 0.0f);
        
        int k_start = warp_id * 64; 
        
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int t_offset = t * 16;
            // CKV Part
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int k_curr = k_start + k * 16;
                wmma::load_matrix_sync(frag_q, smem.q_nope + k_curr, HEAD_DIM_CKV);
                wmma::load_matrix_sync(frag_k, smem.kc_buf[buf_idx] + t_offset * STRIDE_CKV + k_curr, STRIDE_CKV);
                wmma::mma_sync(acc_s[t], frag_q, frag_k, acc_s[t]);
            }
            // KPE Part (Warp 0 only)
            if (warp_id == 0) {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int k_curr = k * 16;
                    wmma::load_matrix_sync(frag_q, smem.q_pe + k_curr, HEAD_DIM_KPE);
                    wmma::load_matrix_sync(frag_k, smem.kp_buf[buf_idx] + t_offset * STRIDE_KPE + k_curr, STRIDE_KPE);
                    wmma::mma_sync(acc_s[t], frag_q, frag_k, acc_s[t]);
                }
            }
        }
        
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            float* dst = smem.scratch.s_partials[warp_id] + t * 16;
            wmma::store_matrix_sync(dst, acc_s[t], S_STRIDE, wmma::mem_row_major);
        }
        __syncthreads();
        
        // --- 2. Reduce S & Softmax ---
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
        
        int row = tid / 16; // Head index
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
                float p;
                if (cur_max == -INFINITY) {
                    p = 0.0f;
                } else {
                    p = expf(val - cur_max);
                }
                sum_p += p;
                smem.p_mat[row * 64 + k] = __float2bfloat16(p);
            }
            #pragma unroll
            for (int mask = 8; mask > 0; mask /= 2)
                sum_p += __shfl_xor_sync(0xffffffff, sum_p, mask);
            
            if (col_lane == 0) {
                float alpha = smem.broadcast_alpha[row];
                smem.lse_sum[row] = smem.lse_sum[row] * alpha + sum_p;
            }
        }
        __syncthreads();
        
        // --- 3. Rescale Output Accumulator ---
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
        
        // --- 4. Accumulate P * V ---
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int t_offset = t * 16;
            wmma::load_matrix_sync(frag_p, smem.p_mat + t_offset, 64);
            
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int col_sub = warp_col_start + k * 16;
                wmma::load_matrix_sync(frag_v, smem.kc_buf[buf_idx] + t_offset * STRIDE_CKV + col_sub, STRIDE_CKV);
                wmma::mma_sync(acc_o[k], frag_p, frag_v, acc_o[k]);
            }
        }
        __syncthreads();
        
        if (next_step < num_chunks) {
            valid_rows = min(CHUNK_SIZE, total_tokens - next_step * CHUNK_SIZE);
        }
    }
    
    // --- Epilogue ---
    int warp_col_start = warp_id * 64;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
         wmma::store_matrix_sync(smem.scratch.o_exchange + warp_col_start + k * 16, acc_o[k], HEAD_DIM_CKV, wmma::mem_row_major);
    }
    __syncthreads();
    
    __nv_bfloat16* out_ptr = temp_out + (batch_idx * num_splits + split_idx) * NUM_HEADS * HEAD_DIM_CKV;
    
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < 16 * HEAD_DIM_CKV) {
            int r = idx / HEAD_DIM_CKV;
            float val = smem.scratch.o_exchange[idx];
            float sum = smem.lse_sum[r];
            float res = (sum == 0.0f) ? 0.0f : (val / sum);
            out_ptr[idx] = __float2bfloat16(res);
        }
    }
    
    if (tid < NUM_HEADS) {
        // Store natural log LSE
        float val = logf(smem.lse_sum[tid]) + smem.lse_max[tid];
        if (smem.lse_sum[tid] == 0.0f) val = -INFINITY;
        temp_lse[(batch_idx * num_splits + split_idx) * NUM_HEADS + tid] = val; 
    }
}

// Reduce Kernel: Grid(batch_size, NUM_HEADS), Block(256)
__global__ void mla_decode_reduce_kernel(
    const __nv_bfloat16* __restrict__ temp_out, // [batch, splits, heads, head_dim]
    const float* __restrict__ temp_lse,          // [batch, splits, heads]
    __nv_bfloat16* __restrict__ output,          // [batch, heads, head_dim]
    float* __restrict__ lse,                     // [batch, heads]
    int num_splits
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    // 1. Load LSEs and find global LSE
    float max_lse = -INFINITY;
    
    // Iterate over splits
    for (int s = 0; s < num_splits; ++s) {
        float val = temp_lse[(batch_idx * num_splits + s) * NUM_HEADS + head_idx];
        if (val > max_lse) max_lse = val;
    }
    
    float sum_exp = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        float val = temp_lse[(batch_idx * num_splits + s) * NUM_HEADS + head_idx];
        if (val != -INFINITY) {
            sum_exp += expf(val - max_lse);
        }
    }
    
    float global_lse = max_lse + logf(sum_exp);
    if (max_lse == -INFINITY) global_lse = -INFINITY;
    
    // Store final LSE in base 2
    if (tid == 0) {
        lse[batch_idx * NUM_HEADS + head_idx] = global_lse * 1.44269504f; // log2(e)
    }
    
    // 2. Accumulate Output
    for (int i = tid; i < HEAD_DIM_CKV; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            float l_val = temp_lse[(batch_idx * num_splits + s) * NUM_HEADS + head_idx];
            if (l_val != -INFINITY) {
                float weight = expf(l_val - global_lse);
                __nv_bfloat16 val = temp_out[((batch_idx * num_splits + s) * NUM_HEADS + head_idx) * HEAD_DIM_CKV + i];
                acc += __bfloat162float(val) * weight;
            }
        }
        output[(batch_idx * NUM_HEADS + head_idx) * HEAD_DIM_CKV + i] = __float2bfloat16(acc);
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
    
    // Heuristic for split count
    int num_splits = 32;
    if (batch_size > 4) num_splits = 16;
    if (batch_size > 16) num_splits = 4;
    if (batch_size > 64) num_splits = 2; // Keep at least 2 to ensure parallelism
    
    // Allocate temps
    auto options = q_nope.options();
    auto temp_out = torch::empty({batch_size, num_splits, NUM_HEADS, HEAD_DIM_CKV}, options);
    auto temp_lse = torch::empty({batch_size, num_splits, NUM_HEADS}, options.dtype(torch::kFloat32));
    
    // Step Kernel
    dim3 grid_step(num_splits, batch_size);
    dim3 block_step(256);
    size_t smem_size = sizeof(SharedStorage);
    
    cudaFuncSetAttribute(mla_decode_step_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    mla_decode_step_kernel<<<grid_step, block_step, smem_size>>>(
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
    
    // Reduce Kernel
    dim3 grid_reduce(batch_size, NUM_HEADS);
    dim3 block_reduce(256);
    
    mla_decode_reduce_kernel<<<grid_reduce, block_reduce>>>(
        reinterpret_cast<__nv_bfloat16*>(temp_out.data_ptr<at::BFloat16>()),
        temp_lse.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        num_splits
    );
}
</cuda_file>

<cpp_file name="main.cpp">
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include "kernel.h"

std::map<std::string, torch::Tensor> run_dict(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float sm_scale
) {
    // Ensure contiguous inputs
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

    run_mla_decode(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr_i32, kv_indices_i32, output, lse, sm_scale);
    
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
</cpp_file>
""".strip()

    return {"mode": "code", "language": "cuda", "code": kernel_xml}

# EVOLVE-BLOCK-END


def run_experiment(**kwargs) -> Dict[str, Any]:
    """
    Entry point called by ShinkaEvolve evaluation.
    """
    from flashinfer_shinka_evaluator import FlashInferShinkaEvaluatorConfig, evaluate_candidate

    cfg = FlashInferShinkaEvaluatorConfig.from_config(kwargs)
    cand = candidate_kernel()
    return evaluate_candidate(cfg=cfg, candidate=cand)

