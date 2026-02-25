#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>

using namespace nvcuda;

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define HEAD_DIM 128
#define GQA_GROUP_SIZE 8

__global__ void gqa_decode_kernel(
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    int num_qo_heads,
    int head_dim,
    float sm_scale)
{
    int batch_idx = blockIdx.x;
    int kv_head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int warp_idx = tid / WARP_SIZE;
    int lane_idx = tid % WARP_SIZE;
    
    int q_head_start = kv_head_idx * GQA_GROUP_SIZE;
    
    int page_start = kv_indptr[batch_idx];
    int page_end = kv_indptr[batch_idx + 1];
    int num_tokens = page_end - page_start;
    
    // Shared Memory Layout
    // s_q: [16, 128] BF16 -> 4KB
    // s_kv: [8, 2, 16, 128] BF16 -> 64KB
    // s_stats: [8, 2, 16] float -> ~1KB
    extern __shared__ char smem[];
    __nv_bfloat16* s_q = (__nv_bfloat16*)smem;
    __nv_bfloat16* s_kv = s_q + 16 * 128;
    float* s_stats = (float*)(s_kv + WARPS_PER_BLOCK * 2 * 16 * 128); 
    
    // Load Q (8 rows, 128 cols)
    int q_base = batch_idx * num_qo_heads * head_dim + q_head_start * head_dim;
    for (int i = tid; i < 16 * 128; i += blockDim.x) {
        int r = i / 128;
        int c = i % 128;
        if (r < 8) {
            s_q[i] = q[q_base + r * 128 + c];
        } else {
            s_q[i] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    
    // Per-warp pointers
    __nv_bfloat16* my_s_k = s_kv + warp_idx * 2 * 16 * 128;
    __nv_bfloat16* my_s_v = my_s_k + 16 * 128;
    
    // Output Accumulators
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[8];
    for (int i = 0; i < 8; ++i) wmma::fill_fragment(acc_o[i], 0.0f);
    
    float l_max[8]; 
    float l_sum[8];
    for (int i = 0; i < 8; ++i) {
        l_max[i] = -INFINITY;
        l_sum[i] = 0.0f;
    }
    
    // Process tokens in chunks, distributed across warps
    for (int tok_base = warp_idx * 16; tok_base < num_tokens; tok_base += WARPS_PER_BLOCK * 16) {
        int valid_rows = min(16, num_tokens - tok_base);
        
        // Vectorized Load K, V (int4 = 8 bf16)
        // Need to load 16 * 128 bf16s = 256 int4s
        // 32 threads/warp -> 8 int4s per thread
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int vec_idx = lane_idx + k * 32;
            int r = vec_idx / 16;
            int c_vec = vec_idx % 16;
            
            int4 k_val = {0,0,0,0};
            int4 v_val = {0,0,0,0};
            
            if (r < valid_rows) {
                int page_id = kv_indices[page_start + tok_base + r];
                long base_addr = (long)page_id * 4 * 128 + kv_head_idx * 128;
                const int4* k_ptr = (const int4*)(k_cache + base_addr);
                const int4* v_ptr = (const int4*)(v_cache + base_addr);
                k_val = k_ptr[c_vec];
                v_val = v_ptr[c_vec];
            }
            ((int4*)my_s_k)[vec_idx] = k_val;
            ((int4*)my_s_v)[vec_idx] = v_val;
        }
        __syncwarp();
        
        // Q * K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s;
        wmma::fill_fragment(acc_s, 0.0f);
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> f_q;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> f_k;
            wmma::load_matrix_sync(f_q, s_q + k * 16, 128);
            wmma::load_matrix_sync(f_k, my_s_k + k * 16, 128);
            wmma::mma_sync(acc_s, f_q, f_k, acc_s);
        }
        
        // Store logits
        float* s_logits = (float*)my_s_k;
        wmma::store_matrix_sync(s_logits, acc_s, 16, wmma::mem_row_major);
        __syncwarp();
        
        // Softmax
        float scale = 0.0f;
        if (lane_idx < 8) {
            int h = lane_idx;
            float m_curr = -INFINITY;
            for (int t = 0; t < valid_rows; ++t) m_curr = fmaxf(m_curr, s_logits[h * 16 + t]);
            
            float m_prev = l_max[h];
            float m_new = fmaxf(m_prev, m_curr * sm_scale);
            if (m_new == -INFINITY) m_new = -1e30f;
            
            float s_prev = expf(m_prev - m_new);
            scale = s_prev;
            
            float p_sum = 0.0f;
            for (int t = 0; t < 16; ++t) {
                float v = 0.0f;
                if (t < valid_rows) {
                    v = expf(s_logits[h * 16 + t] * sm_scale - m_new);
                }
                s_logits[h * 16 + t] = v;
                p_sum += v;
            }
            l_sum[h] = l_sum[h] * s_prev + p_sum;
            l_max[h] = m_new;
        }
        
        float bcast_scale[8];
        #pragma unroll
        for(int h=0; h<8; ++h) bcast_scale[h] = __shfl_sync(0xFFFFFFFF, scale, h);
        
        // Rescale O
        float* scratch = (float*)my_s_k + 256; 
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            wmma::store_matrix_sync(scratch, acc_o[i], 16, wmma::mem_row_major);
            __syncwarp();
            // Vectorized scaling: 2 float4s per thread
            for (int j = 0; j < 2; ++j) {
                int idx = lane_idx + j * 32; 
                int row = idx / 4; 
                if (row < 8) {
                   float4 v = ((float4*)scratch)[idx];
                   float s = bcast_scale[row];
                   v.x *= s; v.y *= s; v.z *= s; v.w *= s;
                   ((float4*)scratch)[idx] = v;
                }
            }
            __syncwarp();
            wmma::load_matrix_sync(acc_o[i], scratch, 16, wmma::mem_row_major);
        }
        
        // P * V
        __nv_bfloat16* s_p = (__nv_bfloat16*)scratch; 
        for (int j = lane_idx; j < 256; j += 32) {
            s_p[j] = __float2bfloat16(s_logits[j]);
        }
        __syncwarp();
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> f_p;
        wmma::load_matrix_sync(f_p, s_p, 16);
        
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> f_v;
            wmma::load_matrix_sync(f_v, my_s_v + j * 16, 128); 
            wmma::mma_sync(acc_o[j], f_p, f_v, acc_o[j]);
        }
    }
    
    // Cross-warp reduction
    float* s_reduce = (float*)s_kv;
    int warp_offset = warp_idx * 2048; 
    for (int i = 0; i < 8; ++i) {
        wmma::store_matrix_sync(s_reduce + warp_offset + i * 256, acc_o[i], 16, wmma::mem_row_major);
    }
    
    if (lane_idx < 8) {
        s_stats[warp_idx * 16 + lane_idx] = l_max[lane_idx];
        s_stats[warp_idx * 16 + 8 + lane_idx] = l_sum[lane_idx];
    }
    __syncthreads();
    
    if (warp_idx == 0) {
        // Warp 0 reduces and writes output
        for (int i = lane_idx; i < 1024; i += 32) {
            int h = i / 128;
            int d = i % 128;
            
            int frag_idx = d / 16;
            int col = d % 16;
            int offset = frag_idx * 256 + h * 16 + col;
            
            float m_final = s_stats[h];
            float s_final = s_stats[8 + h];
            float o_final = s_reduce[offset];
            
            for (int w = 1; w < WARPS_PER_BLOCK; ++w) {
                float mw = s_stats[w * 16 + h];
                float sw = s_stats[w * 16 + 8 + h];
                float ow = s_reduce[w * 2048 + offset];
                
                float m_new = fmaxf(m_final, mw);
                float scale_curr = (m_new == -INFINITY) ? 0.0f : expf(m_final - m_new);
                float scale_new = (m_new == -INFINITY) ? 0.0f : expf(mw - m_new);
                
                s_final = s_final * scale_curr + sw * scale_new;
                o_final = o_final * scale_curr + ow * scale_new;
                m_final = m_new;
            }
            
            int global_h = q_head_start + h;
            int out_idx = batch_idx * num_qo_heads * head_dim + global_h * head_dim + d;
            
            if (s_final > 0.0f) {
                output[out_idx] = __float2bfloat16(o_final / s_final);
            } else {
                output[out_idx] = __float2bfloat16(0.0f);
            }
            
            if (d == 0) {
                lse[batch_idx * num_qo_heads + global_h] = (s_final > 0.0f) ? (m_final + logf(s_final)) * 1.44269504f : -INFINITY;
            }
        }
    }
}

void launch_gqa_decode(
    __nv_bfloat16* output,
    float* lse,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int* kv_indptr,
    const int* kv_indices,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    cudaStream_t stream
) {
    dim3 grid(batch_size, num_kv_heads);
    dim3 block(256); // 8 warps
    int smem_size = 72 * 1024; 
    
    cudaFuncSetAttribute(gqa_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    gqa_decode_kernel<<<grid, block, smem_size, stream>>>(
        output, lse, q, k_cache, v_cache, kv_indptr, kv_indices,
        num_qo_heads, head_dim, sm_scale
    );
}
