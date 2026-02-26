#include "kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>

using namespace nvcuda;

#define BLOCK_M 16
#define BLOCK_N 64
#define HEAD_DIM_CKV 512
#define HEAD_DIM_KPE 64
#define HEAD_DIM_TOTAL 576
// Padding: 576 + 8 = 584. 584 * 2 bytes = 1168 bytes. 1168 % 128 = 16 bytes offset (4 banks).
#define SMEM_STRIDE 584 
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 128

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem), "l"(glob_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
    if (n == 0) asm volatile("cp.async.wait_group 0;");
    else if (n == 1) asm volatile("cp.async.wait_group 1;");
}

__global__ void __launch_bounds__(128) mla_kernel_opt(
    nv_bfloat16* __restrict__ q_nope,
    nv_bfloat16* __restrict__ q_pe,
    nv_bfloat16* __restrict__ ckv_cache,
    nv_bfloat16* __restrict__ kpe_cache,
    int32_t* __restrict__ qo_indptr,
    int32_t* __restrict__ kv_indptr,
    int32_t* __restrict__ kv_indices,
    nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    float sm_scale,
    TileInfo* __restrict__ tile_info
) {
    extern __shared__ char smem[];
    nv_bfloat16* s_K = (nv_bfloat16*)smem;
    nv_bfloat16* s_Q = s_K + 2 * BLOCK_N * SMEM_STRIDE;
    float* s_O = (float*)(s_Q + BLOCK_M * SMEM_STRIDE);
    float* s_S = s_O + BLOCK_M * HEAD_DIM_CKV;
    nv_bfloat16* s_P = (nv_bfloat16*)(s_S + BLOCK_M * BLOCK_N);
    float* s_m = (float*)(s_P + BLOCK_M * BLOCK_N);
    float* s_d = s_m + BLOCK_M;
    float* s_scale = s_d + BLOCK_M;

    int tile_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    TileInfo info = tile_info[tile_idx];
    int batch_idx = info.batch_idx;
    int valid_q_len = info.q_len;
    int q_global_offset = info.q_global_offset;
    int q_start_in_seq = info.q_start_in_seq;

    int kv_start = kv_indptr[batch_idx];
    int kv_len_total = kv_indptr[batch_idx+1] - kv_start;
    
    int prefix_len = kv_len_total - (qo_indptr[batch_idx+1] - qo_indptr[batch_idx]);

    // Init stats/O
    for (int i = tid; i < BLOCK_M * HEAD_DIM_CKV; i += THREADS_PER_BLOCK) s_O[i] = 0.0f;
    if (tid < BLOCK_M) { s_m[tid] = -INFINITY; s_d[tid] = 0.0f; }

    // Load Q
    // Split loops to avoid division: 512 (64*8) and 64 (8*8)
    // CKV part
    for (int i = tid; i < BLOCK_M * 64; i += THREADS_PER_BLOCK) {
        int r = i >> 6;      // i / 64
        int chunk = i & 63;  // i % 64
        int c = chunk * 8;
        
        int s_idx = r * SMEM_STRIDE + c;
        if (r < valid_q_len) {
             int q_idx = q_global_offset + r;
             float4 val = *((float4*)&q_nope[(q_idx * 16 + head_idx) * 512 + c]);
             *((float4*)&s_Q[s_idx]) = val;
        } else {
             *((float4*)&s_Q[s_idx]) = make_float4(0,0,0,0);
        }
    }
    // KPE part
    for (int i = tid; i < BLOCK_M * 8; i += THREADS_PER_BLOCK) {
        int r = i >> 3;     // i / 8
        int chunk = i & 7;  // i % 8
        int c = 512 + chunk * 8;
        
        int s_idx = r * SMEM_STRIDE + c;
        if (r < valid_q_len) {
             int q_idx = q_global_offset + r;
             int c_pe = c - 512;
             float4 val = *((float4*)&q_pe[(q_idx * 16 + head_idx) * 64 + c_pe]);
             *((float4*)&s_Q[s_idx]) = val;
        } else {
             *((float4*)&s_Q[s_idx]) = make_float4(0,0,0,0);
        }
    }

    auto load_k_tile = [&](int step, int buf_idx) {
        int kv_curr = step * BLOCK_N;
        int valid_k = min(BLOCK_N, kv_len_total - kv_curr);
        nv_bfloat16* dst_base = s_K + buf_idx * BLOCK_N * SMEM_STRIDE;
        
        // Load CKV (512 dims = 64 chunks)
        int num_ckv_chunks = valid_k * 64;
        for (int i = tid; i < num_ckv_chunks; i += THREADS_PER_BLOCK) {
            int r = i >> 6;
            int chunk = i & 63;
            
            int page_idx = kv_indices[kv_start + kv_curr + r];
            size_t dst_off = r * SMEM_STRIDE + chunk * 8;
            size_t src_off = (size_t)page_idx * HEAD_DIM_CKV + chunk * 8;
            cp_async_16(&dst_base[dst_off], &ckv_cache[src_off]);
        }
        
        // Load KPE (64 dims = 8 chunks)
        int num_kpe_chunks = valid_k * 8;
        for (int i = tid; i < num_kpe_chunks; i += THREADS_PER_BLOCK) {
            int r = i >> 3;
            int chunk = i & 7;
            
            int page_idx = kv_indices[kv_start + kv_curr + r];
            size_t dst_off = r * SMEM_STRIDE + 512 + chunk * 8;
            size_t src_off = (size_t)page_idx * HEAD_DIM_KPE + chunk * 8;
            cp_async_16(&dst_base[dst_off], &kpe_cache[src_off]);
        }
    };

    __syncthreads(); // Wait for Q

    int num_steps = (kv_len_total + BLOCK_N - 1) / BLOCK_N;
    
    if (num_steps > 0) {
        load_k_tile(0, 0);
        cp_async_commit();
    }

    for (int step = 0; step < num_steps; ++step) {
        int buf_curr = step % 2;
        int buf_next = (step + 1) % 2;
        
        if (step + 1 < num_steps) {
            load_k_tile(step + 1, buf_next);
            cp_async_commit();
        }
        
        cp_async_wait_group(step + 1 < num_steps ? 1 : 0);
        __syncthreads();

        int kv_curr = step * BLOCK_N;
        int valid_k = min(BLOCK_N, kv_len_total - kv_curr);
        nv_bfloat16* curr_K = s_K + buf_curr * BLOCK_N * SMEM_STRIDE;

        // MMA: S = Q * K^T
        int w_col_start = warp_id * 16;
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s;
        wmma::fill_fragment(acc_s, 0.0f);

        for (int k = 0; k < HEAD_DIM_TOTAL; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, nv_bfloat16, wmma::col_major> b;
            
            wmma::load_matrix_sync(a, s_Q + k, SMEM_STRIDE);
            wmma::load_matrix_sync(b, curr_K + w_col_start * SMEM_STRIDE + k, SMEM_STRIDE);
            wmma::mma_sync(acc_s, a, b, acc_s);
        }
        wmma::store_matrix_sync(s_S + w_col_start, acc_s, BLOCK_N, wmma::mem_row_major);
        __syncthreads();

        // Softmax
        for (int r = warp_id; r < BLOCK_M; r += 4) {
            if (r >= valid_q_len) continue;
            
            float m_curr = -INFINITY;
            int q_seq_pos = q_start_in_seq + r;
            int row_base = r * BLOCK_N;
            
            for (int c = lane_id; c < BLOCK_N; c += 32) {
                float val = s_S[row_base + c];
                bool active = (c < valid_k) && (kv_curr + c <= q_seq_pos + prefix_len);
                if (!active) val = -INFINITY;
                else val *= sm_scale;
                s_S[row_base + c] = val;
                m_curr = fmaxf(m_curr, val);
            }
            
            #pragma unroll
            for (int mask = 16; mask > 0; mask /= 2)
                m_curr = fmaxf(m_curr, __shfl_xor_sync(0xffffffff, m_curr, mask));
            
            float m_prev = s_m[r];
            float m_new = fmaxf(m_prev, m_curr);
            float scale = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_new);
            
            if (lane_id == 0) {
                s_scale[r] = scale;
                s_m[r] = m_new;
            }

            float l_sum = 0.0f;
            for (int c = lane_id; c < BLOCK_N; c += 32) {
                float val = s_S[row_base + c];
                float p = 0.0f;
                if (val > -INFINITY) {
                    p = expf(val - m_new);
                }
                s_P[row_base + c] = __float2bfloat16(p);
                l_sum += p;
            }

            #pragma unroll
            for (int mask = 16; mask > 0; mask /= 2)
                l_sum += __shfl_xor_sync(0xffffffff, l_sum, mask);
            
            if (lane_id == 0) {
                s_d[r] = s_d[r] * scale + l_sum;
            }
        }
        __syncthreads();

        // Rescale O
        #pragma unroll
        for (int i = tid; i < BLOCK_M * HEAD_DIM_CKV; i += THREADS_PER_BLOCK) {
            int r = i / HEAD_DIM_CKV;
            if (r < valid_q_len) {
                float sc = s_scale[r];
                if (sc != 1.0f) s_O[i] *= sc;
            }
        }
        __syncthreads();

        // O += P * V
        int v_col_start = warp_id * 128; // 0, 128, 256, 384
        
        #pragma unroll
        for (int chunk = 0; chunk < 8; ++chunk) {
            int col = v_col_start + chunk * 16;
            
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o;
            wmma::load_matrix_sync(acc_o, s_O + col, HEAD_DIM_CKV, wmma::mem_row_major);
            
            for (int k = 0; k < BLOCK_N; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, nv_bfloat16, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, nv_bfloat16, wmma::row_major> v_frag;
                
                wmma::load_matrix_sync(p_frag, s_P + k, BLOCK_N);
                wmma::load_matrix_sync(v_frag, curr_K + k * SMEM_STRIDE + col, SMEM_STRIDE);
                
                wmma::mma_sync(acc_o, p_frag, v_frag, acc_o);
            }
            wmma::store_matrix_sync(s_O + col, acc_o, HEAD_DIM_CKV, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // Output
    for (int i = tid * 8; i < BLOCK_M * HEAD_DIM_CKV; i += THREADS_PER_BLOCK * 8) {
        int r = i / HEAD_DIM_CKV;
        int c = i % HEAD_DIM_CKV;
        
        if (r < valid_q_len) {
            float d = s_d[r];
            float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
            
            float4 v0 = *((float4*)&s_O[i]);
            float4 v1 = *((float4*)&s_O[i+4]);
            
            float* fv0 = (float*)&v0;
            float* fv1 = (float*)&v1;
            
            nv_bfloat16 out_vals[8];
            #pragma unroll
            for(int k=0; k<4; ++k) out_vals[k] = __float2bfloat16(fv0[k] * inv_d);
            #pragma unroll
            for(int k=0; k<4; ++k) out_vals[k+4] = __float2bfloat16(fv1[k] * inv_d);
            
            size_t offset = (size_t)(q_global_offset + r) * 16 * 512 + head_idx * 512 + c;
            *((float4*)&output[offset]) = *((float4*)out_vals);
        }
    }
    
    if (tid < valid_q_len) {
        float d = s_d[tid];
        float m = s_m[tid];
        lse[(q_global_offset + tid) * 16 + head_idx] = (d > 0.0f) ? (logf(d) + m) * 1.44269504f : -INFINITY;
    }
}

void launch_mla_paged_prefill(
    void* q_nope,
    void* q_pe,
    void* ckv_cache,
    void* kpe_cache,
    int32_t* qo_indptr,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    void* output,
    float* lse,
    float sm_scale,
    TileInfo* tile_info,
    int num_tiles,
    cudaStream_t stream
) {
    dim3 grid(num_tiles, 16);
    dim3 block(THREADS_PER_BLOCK);
    int smem_size = 210 * 1024;
    cudaFuncSetAttribute(mla_kernel_opt, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    mla_kernel_opt<<<grid, block, smem_size, stream>>>(
        (nv_bfloat16*)q_nope,
        (nv_bfloat16*)q_pe,
        (nv_bfloat16*)ckv_cache,
        (nv_bfloat16*)kpe_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        (nv_bfloat16*)output,
        lse,
        sm_scale,
        tile_info
    );
}
