#include "kernel.h"
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

// Tuning parameters for H100
// BR=16, BC=16 allows small SMEM usage (~90KB) to fit 2 blocks per SM (228KB capacity).
// This improves occupancy.
#define BR 16
#define BC 16
#define HEAD_DIM_CKV 512
#define HEAD_DIM_KPE 64
#define HEAD_DIM_TOTAL 576
// Pad to 592 to avoid bank conflicts and align to 16 bytes (592*2 = 1184 bytes, 1184%128=32)
#define HEAD_DIM_PADDED 592 
// Pad O accumulator to avoid bank conflicts (520*4 = 2080 bytes, 2080%32!=0)
#define HEAD_DIM_O_PADDED 520

#define WARP_SIZE 32
#define NUM_WARPS 4
#define THREADS 128

__global__ void __launch_bounds__(THREADS) mla_prefill_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int* __restrict__ qo_indptr,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    float sm_scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    const TileInfo* __restrict__ tile_infos,
    int num_heads
) {
    // Shared Memory Layout
    // s_Q: [BR, HEAD_DIM_PADDED] (16 * 592 * 2 = 18944 bytes)
    // s_K: [2, BC, HEAD_DIM_PADDED] (2 * 16 * 592 * 2 = 37888 bytes)
    // s_O: [BR, HEAD_DIM_O_PADDED] (16 * 520 * 4 = 33280 bytes)
    // s_logits: [BR, BC] (16 * 16 * 4 = 1024 bytes) - float
    // s_P: [BR, BC] (16 * 16 * 2 = 512 bytes) - bf16
    // s_stats: [BR] (m, l, scales) (16 * 4 * 3 = 192 bytes)
    // Total: ~92 KB. Fits 2 blocks per SM.
    
    extern __shared__ __align__(16) char smem[];
    __nv_bfloat16* s_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* s_K = s_Q + BR * HEAD_DIM_PADDED;
    float* s_O = (float*)(s_K + 2 * BC * HEAD_DIM_PADDED);
    float* s_logits = s_O + BR * HEAD_DIM_O_PADDED;
    __nv_bfloat16* s_P = (__nv_bfloat16*)(s_logits + BR * BC);
    float* s_m = (float*)(s_P + BR * BC);
    float* s_l = s_m + BR;
    float* s_scales = s_l + BR;

    int tid = threadIdx.x;
    int tile_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    TileInfo info = tile_infos[tile_idx];
    int batch_idx = info.batch_idx;
    int q_start_rel = info.q_start;

    int q_seq_start = qo_indptr[batch_idx];
    int q_seq_end = qo_indptr[batch_idx + 1];
    int q_len_seq = q_seq_end - q_seq_start;
    int q_global_base = q_seq_start + q_start_rel;
    int q_valid = min(BR, q_len_seq - q_start_rel);

    int kv_seq_start = kv_indptr[batch_idx];
    int kv_seq_end = kv_indptr[batch_idx + 1];
    int kv_len = kv_seq_end - kv_seq_start;
    int prefix_len = kv_len - q_len_seq;

    // Initialize O and stats
    for (int i = tid; i < BR * HEAD_DIM_O_PADDED; i += THREADS) {
        s_O[i] = 0.0f;
    }
    if (tid < BR) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    __syncthreads();

    // Load Q
    for (int i = tid; i < BR * HEAD_DIM_PADDED / 8; i += THREADS) {
        int idx = i * 8;
        int r = idx / HEAD_DIM_PADDED;
        int c = idx % HEAD_DIM_PADDED;
        
        if (r < q_valid && c < HEAD_DIM_TOTAL) {
             long long global_idx = q_global_base + r;
             long long offset;
             const __nv_bfloat16* src_ptr;
             if (c < 512) {
                 offset = global_idx * num_heads * 512 + head_idx * 512 + c;
                 src_ptr = q_nope;
             } else {
                 offset = global_idx * num_heads * 64 + head_idx * 64 + (c - 512);
                 src_ptr = q_pe;
             }
             *(int4*)(s_Q + idx) = *(const int4*)(src_ptr + offset);
        } else {
             *(int4*)(s_Q + idx) = make_int4(0,0,0,0);
        }
    }
    __syncthreads();

    int num_kv_blocks = (kv_len + BC - 1) / BC;

    // Causal mask optimization: skip blocks entirely in the future
    int max_q_abs = prefix_len + q_start_rel + q_valid - 1;
    int valid_blocks = num_kv_blocks;
    for (int s = 0; s < num_kv_blocks; ++s) {
        if (s * BC > max_q_abs) {
            valid_blocks = s;
            break;
        }
    }

    auto load_k = [&](int step, int buf_idx) {
        int kv_base = step * BC;
        int kv_valid = min(BC, kv_len - kv_base);
        __nv_bfloat16* dst_base = s_K + buf_idx * BC * HEAD_DIM_PADDED;
        
        // Strided load for scattered K/V
        for (int i = tid; i < BC * HEAD_DIM_PADDED / 8; i += THREADS) {
            int idx = i * 8;
            int r = idx / HEAD_DIM_PADDED;
            int c = idx % HEAD_DIM_PADDED;
            
            __nv_bfloat16* dst_ptr = dst_base + idx;
            
            if (r < kv_valid) {
                if (c < HEAD_DIM_TOTAL) {
                    int page = kv_indices[kv_seq_start + kv_base + r];
                    const void* src;
                    if (c < 512) src = ckv_cache + (long long)page * 512 + c;
                    else src = kpe_cache + (long long)page * 64 + (c - 512);
                    __pipeline_memcpy_async(dst_ptr, src, 16);
                } else {
                    *(int4*)dst_ptr = make_int4(0,0,0,0);
                }
            } else {
                *(int4*)dst_ptr = make_int4(0,0,0,0);
            }
        }
        __pipeline_commit();
    };

    if (valid_blocks > 0) load_k(0, 0);

    for (int step = 0; step < valid_blocks; ++step) {
        int cur_buf = step % 2;
        int next_buf = (step + 1) % 2;

        if (step + 1 < valid_blocks) load_k(step + 1, next_buf);
        __pipeline_wait_prior(step + 1 < valid_blocks ? 1 : 0);
        __syncthreads();

        // 1. GEMM: S = Q * K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s;
        wmma::fill_fragment(acc_s, 0.0f);
        
        __nv_bfloat16* k_ptr = s_K + cur_buf * BC * HEAD_DIM_PADDED;

        if (warp_id == 0) {
            for (int k = 0; k < HEAD_DIM_TOTAL; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
                
                wmma::load_matrix_sync(a, s_Q + k, HEAD_DIM_PADDED);
                // Load K sub-block as B with col_major layout to effectively transpose it.
                // K is stored row-major. Loading as col-major accesses A[c][r] instead of A[r][c].
                wmma::load_matrix_sync(b, k_ptr + k, HEAD_DIM_PADDED);
                wmma::mma_sync(acc_s, a, b, acc_s);
            }
            wmma::store_matrix_sync(s_logits, acc_s, BC, wmma::mem_row_major);
        }
        __syncthreads();

        // 2. Softmax
        if (warp_id == 0) {
            for (int r = tid; r < BR; r += 32) {
                float m_row = -INFINITY;
                int q_idx = q_start_rel + r;
                int q_abs = prefix_len + q_idx;
                
                if (r < q_valid) {
                    for (int c = 0; c < BC; ++c) {
                        int kv_idx = step * BC + c;
                        float val = s_logits[r * BC + c];
                        if (kv_idx >= kv_len || kv_idx > q_abs) {
                            val = -INFINITY;
                        } else {
                            val *= sm_scale;
                        }
                        s_logits[r * BC + c] = val;
                        m_row = fmaxf(m_row, val);
                    }
                }
                
                float m_prev = s_m[r];
                float m_new = fmaxf(m_prev, m_row);
                float d = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_new);
                
                s_m[r] = m_new;
                s_scales[r] = d;
                
                float row_sum = 0.0f;
                if (r < q_valid) {
                    for (int c = 0; c < BC; ++c) {
                        float val = s_logits[r * BC + c];
                        if (val != -INFINITY) {
                            float p = expf(val - m_new);
                            s_logits[r * BC + c] = p; 
                            row_sum += p;
                        } else {
                            s_logits[r * BC + c] = 0.0f;
                        }
                    }
                }
                s_l[r] = s_l[r] * d + row_sum;
            }
        }
        __syncthreads();

        // 3. Scale Accumulators (O)
        for (int i = tid; i < BR * HEAD_DIM_O_PADDED; i += THREADS) {
            int r = i / HEAD_DIM_O_PADDED;
            if (r < BR) {
                s_O[i] *= s_scales[r];
            }
        }
        
        // Convert P to BF16
        for (int i = tid; i < BR * BC; i += THREADS) {
            s_P[i] = __float2bfloat16(s_logits[i]);
        }
        __syncthreads();

        // 4. GEMM 2: O += P * V
        // Split 512 columns among 4 warps -> 128 columns per warp.
        int col_start = warp_id * 128;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_p;
        wmma::load_matrix_sync(frag_p, s_P, BC); 
        
        for (int t = 0; t < 8; ++t) { // 8 tiles of 16 columns = 128 columns
             int c_off = col_start + t * 16;
             wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o;
             wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_v;
             
             wmma::load_matrix_sync(acc_o, s_O + c_off, HEAD_DIM_O_PADDED, wmma::mem_row_major);
             
             // V is subset of K (first 512 cols).
             wmma::load_matrix_sync(frag_v, k_ptr + c_off, HEAD_DIM_PADDED);
             
             wmma::mma_sync(acc_o, frag_p, frag_v, acc_o);
             
             wmma::store_matrix_sync(s_O + c_off, acc_o, HEAD_DIM_O_PADDED, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // Final Write
    for (int i = tid; i < BR * HEAD_DIM_CKV; i += THREADS) {
        int r = i / HEAD_DIM_CKV;
        int c = i % HEAD_DIM_CKV;
        
        if (r < q_valid) {
            float val = s_O[r * HEAD_DIM_O_PADDED + c];
            float l = s_l[r];
            if (l > 0.0f) val /= l;
            
            long long idx = (long long)(q_global_base + r) * num_heads * HEAD_DIM_CKV + head_idx * HEAD_DIM_CKV + c;
            output[idx] = __float2bfloat16(val);
        }
    }
    
    if (tid < BR && tid < q_valid) {
        float m = s_m[tid];
        float l = s_l[tid];
        float lse_val = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
        lse[(long long)(q_global_base + tid) * num_heads + head_idx] = lse_val / logf(2.0f);
    }
}

void run_mla_paged_prefill(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float sm_scale,
    torch::Tensor output,
    torch::Tensor lse,
    torch::Tensor tile_infos
) {
    int num_tiles = tile_infos.size(0) / sizeof(TileInfo);
    int num_heads = q_nope.size(1);
    
    // Shared memory size: ~92KB
    int smem_size = 96000;
    cudaFuncSetAttribute(mla_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    dim3 grid(num_tiles, num_heads);
    dim3 block(THREADS);
    
    mla_prefill_kernel<<<grid, block, smem_size>>>(
        (const __nv_bfloat16*)q_nope.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)q_pe.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)ckv_cache.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)kpe_cache.data_ptr<at::BFloat16>(),
        qo_indptr.data_ptr<int>(),
        kv_indptr.data_ptr<int>(),
        kv_indices.data_ptr<int>(),
        sm_scale,
        (__nv_bfloat16*)output.data_ptr<at::BFloat16>(),
        lse.data_ptr<float>(),
        (const TileInfo*)tile_infos.data_ptr(),
        num_heads
    );
}
