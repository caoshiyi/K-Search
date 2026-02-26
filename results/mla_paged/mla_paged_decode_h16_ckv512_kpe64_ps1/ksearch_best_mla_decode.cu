#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cmath>

using namespace nvcuda;

// H100 Tuning Parameters: Occupancy Focus
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;

constexpr int NUM_HEADS = 16;
constexpr int HEAD_DIM_CKV = 512;
constexpr int HEAD_DIM_KPE = 64;

// Reduce CHUNK_SIZE to 32 to fit 2 blocks per SM (High Occupancy)
constexpr int CHUNK_SIZE = 32; 
constexpr int NUM_TILES_S = CHUNK_SIZE / 16; // 2
constexpr int ACC_O_TILES = (HEAD_DIM_CKV / NUM_WARPS) / 16; // 4

// Memory Layout Constants with padding
// Stride for CKV: 512 + 8 = 520 (1040 bytes, avoids 128-byte bank conflicts)
constexpr int STRIDE_CKV = HEAD_DIM_CKV + 8; 
// Stride for KPE: 64 + 8 = 72 (144 bytes)
constexpr int STRIDE_KPE = HEAD_DIM_KPE + 8; 
// Stride for S (float): 32 + 8 = 40 (160 bytes)
constexpr int S_STRIDE = CHUNK_SIZE + 8; 

struct __align__(128) SharedStorage {
    // KV Buffers (Double Buffered)
    // 2 * 32 * 520 * 2 = 66,560 bytes
    __nv_bfloat16 kc_buf[2][CHUNK_SIZE * STRIDE_CKV]; 
    // 2 * 32 * 72 * 2 = 9,216 bytes
    __nv_bfloat16 kp_buf[2][CHUNK_SIZE * STRIDE_KPE]; 
    
    // P Matrix for Softmax -> Output accumulation
    // 16 * 32 * 2 = 1024 bytes
    __nv_bfloat16 p_mat[NUM_HEADS * CHUNK_SIZE]; 

    // Reused scratch memory
    union {
        // 8 warps * 16 heads * 40 * 4 = 20,480 bytes
        float s_partials[NUM_WARPS][16 * S_STRIDE]; 
        // 8 warps * 256 * 4 = 8192 bytes
        float scale_buf[NUM_WARPS][16 * 16]; 
    } scratch;

    // Softmax statistics
    float lse_max[NUM_HEADS];
    float lse_sum[NUM_HEADS];
    float broadcast_alpha[NUM_HEADS];
};

__device__ __forceinline__ void load_kv_chunk_coop(
    SharedStorage& smem,
    int buf_idx,
    const int* __restrict__ kv_indices,
    int page_start_offset,
    int valid_rows,
    const __nv_bfloat16* __restrict__ ckv_base,
    const __nv_bfloat16* __restrict__ kpe_base,
    int tid
) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Distribute 32 rows across 8 warps -> 4 rows per warp
    constexpr int ROWS_PER_WARP = CHUNK_SIZE / NUM_WARPS; // 4
    int warp_row_start = warp_id * ROWS_PER_WARP;
    
    int my_page_idx = 0;
    
    // Pre-load indices for this warp (4 rows)
    if (lane_id < ROWS_PER_WARP) {
        if (warp_row_start + lane_id < valid_rows) {
            my_page_idx = kv_indices[page_start_offset + warp_row_start + lane_id];
        }
    }
    
    // Distribute work: 4 rows per warp
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = warp_row_start + r;
        
        // Broadcast the index for the current row
        int page_idx = __shfl_sync(0xffffffff, my_page_idx, r);
        
        __nv_bfloat16* dst_kc = smem.kc_buf[buf_idx] + row * STRIDE_CKV;
        __nv_bfloat16* dst_kp = smem.kp_buf[buf_idx] + row * STRIDE_KPE;

        if (row < valid_rows) {
            long long row_offset_ckv = (long long)page_idx * HEAD_DIM_CKV;
            
            // Vectorized Async Load CKV (512 elements = 1024 bytes)
            // 32 threads * 32 bytes/thread = 1024 bytes
            int col1 = lane_id * 8; // 8 bf16s = 16 bytes
            __pipeline_memcpy_async(dst_kc + col1, ckv_base + row_offset_ckv + col1, 16);
            int col2 = col1 + 256;
            __pipeline_memcpy_async(dst_kc + col2, ckv_base + row_offset_ckv + col2, 16);
            
            // KPE (64 elements = 128 bytes)
            long long row_offset_kpe = (long long)page_idx * HEAD_DIM_KPE;
            if (lane_id < 8) {
                int col_k = lane_id * 8; 
                __pipeline_memcpy_async(dst_kp + col_k, kpe_base + row_offset_kpe + col_k, 16);
            }
        } else {
            // Zero padding
            int col1 = lane_id * 8;
            int col2 = col1 + 256;
            *reinterpret_cast<int4*>(dst_kc + col1) = make_int4(0,0,0,0);
            *reinterpret_cast<int4*>(dst_kc + col2) = make_int4(0,0,0,0);
            if (lane_id < 8) {
                int col_k = lane_id * 8;
                *reinterpret_cast<int4*>(dst_kp + col_k) = make_int4(0,0,0,0);
            }
        }
    }
}

__global__ __launch_bounds__(256)
void mla_decode_step_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    __nv_bfloat16* __restrict__ temp_out,
    float* __restrict__ temp_lse,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
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

    // --- Load Q into Registers (Persistent Fragments) ---
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frags[4];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qpe_frags[4];

    const __nv_bfloat16* qn_g = q_nope + batch_idx * NUM_HEADS * HEAD_DIM_CKV;
    int k_start = warp_id * 64; 
    
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        wmma::load_matrix_sync(q_frags[k], qn_g + k_start + k * 16, HEAD_DIM_CKV);
    }
    
    if (warp_id == 0) {
        const __nv_bfloat16* qp_g = q_pe + batch_idx * NUM_HEADS * HEAD_DIM_KPE;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
             wmma::load_matrix_sync(qpe_frags[k], qp_g + k * 16, HEAD_DIM_KPE);
        }
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[ACC_O_TILES];
    #pragma unroll
    for (int i=0; i<ACC_O_TILES; ++i) wmma::fill_fragment(acc_o[i], 0.0f);
    
    int page_start = kv_indptr[batch_idx];
    int total_tokens_all = kv_indptr[batch_idx + 1] - page_start;
    
    int total_tokens = 0;
    int start_token = 0;
    
    if (total_tokens_all > 0) {
        int tokens_per_split = (total_tokens_all + num_splits - 1) / num_splits;
        start_token = split_idx * tokens_per_split;
        int end_token = min(start_token + tokens_per_split, total_tokens_all);
        if (start_token < end_token) {
            total_tokens = end_token - start_token;
        }
    }
    
    if (total_tokens <= 0) {
        if (num_splits > 1) {
            if (tid < NUM_HEADS) {
                long long idx = (long long)batch_idx * NUM_HEADS * num_splits + (long long)tid * num_splits + split_idx;
                temp_lse[idx] = -INFINITY;
            }
            long long base_offset = (long long)batch_idx * NUM_HEADS * num_splits * HEAD_DIM_CKV + 
                                    (long long)split_idx * HEAD_DIM_CKV;
            for (int i = tid; i < NUM_HEADS * HEAD_DIM_CKV; i += BLOCK_SIZE) {
                 int h = i / HEAD_DIM_CKV;
                 int d = i % HEAD_DIM_CKV;
                 long long idx = base_offset + (long long)h * num_splits * HEAD_DIM_CKV + d;
                 temp_out[idx] = __float2bfloat16(0.0f);
            }
        } else {
             if (tid < NUM_HEADS) lse[batch_idx * NUM_HEADS + tid] = -INFINITY;
             for (int i = tid; i < NUM_HEADS * HEAD_DIM_CKV; i += BLOCK_SIZE) {
                 output[(batch_idx * NUM_HEADS * HEAD_DIM_CKV) + i] = __float2bfloat16(0.0f);
             }
        }
        return;
    }
    
    int num_chunks = (total_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // --- Pipelined Prologue ---
    if (total_tokens > 0) {
        int valid_rows_0 = min(CHUNK_SIZE, total_tokens);
        load_kv_chunk_coop(smem, 0, kv_indices, page_start + start_token, valid_rows_0, ckv_cache, kpe_cache, tid);
        __pipeline_commit();

        if (num_chunks > 1) {
            int valid_rows_1 = min(CHUNK_SIZE, total_tokens - CHUNK_SIZE);
            load_kv_chunk_coop(smem, 1, kv_indices, page_start + start_token + CHUNK_SIZE, valid_rows_1, ckv_cache, kpe_cache, tid);
            __pipeline_commit();
        }
    }
    
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_k; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s[NUM_TILES_S]; 
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_p;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_v;

    // --- Pipelined Main Loop ---
    for (int step = 0; step < num_chunks; ++step) {
        if (step == num_chunks - 1) {
             __pipeline_wait_prior(0);
        } else {
             __pipeline_wait_prior(1);
        }
        __syncthreads();
        
        int buf_idx = step % 2;

        #pragma unroll
        for (int i=0; i<NUM_TILES_S; ++i) wmma::fill_fragment(acc_s[i], 0.0f);
        
        // Compute S = Q * K^T + Qpe * Kpe^T
        #pragma unroll
        for (int t = 0; t < NUM_TILES_S; ++t) {
            int t_offset = t * 16;
            
            // CKV Part
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int k_curr = k_start + k * 16;
                wmma::load_matrix_sync(frag_k, smem.kc_buf[buf_idx] + t_offset * STRIDE_CKV + k_curr, STRIDE_CKV);
                wmma::mma_sync(acc_s[t], q_frags[k], frag_k, acc_s[t]);
            }
            
            // KPE Part
            if (warp_id == 0) {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int k_curr = k * 16;
                    wmma::load_matrix_sync(frag_k, smem.kp_buf[buf_idx] + t_offset * STRIDE_KPE + k_curr, STRIDE_KPE);
                    wmma::mma_sync(acc_s[t], qpe_frags[k], frag_k, acc_s[t]);
                }
            }
        }
        
        // Store S partials to smem for reduction
        #pragma unroll
        for (int t = 0; t < NUM_TILES_S; ++t) {
            float* dst = smem.scratch.s_partials[warp_id] + t * 16;
            wmma::store_matrix_sync(dst, acc_s[t], S_STRIDE, wmma::mem_row_major);
        }
        __syncthreads();
        
        // Reduce across warps and Softmax
        for (int i = tid; i < 16 * CHUNK_SIZE; i += BLOCK_SIZE) {
            int h = i / CHUNK_SIZE;
            int c = i % CHUNK_SIZE;
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
        int col_lane = tid % 16; // 0..15
        
        // Block-wide Max/Sum for Softmax. 256 threads.
        if (row < 16) {
             float max_val = -INFINITY;
             for (int k = col_lane; k < CHUNK_SIZE; k += 16) {
                 max_val = max(max_val, smem.scratch.s_partials[0][row * S_STRIDE + k]);
             }
             // Reduce within the 16 threads of the head
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
            for (int k = col_lane; k < CHUNK_SIZE; k += 16) {
                float val = smem.scratch.s_partials[0][row * S_STRIDE + k];
                float p = (cur_max == -INFINITY) ? 0.0f : expf(val - cur_max);
                sum_p += p;
                smem.p_mat[row * CHUNK_SIZE + k] = __float2bfloat16(p);
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
        
        // Output Rescale
        for (int i=0; i<ACC_O_TILES; ++i) {
            wmma::store_matrix_sync(smem.scratch.scale_buf[warp_id], acc_o[i], 16, wmma::mem_row_major);
            __syncwarp();
            
            int lane = tid % 32;
            #pragma unroll
            for(int j=0; j<8; ++j) {
                int idx = lane + j * 32; 
                int r = idx / 16;
                if (idx < 256) {
                    smem.scratch.scale_buf[warp_id][idx] *= smem.broadcast_alpha[r];
                }
            }
            __syncwarp();
            
            wmma::load_matrix_sync(acc_o[i], smem.scratch.scale_buf[warp_id], 16, wmma::mem_row_major);
        }
        
        // Accumulate Output: O += P * V
        int warp_col_start = warp_id * 64;
        #pragma unroll
        for (int t = 0; t < NUM_TILES_S; ++t) {
            int t_offset = t * 16;
            wmma::load_matrix_sync(frag_p, smem.p_mat + t_offset, CHUNK_SIZE); 
            
            #pragma unroll
            for (int k = 0; k < ACC_O_TILES; ++k) { 
                int col_sub = warp_col_start + k * 16;
                wmma::load_matrix_sync(frag_v, smem.kc_buf[buf_idx] + t_offset * STRIDE_CKV + col_sub, STRIDE_CKV);
                wmma::mma_sync(acc_o[k], frag_p, frag_v, acc_o[k]);
            }
        }
        __syncthreads();
        
        // Issue Load for Step + 2
        int next_step_load = step + 2;
        if (next_step_load < num_chunks) {
            int valid_rows_next = min(CHUNK_SIZE, total_tokens - next_step_load * CHUNK_SIZE);
            load_kv_chunk_coop(smem, next_step_load % 2, kv_indices, page_start + start_token + next_step_load * CHUNK_SIZE, valid_rows_next, ckv_cache, kpe_cache, tid);
            __pipeline_commit();
        }
    }
    
    // Store Output
    int warp_col_start = warp_id * 64;
    __nv_bfloat16* out_base;
    long long out_stride_h;
    
    if (num_splits == 1) {
        out_base = output + batch_idx * NUM_HEADS * HEAD_DIM_CKV;
        out_stride_h = HEAD_DIM_CKV;
    } else {
        out_base = temp_out + (long long)batch_idx * NUM_HEADS * num_splits * HEAD_DIM_CKV + 
                   (long long)split_idx * HEAD_DIM_CKV; 
        out_stride_h = (long long)num_splits * HEAD_DIM_CKV;
    }
    
    for (int i=0; i<ACC_O_TILES; ++i) {
         wmma::store_matrix_sync(smem.scratch.scale_buf[warp_id], acc_o[i], 16, wmma::mem_row_major);
         __syncwarp();
         
         int lane = tid % 32;
         #pragma unroll
         for(int j=0; j<8; ++j) {
             int idx = lane + j * 32;
             int r = idx / 16;
             int c = idx % 16;
             if (idx < 256) {
                 float val = smem.scratch.scale_buf[warp_id][idx];
                 float sum = smem.lse_sum[r];
                 float res = (sum == 0.0f) ? 0.0f : (val / sum);
                 int global_col = warp_col_start + i * 16 + c;
                 out_base[r * out_stride_h + global_col] = __float2bfloat16(res);
             }
         }
         __syncwarp();
    }
    
    if (tid < NUM_HEADS) {
        float val = logf(smem.lse_sum[tid]) + smem.lse_max[tid];
        if (smem.lse_sum[tid] == 0.0f) val = -INFINITY;
        
        if (num_splits == 1) {
            if (val != -INFINITY) val *= 1.44269504f;
            lse[batch_idx * NUM_HEADS + tid] = val;
        } else {
            long long idx = (long long)batch_idx * NUM_HEADS * num_splits + (long long)tid * num_splits + split_idx;
            temp_lse[idx] = val; 
        }
    }
}

__global__ void mla_decode_reduce_kernel_opt(
    const __nv_bfloat16* __restrict__ temp_out, 
    const float* __restrict__ temp_lse,          
    __nv_bfloat16* __restrict__ output,          
    float* __restrict__ lse,                     
    int num_splits
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    extern __shared__ float s_lse[]; 
    long long lse_offset_base = (long long)batch_idx * NUM_HEADS * num_splits + (long long)head_idx * num_splits;
    
    // 1. Reduce LSE
    for (int s = tid; s < num_splits; s += blockDim.x) {
        s_lse[s] = temp_lse[lse_offset_base + s];
    }
    __syncthreads();
    
    __shared__ float global_lse_val;
    if (tid == 0) {
        float max_val = -INFINITY;
        for (int s = 0; s < num_splits; ++s) {
            float val = s_lse[s];
            if (val > max_val) max_val = val;
        }
        float sum_exp = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            float val = s_lse[s];
            if (val != -INFINITY) {
                sum_exp += expf(val - max_val);
            }
        }
        float g_lse = (max_val == -INFINITY) ? -INFINITY : (max_val + logf(sum_exp));
        global_lse_val = g_lse;
        lse[batch_idx * NUM_HEADS + head_idx] = (g_lse == -INFINITY) ? -INFINITY : (g_lse * 1.44269504f);
    }
    __syncthreads();
    
    float g_lse = global_lse_val;
    
    // 2. Reduce Output
    int dim_idx = tid * 2;
    if (dim_idx < HEAD_DIM_CKV) {
        long long batch_head_offset = (long long)batch_idx * NUM_HEADS * num_splits * HEAD_DIM_CKV + 
                                      (long long)head_idx * num_splits * HEAD_DIM_CKV;
        int split_stride = HEAD_DIM_CKV;
        
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        
        #pragma unroll 4
        for (int s = 0; s < num_splits; ++s) {
            float l_val = s_lse[s];
            if (l_val != -INFINITY) {
                float weight = expf(l_val - g_lse);
                long long idx = batch_head_offset + (long long)s * split_stride + dim_idx;
                
                const float* ptr = reinterpret_cast<const float*>(temp_out + idx);
                float loaded = *ptr;
                __nv_bfloat16* vec = reinterpret_cast<__nv_bfloat16*>(&loaded);
                
                acc0 += __bfloat162float(vec[0]) * weight;
                acc1 += __bfloat162float(vec[1]) * weight;
            }
        }
        
        __nv_bfloat16 res[2];
        res[0] = __float2bfloat16(acc0);
        res[1] = __float2bfloat16(acc1);
        
        long long out_idx = (long long)(batch_idx * NUM_HEADS + head_idx) * HEAD_DIM_CKV + dim_idx;
        *reinterpret_cast<float*>(output + out_idx) = *reinterpret_cast<float*>(res);
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
    int avg_tokens = total_tokens / batch_size;
    
    // Heuristic for split-K: target filling 264 slots (2 blocks per SM on H100)
    int target_blocks = 264;
    int num_splits = (target_blocks + batch_size - 1) / batch_size;
    
    if (avg_tokens < 128) {
        num_splits = 1;
    } else {
        int max_splits_by_work = (avg_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (num_splits > max_splits_by_work) num_splits = max_splits_by_work;
    }
    
    if (num_splits > 128) num_splits = 128;
    if (num_splits < 1) num_splits = 1;
    
    auto options = q_nope.options();
    torch::Tensor temp_out;
    torch::Tensor temp_lse;
    
    if (num_splits > 1) {
        temp_out = torch::empty({batch_size, NUM_HEADS, num_splits, HEAD_DIM_CKV}, options);
        temp_lse = torch::empty({batch_size, NUM_HEADS, num_splits}, options.dtype(torch::kFloat32));
    }
    
    dim3 grid_step(num_splits, batch_size);
    dim3 block_step(BLOCK_SIZE);
    size_t smem_size = sizeof(SharedStorage);
    
    cudaFuncSetAttribute(mla_decode_step_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    mla_decode_step_kernel<<<grid_step, block_step, smem_size>>>(
        reinterpret_cast<__nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(ckv_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(kpe_cache.data_ptr<at::BFloat16>()),
        kv_indptr.data_ptr<int>(),
        kv_indices.data_ptr<int>(),
        (num_splits > 1) ? reinterpret_cast<__nv_bfloat16*>(temp_out.data_ptr<at::BFloat16>()) : nullptr,
        (num_splits > 1) ? temp_lse.data_ptr<float>() : nullptr,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        sm_scale,
        num_splits
    );
    
    if (num_splits > 1) {
        dim3 grid_reduce(batch_size, NUM_HEADS);
        dim3 block_reduce(256); 
        size_t reduce_smem = num_splits * sizeof(float);
        
        mla_decode_reduce_kernel_opt<<<grid_reduce, block_reduce, reduce_smem>>>(
            reinterpret_cast<__nv_bfloat16*>(temp_out.data_ptr<at::BFloat16>()),
            temp_lse.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            lse.data_ptr<float>(),
            num_splits
        );
    }
}