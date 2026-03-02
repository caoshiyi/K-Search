#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <vector_types.h>

using namespace nvcuda;

// Configuration
constexpr int NUM_QO_HEADS = 16;
constexpr int HEAD_DIM_CKV = 512;
constexpr int HEAD_DIM_KPE = 64;
constexpr int TILE_SIZE = 64; 
constexpr int WARP_SIZE = 32;
constexpr int TOPK = 2048;
// Use 128 threads to match 1 block per SM with high shared memory usage.
// TILE_SIZE=64 with 128 threads means 4 warps split the 64 tokens (16 per warp).
constexpr int BLOCK_THREADS = 128; 

// Strides for shared memory to avoid bank conflicts
// 520 bf16 = 1040 bytes. 1040 % 128 = 16 -> Good bank offset.
constexpr int STRIDE_CKV = 520; 
// 72 bf16 = 144 bytes. 144 % 128 = 16 -> Good.
constexpr int STRIDE_KPE = 72; 
// 72 floats = 288 bytes. 288 % 128 = 32 -> Good.
constexpr int STRIDE_TILE = 72;

// CP.ASYNC Wrapper (Cache All - Standard L2 usage)
#define CP_ASYNC_CA(dst, src) \
    do { \
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst)); \
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(src)); \
    } while(0)

#define CP_ASYNC_COMMIT() asm volatile("cp.async.commit_group;\n")
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" :: "n"(N))

struct alignas(128) SharedStorage {
    // Q matrices
    __nv_bfloat16 q_ckv[NUM_QO_HEADS][STRIDE_CKV]; // 16 * 520 * 2 = 16.6 KB
    __nv_bfloat16 q_pe[NUM_QO_HEADS][STRIDE_KPE];   // 16 * 72 * 2 = 2.3 KB
    
    // Double buffered K caches
    // 2 * 64 * 520 * 2 = 133 KB
    __nv_bfloat16 k_ckv[2][TILE_SIZE][STRIDE_CKV];
    // 2 * 64 * 72 * 2 = 18.4 KB
    __nv_bfloat16 k_pe[2][TILE_SIZE][STRIDE_KPE];
    
    // Cached indices for the current token (All TOPK indices)
    // 2048 * 4 = 8 KB
    int cached_indices[TOPK];

    // Intermediate results
    // 16 * 72 * 4 = 4.6 KB
    float s_logits[NUM_QO_HEADS][STRIDE_TILE]; 
    // 16 * 72 * 2 = 2.3 KB
    __nv_bfloat16 p_matrix[NUM_QO_HEADS][STRIDE_TILE];
    
    // Scratches for reduction/rescale
    float scale_scratch[4][256]; 
    float red_sum[NUM_QO_HEADS];

    // Contiguity and Empty Tile flags
    bool cached_contig[TOPK / TILE_SIZE];
    bool cached_empty[TOPK / TILE_SIZE];
    
    // Total Size ~ 190KB. Fits in B200 SMEM (228KB).
};

__device__ __forceinline__ void load_k_tile_async(
    SharedStorage& smem,
    int buf_idx,
    int step_idx,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    int tid
) {
    if (smem.cached_empty[step_idx]) {
        CP_ASYNC_COMMIT();
        return;
    }

    bool contig = smem.cached_contig[step_idx];
    const int* indices = &smem.cached_indices[step_idx * TILE_SIZE];

    if (contig) {
        int start_node = indices[0];
        
        // Load K_CKV: 64 rows * 512 cols.
        // Block-coalesced load using all 128 threads.
        // Each thread loads 4 * int4 (16 elements = 32 bytes) per iteration? No.
        // 64 * 512 = 32768 elements. 128 threads. 256 elements/thread.
        // 32 * int4 per thread.
        #pragma unroll 4
        for (int i = 0; i < 32; ++i) {
            int task_idx = tid + i * BLOCK_THREADS;
            int r = task_idx >> 6; // / 64
            int c_chunk = task_idx & 63; 
            
            size_t src_offset = (static_cast<size_t>(start_node + r) * HEAD_DIM_CKV + c_chunk * 8) * sizeof(__nv_bfloat16);
            const void* src = reinterpret_cast<const char*>(ckv_cache) + src_offset;
            void* dst = &smem.k_ckv[buf_idx][r][c_chunk * 8];
            CP_ASYNC_CA(dst, src);
        }
        
        // Load K_PE: 64 rows * 64 cols.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int task_idx = tid + i * BLOCK_THREADS;
            int r = task_idx >> 3; // / 8
            int c_chunk = task_idx & 7;

            size_t src_offset = (static_cast<size_t>(start_node + r) * HEAD_DIM_KPE + c_chunk * 8) * sizeof(__nv_bfloat16);
            const void* src = reinterpret_cast<const char*>(kpe_cache) + src_offset;
            void* dst = &smem.k_pe[buf_idx][r][c_chunk * 8];
            CP_ASYNC_CA(dst, src);
        }
    } else {
        // Row-Based Gather Optimization
        // 128 threads, 64 rows.
        // Each row is assigned to 2 threads.
        int r = tid >> 1;     // Row index 0..63
        int sub = tid & 1;    // 0 or 1
        
        int idx = indices[r];
        
        // K_CKV Gather
        if (idx != -1) {
            // Base address for this row in global memory
            // 512 bf16 = 1024 bytes.
            const char* base_src = reinterpret_cast<const char*>(ckv_cache) + static_cast<size_t>(idx) * HEAD_DIM_CKV * sizeof(__nv_bfloat16);
            // 2 threads cover 64 chunks (8 bf16 each). Each thread does 32 chunks.
            // Stride is 2 chunks.
            #pragma unroll 4
            for (int c = sub; c < 64; c += 2) {
                // c * 8 elements * 2 bytes = c * 16 bytes
                size_t offset = c * 16;
                const void* src = base_src + offset;
                void* dst = &smem.k_ckv[buf_idx][r][c * 8];
                CP_ASYNC_CA(dst, src);
            }
        } else {
            // Zero out the row (padding)
            // Cannot use CP_ASYNC for zeroing. Use register stores.
            int4 zero = make_int4(0, 0, 0, 0);
            #pragma unroll 4
            for (int c = sub; c < 64; c += 2) {
                *reinterpret_cast<int4*>(&smem.k_ckv[buf_idx][r][c * 8]) = zero;
            }
        }

        // K_PE Gather
        // 64 cols = 8 chunks.
        // 2 threads cover 8 chunks. Each thread does 4 chunks.
        if (idx != -1) {
            const char* base_src = reinterpret_cast<const char*>(kpe_cache) + static_cast<size_t>(idx) * HEAD_DIM_KPE * sizeof(__nv_bfloat16);
            #pragma unroll
            for (int c = sub; c < 8; c += 2) {
                size_t offset = c * 16;
                const void* src = base_src + offset;
                void* dst = &smem.k_pe[buf_idx][r][c * 8];
                CP_ASYNC_CA(dst, src);
            }
        } else {
            int4 zero = make_int4(0, 0, 0, 0);
            #pragma unroll
            for (int c = sub; c < 8; c += 2) {
                 *reinterpret_cast<int4*>(&smem.k_pe[buf_idx][r][c * 8]) = zero;
            }
        }
    }
    CP_ASYNC_COMMIT();
}

template<typename Accumulator>
__device__ __forceinline__ void compute_tile(
    SharedStorage& smem,
    int cur_buf,
    int step_idx,
    float sm_scale,
    Accumulator (&acc_out)[8],
    float& run_m,
    float& run_l,
    int tid,
    int warp_id,
    int lane_id,
    const wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> (&q_pe_frags)[4]
) {
    if (smem.cached_empty[step_idx]) return;

    // 1. Compute Q * K^T
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s;
    wmma::fill_fragment(acc_s, 0.0f);
    int s_col_offset = warp_id * 16; 

    // Q_CKV * K_CKV^T
    // Q loaded from SMEM, K from SMEM
    #pragma unroll
    for (int k = 0; k < HEAD_DIM_CKV; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag; 
        
        wmma::load_matrix_sync(a_frag, &smem.q_ckv[0][k], STRIDE_CKV);
        wmma::load_matrix_sync(b_frag, &smem.k_ckv[cur_buf][s_col_offset][k], STRIDE_CKV);
        
        wmma::mma_sync(acc_s, a_frag, b_frag, acc_s);
    }
    
    // Q_PE * K_PE^T
    // Q_PE is in REGISTERS (q_pe_frags)
    #pragma unroll
    for (int k_idx = 0; k_idx < 4; ++k_idx) {
        int k = k_idx * 16;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
        
        wmma::load_matrix_sync(b_frag, &smem.k_pe[cur_buf][s_col_offset][k], STRIDE_KPE);
        
        // Use the preloaded Q fragment
        wmma::mma_sync(acc_s, q_pe_frags[k_idx], b_frag, acc_s);
    }

    // Scale scores
    #pragma unroll
    for (int i=0; i<acc_s.num_elements; ++i) acc_s.x[i] *= sm_scale;
    
    // Store scores to SMEM
    float* s_ptr = &smem.s_logits[0][s_col_offset];
    wmma::store_matrix_sync(s_ptr, acc_s, STRIDE_TILE, wmma::mem_row_major);
    __syncthreads();

    // 2. Parallel Softmax
    // Map threads to heads: 128 threads, 16 heads -> 8 threads per head.
    int head = tid / 8;  
    int g_lane = tid % 8; 
    
    float local_max = -FLT_MAX;
    const int* indices = &smem.cached_indices[step_idx * TILE_SIZE];
    
    int my_indices[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        my_indices[i] = indices[g_lane + i * 8];
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int c = g_lane + i * 8;
        if (my_indices[i] != -1) {
            local_max = fmaxf(local_max, smem.s_logits[head][c]);
        }
    }
    
    float tile_max = local_max;
    #pragma unroll
    for (int mask = 4; mask > 0; mask /= 2) {
         tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, mask));
    }
    
    float m_prev = run_m;
    float m_new;
    float group_max = __shfl_sync(0xffffffff, tile_max, lane_id & ~7);
    
    if (g_lane == 0) {
         m_new = fmaxf(m_prev, group_max);
    }
    // Broadcast m_new to all threads in group
    m_new = __shfl_sync(0xffffffff, m_new, lane_id & ~7); 
    m_prev = __shfl_sync(0xffffffff, m_prev, lane_id & ~7);
    
    float exp_factor = (m_prev == -FLT_MAX && m_new == -FLT_MAX) ? 1.0f : expf(m_prev - m_new);
    
    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
         int c = g_lane + i * 8;
         float p_val = 0.0f;
         if (my_indices[i] != -1) {
              float val = smem.s_logits[head][c];
              p_val = expf(val - m_new);
         }
         smem.p_matrix[head][c] = __float2bfloat16(p_val);
         local_sum += p_val;
    }
    
    float tile_sum = local_sum;
    #pragma unroll
    for (int mask = 4; mask > 0; mask /= 2) {
         tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, mask);
    }
    
    if (g_lane == 0) {
         run_l = run_l * exp_factor + tile_sum;
         run_m = m_new;
         smem.red_sum[head] = exp_factor; 
    }
    __syncthreads();

    // 3. Rescale Accumulators
    float* scratch = &smem.scale_scratch[warp_id][0];
    #pragma unroll
    for (int f = 0; f < 8; ++f) {
        wmma::store_matrix_sync(scratch, acc_out[f], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane_id; i < 256; i += 32) {
            int row = i / 16; 
            scratch[i] *= smem.red_sum[row];
        }
        __syncwarp();
        wmma::load_matrix_sync(acc_out[f], scratch, 16, wmma::mem_row_major);
    }

    // 4. Compute P * V
    int out_col_start = warp_id * 128; 
    
    #pragma unroll
    for (int k_in = 0; k_in < TILE_SIZE; k_in += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, &smem.p_matrix[0][k_in], STRIDE_TILE);
        
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int c_off = out_col_start + t * 16;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;
            wmma::load_matrix_sync(v_frag, &smem.k_ckv[cur_buf][k_in][c_off], STRIDE_CKV);
            
            wmma::mma_sync(acc_out[t], p_frag, v_frag, acc_out[t]);
        }
    }
}

__global__ __launch_bounds__(BLOCK_THREADS)
void dsa_attn_kernel_optimized(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int* __restrict__ sparse_indices,
    float sm_scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse
) {
    extern __shared__ char smem_buffer[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 1. Load Q
    {
        const __nv_bfloat16* qn_g = q_nope + token_idx * NUM_QO_HEADS * HEAD_DIM_CKV;
        const __nv_bfloat16* qp_g = q_pe + token_idx * NUM_QO_HEADS * HEAD_DIM_KPE;
        
        // Load Q_CKV
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int task_idx = tid + i * BLOCK_THREADS;
            if (task_idx < NUM_QO_HEADS * 64) { 
                int r = task_idx >> 6; 
                int c = task_idx & 63; 
                const int4* src = reinterpret_cast<const int4*>(qn_g) + task_idx;
                int4* dst = reinterpret_cast<int4*>(&smem.q_ckv[r][c * 8]);
                *dst = *src;
            }
        }

        // Load Q_PE
        if (tid < 128) {
            int r = tid >> 3; 
            int c = tid & 7;  
            const int4* src = reinterpret_cast<const int4*>(qp_g) + tid;
            int4* dst = reinterpret_cast<int4*>(&smem.q_pe[r][c * 8]);
            *dst = *src;
        }
    }

    // 2. Load Indices Cache
    {
        const int* indices_base = sparse_indices + token_idx * TOPK;
        int4* dst = reinterpret_cast<int4*>(smem.cached_indices);
        const int4* src = reinterpret_cast<const int4*>(indices_base);
        #pragma unroll
        for (int i = 0; i < 4; ++i) { 
            int idx = tid + i * BLOCK_THREADS;
            if (idx < (TOPK / 4)) {
                dst[idx] = src[idx];
            }
        }
    }
    __syncthreads();

    // Load Q_PE into registers for reuse
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_pe_frags[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        wmma::load_matrix_sync(q_pe_frags[k], &smem.q_pe[0][k * 16], STRIDE_KPE);
    }

    // 3. Compute Contiguity and Empty Status
    int num_tiles = TOPK / TILE_SIZE; 
    if (tid < num_tiles) {
       int base = tid * TILE_SIZE;
       int start = smem.cached_indices[base];
       bool c = true;
       bool e = true;
       
       if (start == -1) {
           c = false;
           for (int k = 0; k < TILE_SIZE; ++k) {
               if (smem.cached_indices[base + k] != -1) {
                   e = false;
                   break;
               }
           }
       } else {
           e = false;
           for (int k = 1; k < TILE_SIZE; ++k) {
               if (smem.cached_indices[base + k] != start + k) {
                   c = false; 
                   break;
               }
           }
       }
       smem.cached_contig[tid] = c;
       smem.cached_empty[tid] = e;
    }
    __syncthreads();

    // Accumulators
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_out[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) wmma::fill_fragment(acc_out[i], 0.0f);

    float run_m = -FLT_MAX;
    float run_l = 0.0f;
    
    // Pipeline Prologue
    load_k_tile_async(smem, 0, 0, ckv_cache, kpe_cache, tid);

    // Main Loop
    #pragma unroll 1
    for (int step = 0; step < num_tiles - 1; ++step) {
        int cur_buf = step % 2;
        int next_buf = (step + 1) % 2;
        int next_step = step + 1;

        load_k_tile_async(smem, next_buf, next_step, ckv_cache, kpe_cache, tid);
        CP_ASYNC_WAIT_GROUP(1); 
        __syncthreads();

        compute_tile(smem, cur_buf, step, sm_scale, acc_out, run_m, run_l, tid, warp_id, lane_id, q_pe_frags);
        
        __syncthreads();
    }
    
    // Epilogue
    {
        int step = num_tiles - 1;
        int cur_buf = step % 2;
        
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
        
        compute_tile(smem, cur_buf, step, sm_scale, acc_out, run_m, run_l, tid, warp_id, lane_id, q_pe_frags);
        __syncthreads();
    }
    
    // Write LSE
    if ((tid % 8) == 0) { 
        int h = tid / 8;
        float norm = run_l;
        float max_val = run_m;
        if (norm > 0.0f) {
            lse[token_idx * NUM_QO_HEADS + h] = (max_val + logf(norm)) / 0.69314718056f;
        } else {
            lse[token_idx * NUM_QO_HEADS + h] = -INFINITY;
        }
        smem.red_sum[h] = norm; 
    }
    __syncthreads();

    // Store Output with Coalesced Global Write
    // Reuse smem.k_ckv as output buffer since it's no longer needed for K.
    // Cast to flat __nv_bfloat16 array.
    __nv_bfloat16* smem_output = reinterpret_cast<__nv_bfloat16*>(&smem.k_ckv[0][0][0]);
    
    float* scratch = &smem.scale_scratch[warp_id][0];
    int out_col_start = warp_id * 128;
    
    #pragma unroll
    for (int f = 0; f < 8; ++f) {
        wmma::store_matrix_sync(scratch, acc_out[f], 16, wmma::mem_row_major);
        __syncwarp();
        
        for (int i = lane_id; i < 256; i += 32) {
            int row = i / 16;
            int col_loc = i % 16;
            int col_glob = out_col_start + f * 16 + col_loc;
            
            float norm = smem.red_sum[row];
            float val = scratch[i];
            float final_val = (norm > 0.0f) ? (val / norm) : 0.0f;
            
            // Store to shared memory in linear format for coalescing
            smem_output[row * HEAD_DIM_CKV + col_glob] = __float2bfloat16(final_val);
        }
    }
    __syncthreads();
    
    // Coalesced write to global memory
    int num_elements = NUM_QO_HEADS * HEAD_DIM_CKV;
    int base_out_idx = token_idx * num_elements;
    
    for (int i = tid; i < num_elements; i += BLOCK_THREADS) {
        output[base_out_idx + i] = smem_output[i];
    }
}

void run_dsa_attn(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    float sm_scale,
    torch::Tensor output,
    torch::Tensor lse
) {
    int num_tokens = q_nope.size(0);
    
    // Max Shared Memory Config
    int shared_mem_size = sizeof(SharedStorage);
    cudaFuncSetAttribute(dsa_attn_kernel_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    dim3 grid(num_tokens);
    dim3 block(BLOCK_THREADS);

    dsa_attn_kernel_optimized<<<grid, block, shared_mem_size>>>(
        (const __nv_bfloat16*)q_nope.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)q_pe.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)ckv_cache.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)kpe_cache.data_ptr<at::BFloat16>(),
        sparse_indices.data_ptr<int>(),
        sm_scale,
        (__nv_bfloat16*)output.data_ptr<at::BFloat16>(),
        lse.data_ptr<float>()
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}