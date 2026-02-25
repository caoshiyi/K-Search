#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cmath>
#include <algorithm>

using namespace nvcuda;

// H100 Tuning
// Block Size 256 (8 Warps)
// Chunk Size 64
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = 8;
constexpr int CHUNK_SIZE = 64; 

constexpr int NUM_HEADS = 16;
constexpr int HEAD_DIM_CKV = 512;
constexpr int HEAD_DIM_KPE = 64;

// Memory Layouts
constexpr int STRIDE_CKV = HEAD_DIM_CKV + 8; // 520
constexpr int STRIDE_KPE = HEAD_DIM_KPE + 8; // 72
constexpr int STRIDE_S = CHUNK_SIZE + 8; // 72

struct __align__(128) SharedStorage {
    __nv_bfloat16 q_nope[NUM_HEADS * HEAD_DIM_CKV]; 
    __nv_bfloat16 q_pe[NUM_HEADS * HEAD_DIM_KPE];   
    
    // Double Buffered KV
    __nv_bfloat16 kc_buf[2][CHUNK_SIZE * STRIDE_CKV]; 
    __nv_bfloat16 kp_buf[2][CHUNK_SIZE * STRIDE_KPE]; 
    
    // Partials and Scratch
    // S Partials: [8 Warps][16 Heads][72 Stride]
    // Reused as scratch for O rescaling (requires 1KB per warp)
    float s_partials[NUM_WARPS][NUM_HEADS * STRIDE_S]; 
    
    // P Matrix
    __nv_bfloat16 p_mat[NUM_HEADS * CHUNK_SIZE]; 

    // Softmax Stats
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
    // 1024 int4s. 256 threads -> 4 per thread
    #pragma unroll
    for(int i=0; i<4; ++i) dst_qn[i*BLOCK_SIZE + tid] = src_qn[i*BLOCK_SIZE + tid];
    
    const int4* src_qp = reinterpret_cast<const int4*>(qp_g);
    int4* dst_qp = reinterpret_cast<int4*>(smem.q_pe);
    if(tid < 128) dst_qp[tid] = src_qp[tid];
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
    // CKV: 64 rows * 512 cols = 32768 bf16 = 4096 int4s
    // 256 threads -> 16 int4s per thread
    #pragma unroll
    for(int i=0; i<16; ++i) {
        int idx = tid + i * BLOCK_SIZE;
        int row = idx >> 6; // / 64
        int col = idx & 63;
        if(row < valid_rows) {
             int page = kv_indices[page_start_offset + row];
             void* dst = smem.kc_buf[buf_idx] + row * STRIDE_CKV + col * 8;
             const void* src = ckv_base + (long long)page * HEAD_DIM_CKV + col * 8;
             __pipeline_memcpy_async(dst, src, 16);
        }
    }
    // KPE: 64 rows * 64 cols = 4096 bf16 = 512 int4s
    // 256 threads -> 2 int4s per thread
    #pragma unroll
    for(int i=0; i<2; ++i) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < 512) {
             int row = idx >> 3;
             int col = idx & 7;
             if(row < valid_rows) {
                 int page = kv_indices[page_start_offset + row];
                 void* dst = smem.kp_buf[buf_idx] + row * STRIDE_KPE + col * 8;
                 const void* src = kpe_base + (long long)page * HEAD_DIM_KPE + col * 8;
                 __pipeline_memcpy_async(dst, src, 16);
             }
        }
    }
}

__device__ __forceinline__ void zero_invalid_rows(SharedStorage& smem, int buf_idx, int valid, int tid) {
    if(valid == CHUNK_SIZE) return;
    #pragma unroll
    for(int i=0; i<16; ++i) {
        int idx = tid + i * BLOCK_SIZE;
        int row = idx >> 6;
        int col = idx & 63;
        if(row >= valid && row < CHUNK_SIZE) 
            *(int4*)(smem.kc_buf[buf_idx] + row * STRIDE_CKV + col * 8) = make_int4(0,0,0,0);
    }
    #pragma unroll
    for(int i=0; i<2; ++i) {
        int idx = tid + i * BLOCK_SIZE;
        if(idx < 512) {
            int row = idx >> 3;
            int col = idx & 7;
            if(row >= valid && row < CHUNK_SIZE)
                *(int4*)(smem.kp_buf[buf_idx] + row * STRIDE_KPE + col * 8) = make_int4(0,0,0,0);
        }
    }
}

__global__ __launch_bounds__(256, 1)
void mla_decode_step_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    __nv_bfloat16* __restrict__ temp_out,
    float* __restrict__ temp_lse,
    float sm_scale,
    int num_splits
) {
    extern __shared__ char smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);
    
    int split_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    if(tid < NUM_HEADS) {
        smem.lse_max[tid] = -INFINITY;
        smem.lse_sum[tid] = 0.0f;
    }

    load_q(smem, q_nope + batch_idx * NUM_HEADS * HEAD_DIM_CKV, 
                 q_pe + batch_idx * NUM_HEADS * HEAD_DIM_KPE, tid);
    __syncthreads();

    // Cache Q in registers (8 warps split 512 dims -> 64 dims/warp)
    // 64 dims = 4 tiles of 16.
    // Q [16 heads, 64 dims].
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_q[4];
    int q_col_start = warp_id * 64;
    #pragma unroll
    for(int k=0; k<4; ++k) {
        wmma::load_matrix_sync(frag_q[k], smem.q_nope + q_col_start + k*16, HEAD_DIM_CKV);
    }
    
    // Output Acc: 4 fragments for [16 heads, 64 cols] (Split-N output)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[4];
    #pragma unroll
    for(int i=0; i<4; ++i) wmma::fill_fragment(acc_o[i], 0.0f);

    // Tokens
    int page_start = kv_indptr[batch_idx];
    int total_all = kv_indptr[batch_idx+1] - page_start;
    int per_split = (total_all + num_splits - 1) / num_splits;
    int start = split_idx * per_split;
    int end = min(start + per_split, total_all);
    int total = max(0, end - start);
    
    if(total <= 0) {
        if(tid < NUM_HEADS) temp_lse[(batch_idx*num_splits + split_idx)*NUM_HEADS + tid] = -INFINITY;
        // Zero out slice
        __nv_bfloat16* dst = temp_out + (batch_idx*num_splits + split_idx)*NUM_HEADS*HEAD_DIM_CKV;
        int base = warp_id * 64;
        for(int k=0; k<4; ++k) { // 4 tiles -> 64 cols
             int col_off = base + k*16;
             for(int i=tid%32; i<256; i+=32) {
                 int r = i/16; int c = i%16;
                 dst[r*HEAD_DIM_CKV + col_off + c] = __float2bfloat16(0.0f);
             }
        }
        return;
    }
    
    int num_chunks = (total + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int valid = min(CHUNK_SIZE, total);
    
    load_kv_chunk(smem, 0, kv_indices, page_start + start, valid, ckv_cache, kpe_cache, tid);
    __pipeline_commit();
    
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_k;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s[4]; // 4 tiles of 16 -> 64 tokens

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_p;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_v;

    for(int step=0; step<num_chunks; ++step) {
        __pipeline_wait_prior(0);
        int buf = step % 2;
        int next = step + 1;
        zero_invalid_rows(smem, buf, valid, tid);
        __syncthreads();
        
        if(next < num_chunks) {
            int nv = min(CHUNK_SIZE, total - next * CHUNK_SIZE);
            load_kv_chunk(smem, next % 2, kv_indices, page_start + start + next * CHUNK_SIZE, nv, ckv_cache, kpe_cache, tid);
            __pipeline_commit();
        }
        
        // --- Score ---
        #pragma unroll
        for(int t=0; t<4; ++t) wmma::fill_fragment(acc_s[t], 0.0f);
        
        // CKV (4 tiles of K-dim)
        #pragma unroll
        for(int k=0; k<4; ++k) {
             int k_off = q_col_start + k * 16;
             #pragma unroll
             for(int t=0; t<4; ++t) {
                 wmma::load_matrix_sync(frag_k, smem.kc_buf[buf] + t * 16 * STRIDE_CKV + k_off, STRIDE_CKV);
                 wmma::mma_sync(acc_s[t], frag_q[k], frag_k, acc_s[t]);
             }
        }
        // KPE (Warp 0)
        if(warp_id == 0) {
             wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> f_qpe;
             #pragma unroll
             for(int k=0; k<4; ++k) {
                 wmma::load_matrix_sync(f_qpe, smem.q_pe + k*16, HEAD_DIM_KPE);
                 #pragma unroll
                 for(int t=0; t<4; ++t) {
                     wmma::load_matrix_sync(frag_k, smem.kp_buf[buf] + t * 16 * STRIDE_KPE + k*16, STRIDE_KPE);
                     wmma::mma_sync(acc_s[t], f_qpe, frag_k, acc_s[t]);
                 }
             }
        }
        
        // Dump S partials
        #pragma unroll
        for(int t=0; t<4; ++t) {
            wmma::store_matrix_sync(smem.s_partials[warp_id] + t * 16, acc_s[t], STRIDE_S, wmma::mem_row_major);
        }
        __syncthreads();
        
        // Reduce S
        #pragma unroll
        for(int i=0; i<4; ++i) { // 256 threads cover 1024 elems (16*64). 4 per thread
            int idx = tid + i * 256;
            int r = idx / 64; 
            int c = idx & 63;
            float sum = 0.0f;
            int off = r * STRIDE_S + c;
            #pragma unroll
            for(int w=0; w<NUM_WARPS; ++w) sum += smem.s_partials[w][off];
            
            if(step * CHUNK_SIZE + c >= total) sum = -INFINITY;
            else sum *= sm_scale;
            
            smem.s_partials[0][off] = sum;
        }
        __syncthreads();
        
        // Softmax
        int r = tid / 16;
        int lane = tid % 16;
        if(r < 16) {
             float m = -INFINITY;
             for(int c=lane; c<64; c+=16) m = max(m, smem.s_partials[0][r * STRIDE_S + c]);
             #pragma unroll
             for(int mask=8; mask>0; mask/=2) m = max(m, __shfl_xor_sync(0xffffffff, m, mask));
             
             if(lane == 0) {
                 float pm = smem.lse_max[r];
                 float cm = max(pm, m);
                 float a = (pm == -INFINITY) ? 0.0f : expf(pm - cm);
                 if(pm == -INFINITY && cm == -INFINITY) a = 1.0f;
                 smem.lse_max[r] = cm;
                 smem.broadcast_alpha[r] = a;
             }
        }
        __syncthreads();
        
        if(r < 16) {
             float cm = smem.lse_max[r];
             float row_sum = 0.0f;
             for(int c=lane; c<64; c+=16) {
                 float v = smem.s_partials[0][r * STRIDE_S + c];
                 float p = (cm == -INFINITY) ? 0.0f : expf(v - cm);
                 smem.p_mat[r * 64 + c] = __float2bfloat16(p);
                 row_sum += p;
             }
             #pragma unroll
             for(int mask=8; mask>0; mask/=2) row_sum += __shfl_xor_sync(0xffffffff, row_sum, mask);
             if(lane == 0) smem.lse_sum[r] = smem.lse_sum[r] * smem.broadcast_alpha[r] + row_sum;
        }
        __syncthreads();
        
        // Rescale Output (using scratch)
        for(int k=0; k<4; ++k) {
             wmma::store_matrix_sync(smem.s_partials[warp_id], acc_o[k], 16, wmma::mem_row_major);
             __syncwarp();
             for(int i=0; i<8; ++i) { // 32 threads, 256 elems
                 int idx = tid%32 + i*32;
                 int rr = idx/16;
                 smem.s_partials[warp_id][idx] *= smem.broadcast_alpha[rr];
             }
             __syncwarp();
             wmma::load_matrix_sync(acc_o[k], smem.s_partials[warp_id], 16, wmma::mem_row_major);
        }
        __syncthreads(); // Wait for P ready
        
        // Output Accumulation
        int v_col_base = warp_id * 64;
        #pragma unroll
        for(int t=0; t<4; ++t) { // 4 tiles of P (64 cols of P = 64 rows of V)
            wmma::load_matrix_sync(frag_p, smem.p_mat + t*16, 64);
            #pragma unroll
            for(int k=0; k<4; ++k) {
                wmma::load_matrix_sync(frag_v, smem.kc_buf[buf] + t * 16 * STRIDE_CKV + v_col_base + k*16, STRIDE_CKV);
                wmma::mma_sync(acc_o[k], frag_p, frag_v, acc_o[k]);
            }
        }
        __syncthreads();
        if(next < num_chunks) valid = min(CHUNK_SIZE, total - next * CHUNK_SIZE);
    }
    
    // Final Store
    int v_col_base = warp_id * 64;
    __nv_bfloat16* dst = temp_out + (batch_idx*num_splits + split_idx)*NUM_HEADS*HEAD_DIM_CKV;
    
    for(int k=0; k<4; ++k) {
        wmma::store_matrix_sync(smem.s_partials[warp_id], acc_o[k], 16, wmma::mem_row_major);
        __syncwarp();
        for(int i=0; i<8; ++i) {
             int idx = tid%32 + i*32;
             int rr = idx/16;
             int cc = idx%16;
             float v = smem.s_partials[warp_id][idx];
             float s = smem.lse_sum[rr];
             float res = (s == 0.0f) ? 0.0f : (v/s);
             dst[rr * HEAD_DIM_CKV + v_col_base + k*16 + cc] = __float2bfloat16(res);
        }
    }
    
    if(tid < NUM_HEADS) {
        float l = smem.lse_sum[tid];
        float m = smem.lse_max[tid];
        float val = (l == 0.0f) ? -INFINITY : (logf(l) + m);
        temp_lse[(batch_idx*num_splits + split_idx)*NUM_HEADS + tid] = val;
    }
}

__global__ void mla_decode_reduce_kernel(
    const __nv_bfloat16* __restrict__ temp_out, 
    const float* __restrict__ temp_lse,          
    __nv_bfloat16* __restrict__ output,          
    float* __restrict__ lse,                     
    int num_splits
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    float max_lse = -INFINITY;
    for (int s = 0; s < num_splits; ++s) {
        float val = temp_lse[(batch_idx * num_splits + s) * NUM_HEADS + head_idx];
        if (val > max_lse) max_lse = val;
    }
    
    float sum_exp = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        float val = temp_lse[(batch_idx * num_splits + s) * NUM_HEADS + head_idx];
        if (val != -INFINITY) sum_exp += expf(val - max_lse);
    }
    
    float global_lse = max_lse + logf(sum_exp);
    if (max_lse == -INFINITY) global_lse = -INFINITY;
    
    if (tid == 0) lse[batch_idx * NUM_HEADS + head_idx] = global_lse * 1.44269504f;
    
    int offset = (batch_idx * NUM_HEADS + head_idx) * HEAD_DIM_CKV;
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
        output[offset + i] = __float2bfloat16(acc);
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
    
    // Heuristic: ~256+ blocks total
    int num_splits = (256 + batch_size - 1) / batch_size;
    if (num_splits < 2) num_splits = 2;
    if (num_splits > 64) num_splits = 64; 
    
    auto options = q_nope.options();
    auto temp_out = torch::empty({batch_size, num_splits, NUM_HEADS, HEAD_DIM_CKV}, options);
    auto temp_lse = torch::empty({batch_size, num_splits, NUM_HEADS}, options.dtype(torch::kFloat32));
    
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
