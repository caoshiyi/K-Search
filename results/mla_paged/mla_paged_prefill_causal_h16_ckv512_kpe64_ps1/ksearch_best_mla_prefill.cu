#include "kernel.h"
#include <mma.h>
#include <cuda_bf16.h>

using namespace nvcuda;

// Constants optimized for H100 and the specific problem dimensions
constexpr int HEAD_DIM_CKV = 512;
constexpr int HEAD_DIM_KPE = 64;
constexpr int TILE_Q = 16;
constexpr int TILE_KV = 16;
constexpr int THREADS_PER_BLOCK = 128;

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* global) {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        : : "r"(smem_int), "l"(global)
    );
}

__device__ void fill_zero(nv_bfloat16* ptr, int size, int tid, int num_threads) {
    int4* ptr4 = reinterpret_cast<int4*>(ptr);
    int size4 = size / 8;
    for (int i = tid; i < size4; i += num_threads) {
        ptr4[i] = make_int4(0, 0, 0, 0);
    }
}

__device__ void load_q_tile_async(
    nv_bfloat16* __restrict__ s_q,
    const nv_bfloat16* __restrict__ g_q,
    int q_len,
    int dim,
    int stride_bytes,
    int tid,
    int num_threads
) {
    int vecs = dim / 8; // 8 bf16 per 16 bytes
    int total_vecs = q_len * vecs;

    for (int i = tid; i < total_vecs; i += num_threads) {
        int r = i / vecs;
        int v = i % vecs;

        char* dst = (char*)s_q + r * dim * 2 + v * 16;
        const char* src = (const char*)g_q + (long long)r * stride_bytes + v * 16;

        cp_async_16(dst, src);
    }
}

__device__ void load_kv_gather(
    nv_bfloat16* __restrict__ s_kv,
    const nv_bfloat16* __restrict__ g_base,
    const int* __restrict__ indices,
    int dim,
    int current_rows,
    int idx_offset,
    int tid,
    int num_threads
) {
    // dim=512 -> 1024 bytes -> 64 chunks of 16 bytes.
    // dim=64  -> 128 bytes  -> 8 chunks of 16 bytes.

    int page_idx = tid / 8;
    int lane = tid % 8;

    if (page_idx < current_rows) {
        int g_idx = indices[idx_offset + page_idx];

        int bytes_per_page = dim * 2;
        int chunks_per_page = bytes_per_page / 16;
        int chunks_per_thread = chunks_per_page / 8;

        const char* src_ptr_base = (const char*)g_base + (long long)g_idx * bytes_per_page;
        char* dst_ptr_base = (char*)s_kv + page_idx * bytes_per_page;

        #pragma unroll
        for (int c = 0; c < chunks_per_thread; ++c) {
            int chunk_idx = lane + c * 8;
            cp_async_16(dst_ptr_base + chunk_idx * 16, src_ptr_base + chunk_idx * 16);
        }
    }
}

__device__ inline void unpack_bf16_4(int2 val, float* out) {
    nv_bfloat16* v = reinterpret_cast<nv_bfloat16*>(&val);
    out[0] = __bfloat162float(v[0]);
    out[1] = __bfloat162float(v[1]);
    out[2] = __bfloat162float(v[2]);
    out[3] = __bfloat162float(v[3]);
}

__device__ int find_batch_idx(int q_idx, const int* indptr, int batch_size) {
    // Optimization: Linear scan for small batches (common case)
    if (batch_size <= 8) {
        for (int i = 0; i < batch_size; ++i) {
            if (q_idx < indptr[i+1]) return i;
        }
        return batch_size - 1;
    }

    int low = 0;
    int high = batch_size - 1;
    int ans = 0;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (indptr[mid + 1] > q_idx) {
            ans = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return ans;
}

extern __shared__ char smem_buffer[];

__device__ void process_batch_tile(
    int valid_rows_start,
    int valid_rows_count,
    int tile_global_q_start,
    int batch_q_start,
    int batch_q_len,
    int kv_start,
    int kv_len,
    const nv_bfloat16* __restrict__ ckv_cache,
    const nv_bfloat16* __restrict__ kpe_cache,
    const int* __restrict__ kv_indices,
    nv_bfloat16* s_qn,
    nv_bfloat16* s_qp,
    nv_bfloat16* s_kc,
    nv_bfloat16* s_kp,
    float* s_stats,
    float sm_scale,
    nv_bfloat16* output,
    float* lse,
    int head,
    int tid,
    int wid,
    int num_threads
) {
    float* s_scores_parts = s_stats;
    float* s_scores = s_scores_parts + 1024;
    float* s_p = s_scores + 256;
    float* s_m = s_p + 256;
    float* s_d = s_m + 16;
    float* s_alpha = s_d + 16;

    float r_acc[16][4];
    #pragma unroll
    for(int i=0;i<16;++i)
        #pragma unroll
        for(int j=0;j<4;++j) r_acc[i][j] = 0.0f;

    if (tid < 16) {
        s_m[tid] = -INFINITY;
        s_d[tid] = 0.0f;
    }

    int prefix_len = kv_len - batch_q_len;
    int num_steps = (kv_len + TILE_KV - 1) / TILE_KV;

    // Lambda to process one step of KV
    auto compute_step = [&](int step) {
        int buf_idx = step % 2;
        nv_bfloat16* k_ptr = s_kc + buf_idx * TILE_KV * HEAD_DIM_CKV;
        nv_bfloat16* kp_ptr = s_kp + buf_idx * TILE_KV * HEAD_DIM_KPE;

        // 1. Compute Scores (Q * K^T)
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        int chunks_per_warp = 9; // 576 / 16 / 4 = 9
        int start_chunk = wid * chunks_per_warp;
        int end_chunk = start_chunk + chunks_per_warp;

        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            int chunk = start_chunk + k;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, nv_bfloat16, wmma::col_major> b_frag;

            if (chunk < 32) {
                int offset = chunk * 16;
                wmma::load_matrix_sync(a_frag, s_qn + offset, HEAD_DIM_CKV);
                wmma::load_matrix_sync(b_frag, k_ptr + offset, HEAD_DIM_CKV);
            } else {
                int offset = (chunk - 32) * 16;
                wmma::load_matrix_sync(a_frag, s_qp + offset, HEAD_DIM_KPE);
                wmma::load_matrix_sync(b_frag, kp_ptr + offset, HEAD_DIM_KPE);
            }
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        wmma::store_matrix_sync(s_scores_parts + wid * 256, acc, 16, wmma::mem_row_major);
        __syncthreads();

        // 2. Reduce scores across warps
        #pragma unroll
        for (int i = tid; i < 256; i += num_threads) {
            float sum = s_scores_parts[i] + s_scores_parts[i+256] + s_scores_parts[i+512] + s_scores_parts[i+768];
            s_scores[i] = sum;
        }
        __syncthreads();

        // 3. Softmax
        if (tid < 16) {
            bool row_valid = (tid >= valid_rows_start) && (tid < valid_rows_start + valid_rows_count);
            int cur_kv_rows = min(TILE_KV, kv_len - step * TILE_KV);
            float m_local = -INFINITY;

            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float val = s_scores[tid * 16 + c] * sm_scale;

                int abs_q = tile_global_q_start + tid;
                int rel_q = abs_q - batch_q_start;
                int abs_kv = step * TILE_KV + c;

                // Mask: valid row, valid kv, causal constraint
                bool mask = row_valid && (c < cur_kv_rows) && (abs_kv <= prefix_len + rel_q);

                if (!mask) val = -INFINITY;
                s_scores[tid * 16 + c] = val;
                m_local = max(m_local, val);
            }

            float m_prev = s_m[tid];
            float m_new = max(m_prev, m_local);
            float alpha = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_new);
            float beta = (m_local == -INFINITY) ? 0.0f : expf(m_local - m_new);
            s_m[tid] = m_new;
            s_alpha[tid] = alpha;

            float d_local = 0.0f;
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float val = s_scores[tid * 16 + c];
                float p = 0.0f;
                if (val > -INFINITY) {
                    p = expf(val - m_local) * beta;
                    d_local += expf(val - m_local);
                }
                s_p[tid * 16 + c] = p;
            }
            s_d[tid] = s_d[tid] * alpha + d_local * beta;
        }
        __syncthreads();

        // 4. Update Output Accumulators
        int col_start = wid * 128 + (tid % 32) * 4;

        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            float a = s_alpha[r];
            #pragma unroll
            for (int j = 0; j < 4; ++j) r_acc[r][j] *= a;
        }

        for (int k = 0; k < 16; ++k) {
            int2 v_packed = *reinterpret_cast<int2*>(&k_ptr[k * HEAD_DIM_CKV + col_start]);
            float v[4];
            unpack_bf16_4(v_packed, v);

            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                float p = s_p[r * 16 + k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    r_acc[r][j] += p * v[j];
                }
            }
        }
    };

    // Pipeline
    if (num_steps > 0) {
        // Prologue
        int cur_kv_rows = min(TILE_KV, kv_len);
        load_kv_gather(s_kc, ckv_cache, kv_indices, HEAD_DIM_CKV, cur_kv_rows, kv_start, tid, num_threads);
        load_kv_gather(s_kp, kpe_cache, kv_indices, HEAD_DIM_KPE, cur_kv_rows, kv_start, tid, num_threads);
        cp_async_commit();

        // Body
        for (int step = 0; step < num_steps - 1; ++step) {
            int next_kv_t = (step + 1) * TILE_KV;
            int next_kv_rows = min(TILE_KV, kv_len - next_kv_t);
            int buf_idx = (step + 1) % 2;
            load_kv_gather(s_kc + buf_idx * TILE_KV * HEAD_DIM_CKV, ckv_cache, kv_indices, HEAD_DIM_CKV, next_kv_rows, kv_start + next_kv_t, tid, num_threads);
            load_kv_gather(s_kp + buf_idx * TILE_KV * HEAD_DIM_KPE, kpe_cache, kv_indices, HEAD_DIM_KPE, next_kv_rows, kv_start + next_kv_t, tid, num_threads);
            cp_async_commit();

            cp_async_wait_group<1>();
            __syncthreads();

            compute_step(step);
            __syncthreads();
        }

        // Epilogue
        cp_async_wait_group<0>();
        __syncthreads();
        compute_step(num_steps - 1);
        __syncthreads();
    }

    // Write Output
    int col_start = wid * 128 + (tid % 32) * 4;
    int out_stride = gridDim.y * HEAD_DIM_CKV;

    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        if (r >= valid_rows_start && r < valid_rows_start + valid_rows_count) {
            float d = s_d[r];
            float inv_d = (d > 1e-6f) ? (1.0f / d) : 0.0f;
            int abs_r = tile_global_q_start + r;

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int c = col_start + j;
                if (c < HEAD_DIM_CKV) {
                    float val = r_acc[r][j] * inv_d;
                    long long dst_idx = (long long)abs_r * out_stride + head * HEAD_DIM_CKV + c;
                    output[dst_idx] = __float2bfloat16(val);
                }
            }
        }
    }

    if (tid < 16) {
        int r = tid;
        if (r >= valid_rows_start && r < valid_rows_start + valid_rows_count) {
            float m = s_m[r];
            float d = s_d[r];
            float res = -INFINITY;
            if (m > -INFINITY) {
                res = (m + logf(max(d, 1e-6f))) * 1.44269504089f; // log2(e)
            }
            int abs_r = tile_global_q_start + r;
            lse[(long long)abs_r * gridDim.y + head] = res;
        }
    }
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK) mla_kernel_flattened(
    const nv_bfloat16* __restrict__ q_nope,
    const nv_bfloat16* __restrict__ q_pe,
    const nv_bfloat16* __restrict__ ckv_cache,
    const nv_bfloat16* __restrict__ kpe_cache,
    const int* __restrict__ qo_indptr,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    float sm_scale,
    int total_q,
    int batch_size
) {
    int head = blockIdx.y;
    int tile_idx = blockIdx.x;
    int global_q_start = tile_idx * TILE_Q;

    if (global_q_start >= total_q) return;

    int tid = threadIdx.x;
    int wid = tid / 32;
    int num_threads = blockDim.x;

    // SMEM Layout Calculation
    // s_qn: 16 * 512 * 2 = 16384 bytes
    // s_qp: 16 * 64 * 2 = 2048 bytes
    // s_kc: 2 * 16 * 512 * 2 = 32768 bytes
    // s_kp: 2 * 16 * 64 * 2 = 4096 bytes
    // s_stats: ~6KB
    nv_bfloat16* s_qn = (nv_bfloat16*)smem_buffer;
    nv_bfloat16* s_qp = s_qn + TILE_Q * HEAD_DIM_CKV;
    nv_bfloat16* s_kc = s_qp + TILE_Q * HEAD_DIM_KPE;
    nv_bfloat16* s_kp = s_kc + 2 * TILE_KV * HEAD_DIM_CKV;
    float* s_stats = (float*)(s_kp + 2 * TILE_KV * HEAD_DIM_KPE);

    int q_rows = min(TILE_Q, total_q - global_q_start);
    int heads_stride = gridDim.y;

    fill_zero(s_qn, TILE_Q * HEAD_DIM_CKV, tid, num_threads);
    fill_zero(s_qp, TILE_Q * HEAD_DIM_KPE, tid, num_threads);

    load_q_tile_async(s_qn, q_nope + global_q_start * heads_stride * HEAD_DIM_CKV + head * HEAD_DIM_CKV,
                q_rows, HEAD_DIM_CKV, heads_stride * HEAD_DIM_CKV * 2, tid, num_threads);
    load_q_tile_async(s_qp, q_pe + global_q_start * heads_stride * HEAD_DIM_KPE + head * HEAD_DIM_KPE,
                q_rows, HEAD_DIM_KPE, heads_stride * HEAD_DIM_KPE * 2, tid, num_threads);

    cp_async_commit();
    cp_async_wait_group<0>();
    __syncthreads();

    int current_q = global_q_start;
    int end_q = global_q_start + q_rows;

    while (current_q < end_q) {
        int b = find_batch_idx(current_q, qo_indptr, batch_size);
        int batch_start_q = qo_indptr[b];
        int batch_end_q = qo_indptr[b+1];

        int next_boundary = min(end_q, batch_end_q);
        int num_rows = next_boundary - current_q;

        int kv_start = kv_indptr[b];
        int kv_end = kv_indptr[b+1];
        int kv_len = kv_end - kv_start;
        int batch_q_len = batch_end_q - batch_start_q;

        int valid_rows_start = current_q - global_q_start;

        process_batch_tile(
            valid_rows_start,
            num_rows,
            global_q_start,
            batch_start_q,
            batch_q_len,
            kv_start,
            kv_len,
            ckv_cache, kpe_cache, kv_indices,
            s_qn, s_qp, s_kc, s_kp, s_stats,
            sm_scale, output, lse, head, tid, wid, num_threads
        );
        __syncthreads();

        current_q = next_boundary;
    }
}

void launch_mla_paged(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    torch::Tensor output,
    torch::Tensor lse,
    float sm_scale
) {
    int total_q = q_nope.size(0);
    int num_heads = q_nope.size(1);
    int batch_size = qo_indptr.size(0) - 1;

    if (total_q == 0) return;

    dim3 grid((total_q + TILE_Q - 1) / TILE_Q, num_heads, 1);
    dim3 block(THREADS_PER_BLOCK);

    // We need approx 60KB SMEM. H100 allows up to 227KB.
    int smem_size = 65536;
    cudaFuncSetAttribute(mla_kernel_flattened, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    mla_kernel_flattened<<<grid, block, smem_size>>>(
        (nv_bfloat16*)q_nope.data_ptr<at::BFloat16>(),
        (nv_bfloat16*)q_pe.data_ptr<at::BFloat16>(),
        (nv_bfloat16*)ckv_cache.data_ptr<at::BFloat16>(),
        (nv_bfloat16*)kpe_cache.data_ptr<at::BFloat16>(),
        qo_indptr.data_ptr<int>(),
        kv_indptr.data_ptr<int>(),
        kv_indices.data_ptr<int>(),
        (nv_bfloat16*)output.data_ptr<at::BFloat16>(),
        lse.data_ptr<float>(),
        sm_scale,
        total_q,
        batch_size
    );
}