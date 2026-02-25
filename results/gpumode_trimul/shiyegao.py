from __future__ import annotations

from typing import Any, Dict, Tuple

import os

import torch

_EXT = None
_EXT_LOCK = None


def _lazy_import_extension_utils():
    from torch.utils.cpp_extension import load_inline  

    return load_inline


def _get_ext():
    global _EXT, _EXT_LOCK
    if _EXT is not None:
        return _EXT
    if _EXT_LOCK is None:
        import threading  

        _EXT_LOCK = threading.Lock()
    with _EXT_LOCK:
        if _EXT is not None:
            return _EXT

        load_inline = _lazy_import_extension_utils()

        
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

        cpp_src = r"""
#include <torch/extension.h>

torch::Tensor trimul_fwd(
    torch::Tensor x,
    torch::Tensor mask_h,
    torch::Tensor ln1_w,
    torch::Tensor ln1_b,
    torch::Tensor w_left,
    torch::Tensor w_right,
    torch::Tensor w_lg,
    torch::Tensor w_rg,
    torch::Tensor w_og,
    torch::Tensor ln2_w,
    torch::Tensor ln2_b,
    torch::Tensor w_out,
    int64_t dim,
    int64_t hidden);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &trimul_fwd, "trimul forward (cuda)");
}
"""

        cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <mutex>

namespace {

static inline void checkCuda(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
  }
}

static inline void checkCublas(cublasStatus_t s, const char* msg) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(msg) + ": cublas status=" + std::to_string((int)s));
  }
}

struct CublasHandleHolder {
  cublasHandle_t handle = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = (size_t)64 * 1024 * 1024;
  CublasHandleHolder() {
    checkCublas(cublasCreate(&handle), "cublasCreate");
    checkCuda(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc_cublas_workspace");
    checkCublas(cublasSetWorkspace(handle, workspace, workspace_bytes), "cublasSetWorkspace");
    // 强制启用 tensor op math（half 输入的 GEMM 明确走张量核）
    checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
  }
  ~CublasHandleHolder() {
    if (workspace) {
      cudaFree(workspace);
      workspace = nullptr;
    }
    if (handle) {
      cublasDestroy(handle);
      handle = nullptr;
    }
  }
};

static CublasHandleHolder* get_cublas() {
  static std::once_flag once;
  static CublasHandleHolder* holder = nullptr;
  std::call_once(once, []() { holder = new CublasHandleHolder(); });
  return holder;
}

// ---------------- cuBLASLt（用于 contract / GEMM1 / GEMM2） ----------------
// 目标：对重复出现的形状缓存 descriptor + heuristic algo，减少选择开销并挖潜性能。

struct LtContractCacheEntry {
  int n = 0;
  int batch = 0;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatmulAlgo_t algo;
  size_t algo_workspace = 0;
  // contract 同样做一次性择优：避免 heuristic 在部分形状上误判导致尾部回退
  int algo_count = 0;
  cublasLtMatmulAlgo_t algo_list[8];
  size_t algo_ws_list[8];
  bool tuned = false;
  bool ready = false;
};

static inline void lt_destroy_entry(LtContractCacheEntry* e) {
  if (e->op) {
    cublasLtMatmulDescDestroy(e->op);
    e->op = nullptr;
  }
  if (e->a) {
    cublasLtMatrixLayoutDestroy(e->a);
    e->a = nullptr;
  }
  if (e->b) {
    cublasLtMatrixLayoutDestroy(e->b);
    e->b = nullptr;
  }
  if (e->c) {
    cublasLtMatrixLayoutDestroy(e->c);
    e->c = nullptr;
  }
  e->n = 0;
  e->batch = 0;
  e->algo_workspace = 0;
  e->algo_count = 0;
  e->tuned = false;
  e->ready = false;
}

struct LtGemmCacheEntry {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatmulAlgo_t algo;
  size_t algo_workspace = 0;
  // 保存候选 algo，用于首次出现该形状时做一次轻量择优（仅发生一次，随后复用）。
  int algo_count = 0;
  cublasLtMatmulAlgo_t algo_list[8];
  size_t algo_ws_list[8];
  bool tuned = false;
  bool ready = false;
};

static inline void lt_destroy_gemm(LtGemmCacheEntry* e) {
  if (e->op) {
    cublasLtMatmulDescDestroy(e->op);
    e->op = nullptr;
  }
  if (e->a) {
    cublasLtMatrixLayoutDestroy(e->a);
    e->a = nullptr;
  }
  if (e->b) {
    cublasLtMatrixLayoutDestroy(e->b);
    e->b = nullptr;
  }
  if (e->c) {
    cublasLtMatrixLayoutDestroy(e->c);
    e->c = nullptr;
  }
  e->m = 0;
  e->n = 0;
  e->k = 0;
  e->algo_workspace = 0;
  e->algo_count = 0;
  e->tuned = false;
  e->ready = false;
}

struct CublasLtHolder {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  // 经验：增大 Lt workspace 往往能解锁更快的 algo（尤其是大 N 的 contract / 大 M 的 gemm1）。
  // 该 workspace 仅分配一次；H100 80GB 显存充足，优先换取更高吞吐。
  size_t workspace_bytes = (size_t)768 * 1024 * 1024;
  std::mutex mu;
  // 形状组合：n∈{256,512,768,1024}；batch=bs*hidden（常见 hidden∈{128,256}，bs∈{1,2}），
  // 至少 4*4=16 种；cache=8 在 tests/benchmark 交错时容易 eviction，触发 descriptor 重建。
  LtContractCacheEntry cache[32];
  int next_slot = 0;
  // 形状组合：M=bs*N*N（bs∈{1,2}，N∈{256,512,768,1024}）且 dim∈{128,384}，
  // gemm1/gemm2 至少 16 种 (M,N,K)；cache=8 会发生反复 eviction 触发 descriptor/tune 抖动。
  // 进一步覆盖 dim/hidden/batch 的组合，避免 cache 边界抖动（descriptor/tune 重复触发）。
  LtGemmCacheEntry g1_cache[64];
  int g1_next = 0;
  LtGemmCacheEntry g2_cache[64];
  int g2_next = 0;

  CublasLtHolder() {
    checkCublas(cublasLtCreate(&handle), "cublasLtCreate");
    checkCuda(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc_cublasLt_workspace");
  }
  ~CublasLtHolder() {
    for (int i = 0; i < 32; ++i) {
      lt_destroy_entry(&cache[i]);
    }
    for (int i = 0; i < 64; ++i) {
      lt_destroy_gemm(&g1_cache[i]);
      lt_destroy_gemm(&g2_cache[i]);
    }
    if (workspace) {
      cudaFree(workspace);
      workspace = nullptr;
    }
    if (handle) {
      cublasLtDestroy(handle);
      handle = nullptr;
    }
  }

  LtContractCacheEntry* get_contract(int n, int batch) {
    std::lock_guard<std::mutex> lock(mu);
    for (int i = 0; i < 32; ++i) {
      if (cache[i].ready && cache[i].n == n && cache[i].batch == batch) {
        return &cache[i];
      }
    }

    LtContractCacheEntry* e = &cache[next_slot & 31];
    next_slot++;
    lt_destroy_entry(e);
    e->n = n;
    e->batch = batch;

    checkCublas(
        cublasLtMatmulDescCreate(&e->op, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "cublasLtMatmulDescCreate_contract");
    cublasOperation_t op_a = CUBLAS_OP_T;
    cublasOperation_t op_b = CUBLAS_OP_N;
    checkCublas(
        cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)),
        "cublasLtMatmulDescSetAttribute_contract_a");
    checkCublas(
        cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)),
        "cublasLtMatmulDescSetAttribute_contract_b");

    checkCublas(cublasLtMatrixLayoutCreate(&e->a, CUDA_R_16F, n, n, n), "cublasLtMatrixLayoutCreate_a");
    checkCublas(cublasLtMatrixLayoutCreate(&e->b, CUDA_R_16F, n, n, n), "cublasLtMatrixLayoutCreate_b");
    checkCublas(cublasLtMatrixLayoutCreate(&e->c, CUDA_R_16F, n, n, n), "cublasLtMatrixLayoutCreate_c");

    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
        "cublasLtMatrixLayoutSetAttribute_order_a");
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
        "cublasLtMatrixLayoutSetAttribute_order_b");
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
        "cublasLtMatrixLayoutSetAttribute_order_c");

    long long stride_elems = (long long)n * (long long)n;
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)),
        "cublasLtMatrixLayoutSetAttribute_batch_a");
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(
            e->a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_elems, sizeof(stride_elems)),
        "cublasLtMatrixLayoutSetAttribute_stride_a");

    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)),
        "cublasLtMatrixLayoutSetAttribute_batch_b");
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(
            e->b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_elems, sizeof(stride_elems)),
        "cublasLtMatrixLayoutSetAttribute_stride_b");

    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)),
        "cublasLtMatrixLayoutSetAttribute_batch_c");
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(
            e->c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_elems, sizeof(stride_elems)),
        "cublasLtMatrixLayoutSetAttribute_stride_c");

    cublasLtMatmulPreference_t pref = nullptr;
    checkCublas(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
    checkCublas(
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)),
        "cublasLtMatmulPreferenceSetAttribute_workspace");

    constexpr int MAX_ALGOS = 32;
    cublasLtMatmulHeuristicResult_t heurs[MAX_ALGOS];
    int got = 0;
    checkCublas(
        cublasLtMatmulAlgoGetHeuristic(handle, e->op, e->a, e->b, e->c, e->c, pref, MAX_ALGOS, heurs, &got),
        "cublasLtMatmulAlgoGetHeuristic_contract");
    cublasLtMatmulPreferenceDestroy(pref);

    if (got <= 0) {
      throw std::runtime_error("cublasLt: no heuristic algo for contract");
    }

    e->algo_count = 0;
    for (int i = 0; i < got && e->algo_count < 8; ++i) {
      if (heurs[i].state == CUBLAS_STATUS_SUCCESS && heurs[i].workspaceSize <= workspace_bytes) {
        e->algo_list[e->algo_count] = heurs[i].algo;
        e->algo_ws_list[e->algo_count] = heurs[i].workspaceSize;
        e->algo_count++;
      }
    }
    if (e->algo_count <= 0) {
      throw std::runtime_error("cublasLt: workspace too small for all contract algos");
    }

    // 默认先用 heuristic 的第一个；首次调用点做一次轻量择优（不污染 benchmark）。
    e->algo = e->algo_list[0];
    e->algo_workspace = e->algo_ws_list[0];
    e->tuned = false;
    e->ready = true;
    return e;
  }

  LtGemmCacheEntry* get_gemm1(int64_t M, int64_t N, int64_t K) {
    std::lock_guard<std::mutex> lock(mu);
    for (int i = 0; i < 64; ++i) {
      if (g1_cache[i].ready && g1_cache[i].m == M && g1_cache[i].n == N && g1_cache[i].k == K) {
        return &g1_cache[i];
      }
    }

    LtGemmCacheEntry* e = &g1_cache[g1_next & 63];
    g1_next++;
    lt_destroy_gemm(e);
    e->m = M;
    e->n = N;
    e->k = K;

    checkCublas(
        cublasLtMatmulDescCreate(&e->op, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "cublasLtMatmulDescCreate_gemm1");
    cublasOperation_t op_a = CUBLAS_OP_T;
    cublasOperation_t op_b = CUBLAS_OP_N;
    checkCublas(
        cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)),
        "cublasLtMatmulDescSetAttribute_gemm1_a");
    checkCublas(
        cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)),
        "cublasLtMatmulDescSetAttribute_gemm1_b");

    // A: x_rm [M,K] row-major -> 视作 column-major [K,M]
    checkCublas(cublasLtMatrixLayoutCreate(&e->a, CUDA_R_16F, K, M, K), "cublasLtMatrixLayoutCreate_gemm1_a");
    // B: w_rm [N,K] row-major -> 视作 column-major [K,N]
    checkCublas(cublasLtMatrixLayoutCreate(&e->b, CUDA_R_16F, K, N, K), "cublasLtMatrixLayoutCreate_gemm1_b");
    // C: out [N,M] row-major -> 视作 column-major [M,N]
    checkCublas(cublasLtMatrixLayoutCreate(&e->c, CUDA_R_16F, M, N, M), "cublasLtMatrixLayoutCreate_gemm1_c");

    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
        "cublasLtMatrixLayoutSetAttribute_gemm1_order_a");
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
        "cublasLtMatrixLayoutSetAttribute_gemm1_order_b");
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
        "cublasLtMatrixLayoutSetAttribute_gemm1_order_c");

    cublasLtMatmulPreference_t pref = nullptr;
    checkCublas(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate_gemm1");
    checkCublas(
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)),
        "cublasLtMatmulPreferenceSetAttribute_gemm1_ws");

    constexpr int MAX_ALGOS = 32;
    cublasLtMatmulHeuristicResult_t heurs[MAX_ALGOS];
    int got = 0;
    checkCublas(
        cublasLtMatmulAlgoGetHeuristic(handle, e->op, e->a, e->b, e->c, e->c, pref, MAX_ALGOS, heurs, &got),
        "cublasLtMatmulAlgoGetHeuristic_gemm1");
    cublasLtMatmulPreferenceDestroy(pref);

    if (got <= 0) {
      throw std::runtime_error("cublasLt: no heuristic algo for gemm1");
    }

    e->algo_count = 0;
    for (int i = 0; i < got && e->algo_count < 8; ++i) {
      if (heurs[i].state == CUBLAS_STATUS_SUCCESS && heurs[i].workspaceSize <= workspace_bytes) {
        e->algo_list[e->algo_count] = heurs[i].algo;
        e->algo_ws_list[e->algo_count] = heurs[i].workspaceSize;
        e->algo_count++;
      }
    }
    if (e->algo_count <= 0) {
      throw std::runtime_error("cublasLt: workspace too small for gemm1");
    }

    // 默认先用 heuristic 返回的第一个；首次调用点会做一次轻量择优（不污染 benchmark）。
    e->algo = e->algo_list[0];
    e->algo_workspace = e->algo_ws_list[0];
    e->tuned = false;
    e->ready = true;
    return e;
  }

  LtGemmCacheEntry* get_gemm2(int64_t M, int64_t N, int64_t K) {
    std::lock_guard<std::mutex> lock(mu);
    for (int i = 0; i < 64; ++i) {
      if (g2_cache[i].ready && g2_cache[i].m == M && g2_cache[i].n == N && g2_cache[i].k == K) {
        return &g2_cache[i];
      }
    }

    LtGemmCacheEntry* e = &g2_cache[g2_next & 63];
    g2_next++;
    lt_destroy_gemm(e);
    e->m = M;
    e->n = N;
    e->k = K;

    checkCublas(
        cublasLtMatmulDescCreate(&e->op, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "cublasLtMatmulDescCreate_gemm2");
    cublasOperation_t op_a = CUBLAS_OP_N;
    cublasOperation_t op_b = CUBLAS_OP_N;
    checkCublas(
        cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)),
        "cublasLtMatmulDescSetAttribute_gemm2_a");
    checkCublas(
        cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)),
        "cublasLtMatmulDescSetAttribute_gemm2_b");

    // A: a_dmaj_rm [K,M] row-major -> 视作 column-major [M,K]
    checkCublas(cublasLtMatrixLayoutCreate(&e->a, CUDA_R_16F, M, K, M), "cublasLtMatrixLayoutCreate_gemm2_a");
    // B: w_rm [N,K] row-major -> 视作 column-major [K,N]
    checkCublas(cublasLtMatrixLayoutCreate(&e->b, CUDA_R_16F, K, N, K), "cublasLtMatrixLayoutCreate_gemm2_b");
    // C: y_t_rm [N,M] row-major -> 视作 column-major [M,N]
    checkCublas(cublasLtMatrixLayoutCreate(&e->c, CUDA_R_16F, M, N, M), "cublasLtMatrixLayoutCreate_gemm2_c");

    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
        "cublasLtMatrixLayoutSetAttribute_gemm2_order_a");
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
        "cublasLtMatrixLayoutSetAttribute_gemm2_order_b");
    checkCublas(
        cublasLtMatrixLayoutSetAttribute(e->c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
        "cublasLtMatrixLayoutSetAttribute_gemm2_order_c");

    cublasLtMatmulPreference_t pref = nullptr;
    checkCublas(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate_gemm2");
    checkCublas(
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)),
        "cublasLtMatmulPreferenceSetAttribute_gemm2_ws");

    constexpr int MAX_ALGOS = 32;
    cublasLtMatmulHeuristicResult_t heurs[MAX_ALGOS];
    int got = 0;
    checkCublas(
        cublasLtMatmulAlgoGetHeuristic(handle, e->op, e->a, e->b, e->c, e->c, pref, MAX_ALGOS, heurs, &got),
        "cublasLtMatmulAlgoGetHeuristic_gemm2");
    cublasLtMatmulPreferenceDestroy(pref);

    if (got <= 0) {
      throw std::runtime_error("cublasLt: no heuristic algo for gemm2");
    }

    e->algo_count = 0;
    for (int i = 0; i < got && e->algo_count < 8; ++i) {
      if (heurs[i].state == CUBLAS_STATUS_SUCCESS && heurs[i].workspaceSize <= workspace_bytes) {
        e->algo_list[e->algo_count] = heurs[i].algo;
        e->algo_ws_list[e->algo_count] = heurs[i].workspaceSize;
        e->algo_count++;
      }
    }
    if (e->algo_count <= 0) {
      throw std::runtime_error("cublasLt: workspace too small for gemm2");
    }

    e->algo = e->algo_list[0];
    e->algo_workspace = e->algo_ws_list[0];
    e->tuned = false;
    e->ready = true;
    return e;
  }
};

static CublasLtHolder* get_cublas_lt() {
  static std::once_flag once;
  static CublasLtHolder* holder = nullptr;
  std::call_once(once, []() { holder = new CublasLtHolder(); });
  return holder;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int d = 16; d > 0; d >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, d);
  }
  return v;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
  float z = __expf(-x);
  return 1.0f / (1.0f + z);
}

// ---------------- 权重打包/转换（每次调用都执行，禁止跨调用复用中间结果） ----------------

__global__ void pack_w_cat5_and_out_f32_to_f16_v4(
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    const float* __restrict__ w4,
    const float* __restrict__ w5,
    half* __restrict__ out_cat, // [5*hidden, dim] row-major
    half* __restrict__ out_wout, // [dim, hidden] row-major
    int64_t dim,
    int64_t hidden) {
  int m = (int)blockIdx.y; // 0..5
  const float* src = (m == 0) ? w0 : (m == 1) ? w1 : (m == 2) ? w2 : (m == 3) ? w3 : (m == 4) ? w4 : w5;

  int64_t n = hidden * dim;
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x * 4 + (int64_t)threadIdx.x * 4;
  if (i >= n) return;

  half* dst = (m < 5) ? (out_cat + (int64_t)m * n) : out_wout;

  if (i + 3 < n) {
    float4 v = *reinterpret_cast<const float4*>(src + i);
    half2 h0 = __floats2half2_rn(v.x, v.y);
    half2 h1 = __floats2half2_rn(v.z, v.w);
    half2* out2 = reinterpret_cast<half2*>(dst + i);
    out2[0] = h0;
    out2[1] = h1;
  } else {
    for (int k = 0; k < 4; ++k) {
      int64_t j = i + (int64_t)k;
      if (j < n) {
        dst[j] = __float2half_rn(src[j]);
      }
    }
  }
}

static void launch_pack_weights(
    torch::Tensor w_left,
    torch::Tensor w_right,
    torch::Tensor w_lg,
    torch::Tensor w_rg,
    torch::Tensor w_og,
    torch::Tensor w_out,
    torch::Tensor w_cat_h,
    torch::Tensor w_out_h) {
  int64_t hidden = w_left.size(0);
  int64_t dim = w_left.size(1);

  {
    int64_t n = hidden * dim;
    dim3 block(256, 1, 1);
    dim3 grid((unsigned)((n + (int64_t)block.x * 4 - 1) / ((int64_t)block.x * 4)), 6, 1);
    pack_w_cat5_and_out_f32_to_f16_v4<<<grid, block>>>(
        (const float*)w_left.data_ptr(),
        (const float*)w_right.data_ptr(),
        (const float*)w_lg.data_ptr(),
        (const float*)w_rg.data_ptr(),
        (const float*)w_og.data_ptr(),
        (const float*)w_out.data_ptr(),
        (half*)w_cat_h.data_ptr(),
        (half*)w_out_h.data_ptr(),
        dim,
        hidden);
    checkCuda(cudaGetLastError(), "pack_w_cat5_and_out_f32_to_f16_v4");
  }
}

// ---------------- LN1 ----------------
// 关键优化：减少 block 数量（每个 block 处理多个 row），显著降低大 M 场景下的调度/启动开销。

__global__ void ln1_128_f16_warp8_r8(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    half* __restrict__ y,
    int64_t rows) {
  // 进一步减少 CTA 数量：每个 warp 顺序处理 8 个 row（dim=128 时每 row=1 warp）。
  int tid = (int)threadIdx.x;     // 0..255
  int lane = tid & 31;            // 0..31
  int warp_id = tid >> 5;         // 0..7
  int64_t row0 = (int64_t)blockIdx.x * 64 + (int64_t)warp_id * 8;
  if (row0 >= rows) return;

  const float4* w4 = reinterpret_cast<const float4*>(w);
  const float4* b4 = reinterpret_cast<const float4*>(b);
  float4 gw = w4[lane];
  float4 gb = b4[lane];

  #pragma unroll
  for (int rr = 0; rr < 8; ++rr) {
    int64_t row = row0 + (int64_t)rr;
    if (row >= rows) return;
    const float4* x4 = reinterpret_cast<const float4*>(x + row * 128);
    float4 v = x4[lane];
    float s = v.x + v.y + v.z + v.w;
    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    s = warp_sum(s);
    ss = warp_sum(ss);
    s = __shfl_sync(0xffffffff, s, 0);
    ss = __shfl_sync(0xffffffff, ss, 0);
    float mean = s * (1.0f / 128.0f);
    float var = ss * (1.0f / 128.0f) - mean * mean;
    float inv = rsqrtf(var + 1e-5f);

    float y0 = (v.x - mean) * inv * gw.x + gb.x;
    float y1 = (v.y - mean) * inv * gw.y + gb.y;
    float y2 = (v.z - mean) * inv * gw.z + gb.z;
    float y3 = (v.w - mean) * inv * gw.w + gb.w;

    half2* y2p = reinterpret_cast<half2*>(y + row * 128 + lane * 4);
    y2p[0] = __floats2half2_rn(y0, y1);
    y2p[1] = __floats2half2_rn(y2, y3);
  }
}

__global__ void ln1_384_f16_rows8(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    half* __restrict__ y,
    int64_t rows) {
  // 关键优化：每个 row 需要 3 warps（384=96*4），将单 CTA 的 row 数从 4 提升到 8，
  // 将 CTA 数量减少约 2x，降低调度/启动开销。
  int tid = (int)threadIdx.x;     // 0..383
  int lane = tid & 31;            // 0..31
  int warp_id = tid >> 5;         // 0..11

  int row_in_block = warp_id / 3;       // 0..3
  int warp_in_row = warp_id - row_in_block * 3; // 0..2
  int tid_row = warp_in_row * 32 + lane; // 0..95

  __shared__ float warp_s[4][3];
  __shared__ float warp_ss[4][3];
  __shared__ float tot_s[4];
  __shared__ float tot_ss[4];

  #pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    int64_t row = (int64_t)blockIdx.x * 8 + (int64_t)row_in_block * 2 + (int64_t)rr;
    bool row_ok = row < rows;

    float4 v;
    if (row_ok) {
      const float4* x4 = reinterpret_cast<const float4*>(x + row * 384);
      v = x4[tid_row];
    } else {
      v.x = 0.0f;
      v.y = 0.0f;
      v.z = 0.0f;
      v.w = 0.0f;
    }

    float s = v.x + v.y + v.z + v.w;
    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    s = warp_sum(s);
    ss = warp_sum(ss);

    if (lane == 0) {
      warp_s[row_in_block][warp_in_row] = s;
      warp_ss[row_in_block][warp_in_row] = ss;
    }
    __syncthreads();

    if (warp_in_row == 0) {
      float sum = 0.0f;
      float sq = 0.0f;
      if (lane < 3) {
        sum = warp_s[row_in_block][lane];
        sq = warp_ss[row_in_block][lane];
      }
      sum = warp_sum(sum);
      sq = warp_sum(sq);
      if (lane == 0) {
        tot_s[row_in_block] = sum;
        tot_ss[row_in_block] = sq;
      }
    }
    __syncthreads();

    float sum = tot_s[row_in_block];
    float sq = tot_ss[row_in_block];

    float mean = sum * (1.0f / 384.0f);
    float var = sq * (1.0f / 384.0f) - mean * mean;
    float inv = rsqrtf(var + 1e-5f);

    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4 gw = w4[tid_row];
    float4 gb = b4[tid_row];

    float y0 = (v.x - mean) * inv * gw.x + gb.x;
    float y1 = (v.y - mean) * inv * gw.y + gb.y;
    float y2 = (v.z - mean) * inv * gw.z + gb.z;
    float y3 = (v.w - mean) * inv * gw.w + gb.w;

    if (row_ok) {
      half2 h0 = __floats2half2_rn(y0, y1);
      half2 h1 = __floats2half2_rn(y2, y3);
      half2* y2p = reinterpret_cast<half2*>(y + row * 384 + tid_row * 4);
      y2p[0] = h0;
      y2p[1] = h1;
    }
  }
}

__global__ void ln1_256_f16_rows8(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    half* __restrict__ y,
    int64_t rows) {
  // dim=256：每 row 2 warps（256=64*4）；单 CTA 处理 8 个 row，进一步摊薄调度/归约开销
  int tid = (int)threadIdx.x; // 0..255
  int lane = tid & 31;
  int warp_id = tid >> 5; // 0..7

  int row_in_block = warp_id >> 1;     // 0..3
  int warp_in_row = warp_id & 1;       // 0..1
  int tid_row = warp_in_row * 32 + lane; // 0..63 (float4 index)

  __shared__ float warp_s[4][2];
  __shared__ float warp_ss[4][2];
  __shared__ float tot_s[4];
  __shared__ float tot_ss[4];

  #pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    int64_t row = (int64_t)blockIdx.x * 8 + (int64_t)row_in_block * 2 + (int64_t)rr;
    bool row_ok = row < rows;

    float4 v;
    if (row_ok) {
      const float4* x4 = reinterpret_cast<const float4*>(x + row * 256);
      v = x4[tid_row];
    } else {
      v.x = 0.0f;
      v.y = 0.0f;
      v.z = 0.0f;
      v.w = 0.0f;
    }

    float s = v.x + v.y + v.z + v.w;
    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    s = warp_sum(s);
    ss = warp_sum(ss);

    if (lane == 0) {
      warp_s[row_in_block][warp_in_row] = s;
      warp_ss[row_in_block][warp_in_row] = ss;
    }
    __syncthreads();

    if (warp_in_row == 0) {
      float sum = (lane < 2) ? warp_s[row_in_block][lane] : 0.0f;
      float sq = (lane < 2) ? warp_ss[row_in_block][lane] : 0.0f;
      sum = warp_sum(sum);
      sq = warp_sum(sq);
      if (lane == 0) {
        tot_s[row_in_block] = sum;
        tot_ss[row_in_block] = sq;
      }
    }
    __syncthreads();

    float sum = tot_s[row_in_block];
    float sq = tot_ss[row_in_block];
    float mean = sum * (1.0f / 256.0f);
    float var = sq * (1.0f / 256.0f) - mean * mean;
    float inv = rsqrtf(var + 1e-5f);

    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4 gw = w4[tid_row];
    float4 gb = b4[tid_row];

    float y0 = (v.x - mean) * inv * gw.x + gb.x;
    float y1 = (v.y - mean) * inv * gw.y + gb.y;
    float y2 = (v.z - mean) * inv * gw.z + gb.z;
    float y3 = (v.w - mean) * inv * gw.w + gb.w;

    if (row_ok) {
      half2* y2p = reinterpret_cast<half2*>(y + row * 256 + tid_row * 4);
      y2p[0] = __floats2half2_rn(y0, y1);
      y2p[1] = __floats2half2_rn(y2, y3);
    }
  }
}

__global__ void ln1_768_f16_rows4(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    half* __restrict__ y,
    int64_t rows) {
  // dim=768：每 row 6 warps（768=192*4）；单 CTA 处理 4 个 row，降低 block 数量
  int tid = (int)threadIdx.x; // 0..383
  int lane = tid & 31;
  int warp_id = tid >> 5; // 0..11

  int row_in_block = warp_id / 6;           // 0..1
  int warp_in_row = warp_id - row_in_block * 6; // 0..5
  int tid_row = warp_in_row * 32 + lane;    // 0..191 (float4 index)

  __shared__ float warp_s[2][6];
  __shared__ float warp_ss[2][6];
  __shared__ float tot_s[2];
  __shared__ float tot_ss[2];

  #pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    int64_t row = (int64_t)blockIdx.x * 4 + (int64_t)row_in_block * 2 + (int64_t)rr;
    bool row_ok = row < rows;

    float4 v;
    if (row_ok) {
      const float4* x4 = reinterpret_cast<const float4*>(x + row * 768);
      v = x4[tid_row];
    } else {
      v.x = 0.0f;
      v.y = 0.0f;
      v.z = 0.0f;
      v.w = 0.0f;
    }

    float s = v.x + v.y + v.z + v.w;
    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    s = warp_sum(s);
    ss = warp_sum(ss);

    if (lane == 0) {
      warp_s[row_in_block][warp_in_row] = s;
      warp_ss[row_in_block][warp_in_row] = ss;
    }
    __syncthreads();

    if (warp_in_row == 0) {
      float sum = (lane < 6) ? warp_s[row_in_block][lane] : 0.0f;
      float sq = (lane < 6) ? warp_ss[row_in_block][lane] : 0.0f;
      sum = warp_sum(sum);
      sq = warp_sum(sq);
      if (lane == 0) {
        tot_s[row_in_block] = sum;
        tot_ss[row_in_block] = sq;
      }
    }
    __syncthreads();

    float sum = tot_s[row_in_block];
    float sq = tot_ss[row_in_block];
    float mean = sum * (1.0f / 768.0f);
    float var = sq * (1.0f / 768.0f) - mean * mean;
    float inv = rsqrtf(var + 1e-5f);

    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4 gw = w4[tid_row];
    float4 gb = b4[tid_row];

    float y0 = (v.x - mean) * inv * gw.x + gb.x;
    float y1 = (v.y - mean) * inv * gw.y + gb.y;
    float y2 = (v.z - mean) * inv * gw.z + gb.z;
    float y3 = (v.w - mean) * inv * gw.w + gb.w;

    if (row_ok) {
      half2* y2p = reinterpret_cast<half2*>(y + row * 768 + tid_row * 4);
      y2p[0] = __floats2half2_rn(y0, y1);
      y2p[1] = __floats2half2_rn(y2, y3);
    }
  }
}

__global__ void ln1_generic_f16(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    half* __restrict__ y,
    int dim,
    int64_t rows) {
  int64_t row = (int64_t)blockIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  float sq = 0.0f;
  int64_t base = row * (int64_t)dim;
  for (int c = (int)threadIdx.x; c < dim; c += (int)blockDim.x) {
    float v = x[base + c];
    sum += v;
    sq += v * v;
  }

  __shared__ float shm_sum[256];
  __shared__ float shm_sq[256];
  int t = (int)threadIdx.x;
  shm_sum[t] = sum;
  shm_sq[t] = sq;
  __syncthreads();

  for (int stride = ((int)blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      shm_sum[t] += shm_sum[t + stride];
      shm_sq[t] += shm_sq[t + stride];
    }
    __syncthreads();
  }

  float mean = shm_sum[0] / (float)dim;
  float var = shm_sq[0] / (float)dim - mean * mean;
  float inv = rsqrtf(var + 1e-5f);

  for (int c = (int)threadIdx.x; c < dim; c += (int)blockDim.x) {
    float v = x[base + c];
    float yv = (v - mean) * inv * w[c] + b[c];
    y[base + c] = __float2half_rn(yv);
  }
}

static void launch_ln1(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor y) {
  int dim = (int)x.size(1);
  auto rows = x.size(0);
  if (dim == 128) {
    // 8 warps/CTA；每个 warp 顺序处理 8 个 row（进一步减少 CTA 数量）
    dim3 block(256, 1, 1);
    dim3 grid((unsigned)((rows + 63) / 64), 1, 1);
    ln1_128_f16_warp8_r8<<<grid, block>>>(
        (const float*)x.data_ptr(),
        (const float*)w.data_ptr(),
        (const float*)b.data_ptr(),
        (half*)y.data_ptr(),
        (int64_t)rows);
    checkCuda(cudaGetLastError(), "ln1_128_f16_warp8_r8");
  } else if (dim == 256) {
    // 8 warps/CTA；每 row 2 warps，单 CTA 处理 8 row
    dim3 block(256, 1, 1);
    dim3 grid((unsigned)((rows + 7) / 8), 1, 1);
    ln1_256_f16_rows8<<<grid, block>>>(
        (const float*)x.data_ptr(),
        (const float*)w.data_ptr(),
        (const float*)b.data_ptr(),
        (half*)y.data_ptr(),
        (int64_t)rows);
    checkCuda(cudaGetLastError(), "ln1_256_f16_rows8");
  } else if (dim == 384) {
    // 12 warps/CTA；每 3 个 warp 处理 1 个 row（单 CTA 处理 8 个 row）
    dim3 block(384, 1, 1);
    dim3 grid((unsigned)((rows + 7) / 8), 1, 1);
    ln1_384_f16_rows8<<<grid, block>>>(
        (const float*)x.data_ptr(),
        (const float*)w.data_ptr(),
        (const float*)b.data_ptr(),
        (half*)y.data_ptr(),
        (int64_t)rows);
    checkCuda(cudaGetLastError(), "ln1_384_f16_rows8");
  } else if (dim == 768) {
    // 12 warps/CTA；每 row 6 warps，单 CTA 处理 4 row
    dim3 block(384, 1, 1);
    dim3 grid((unsigned)((rows + 3) / 4), 1, 1);
    ln1_768_f16_rows4<<<grid, block>>>(
        (const float*)x.data_ptr(),
        (const float*)w.data_ptr(),
        (const float*)b.data_ptr(),
        (half*)y.data_ptr(),
        (int64_t)rows);
    checkCuda(cudaGetLastError(), "ln1_768_f16_rows4");
  } else {
    dim3 block(256, 1, 1);
    dim3 grid((unsigned)rows, 1, 1);
    ln1_generic_f16<<<grid, block>>>(
        (const float*)x.data_ptr(),
        (const float*)w.data_ptr(),
        (const float*)b.data_ptr(),
        (half*)y.data_ptr(),
        dim,
        (int64_t)rows);
    checkCuda(cudaGetLastError(), "ln1_generic_f16");
  }
}

// --------------- pack (projT -> left/right) ---------------
// projT 形状为 [5H, M]（row-major，最后一维 M 连续），避免额外转置。
// mask 支持 half/float32：
// - half：维持历史版本（节省读带宽）
// - float32：避免 Python 侧额外的 dtype 转换核与写回，再在 kernel 内就地转 half2（0/1 掩码不会引入误差）

__global__ void pack4_lr_dmaj_f16(
    const half* __restrict__ projT,  // [5H, M]
    const half* __restrict__ mask_h, // [bs, nn]
    half* __restrict__ left,         // [bs*H, nn]
    half* __restrict__ right,        // [bs*H, nn]
    int64_t nn,
    int64_t M,
    int hidden) {
  int b = (int)blockIdx.y; // 0..bs-1
  int d = (int)blockIdx.z; // 0..hidden-1
  int bd = b * hidden + d;

  // 向量化：每线程处理 4 个元素（2x half2），减少 grid.x 约 2x
  int64_t p0 = (int64_t)blockIdx.x * (int64_t)blockDim.x * 4 + (int64_t)threadIdx.x * 4;
  if (p0 >= nn) return;

  const half* mask_row = mask_h + (int64_t)b * nn;
  int64_t idx0 = (int64_t)b * nn + p0;

  const half* row_l = projT + (int64_t)d * M;
  const half* row_r = projT + (int64_t)(hidden + d) * M;
  const half* row_gl = projT + (int64_t)(2 * hidden + d) * M;
  const half* row_gr = projT + (int64_t)(3 * hidden + d) * M;

  half* out_l = left + (int64_t)bd * nn;
  half* out_r = right + (int64_t)bd * nn;

  // 主路径：一次处理 4 个元素（2x half2）
  if (p0 + 3 < nn) {
    half2 m0 = *reinterpret_cast<const half2*>(mask_row + p0);
    half2 m1 = *reinterpret_cast<const half2*>(mask_row + p0 + 2);

    half2 l0 = *reinterpret_cast<const half2*>(row_l + idx0);
    half2 l1 = *reinterpret_cast<const half2*>(row_l + idx0 + 2);
    half2 r0 = *reinterpret_cast<const half2*>(row_r + idx0);
    half2 r1 = *reinterpret_cast<const half2*>(row_r + idx0 + 2);

    half2 gl0 = *reinterpret_cast<const half2*>(row_gl + idx0);
    half2 gl1 = *reinterpret_cast<const half2*>(row_gl + idx0 + 2);
    half2 gr0 = *reinterpret_cast<const half2*>(row_gr + idx0);
    half2 gr1 = *reinterpret_cast<const half2*>(row_gr + idx0 + 2);

    // gate：half2 -> float2 做 sigmoid，再回写 half2；乘法用 half2 指令（吞吐更友好）
    float2 glf0 = __half22float2(gl0);
    float2 glf1 = __half22float2(gl1);
    float2 grf0 = __half22float2(gr0);
    float2 grf1 = __half22float2(gr1);

    glf0.x = fast_sigmoid(glf0.x);
    glf0.y = fast_sigmoid(glf0.y);
    glf1.x = fast_sigmoid(glf1.x);
    glf1.y = fast_sigmoid(glf1.y);
    grf0.x = fast_sigmoid(grf0.x);
    grf0.y = fast_sigmoid(grf0.y);
    grf1.x = fast_sigmoid(grf1.x);
    grf1.y = fast_sigmoid(grf1.y);

    half2 glh0 = __floats2half2_rn(glf0.x, glf0.y);
    half2 glh1 = __floats2half2_rn(glf1.x, glf1.y);
    half2 grh0 = __floats2half2_rn(grf0.x, grf0.y);
    half2 grh1 = __floats2half2_rn(grf1.x, grf1.y);

    *reinterpret_cast<half2*>(out_l + p0) = __hmul2(__hmul2(l0, glh0), m0);
    *reinterpret_cast<half2*>(out_r + p0) = __hmul2(__hmul2(r0, grh0), m0);
    *reinterpret_cast<half2*>(out_l + p0 + 2) = __hmul2(__hmul2(l1, glh1), m1);
    *reinterpret_cast<half2*>(out_r + p0 + 2) = __hmul2(__hmul2(r1, grh1), m1);
  } else {
    for (int t = 0; t < 4; ++t) {
      int64_t p = p0 + t;
      if (p < nn) {
        half m = mask_row[p];
        float ml = __half2float(m);
        float l = __half2float(row_l[(int64_t)b * nn + p]);
        float r = __half2float(row_r[(int64_t)b * nn + p]);
        float glf = fast_sigmoid(__half2float(row_gl[(int64_t)b * nn + p]));
        float grf = fast_sigmoid(__half2float(row_gr[(int64_t)b * nn + p]));
        out_l[p] = __float2half_rn(l * glf * ml);
        out_r[p] = __float2half_rn(r * grf * ml);
      }
    }
  }
}

__global__ void pack4_lr_dmaj_f16_full(
    const half* __restrict__ projT,  // [5H, M]
    const half* __restrict__ mask_h, // [bs, nn]
    half* __restrict__ left,         // [bs*H, nn]
    half* __restrict__ right,        // [bs*H, nn]
    int64_t nn,
    int64_t M,
    int hidden) {
  int b = (int)blockIdx.y; // 0..bs-1
  int d = (int)blockIdx.z; // 0..hidden-1
  int bd = b * hidden + d;

  int64_t p0 = (int64_t)blockIdx.x * (int64_t)blockDim.x * 4 + (int64_t)threadIdx.x * 4;

  const half* mask_row = mask_h + (int64_t)b * nn;
  int64_t idx0 = (int64_t)b * nn + p0;

  const half* row_l = projT + (int64_t)d * M;
  const half* row_r = projT + (int64_t)(hidden + d) * M;
  const half* row_gl = projT + (int64_t)(2 * hidden + d) * M;
  const half* row_gr = projT + (int64_t)(3 * hidden + d) * M;

  half* out_l = left + (int64_t)bd * nn;
  half* out_r = right + (int64_t)bd * nn;

  half2 m0 = *reinterpret_cast<const half2*>(mask_row + p0);
  half2 m1 = *reinterpret_cast<const half2*>(mask_row + p0 + 2);

  half2 l0 = *reinterpret_cast<const half2*>(row_l + idx0);
  half2 l1 = *reinterpret_cast<const half2*>(row_l + idx0 + 2);
  half2 r0 = *reinterpret_cast<const half2*>(row_r + idx0);
  half2 r1 = *reinterpret_cast<const half2*>(row_r + idx0 + 2);

  half2 gl0 = *reinterpret_cast<const half2*>(row_gl + idx0);
  half2 gl1 = *reinterpret_cast<const half2*>(row_gl + idx0 + 2);
  half2 gr0 = *reinterpret_cast<const half2*>(row_gr + idx0);
  half2 gr1 = *reinterpret_cast<const half2*>(row_gr + idx0 + 2);

  float2 glf0 = __half22float2(gl0);
  float2 glf1 = __half22float2(gl1);
  float2 grf0 = __half22float2(gr0);
  float2 grf1 = __half22float2(gr1);

  glf0.x = fast_sigmoid(glf0.x);
  glf0.y = fast_sigmoid(glf0.y);
  glf1.x = fast_sigmoid(glf1.x);
  glf1.y = fast_sigmoid(glf1.y);
  grf0.x = fast_sigmoid(grf0.x);
  grf0.y = fast_sigmoid(grf0.y);
  grf1.x = fast_sigmoid(grf1.x);
  grf1.y = fast_sigmoid(grf1.y);

  half2 glh0 = __floats2half2_rn(glf0.x, glf0.y);
  half2 glh1 = __floats2half2_rn(glf1.x, glf1.y);
  half2 grh0 = __floats2half2_rn(grf0.x, grf0.y);
  half2 grh1 = __floats2half2_rn(grf1.x, grf1.y);

  *reinterpret_cast<half2*>(out_l + p0) = __hmul2(__hmul2(l0, glh0), m0);
  *reinterpret_cast<half2*>(out_r + p0) = __hmul2(__hmul2(r0, grh0), m0);
  *reinterpret_cast<half2*>(out_l + p0 + 2) = __hmul2(__hmul2(l1, glh1), m1);
  *reinterpret_cast<half2*>(out_r + p0 + 2) = __hmul2(__hmul2(r1, grh1), m1);
}

__global__ void pack4_lr_dmaj_f16_mf32(
    const half* __restrict__ projT,   // [5H, M]
    const float* __restrict__ mask_f, // [bs, nn]
    half* __restrict__ left,          // [bs*H, nn]
    half* __restrict__ right,         // [bs*H, nn]
    int64_t nn,
    int64_t M,
    int hidden) {
  int b = (int)blockIdx.y;
  int d = (int)blockIdx.z;
  int bd = b * hidden + d;

  int64_t p0 = (int64_t)blockIdx.x * (int64_t)blockDim.x * 4 + (int64_t)threadIdx.x * 4;
  if (p0 >= nn) return;

  const float* mask_row = mask_f + (int64_t)b * nn;
  int64_t idx0 = (int64_t)b * nn + p0;

  const half* row_l = projT + (int64_t)d * M;
  const half* row_r = projT + (int64_t)(hidden + d) * M;
  const half* row_gl = projT + (int64_t)(2 * hidden + d) * M;
  const half* row_gr = projT + (int64_t)(3 * hidden + d) * M;

  half* out_l = left + (int64_t)bd * nn;
  half* out_r = right + (int64_t)bd * nn;

  if (p0 + 3 < nn) {
    float4 mv = *reinterpret_cast<const float4*>(mask_row + p0);
    half2 m0 = __floats2half2_rn(mv.x, mv.y);
    half2 m1 = __floats2half2_rn(mv.z, mv.w);

    half2 l0 = *reinterpret_cast<const half2*>(row_l + idx0);
    half2 l1 = *reinterpret_cast<const half2*>(row_l + idx0 + 2);
    half2 r0 = *reinterpret_cast<const half2*>(row_r + idx0);
    half2 r1 = *reinterpret_cast<const half2*>(row_r + idx0 + 2);

    half2 gl0 = *reinterpret_cast<const half2*>(row_gl + idx0);
    half2 gl1 = *reinterpret_cast<const half2*>(row_gl + idx0 + 2);
    half2 gr0 = *reinterpret_cast<const half2*>(row_gr + idx0);
    half2 gr1 = *reinterpret_cast<const half2*>(row_gr + idx0 + 2);

    float2 glf0 = __half22float2(gl0);
    float2 glf1 = __half22float2(gl1);
    float2 grf0 = __half22float2(gr0);
    float2 grf1 = __half22float2(gr1);

    glf0.x = fast_sigmoid(glf0.x);
    glf0.y = fast_sigmoid(glf0.y);
    glf1.x = fast_sigmoid(glf1.x);
    glf1.y = fast_sigmoid(glf1.y);
    grf0.x = fast_sigmoid(grf0.x);
    grf0.y = fast_sigmoid(grf0.y);
    grf1.x = fast_sigmoid(grf1.x);
    grf1.y = fast_sigmoid(grf1.y);

    half2 glh0 = __floats2half2_rn(glf0.x, glf0.y);
    half2 glh1 = __floats2half2_rn(glf1.x, glf1.y);
    half2 grh0 = __floats2half2_rn(grf0.x, grf0.y);
    half2 grh1 = __floats2half2_rn(grf1.x, grf1.y);

    *reinterpret_cast<half2*>(out_l + p0) = __hmul2(__hmul2(l0, glh0), m0);
    *reinterpret_cast<half2*>(out_r + p0) = __hmul2(__hmul2(r0, grh0), m0);
    *reinterpret_cast<half2*>(out_l + p0 + 2) = __hmul2(__hmul2(l1, glh1), m1);
    *reinterpret_cast<half2*>(out_r + p0 + 2) = __hmul2(__hmul2(r1, grh1), m1);
  } else {
    for (int t = 0; t < 4; ++t) {
      int64_t p = p0 + t;
      if (p < nn) {
        float ml = mask_row[p];
        float l = __half2float(row_l[(int64_t)b * nn + p]);
        float r = __half2float(row_r[(int64_t)b * nn + p]);
        float glf = fast_sigmoid(__half2float(row_gl[(int64_t)b * nn + p]));
        float grf = fast_sigmoid(__half2float(row_gr[(int64_t)b * nn + p]));
        out_l[p] = __float2half_rn(l * glf * ml);
        out_r[p] = __float2half_rn(r * grf * ml);
      }
    }
  }
}

__global__ void pack4_lr_dmaj_f16_mf32_full(
    const half* __restrict__ projT,
    const float* __restrict__ mask_f,
    half* __restrict__ left,
    half* __restrict__ right,
    int64_t nn,
    int64_t M,
    int hidden) {
  int b = (int)blockIdx.y;
  int d = (int)blockIdx.z;
  int bd = b * hidden + d;

  int64_t p0 = (int64_t)blockIdx.x * (int64_t)blockDim.x * 4 + (int64_t)threadIdx.x * 4;

  const float* mask_row = mask_f + (int64_t)b * nn;
  int64_t idx0 = (int64_t)b * nn + p0;

  const half* row_l = projT + (int64_t)d * M;
  const half* row_r = projT + (int64_t)(hidden + d) * M;
  const half* row_gl = projT + (int64_t)(2 * hidden + d) * M;
  const half* row_gr = projT + (int64_t)(3 * hidden + d) * M;

  half* out_l = left + (int64_t)bd * nn;
  half* out_r = right + (int64_t)bd * nn;

  float4 mv = *reinterpret_cast<const float4*>(mask_row + p0);
  half2 m0 = __floats2half2_rn(mv.x, mv.y);
  half2 m1 = __floats2half2_rn(mv.z, mv.w);

  half2 l0 = *reinterpret_cast<const half2*>(row_l + idx0);
  half2 l1 = *reinterpret_cast<const half2*>(row_l + idx0 + 2);
  half2 r0 = *reinterpret_cast<const half2*>(row_r + idx0);
  half2 r1 = *reinterpret_cast<const half2*>(row_r + idx0 + 2);

  half2 gl0 = *reinterpret_cast<const half2*>(row_gl + idx0);
  half2 gl1 = *reinterpret_cast<const half2*>(row_gl + idx0 + 2);
  half2 gr0 = *reinterpret_cast<const half2*>(row_gr + idx0);
  half2 gr1 = *reinterpret_cast<const half2*>(row_gr + idx0 + 2);

  float2 glf0 = __half22float2(gl0);
  float2 glf1 = __half22float2(gl1);
  float2 grf0 = __half22float2(gr0);
  float2 grf1 = __half22float2(gr1);

  glf0.x = fast_sigmoid(glf0.x);
  glf0.y = fast_sigmoid(glf0.y);
  glf1.x = fast_sigmoid(glf1.x);
  glf1.y = fast_sigmoid(glf1.y);
  grf0.x = fast_sigmoid(grf0.x);
  grf0.y = fast_sigmoid(grf0.y);
  grf1.x = fast_sigmoid(grf1.x);
  grf1.y = fast_sigmoid(grf1.y);

  half2 glh0 = __floats2half2_rn(glf0.x, glf0.y);
  half2 glh1 = __floats2half2_rn(glf1.x, glf1.y);
  half2 grh0 = __floats2half2_rn(grf0.x, grf0.y);
  half2 grh1 = __floats2half2_rn(grf1.x, grf1.y);

  *reinterpret_cast<half2*>(out_l + p0) = __hmul2(__hmul2(l0, glh0), m0);
  *reinterpret_cast<half2*>(out_r + p0) = __hmul2(__hmul2(r0, grh0), m0);
  *reinterpret_cast<half2*>(out_l + p0 + 2) = __hmul2(__hmul2(l1, glh1), m1);
  *reinterpret_cast<half2*>(out_r + p0 + 2) = __hmul2(__hmul2(r1, grh1), m1);
}

static void launch_pack4(
    torch::Tensor projT,
    torch::Tensor mask,
    torch::Tensor left,
    torch::Tensor right,
    int bs,
    int n,
    int hidden) {
  int64_t nn = (int64_t)n * (int64_t)n;
  int64_t M = (int64_t)bs * nn;
  dim3 block(256, 1, 1);
  int64_t vec = (int64_t)block.x * 4;
  bool full = ((nn % vec) == 0);
  dim3 grid((unsigned)(full ? (nn / vec) : ((nn + vec - 1) / vec)), (unsigned)bs, (unsigned)hidden);
  if (mask.scalar_type() == torch::kFloat16) {
    if (full) {
      pack4_lr_dmaj_f16_full<<<grid, block>>>(
          (const half*)projT.data_ptr(),
          (const half*)mask.data_ptr(),
          (half*)left.data_ptr(),
          (half*)right.data_ptr(),
          nn,
          M,
          hidden);
      checkCuda(cudaGetLastError(), "pack4_lr_dmaj_f16_full");
    } else {
      pack4_lr_dmaj_f16<<<grid, block>>>(
          (const half*)projT.data_ptr(),
          (const half*)mask.data_ptr(),
          (half*)left.data_ptr(),
          (half*)right.data_ptr(),
          nn,
          M,
          hidden);
      checkCuda(cudaGetLastError(), "pack4_lr_dmaj_f16");
    }
  } else {
    if (full) {
      pack4_lr_dmaj_f16_mf32_full<<<grid, block>>>(
          (const half*)projT.data_ptr(),
          (const float*)mask.data_ptr(),
          (half*)left.data_ptr(),
          (half*)right.data_ptr(),
          nn,
          M,
          hidden);
      checkCuda(cudaGetLastError(), "pack4_lr_dmaj_f16_mf32_full");
    } else {
      pack4_lr_dmaj_f16_mf32<<<grid, block>>>(
          (const half*)projT.data_ptr(),
          (const float*)mask.data_ptr(),
          (half*)left.data_ptr(),
          (half*)right.data_ptr(),
          nn,
          M,
          hidden);
      checkCuda(cudaGetLastError(), "pack4_lr_dmaj_f16_mf32");
    }
  }
}

// --------------- LN2 + gate ---------------
// out_acc: [bs*H, nn]；gate 原始值来自 projT 的第 5 段（out_gate），避免 ogate 中间张量写回/读回。
// 输出 out_norm_T: [H, M]（row-major，最后一维 M 连续）。
// 关键点：按 p 连续加载；用少量同步做跨 warp 规约。
// 本版本将 out_acc 以 half 存储，减轻 LN2 的带宽压力与 shared footprint。

template<int WARPS>
__global__ void ln2_gate_tile64_f16_h2(
    const half* __restrict__ out_acc, // [bs*H, nn] (half)
    const half* __restrict__ projT,   // [5H, M] (half)
    const float* __restrict__ w,      // [H]
    const float* __restrict__ b,      // [H]
    half* __restrict__ out_norm_T,    // [H, M]
    int64_t nn,
    int hidden) {
  int bb = (int)blockIdx.y;
  int64_t p0 = (int64_t)blockIdx.x * 64;
  int tid = (int)threadIdx.x;
  int lane = tid & 31;
  int wid = tid >> 5;
  int64_t p = p0 + (int64_t)lane * 2;
  bool p_ok = (p < nn);
  bool pair_ok = (p + 1 < nn);

  int64_t M = nn * (int64_t)gridDim.y;
  int64_t out_p = (int64_t)bb * nn + p;

  extern __shared__ unsigned char smem_u8[];
  half2* sh_x2 = (half2*)smem_u8; // hidden*32 (half2)
  float2* sh_sum2 = (float2*)(smem_u8 + (size_t)((int64_t)hidden * 32) * sizeof(half2));
  float2* sh_sq2 = sh_sum2 + WARPS * 32;
  float2* sh_mean2 = sh_sq2 + WARPS * 32;
  float2* sh_inv2 = sh_mean2 + 32;

  float2 sum2;
  sum2.x = 0.0f;
  sum2.y = 0.0f;
  float2 sq2;
  sq2.x = 0.0f;
  sq2.y = 0.0f;

  for (int d = wid; d < hidden; d += WARPS) {
    half2 hx2;
    if (pair_ok) {
      int64_t idx = ((int64_t)bb * (int64_t)hidden + (int64_t)d) * nn + p;
      hx2 = *reinterpret_cast<const half2*>(out_acc + idx);
    } else if (p_ok) {
      int64_t idx = ((int64_t)bb * (int64_t)hidden + (int64_t)d) * nn + p;
      hx2 = __halves2half2(out_acc[idx], __float2half_rn(0.0f));
    } else {
      hx2 = __float2half2_rn(0.0f);
    }
    sh_x2[(int64_t)d * 32 + lane] = hx2;
    float2 xf = __half22float2(hx2);
    sum2.x += xf.x;
    sum2.y += xf.y;
    sq2.x += xf.x * xf.x;
    sq2.y += xf.y * xf.y;
  }

  sh_sum2[wid * 32 + lane] = sum2;
  sh_sq2[wid * 32 + lane] = sq2;
  __syncthreads();

  if (wid == 0) {
    float2 tot;
    tot.x = 0.0f;
    tot.y = 0.0f;
    float2 tot_sq;
    tot_sq.x = 0.0f;
    tot_sq.y = 0.0f;
#pragma unroll
    for (int w_id = 0; w_id < WARPS; ++w_id) {
      float2 s = sh_sum2[w_id * 32 + lane];
      float2 q = sh_sq2[w_id * 32 + lane];
      tot.x += s.x;
      tot.y += s.y;
      tot_sq.x += q.x;
      tot_sq.y += q.y;
    }
    float inv_n = 1.0f / (float)hidden;
    float2 mean;
    mean.x = tot.x * inv_n;
    mean.y = tot.y * inv_n;
    float2 var;
    var.x = tot_sq.x * inv_n - mean.x * mean.x;
    var.y = tot_sq.y * inv_n - mean.y * mean.y;
    float2 inv;
    inv.x = rsqrtf(var.x + 1e-5f);
    inv.y = rsqrtf(var.y + 1e-5f);
    sh_mean2[lane] = mean;
    sh_inv2[lane] = inv;
  }
  __syncthreads();

  float2 mean = sh_mean2[lane];
  float2 inv = sh_inv2[lane];

  if (!p_ok) return;

  for (int d = wid; d < hidden; d += WARPS) {
    half2 hx2 = sh_x2[(int64_t)d * 32 + lane];
    float2 xf = __half22float2(hx2);

    // LN2
    float wv = w[d];
    float bv = b[d];
    float y0 = (xf.x - mean.x) * inv.x * wv + bv;
    float y1 = (xf.y - mean.y) * inv.y * wv + bv;

    // gate out（sigmoid）
    int64_t gate_idx = ((int64_t)(4 * hidden + d) * M) + out_p;
    half2 ogx2;
    if (pair_ok) {
      ogx2 = *reinterpret_cast<const half2*>(projT + gate_idx);
    } else {
      ogx2 = __halves2half2(projT[gate_idx], __float2half_rn(0.0f));
    }
    float2 gf = __half22float2(ogx2);
    gf.x = fast_sigmoid(gf.x);
    gf.y = fast_sigmoid(gf.y);

    y0 *= gf.x;
    y1 *= gf.y;

    // 写回 out_norm_T： [H, M] row-major，最后一维 M 连续
    int64_t out_idx = (int64_t)d * M + out_p;
    half2 out2 = __floats2half2_rn(y0, y1);
    if (pair_ok) {
      *reinterpret_cast<half2*>(out_norm_T + out_idx) = out2;
    } else {
      out_norm_T[out_idx] = __float2half_rn(y0);
    }
  }
}

template<int WARPS>
__global__ void ln2_gate_tile64_f16_h2_full(
    const half* __restrict__ out_acc,
    const half* __restrict__ projT,
    const float* __restrict__ w,
    const float* __restrict__ b,
    half* __restrict__ out_norm_T,
    int64_t nn,
    int hidden) {
  int bb = (int)blockIdx.y;
  int64_t p0 = (int64_t)blockIdx.x * 64;
  int tid = (int)threadIdx.x;
  int lane = tid & 31;
  int wid = tid >> 5;
  int64_t p = p0 + (int64_t)lane * 2;

  int64_t M = nn * (int64_t)gridDim.y;
  int64_t out_p = (int64_t)bb * nn + p;

  extern __shared__ unsigned char smem_u8[];
  half2* sh_x2 = (half2*)smem_u8;
  float2* sh_sum2 = (float2*)(smem_u8 + (size_t)((int64_t)hidden * 32) * sizeof(half2));
  float2* sh_sq2 = sh_sum2 + WARPS * 32;
  float2* sh_mean2 = sh_sq2 + WARPS * 32;
  float2* sh_inv2 = sh_mean2 + 32;

  float2 sum2;
  sum2.x = 0.0f;
  sum2.y = 0.0f;
  float2 sq2;
  sq2.x = 0.0f;
  sq2.y = 0.0f;

  for (int d = wid; d < hidden; d += WARPS) {
    int64_t idx = ((int64_t)bb * (int64_t)hidden + (int64_t)d) * nn + p;
    half2 hx2 = *reinterpret_cast<const half2*>(out_acc + idx);
    sh_x2[(int64_t)d * 32 + lane] = hx2;
    float2 xf = __half22float2(hx2);
    sum2.x += xf.x;
    sum2.y += xf.y;
    sq2.x += xf.x * xf.x;
    sq2.y += xf.y * xf.y;
  }

  sh_sum2[wid * 32 + lane] = sum2;
  sh_sq2[wid * 32 + lane] = sq2;
  __syncthreads();

  if (wid == 0) {
    float2 tot;
    tot.x = 0.0f;
    tot.y = 0.0f;
    float2 tot_sq;
    tot_sq.x = 0.0f;
    tot_sq.y = 0.0f;
#pragma unroll
    for (int w_id = 0; w_id < WARPS; ++w_id) {
      float2 s = sh_sum2[w_id * 32 + lane];
      float2 q = sh_sq2[w_id * 32 + lane];
      tot.x += s.x;
      tot.y += s.y;
      tot_sq.x += q.x;
      tot_sq.y += q.y;
    }
    float inv_n = 1.0f / (float)hidden;
    float2 mean;
    mean.x = tot.x * inv_n;
    mean.y = tot.y * inv_n;
    float2 var;
    var.x = tot_sq.x * inv_n - mean.x * mean.x;
    var.y = tot_sq.y * inv_n - mean.y * mean.y;
    float2 inv;
    inv.x = rsqrtf(var.x + 1e-5f);
    inv.y = rsqrtf(var.y + 1e-5f);
    sh_mean2[lane] = mean;
    sh_inv2[lane] = inv;
  }
  __syncthreads();

  float2 mean = sh_mean2[lane];
  float2 inv = sh_inv2[lane];

  const half* og_base = projT + (int64_t)(4 * hidden) * M + out_p;

  for (int d = wid; d < hidden; d += WARPS) {
    half2 hx2 = sh_x2[(int64_t)d * 32 + lane];
    float2 xf = __half22float2(hx2);
    float y0 = (xf.x - mean.x) * inv.x * w[d] + b[d];
    float y1 = (xf.y - mean.y) * inv.y * w[d] + b[d];

    half2 ogx2 = *reinterpret_cast<const half2*>(og_base + (int64_t)d * M);
    float2 gf = __half22float2(ogx2);
    gf.x = fast_sigmoid(gf.x);
    gf.y = fast_sigmoid(gf.y);
    y0 *= gf.x;
    y1 *= gf.y;

    *reinterpret_cast<half2*>(out_norm_T + (int64_t)d * M + out_p) = __floats2half2_rn(y0, y1);
  }
}

static void launch_ln2(
    torch::Tensor out_acc,
    torch::Tensor projT,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor out_norm_T,
    int bs,
    int n,
    int hidden) {
  int64_t nn = (int64_t)n * (int64_t)n;
  dim3 grid((unsigned)((nn + 63) / 64), (unsigned)bs, 1);
  bool full = ((nn % 64) == 0);

  if (hidden <= 32) {
    constexpr int WARPS = 1;
    dim3 block(WARPS * 32, 1, 1);
    size_t shmem = (size_t)((int64_t)hidden * 32) * sizeof(half2)
                 + (size_t)((int64_t)WARPS * 32 * 2 + 64) * sizeof(float2);
    if (full) {
      ln2_gate_tile64_f16_h2_full<WARPS><<<grid, block, shmem>>>(
          (const half*)out_acc.data_ptr(),
          (const half*)projT.data_ptr(),
          (const float*)w.data_ptr(),
          (const float*)b.data_ptr(),
          (half*)out_norm_T.data_ptr(),
          nn,
          hidden);
    } else {
      ln2_gate_tile64_f16_h2<WARPS><<<grid, block, shmem>>>(
          (const half*)out_acc.data_ptr(),
          (const half*)projT.data_ptr(),
          (const float*)w.data_ptr(),
          (const float*)b.data_ptr(),
          (half*)out_norm_T.data_ptr(),
          nn,
          hidden);
    }
    checkCuda(cudaGetLastError(), "ln2_gate_tile64_f16_h2_1w");
  } else if (hidden <= 64) {
    constexpr int WARPS = 2;
    dim3 block(WARPS * 32, 1, 1);
    size_t shmem = (size_t)((int64_t)hidden * 32) * sizeof(half2)
                 + (size_t)((int64_t)WARPS * 32 * 2 + 64) * sizeof(float2);
    if (full) {
      ln2_gate_tile64_f16_h2_full<WARPS><<<grid, block, shmem>>>(
          (const half*)out_acc.data_ptr(),
          (const half*)projT.data_ptr(),
          (const float*)w.data_ptr(),
          (const float*)b.data_ptr(),
          (half*)out_norm_T.data_ptr(),
          nn,
          hidden);
    } else {
      ln2_gate_tile64_f16_h2<WARPS><<<grid, block, shmem>>>(
          (const half*)out_acc.data_ptr(),
          (const half*)projT.data_ptr(),
          (const float*)w.data_ptr(),
          (const float*)b.data_ptr(),
          (half*)out_norm_T.data_ptr(),
          nn,
          hidden);
    }
    checkCuda(cudaGetLastError(), "ln2_gate_tile64_f16_h2_2w");
  } else if (hidden <= 128) {
    // hidden=128 是评测主形状，增加 warps 提升并行度（每 warp 处理 16 个通道）
    constexpr int WARPS = 8;
    dim3 block(WARPS * 32, 1, 1);
    size_t shmem = (size_t)((int64_t)hidden * 32) * sizeof(half2)
                 + (size_t)((int64_t)WARPS * 32 * 2 + 64) * sizeof(float2);
    if (full) {
      ln2_gate_tile64_f16_h2_full<WARPS><<<grid, block, shmem>>>(
          (const half*)out_acc.data_ptr(),
          (const half*)projT.data_ptr(),
          (const float*)w.data_ptr(),
          (const float*)b.data_ptr(),
          (half*)out_norm_T.data_ptr(),
          nn,
          hidden);
    } else {
      ln2_gate_tile64_f16_h2<WARPS><<<grid, block, shmem>>>(
          (const half*)out_acc.data_ptr(),
          (const half*)projT.data_ptr(),
          (const float*)w.data_ptr(),
          (const float*)b.data_ptr(),
          (half*)out_norm_T.data_ptr(),
          nn,
          hidden);
    }
    checkCuda(cudaGetLastError(), "ln2_gate_tile64_f16_h2_8w_h128");
  } else if (hidden <= 256) {
    constexpr int WARPS = 8;
    dim3 block(WARPS * 32, 1, 1);
    size_t shmem = (size_t)((int64_t)hidden * 32) * sizeof(half2)
                 + (size_t)((int64_t)WARPS * 32 * 2 + 64) * sizeof(float2);
    if (full) {
      ln2_gate_tile64_f16_h2_full<WARPS><<<grid, block, shmem>>>(
          (const half*)out_acc.data_ptr(),
          (const half*)projT.data_ptr(),
          (const float*)w.data_ptr(),
          (const float*)b.data_ptr(),
          (half*)out_norm_T.data_ptr(),
          nn,
          hidden);
    } else {
      ln2_gate_tile64_f16_h2<WARPS><<<grid, block, shmem>>>(
          (const half*)out_acc.data_ptr(),
          (const half*)projT.data_ptr(),
          (const float*)w.data_ptr(),
          (const float*)b.data_ptr(),
          (half*)out_norm_T.data_ptr(),
          nn,
          hidden);
    }
    checkCuda(cudaGetLastError(), "ln2_gate_tile64_f16_h2_8w");
  } else {
    throw std::runtime_error("hidden_dim too large");
  }
}

// ---------------- GEMM helpers ----------------

// 约定：所有矩阵都来自 row-major Tensor，但用 cuBLAS 的 column-major 语义解释，
// 通过精确设置 m/n/k 与 lda/ldb/ldc 得到想要的布局，避免额外转置核。

static void tune_gemm_once(
    CublasLtHolder* lt_holder,
    LtGemmCacheEntry* e,
    const void* a_ptr,
    const void* b_ptr,
    void* c_ptr) {
  if (e->tuned) return;
  if (e->algo_count <= 1) {
    e->tuned = true;
    return;
  }

  cudaEvent_t ev0;
  cudaEvent_t ev1;
  checkCuda(cudaEventCreate(&ev0), "cudaEventCreate_ev0_gemm");
  checkCuda(cudaEventCreate(&ev1), "cudaEventCreate_ev1_gemm");

  float alpha = 1.0f;
  float beta = 0.0f;
  float best_ms = 1e30f;
  int best_i = 0;
  bool any_ok = false;

  constexpr int REPS = 3;
  for (int i = 0; i < e->algo_count; ++i) {
    // 先跑一次确保该候选可用（并顺带做 warmup）
    cublasStatus_t st0 = cublasLtMatmul(
        lt_holder->handle,
        e->op,
        &alpha,
        a_ptr,
        e->a,
        b_ptr,
        e->b,
        &beta,
        c_ptr,
        e->c,
        c_ptr,
        e->c,
        &e->algo_list[i],
        lt_holder->workspace,
        lt_holder->workspace_bytes,
        0);
    if (st0 != CUBLAS_STATUS_SUCCESS) continue;

    checkCuda(cudaEventRecord(ev0, 0), "cudaEventRecord_ev0_gemm");
    bool ok = true;
    for (int r = 0; r < REPS; ++r) {
      cublasStatus_t st = cublasLtMatmul(
          lt_holder->handle,
          e->op,
          &alpha,
          a_ptr,
          e->a,
          b_ptr,
          e->b,
          &beta,
          c_ptr,
          e->c,
          c_ptr,
          e->c,
          &e->algo_list[i],
          lt_holder->workspace,
          lt_holder->workspace_bytes,
          0);
      if (st != CUBLAS_STATUS_SUCCESS) {
        ok = false;
        break;
      }
    }
    checkCuda(cudaEventRecord(ev1, 0), "cudaEventRecord_ev1_gemm");
    checkCuda(cudaEventSynchronize(ev1), "cudaEventSynchronize_ev1_gemm");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, ev0, ev1), "cudaEventElapsedTime_gemm");
    if (ok && ms < best_ms) {
      best_ms = ms;
      best_i = i;
      any_ok = true;
    }
  }

  checkCuda(cudaEventDestroy(ev0), "cudaEventDestroy_ev0_gemm");
  checkCuda(cudaEventDestroy(ev1), "cudaEventDestroy_ev1_gemm");

  if (!any_ok) {
    throw std::runtime_error("cublasLt: all tuned gemm algos failed");
  }

  e->algo = e->algo_list[best_i];
  e->algo_workspace = e->algo_ws_list[best_i];
  e->tuned = true;
}

static void tune_contract_once(
    CublasLtHolder* lt_holder,
    LtContractCacheEntry* e,
    const void* a_ptr,
    const void* b_ptr,
    void* c_ptr) {
  if (e->tuned) return;
  if (e->algo_count <= 1) {
    e->tuned = true;
    return;
  }

  cudaEvent_t ev0;
  cudaEvent_t ev1;
  checkCuda(cudaEventCreate(&ev0), "cudaEventCreate_ev0_contract");
  checkCuda(cudaEventCreate(&ev1), "cudaEventCreate_ev1_contract");

  float alpha = 1.0f;
  float beta = 0.0f;
  float best_ms = 1e30f;
  int best_i = 0;
  bool any_ok = false;

  // contract 单次耗时较大，择优用更小 REPS 控制 warmup 代价
  constexpr int REPS = 2;
  for (int i = 0; i < e->algo_count; ++i) {
    // 先跑一次确保该候选可用（并顺带做 warmup）
    cublasStatus_t st0 = cublasLtMatmul(
        lt_holder->handle,
        e->op,
        &alpha,
        a_ptr,
        e->a,
        b_ptr,
        e->b,
        &beta,
        c_ptr,
        e->c,
        c_ptr,
        e->c,
        &e->algo_list[i],
        lt_holder->workspace,
        lt_holder->workspace_bytes,
        0);
    if (st0 != CUBLAS_STATUS_SUCCESS) continue;

    checkCuda(cudaEventRecord(ev0, 0), "cudaEventRecord_ev0_contract");
    bool ok = true;
    for (int r = 0; r < REPS; ++r) {
      cublasStatus_t st = cublasLtMatmul(
          lt_holder->handle,
          e->op,
          &alpha,
          a_ptr,
          e->a,
          b_ptr,
          e->b,
          &beta,
          c_ptr,
          e->c,
          c_ptr,
          e->c,
          &e->algo_list[i],
          lt_holder->workspace,
          lt_holder->workspace_bytes,
          0);
      if (st != CUBLAS_STATUS_SUCCESS) {
        ok = false;
        break;
      }
    }
    checkCuda(cudaEventRecord(ev1, 0), "cudaEventRecord_ev1_contract");
    checkCuda(cudaEventSynchronize(ev1), "cudaEventSynchronize_ev1_contract");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, ev0, ev1), "cudaEventElapsedTime_contract");
    if (ok && ms < best_ms) {
      best_ms = ms;
      best_i = i;
      any_ok = true;
    }
  }

  checkCuda(cudaEventDestroy(ev0), "cudaEventDestroy_ev0_contract");
  checkCuda(cudaEventDestroy(ev1), "cudaEventDestroy_ev1_contract");

  if (!any_ok) {
    throw std::runtime_error("cublasLt: all tuned contract algos failed");
  }

  e->algo = e->algo_list[best_i];
  e->algo_workspace = e->algo_ws_list[best_i];
  e->tuned = true;
}

static void gemm1_x_wt_to_dmaj_f16(
    cublasHandle_t h,
    const half* x_rm,  // [M, K] row-major
    const half* w_rm,  // [N, K] row-major
    half* c_dmaj_rm,   // [N, M] row-major (等价于 column-major [M, N])
    int64_t M,
    int64_t N,
    int64_t K) {
  (void)h;
  auto* lt_holder = get_cublas_lt();
  LtGemmCacheEntry* e = lt_holder->get_gemm1(M, N, K);
  if (!e->tuned) {
    std::lock_guard<std::mutex> lock(lt_holder->mu);
    if (!e->tuned) {
      tune_gemm_once(lt_holder, e, (const void*)x_rm, (const void*)w_rm, (void*)c_dmaj_rm);
    }
  }
  float alpha = 1.0f;
  float beta = 0.0f;
  checkCublas(
      cublasLtMatmul(
          lt_holder->handle,
          e->op,
          &alpha,
          (const void*)x_rm,
          e->a,
          (const void*)w_rm,
          e->b,
          &beta,
          (const void*)c_dmaj_rm,
          e->c,
          (void*)c_dmaj_rm,
          e->c,
          &e->algo,
          lt_holder->workspace,
          lt_holder->workspace_bytes,
          0),
      "cublasLtMatmul_gemm1");
}

static void gemm2_dmaj_to_y_t_f16_f16(
    cublasHandle_t h,
    const half* a_dmaj_rm, // [K, M] row-major (等价于 column-major [M, K])
    const half* w_rm,      // [N, K] row-major (等价于 column-major [K, N])
    half* y_t_rm,          // [N, M] row-major (等价于 column-major [M, N])
    int64_t M,
    int64_t N,
    int64_t K) {
  (void)h;
  auto* lt_holder = get_cublas_lt();
  LtGemmCacheEntry* e = lt_holder->get_gemm2(M, N, K);
  if (!e->tuned) {
    std::lock_guard<std::mutex> lock(lt_holder->mu);
    if (!e->tuned) {
      tune_gemm_once(lt_holder, e, (const void*)a_dmaj_rm, (const void*)w_rm, (void*)y_t_rm);
    }
  }
  float alpha = 1.0f;
  float beta = 0.0f;
  checkCublas(
      cublasLtMatmul(
          lt_holder->handle,
          e->op,
          &alpha,
          (const void*)a_dmaj_rm,
          e->a,
          (const void*)w_rm,
          e->b,
          &beta,
          (const void*)y_t_rm,
          e->c,
          (void*)y_t_rm,
          e->c,
          &e->algo,
          lt_holder->workspace,
          lt_holder->workspace_bytes,
          0),
      "cublasLtMatmul_gemm2");
}

static void gemm_contract_batched_f16_f16(
    cublasHandle_t h,
    const half* left_row,   // [B,M,K] row-major [batch,n,n]
    const half* right_row,  // [B,N,K] row-major [batch,n,n]
    half* out_row,          // [B,M,N] row-major [batch,n,n] (half)
    int batch,
    int n) {
  float alpha = 1.0f;
  float beta = 0.0f;
  long long strideA = (long long)n * (long long)n;
  long long strideB = (long long)n * (long long)n;
  long long strideC = (long long)n * (long long)n;

  checkCublas(
      cublasGemmStridedBatchedEx(
          h,
          CUBLAS_OP_T, CUBLAS_OP_N,
          n, n, n,
          &alpha,
          right_row, CUDA_R_16F, n, strideB,
          left_row, CUDA_R_16F, n, strideA,
          &beta,
          out_row, CUDA_R_16F, n, strideC,
          batch,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cublasGemmStridedBatchedEx_contract");
}

static void contract_batched_f16_f16_auto(
    cublasHandle_t h,
    const half* left_row,
    const half* right_row,
    half* out_row,
    int batch,
    int n) {
  (void)h;
  auto* lt_holder = get_cublas_lt();
  LtContractCacheEntry* e = lt_holder->get_contract(n, batch);
  if (!e->tuned) {
    std::lock_guard<std::mutex> lock(lt_holder->mu);
    if (!e->tuned) {
      tune_contract_once(lt_holder, e, (const void*)right_row, (const void*)left_row, (void*)out_row);
    }
  }
  float alpha = 1.0f;
  float beta = 0.0f;
  checkCublas(
      cublasLtMatmul(
          lt_holder->handle,
          e->op,
          &alpha,
          (const void*)right_row,
          e->a,
          (const void*)left_row,
          e->b,
          &beta,
          (const void*)out_row,
          e->c,
          (void*)out_row,
          e->c,
          &e->algo,
          lt_holder->workspace,
          lt_holder->workspace_bytes,
          0),
      "cublasLtMatmul_contract");
}

} // namespace

torch::Tensor trimul_fwd(
    torch::Tensor x,
    torch::Tensor mask_h,
    torch::Tensor ln1_w,
    torch::Tensor ln1_b,
    torch::Tensor w_left,
    torch::Tensor w_right,
    torch::Tensor w_lg,
    torch::Tensor w_rg,
    torch::Tensor w_og,
    torch::Tensor ln2_w,
    torch::Tensor ln2_b,
    torch::Tensor w_out,
    int64_t dim,
    int64_t hidden) {
  if (!x.is_cuda() || !mask_h.is_cuda()) {
    throw std::runtime_error("cuda only");
  }
  if (x.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("x must be float32");
  }
  if (mask_h.scalar_type() != torch::kFloat16 && mask_h.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("mask must be float16/float32");
  }
  if (dim != x.size(3)) {
    throw std::runtime_error("dim mismatch");
  }
  if (w_left.scalar_type() != torch::kFloat32 || w_right.scalar_type() != torch::kFloat32 ||
      w_lg.scalar_type() != torch::kFloat32 || w_rg.scalar_type() != torch::kFloat32 ||
      w_og.scalar_type() != torch::kFloat32 || w_out.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("proj weights must be float32");
  }
  if (w_left.sizes() != torch::IntArrayRef({hidden, dim})) throw std::runtime_error("left_proj.weight shape mismatch");
  if (w_right.sizes() != torch::IntArrayRef({hidden, dim})) throw std::runtime_error("right_proj.weight shape mismatch");
  if (w_lg.sizes() != torch::IntArrayRef({hidden, dim})) throw std::runtime_error("left_gate.weight shape mismatch");
  if (w_rg.sizes() != torch::IntArrayRef({hidden, dim})) throw std::runtime_error("right_gate.weight shape mismatch");
  if (w_og.sizes() != torch::IntArrayRef({hidden, dim})) throw std::runtime_error("out_gate.weight shape mismatch");
  if (w_out.sizes() != torch::IntArrayRef({dim, hidden})) throw std::runtime_error("to_out.weight shape mismatch");

  int bs = (int)x.size(0);
  int n = (int)x.size(1);
  int64_t nn = (int64_t)n * (int64_t)n;
  int64_t M = (int64_t)bs * nn;

  auto x2d = x.view({M, dim});
  auto xhat = torch::empty({M, dim}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat16));
  launch_ln1(x2d, ln1_w, ln1_b, xhat);

  // 每次调用都做权重拼接/转换（禁止跨调用复用）
  int64_t out_ch = hidden * 5;
  auto w_cat_h = torch::empty({out_ch, dim}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat16));
  auto w_out_h = torch::empty({dim, hidden}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat16));
  launch_pack_weights(w_left, w_right, w_lg, w_rg, w_og, w_out, w_cat_h, w_out_h);

  // projT: [5H, M]（d-major，最后一维连续）
  auto projT = torch::empty({out_ch, M}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat16));

  auto* holder = get_cublas();
  cublasHandle_t h = holder->handle;
  gemm1_x_wt_to_dmaj_f16(
      h,
      (const half*)xhat.data_ptr(),
      (const half*)w_cat_h.data_ptr(),
      (half*)projT.data_ptr(),
      M,
      out_ch,
      dim);

  auto left = torch::empty({bs * (int)hidden, nn}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat16));
  auto right = torch::empty({bs * (int)hidden, nn}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat16));

  // pack：projT[前 4 段, M] + mask[bs,nn] -> left/right[bs*H,nn]
  // out_gate（第 5 段）留在 projT 中，后续由 LN2 kernel 直接读取并 sigmoid。
  launch_pack4(projT, mask_h.view({bs, nn}), left, right, bs, n, (int)hidden);

  auto left3 = left.view({bs * (int)hidden, n, n});
  auto right3 = right.view({bs * (int)hidden, n, n});
  // out_acc: half（仍由 GEMM 做 FP32 累加，只降低写回精度）
  auto out_acc = torch::empty({bs * (int)hidden, n, n}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat16));

  contract_batched_f16_f16_auto(
      h,
      (const half*)left3.data_ptr(),
      (const half*)right3.data_ptr(),
      (half*)out_acc.data_ptr(),
      bs * (int)hidden, n);

  // LN2 + gate：输出 out_norm_T[H,M] half
  auto out_norm_T = torch::empty({hidden, M}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat16));
  launch_ln2(out_acc, projT, ln2_w, ln2_b, out_norm_T, bs, n, (int)hidden);

  // gemm2：y_T[dim,M] half（降低最终写回带宽；仍 FP32 累加）
  auto y_T = torch::empty({dim, M}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat16));
  gemm2_dmaj_to_y_t_f16_f16(
      h,
      (const half*)out_norm_T.data_ptr(),
      (const half*)w_out_h.data_ptr(),
      (half*)y_T.data_ptr(),
      M,
      dim,
      hidden);

  return y_T.view({dim, bs, n, n}).permute({1, 2, 3, 0});
}
"""

        name = "trimul_ext_mod22_lt_cache64_ws768_ct32"

        
        
        extra_cuda_cflags = [
            "-O2",
            "--use_fast_math",
        ]
        extra_cflags = [
            "-O2",
        ]

        extra_ldflags = [
            "-lcublas",
            "-lcublasLt",
        ]

        _EXT = load_inline(
            name=name,
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=None,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            with_cuda=True,
            verbose=False,
        )
        return _EXT


@torch.inference_mode()
def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]) -> torch.Tensor:
    x, mask, weights, config = data

    dim = int(config["dim"])
    hidden = int(config["hidden_dim"])

    if x.dtype != torch.float32:
        x = x.to(dtype=torch.float32)
    
    if mask.dtype not in (torch.float16, torch.float32):
        mask = mask.to(dtype=torch.float16)

    x = x.contiguous()
    mask = mask.contiguous()

    ext = _get_ext()
    return ext.fwd(
        x,
        mask,
        weights["norm.weight"].contiguous(),
        weights["norm.bias"].contiguous(),
        weights["left_proj.weight"].contiguous(),
        weights["right_proj.weight"].contiguous(),
        weights["left_gate.weight"].contiguous(),
        weights["right_gate.weight"].contiguous(),
        weights["out_gate.weight"].contiguous(),
        weights["to_out_norm.weight"].contiguous(),
        weights["to_out_norm.bias"].contiguous(),
        weights["to_out.weight"].contiguous(),
        dim,
        hidden,
    )


__all__ = ["custom_kernel"]
