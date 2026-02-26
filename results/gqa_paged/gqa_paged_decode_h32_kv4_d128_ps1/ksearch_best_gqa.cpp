#include <torch/extension.h>
#include <vector>
#include "ksearch_best_gqa.h"

std::vector<torch::Tensor> run(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale
) {
    q = q.contiguous();
    k_cache = k_cache.contiguous();
    v_cache = v_cache.contiguous();
    kv_indptr = kv_indptr.to(torch::kInt).contiguous();
    kv_indices = kv_indices.to(torch::kInt).contiguous();

    int batch_size = q.size(0);
    int num_qo_heads = q.size(1);
    int head_dim = q.size(2);
    int num_kv_heads = k_cache.size(2);

    auto output = torch::empty({batch_size, num_qo_heads, head_dim}, q.options());
    auto lse = torch::empty({batch_size, num_qo_heads}, q.options().dtype(torch::kFloat32));

    int split_k = 1;
    auto tmp_buffer = torch::empty({1}, q.options().dtype(torch::kFloat32));
    auto tmp_stats = torch::empty({1}, q.options().dtype(torch::kFloat32));
    auto semaphores = torch::empty({1}, q.options().dtype(torch::kInt32));

    if (split_k > 1) {
        tmp_buffer = torch::empty({batch_size, num_kv_heads, split_k, 8, head_dim}, q.options().dtype(torch::kFloat32));
        tmp_stats = torch::empty({batch_size, num_kv_heads, split_k, 16}, q.options().dtype(torch::kFloat32));
        semaphores = torch::zeros({batch_size, num_kv_heads}, q.options().dtype(torch::kInt32));
    }

    launch_gqa_decode(
        output,
        lse,
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        static_cast<float>(sm_scale),
        tmp_buffer,
        tmp_stats,
        semaphores,
        split_k
    );

    return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "GQA Paged Decode");
}
