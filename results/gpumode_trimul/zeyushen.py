import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel_v44(X, LN_W, LN_B, Out,
                        stride_xb, stride_xn1, stride_xn2, stride_xc,
                        B, N, C, 
                        BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_b, pid_n1 = tl.program_id(0), tl.program_id(1)
    pid_n2_start = tl.program_id(2) * BLOCK_N
    offs_n2 = pid_n2_start + tl.arange(0, BLOCK_N)
    mask_n2 = offs_n2 < N
    
    m1 = tl.zeros([BLOCK_N], dtype=tl.float32)
    m2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        x_ptr = X + pid_b * stride_xb + pid_n1 * stride_xn1 + offs_n2[:, None] * stride_xn2 + offs_c[None, :]
        x = tl.load(x_ptr, mask=(mask_n2[:, None] & mask_c[None, :]), other=0.0).to(tl.float32)
        m1 += tl.sum(x, axis=1)
        m2 += tl.sum(x * x, axis=1)
    
    mean = m1 / C
    var = tl.maximum(0.0, (m2 / C) - (mean * mean))
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        x_ptr = X + pid_b * stride_xb + pid_n1 * stride_xn1 + offs_n2[:, None] * stride_xn2 + offs_c[None, :]
        x = tl.load(x_ptr, mask=(mask_n2[:, None] & mask_c[None, :]), other=0.0).to(tl.float32)
        ln_w = tl.load(LN_W + offs_c, mask=mask_c, other=0.0)
        ln_b = tl.load(LN_B + offs_c, mask=mask_c, other=0.0)
        x_hat = (x - mean[:, None]) * rstd[:, None] * ln_w[None, :] + ln_b[None, :]
        out_ptr = Out + pid_b * stride_xb + pid_n1 * stride_xn1 + offs_n2[:, None] * stride_xn2 + offs_c[None, :]
        tl.store(out_ptr, x_hat.to(tl.float16), mask=(mask_n2[:, None] & mask_c[None, :]))

@triton.jit
def projection_kernel_v44(
    X_norm, Mask, W_concat,
    L_out, R_out,
    stride_xb, stride_xn1, stride_xn2, stride_xc,
    stride_mb, stride_mn1, stride_mn2,
    stride_ob, stride_od, stride_on1, stride_on2,
    B, N, C, D: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr
):
    pid_b, pid_n1 = tl.program_id(0), tl.program_id(1)
    pid_n2_start = tl.program_id(2) * BLOCK_N
    offs_n2 = pid_n2_start + tl.arange(0, BLOCK_N)
    mask_n2 = offs_n2 < N
    
    acc_l = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    acc_r = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    acc_lg = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    acc_rg = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    x_block_ptr = tl.make_block_ptr(
        base=X_norm + pid_b * stride_xb + pid_n1 * stride_xn1,
        shape=(N, C),
        strides=(stride_xn2, stride_xc),
        offsets=(pid_n2_start, 0),
        block_shape=(BLOCK_N, BLOCK_C),
        order=(1, 0)
    )
    
    w_l_ptr = tl.make_block_ptr(base=W_concat, shape=(4*D, C), strides=(C, 1), offsets=(0*D, 0), block_shape=(D, BLOCK_C), order=(1, 0))
    w_r_ptr = tl.make_block_ptr(base=W_concat, shape=(4*D, C), strides=(C, 1), offsets=(1*D, 0), block_shape=(D, BLOCK_C), order=(1, 0))
    w_lg_ptr = tl.make_block_ptr(base=W_concat, shape=(4*D, C), strides=(C, 1), offsets=(2*D, 0), block_shape=(D, BLOCK_C), order=(1, 0))
    w_rg_ptr = tl.make_block_ptr(base=W_concat, shape=(4*D, C), strides=(C, 1), offsets=(3*D, 0), block_shape=(D, BLOCK_C), order=(1, 0))

    for c_start in range(0, C, BLOCK_C):
        x = tl.load(x_block_ptr, boundary_check=(0, 1)).to(tl.float16)
        w_l = tl.load(w_l_ptr, boundary_check=(1,)).to(tl.float16)
        w_r = tl.load(w_r_ptr, boundary_check=(1,)).to(tl.float16)
        w_lg = tl.load(w_lg_ptr, boundary_check=(1,)).to(tl.float16)
        w_rg = tl.load(w_rg_ptr, boundary_check=(1,)).to(tl.float16)

        acc_l += tl.dot(x, tl.trans(w_l))
        acc_r += tl.dot(x, tl.trans(w_r))
        acc_lg += tl.dot(x, tl.trans(w_lg))
        acc_rg += tl.dot(x, tl.trans(w_rg))
        
        x_block_ptr = tl.advance(x_block_ptr, [0, BLOCK_C])
        w_l_ptr = tl.advance(w_l_ptr, [0, BLOCK_C])
        w_r_ptr = tl.advance(w_r_ptr, [0, BLOCK_C])
        w_lg_ptr = tl.advance(w_lg_ptr, [0, BLOCK_C])
        w_rg_ptr = tl.advance(w_rg_ptr, [0, BLOCK_C])

    m_ptr = Mask + pid_b * stride_mb + pid_n1 * stride_mn1 + offs_n2 * stride_mn2
    mask_val = tl.load(m_ptr, mask=mask_n2, other=0.0).to(tl.float32)

    l_final = acc_l * (mask_val[:, None] * tl.sigmoid(acc_lg))
    r_final = acc_r * (mask_val[:, None] * tl.sigmoid(acc_rg))

    out_off = pid_b * stride_ob + tl.arange(0, D)[:, None] * stride_od + pid_n1 * stride_on1 + offs_n2[None, :] * stride_on2
    tl.store(L_out + out_off, tl.trans(l_final).to(tl.float16), mask=mask_n2[None, :])
    tl.store(R_out + out_off, tl.trans(r_final).to(tl.float16), mask=mask_n2[None, :])

@triton.jit
def post_process_kernel_v44(
    matmul_out, X_norm, W_post_concat, LN_W, LN_B, Out,
    stride_mb, stride_md, stride_mn, stride_mm,
    stride_xb, stride_xn1, stride_xn2, stride_xc,
    stride_ob, stride_on1, stride_on2, stride_oc,
    B, N, D: tl.constexpr, C: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr
):
    pid_b, pid_n1 = tl.program_id(0), tl.program_id(1)
    pid_n2_start = tl.program_id(2) * BLOCK_N
    offs_n2 = pid_n2_start + tl.arange(0, BLOCK_N)
    mask_n2 = offs_n2 < N
    offs_d = tl.arange(0, D)

    m_block_ptr = tl.make_block_ptr(
        base=matmul_out + pid_b * stride_mb + pid_n1 * stride_mn,
        shape=(D, N),
        strides=(stride_md, stride_mm),
        offsets=(0, pid_n2_start),
        block_shape=(D, BLOCK_N),
        order=(0, 1)
    )
    x = tl.trans(tl.load(m_block_ptr, boundary_check=(1,))).to(tl.float32)

    mean = tl.sum(x, axis=1) / D
    diff = x - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    ln_w = tl.load(LN_W + offs_d)
    ln_b = tl.load(LN_B + offs_d)
    x_normed = ((diff * rstd[:, None]) * ln_w[None, :] + ln_b[None, :])

    acc_og = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    x_norm_ptr = tl.make_block_ptr(
        base=X_norm + pid_b * stride_xb + pid_n1 * stride_xn1,
        shape=(N, C),
        strides=(stride_xn2, stride_xc),
        offsets=(pid_n2_start, 0),
        block_shape=(BLOCK_N, BLOCK_C),
        order=(1, 0)
    )
    w_og_ptr = tl.make_block_ptr(base=W_post_concat, shape=(2*D, C), strides=(C, 1), offsets=(0, 0), block_shape=(D, BLOCK_C), order=(1, 0))

    for c_start in range(0, C, BLOCK_C):
        xn = tl.load(x_norm_ptr, boundary_check=(0, 1)).to(tl.float16)
        w_og = tl.load(w_og_ptr, boundary_check=(1,)).to(tl.float16)
        acc_og += tl.dot(xn, tl.trans(w_og))
        x_norm_ptr = tl.advance(x_norm_ptr, [0, BLOCK_C])
        w_og_ptr = tl.advance(w_og_ptr, [0, BLOCK_C])

    og = tl.sigmoid(acc_og)
    x_normed_gated = (x_normed * og).to(tl.float16)

    w_out_ptr = tl.make_block_ptr(base=W_post_concat, shape=(2*D, C), strides=(C, 1), offsets=(D, 0), block_shape=(D, BLOCK_C), order=(1, 0))
    for c_start in range(0, C, BLOCK_C):
        w_out = tl.load(w_out_ptr, boundary_check=(1,)).to(tl.float16)
        out_chunk = tl.dot(x_normed_gated, w_out)
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        out_ptr = Out + pid_b * stride_ob + pid_n1 * stride_on1 + offs_n2[:, None] * stride_on2 + offs_c[None, :] * stride_oc
        tl.store(out_ptr, out_chunk.to(tl.float32), mask=(mask_n2[:, None] & mask_c[None, :]))
        w_out_ptr = tl.advance(w_out_ptr, [0, BLOCK_C])

def custom_kernel(data):
    x, mask, weights, config = data
    B, N, _, C = x.shape
    D = config["hidden_dim"]
    device = x.device

    x_norm = torch.empty_like(x, dtype=torch.float16)
    layernorm_kernel_v44[(B, N, (N + 127) // 128)](
        x, weights["norm.weight"], weights["norm.bias"], x_norm,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        B, N, C, BLOCK_N=128, BLOCK_C=128, num_warps=8
    )

    W_concat = torch.cat([
        weights["left_proj.weight"], weights["right_proj.weight"], 
        weights["left_gate.weight"], weights["right_gate.weight"]
    ], dim=0).to(device=device, dtype=torch.float16)

    L_p = torch.empty((B, D, N, N), device=device, dtype=torch.float16)
    R_p = torch.empty((B, D, N, N), device=device, dtype=torch.float16)
    
    projection_kernel_v44[(B, N, (N + 63) // 64)](
        x_norm, mask, W_concat, L_p, R_p,
        x_norm.stride(0), x_norm.stride(1), x_norm.stride(2), x_norm.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2),
        L_p.stride(0), L_p.stride(1), L_p.stride(2), L_p.stride(3),
        B, N, C, D, BLOCK_N=64, BLOCK_C=64, num_warps=8, num_stages=3
    )

    matmul_out = torch.matmul(L_p, R_p.transpose(-1, -2))
    
    W_post_concat = torch.cat([
        weights["out_gate.weight"],
        weights["to_out.weight"].t().contiguous()
    ], dim=0).to(device=device, dtype=torch.float16)

    out = torch.empty((B, N, N, C), device=device, dtype=torch.float32)
    
    post_process_kernel_v44[(B, N, (N + 63) // 64)](
        matmul_out, x_norm, W_post_concat, weights["to_out_norm.weight"], weights["to_out_norm.bias"], out,
        matmul_out.stride(0), matmul_out.stride(1), matmul_out.stride(2), matmul_out.stride(3),
        x_norm.stride(0), x_norm.stride(1), x_norm.stride(2), x_norm.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, N, D, C, BLOCK_N=64, BLOCK_C=64, num_warps=8, num_stages=3
    )

    return out
