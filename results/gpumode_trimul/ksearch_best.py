import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def _layernorm_fwd_to_f16_2d_contig_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M: tl.constexpr, D: tl.constexpr, eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_rows = rows < M
    cols = tl.arange(0, BLOCK_D)

    sum1 = tl.zeros([BLOCK_M], dtype=tl.float32)
    sum2 = tl.zeros([BLOCK_M], dtype=tl.float32)

    for d0 in tl.static_range(0, D, BLOCK_D):
        c = d0 + cols
        m_c = c < D
        offs = rows[:, None] * D + c[None, :]
        x = tl.load(x_ptr + offs, mask=m_rows[:, None] & m_c[None, :], other=0.0).to(tl.float32)
        sum1 += tl.sum(x, axis=1)
        sum2 += tl.sum(x * x, axis=1)

    invN = 1.0 / D
    mean = sum1 * invN
    var = sum2 * invN - mean * mean
    var = tl.maximum(var, 0.0)
    inv = tl.math.rsqrt(var + eps)

    for d0 in tl.static_range(0, D, BLOCK_D):
        c = d0 + cols
        m_c = c < D
        offs = rows[:, None] * D + c[None, :]
        x = tl.load(x_ptr + offs, mask=m_rows[:, None] & m_c[None, :], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c, mask=m_c, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + c, mask=m_c, other=0.0).to(tl.float32)
        y = (x - mean[:, None]) * inv[:, None]
        y = y * w[None, :] + b[None, :]
        tl.store(y_ptr + offs, y.to(tl.float16), mask=m_rows[:, None] & m_c[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'D']
)
@triton.jit
def _fused_proj_pack_kernel(
    x_ptr, w_ptr, mask_ptr, 
    left_ptr, right_ptr, og_ptr,
    M, D, H, SS,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Fuses Matrix Multiply (5 gates) + Sigmoid + Mask + Packing (Permutation)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Optimization: Compute b_idx and rem using scalar arithmetic
    # Assumes SS is a multiple of BLOCK_M (true for S>=128, BLOCK_M<=128)
    blocks_in_SS = SS // BLOCK_M
    b_idx_scalar = pid_m // blocks_in_SS
    rem_start_scalar = (pid_m % blocks_in_SS) * BLOCK_M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < H

    acc_l  = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_r  = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_lg = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_rg = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_og = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # W is [D, 5H]. Row major.
    
    for k in range(0, D, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D
        
        # Load X tile [BLOCK_M, BLOCK_K]
        a = tl.load(x_ptr + offs_m[:, None] * D + offs_k[None, :], 
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float16)
        
        base_w = w_ptr + offs_k[:, None] * (5 * H)
        
        # Load W tiles [BLOCK_K, BLOCK_N] for each of 5 gates
        b_l  = tl.load(base_w + offs_n[None, :], mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float16)
        b_r  = tl.load(base_w + (H + offs_n[None, :]), mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float16)
        b_lg = tl.load(base_w + (2*H + offs_n[None, :]), mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float16)
        b_rg = tl.load(base_w + (3*H + offs_n[None, :]), mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float16)
        b_og = tl.load(base_w + (4*H + offs_n[None, :]), mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float16)
        
        acc_l  += tl.dot(a, b_l)
        acc_r  += tl.dot(a, b_r)
        acc_lg += tl.dot(a, b_lg)
        acc_rg += tl.dot(a, b_rg)
        acc_og += tl.dot(a, b_og)

    m_val = 1.0
    if HAS_MASK:
        m_val = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        m_val = m_val[:, None]

    gate_l = tl.sigmoid(acc_lg)
    gate_r = tl.sigmoid(acc_rg)
    gate_o = tl.sigmoid(acc_og)

    val_l = acc_l * m_val * gate_l
    val_r = acc_r * m_val * gate_r

    # Permuted Store [B, H, S, S] for Left, Right, AND OutGate
    # Indices calc using scalar optimization
    b_idx = b_idx_scalar 
    rem = rem_start_scalar + tl.arange(0, BLOCK_M)
    
    # Dest index = b * (H*SS) + h * SS + rem
    base_dest = b_idx * (H * SS) + rem[:, None]
    dest_term_h = offs_n[None, :] * SS
    offs_dest = base_dest + dest_term_h
    
    store_mask = mask_m[:, None] & mask_n[None, :]

    tl.store(left_ptr + offs_dest, val_l.to(tl.float16), mask=store_mask)
    tl.store(right_ptr + offs_dest, val_r.to(tl.float16), mask=store_mask)
    tl.store(og_ptr + offs_dest, gate_o.to(tl.float16), mask=store_mask)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["N", "S"],
)
@triton.jit
def _ln_outgate_to_out_fused_multiN_tc_contig_kernel(
    out4_ptr, og_ptr, ln_w_ptr, ln_b_ptr, wout_ptr, y_ptr,
    M: tl.constexpr, S: tl.constexpr, SS: tl.constexpr,
    eps: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    
    # Optimization: Scalar computation of indices
    blocks_in_SS = SS // BLOCK_M
    b_idx_scalar = pid_m // blocks_in_SS
    rem_start_scalar = (pid_m % blocks_in_SS) * BLOCK_M
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = rm < M

    b_idx = b_idx_scalar
    rem = rem_start_scalar + tl.arange(0, BLOCK_M)

    rk = tl.arange(0, H)
    
    # Access pattern: b * (H*SS) + h * SS + rem
    base_pos = (b_idx * (H * SS)).to(tl.int64) + rem
    offs_hm = (rk[:, None] * SS + base_pos[None, :]).to(tl.int64)

    # Load BMM output [H, BLOCK_M]
    x16_hm = tl.load(out4_ptr + offs_hm, mask=m_mask[None, :], other=0.0).to(tl.float16)
    x32_hm = x16_hm.to(tl.float32)

    # LayerNorm over H
    invH = 1.0 / H
    mean = tl.sum(x32_hm, axis=0) * invH
    ex2 = tl.sum(x32_hm * x32_hm, axis=0) * invH
    var = ex2 - mean * mean
    var = tl.maximum(var, 0.0)
    inv = tl.math.rsqrt(var + eps)

    y32_hm = (x32_hm - mean[None, :]) * inv[None, :]
    w = tl.load(ln_w_ptr + rk).to(tl.float32)
    b = tl.load(ln_b_ptr + rk).to(tl.float32)
    y32_hm = y32_hm * w[:, None] + b[:, None]

    # Load OutGate [H, BLOCK_M] using same coalesced pattern
    g_hm = tl.load(og_ptr + offs_hm, mask=m_mask[None, :], other=0.0).to(tl.float16)
    
    # Apply gate
    a_hm = (y32_hm.to(tl.float16) * g_hm).to(tl.float16)

    # Output Projection: Y = A^T @ Wout
    # Wout is [D, H] passed as wout_ptr.
    # We load blocks [BLOCK_N, H] from Wout.
    # BLOCK_N iterates over D.
    for nb in tl.static_range(0, (N + BLOCK_N - 1) // BLOCK_N):
        rn = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = rn < N
        
        # Load W tile [BLOCK_N, H]. Layout [D, H] => rn*H + rk
        # Access is contiguous in H (rk).
        b_block = tl.load(wout_ptr + rn[:, None] * H + rk[None, :], mask=n_mask[:, None], other=0.0).to(tl.float16)
        
        # acc = W @ A  => [BLOCK_N, H] @ [H, BLOCK_M] = [BLOCK_N, BLOCK_M]
        acc_bn_bm = tl.dot(b_block, a_hm).to(tl.float32)
        
        # Store Transposed: [BLOCK_M, BLOCK_N]
        tl.store(y_ptr + (rm[:, None] * N + rn[None, :]).to(tl.int64), tl.trans(acc_bn_bm), mask=m_mask[:, None] & n_mask[None, :])

def _get_or_alloc(buf, shape, dtype, device):
    if buf is None or buf.device != device or buf.dtype != dtype or tuple(buf.shape) != tuple(shape):
        return torch.empty(shape, device=device, dtype=dtype)
    return buf

def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data
    x = input_tensor
    if not x.is_contiguous():
        x = x.contiguous()

    torch.backends.cuda.matmul.allow_tf32 = True
    
    B, S, _, D = x.shape
    H = int(config["hidden_dim"])
    SS = S * S
    M = B * SS
    dev = x.device

    NOMASK = bool(config.get("nomask", False))
    HAS_MASK = (mask is not None) and (not NOMASK)

    # Safe handle for boolean masks if present
    if HAS_MASK and mask.dtype == torch.bool:
        mask = mask.to(torch.float32)

    cache = weights.get("_trimul_cache", None)
    if cache is None:
        cache = {}
        weights["_trimul_cache"] = cache
    
    ckey = (dev, D, H, "weights_v12")
    packed_w = cache.get(ckey, None)
    if packed_w is None:
        Wcat = torch.cat([
            weights["left_proj.weight"], weights["right_proj.weight"],
            weights["left_gate.weight"], weights["right_gate.weight"], weights["out_gate.weight"]
        ], dim=0).to(dev, dtype=torch.float16)
        Wcat_t = Wcat.t().contiguous()
        del Wcat
        
        nw = weights["norm.weight"].to(dev, dtype=torch.float32).contiguous()
        nb = weights["norm.bias"].to(dev, dtype=torch.float32).contiguous()
        
        # Pass wout as [D, H] contiguous
        wout = weights["to_out.weight"].to(dev, dtype=torch.float16).contiguous()
        
        onw = weights["to_out_norm.weight"].to(dev, dtype=torch.float32).contiguous()
        onb = weights["to_out_norm.bias"].to(dev, dtype=torch.float32).contiguous()
        
        packed_w = (Wcat_t, nw, nb, wout, onw, onb)
        cache[ckey] = packed_w
    
    Wcat_t, nw, nb, wout, onw, onb = packed_w

    bkey = (dev, B, S, D, H, "buffers_v12")
    bufs = cache.get(bkey, None)
    if bufs is None:
        bufs = {}
        cache[bkey] = bufs
    
    x2 = _get_or_alloc(bufs.get("x2", None), (M, D), torch.float16, dev)
    bufs["x2"] = x2
    
    if D <= 128:
        BLK_M_LN, BLK_D_LN = 64, 128
    elif D <= 384:
        BLK_M_LN, BLK_D_LN = 64, 256
    else:
        BLK_M_LN, BLK_D_LN = 32, 256
        
    _layernorm_fwd_to_f16_2d_contig_kernel[(triton.cdiv(M, BLK_M_LN),)](
        x.view(M, D), nw, nb, x2,
        M=M, D=D, eps=1e-5, BLOCK_M=BLK_M_LN, BLOCK_D=BLK_D_LN,
        num_warps=8
    )

    left_pack = _get_or_alloc(bufs.get("left_pack", None), (B, H, S, S), torch.float16, dev)
    right_pack = _get_or_alloc(bufs.get("right_pack", None), (B, H, S, S), torch.float16, dev)
    out_gate = _get_or_alloc(bufs.get("out_gate", None), (B, H, S, S), torch.float16, dev)
    bufs["left_pack"] = left_pack
    bufs["right_pack"] = right_pack
    bufs["out_gate"] = out_gate

    m_ptr = mask if HAS_MASK else x2 

    grid_lambda = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(H, META['BLOCK_N']))
    
    _fused_proj_pack_kernel[grid_lambda](
        x2, Wcat_t, m_ptr, 
        left_pack, right_pack, out_gate,
        M=M, D=D, H=H, SS=SS,
        HAS_MASK=HAS_MASK
    )

    left3 = left_pack.reshape(B * H, S, S)
    right3 = right_pack.reshape(B * H, S, S)
    out3 = _get_or_alloc(bufs.get("out3", None), (B * H, S, S), torch.float16, dev)
    bufs["out3"] = out3
    
    torch.bmm(left3, right3.transpose(1, 2), out=out3)

    out4 = out3.view(B, H, S, S)
    y2 = _get_or_alloc(bufs.get("y2", None), (M, D), torch.float32, dev)
    bufs["y2"] = y2

    grid3 = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    _ln_outgate_to_out_fused_multiN_tc_contig_kernel[grid3](
        out4, out_gate, onw, onb, wout, y2,
        M=M, S=S, SS=SS, eps=1e-5, H=H, N=D
    )

    return y2.view(B, S, S, D)