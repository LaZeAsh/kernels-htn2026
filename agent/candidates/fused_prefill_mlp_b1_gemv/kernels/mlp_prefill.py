"""Full-prompt gate/up projection and SwiGLU from original BF16 weights."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BK": bk}, num_warps=warps, num_stages=2)
        for bk in (64, 128) for warps in (4, 8)
    ],
    key=["M"], warmup=10, rep=50, use_cuda_graph=False,
)
@triton.jit
def _prefill_swiglu(X, GW, UW, PRODUCT, M: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * 64 + tl.arange(0, 64)
    col = tl.program_id(1) * 64 + tl.arange(0, 64)
    kk = tl.arange(0, BK)
    gate = tl.full((64, 64), 0.0, tl.float32)
    up = tl.full((64, 64), 0.0, tl.float32)
    for block in range(2560 // BK):
        k = block * BK + kk
        x = tl.load(X + row[:, None] * 2560 + k[None, :],
                    row[:, None] < M, 0)
        offsets = col[:, None] * 2560 + k[None, :]
        gw = tl.load(GW + offsets)
        uw = tl.load(UW + offsets)
        gate = tl.dot(x, tl.trans(gw), gate)
        up = tl.dot(x, tl.trans(uw), up)
    gate = gate.to(tl.bfloat16)
    up = up.to(tl.bfloat16)
    gf = gate.to(tl.float32)
    activated = (gf / (1.0 + tl.exp(-gf))).to(tl.bfloat16)
    product = (activated.to(tl.float32) * up.to(tl.float32)).to(tl.bfloat16)
    tl.store(PRODUCT + row[:, None] * 9728 + col[None, :], product,
             row[:, None] < M)


def fused_prefill_swiglu(x, gate_weight, up_weight):
    if (x.ndim != 3 or x.shape[-1] != 2560 or x.shape[0] * x.shape[1] < 64
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or gate_weight.shape != (9728, 2560)
            or up_weight.shape != gate_weight.shape
            or gate_weight.dtype != x.dtype or up_weight.dtype != x.dtype
            or not gate_weight.is_contiguous() or not up_weight.is_contiguous()):
        raise ValueError("invalid full-prompt MLP input or original weights")
    batch, length, _ = x.shape
    rows = batch * length
    product = torch.empty((batch, length, 9728), dtype=x.dtype, device=x.device)
    _prefill_swiglu[(triton.cdiv(rows, 64), 152)](
        x, gate_weight, up_weight, product, rows,
        enable_fp_fusion=False,
    )
    return product
