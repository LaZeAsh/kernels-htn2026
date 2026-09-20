"""Experimental decode MLP split-K gate/up projections and SwiGLU."""

import torch
import triton
import triton.language as tl

from kernels.compiler_diag import log_compiler_once


@triton.autotune(
    configs=[
        triton.Config({"BN": bn, "BK": bk}, num_warps=4, num_stages=2)
        for bn in (64, 128) for bk in (64, 128)
    ],
    key=["B"], warmup=10, rep=50, use_cuda_graph=False,
)
@triton.jit
def _project(X, GW, UW, GP, UP, B: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    # BM64 may enable Hopper MMA v3 lowering; rows beyond B are masked.
    rows = tl.arange(0, 64)
    cols = tile * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate = tl.full((64, BN), 0, tl.float32)
    up = tl.full((64, BN), 0, tl.float32)
    for block in range(640 // BK):
        k = split * 640 + block * BK + kk
        x = tl.load(X + rows[:, None] * 2560 + k[None, :],
                    rows[:, None] < B, 0)
        go = cols[:, None] * 2560 + k[None, :]
        gw = tl.load(GW + go)
        uw = tl.load(UW + go)
        gate = tl.dot(x, tl.trans(gw), gate)
        up = tl.dot(x, tl.trans(uw), up)
    partial_offset = (rows[:, None] * 4 + split) * 9728 + cols[None, :]
    tl.store(GP + partial_offset, gate, rows[:, None] < B)
    tl.store(UP + partial_offset, up, rows[:, None] < B)


@triton.jit
def _reduce_swiglu(GP, UP, PRODUCT):
    b = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * 256 + tl.arange(0, 256)
    base = b * 4 * 9728 + col
    gate = (tl.load(GP + base) + tl.load(GP + base + 9728)
            + tl.load(GP + base + 2 * 9728)
            + tl.load(GP + base + 3 * 9728)).to(tl.bfloat16)
    up = (tl.load(UP + base) + tl.load(UP + base + 9728)
          + tl.load(UP + base + 2 * 9728)
          + tl.load(UP + base + 3 * 9728)).to(tl.bfloat16)
    gf = gate.to(tl.float32)
    activated = (gf / (1.0 + tl.exp(-gf))).to(tl.bfloat16)
    product = (activated.to(tl.float32) * up.to(tl.float32)).to(tl.bfloat16)
    tl.store(PRODUCT + b * 9728 + col, product)


def large_split_swiglu(x, gate_weight, up_weight):
    """BF16 [17..64,1,2560] x original weights to [B,1,9728]."""
    batch = x.shape[0]
    if (batch < 17 or batch > 64 or x.shape != (batch, 1, 2560)
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or gate_weight.shape != (9728, 2560)
            or up_weight.shape != gate_weight.shape
            or gate_weight.dtype != x.dtype or up_weight.dtype != x.dtype
            or not gate_weight.is_contiguous() or not up_weight.is_contiguous()):
        raise ValueError("invalid decode MLP x or original weights")
    gate_partial = torch.empty((batch, 4, 9728), dtype=torch.float32, device=x.device)
    up_partial = torch.empty_like(gate_partial)
    product = torch.empty((batch, 1, 9728), dtype=x.dtype, device=x.device)
    compiled = _project[(lambda meta: (triton.cdiv(9728, meta["BN"]), 4))](
        x, gate_weight, up_weight, gate_partial, up_partial, batch,
    )
    log_compiler_once("large_mlp_bm64_project", compiled)
    _reduce_swiglu[(batch, 38)](
        gate_partial, up_partial, product,
        num_warps=4, enable_fp_fusion=False,
    )
    return product
