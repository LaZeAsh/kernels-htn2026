"""Decode split-K MLP using interleaved original BF16 gate/up rows."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BN": bn, "BK": bk}, num_warps=4, num_stages=2)
        for bn in (64, 128) for bk in (64, 128)
    ],
    key=["B"], warmup=10, rep=50, use_cuda_graph=False,
)
@triton.jit
def _project(X, W, PART, B: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    rows = tl.arange(0, 16)
    cols = tile * (2 * BN) + tl.arange(0, 2 * BN)
    kk = tl.arange(0, BK)
    acc = tl.full((16, 2 * BN), 0, tl.float32)
    for block in range(640 // BK):
        k = split * 640 + block * BK + kk
        x = tl.load(X + rows[:, None] * 2560 + k[None, :],
                    rows[:, None] < B, 0)
        offset = cols[:, None] * 2560 + k[None, :]
        w = tl.load(W + offset)
        acc = tl.dot(x, tl.trans(w), acc)
    partial_offset = (rows[:, None] * 4 + split) * 19456 + cols[None, :]
    tl.store(PART + partial_offset, acc, rows[:, None] < B)


@triton.jit
def _reduce_swiglu(PART, PRODUCT):
    b = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * 256 + tl.arange(0, 256)
    base = b * 4 * 19456 + 2 * col
    gate = (tl.load(PART + base) + tl.load(PART + base + 19456)
            + tl.load(PART + base + 2 * 19456)
            + tl.load(PART + base + 3 * 19456)).to(tl.bfloat16)
    up = (tl.load(PART + base + 1) + tl.load(PART + base + 19457)
          + tl.load(PART + base + 2 * 19456 + 1)
          + tl.load(PART + base + 3 * 19456 + 1)).to(tl.bfloat16)
    gf = gate.to(tl.float32)
    activated = (gf / (1.0 + tl.exp(-gf))).to(tl.bfloat16)
    product = (activated.to(tl.float32) * up.to(tl.float32)).to(tl.bfloat16)
    tl.store(PRODUCT + b * 9728 + col, product)


def split_swiglu(x, interleaved_weight):
    """BF16 [B,1,2560] x [2*9728,2560] interleaved gate/up rows."""
    batch = x.shape[0]
    if (batch < 1 or batch > 16 or x.shape != (batch, 1, 2560)
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or interleaved_weight.shape != (19456, 2560)
            or interleaved_weight.dtype != x.dtype
            or not interleaved_weight.is_contiguous()):
        raise ValueError("invalid decode MLP x or original weights")
    partial = torch.empty((batch, 4, 19456), dtype=torch.float32, device=x.device)
    product = torch.empty((batch, 1, 9728), dtype=x.dtype, device=x.device)
    _project[(lambda meta: (triton.cdiv(9728, meta["BN"]), 4))](
        x, interleaved_weight, partial, batch,
    )
    _reduce_swiglu[(batch, 38)](
        partial, product,
        num_warps=4, enable_fp_fusion=False,
    )
    return product
