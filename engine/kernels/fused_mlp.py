"""Decode-only gate/up GEMMs and SwiGLU with one input read per K tile.

The source weights remain separate BF16 [I,H] parameters. A program computes
[B tile,I tile] gate and up FP32 accumulators, then rounds at native BF16
projection, SiLU and product boundaries. Output is consumed by native down_proj.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up_swiglu(X, WG, WU, OUT,
                    B: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate_acc = tl.full((BM, BN), 0.0, tl.float32)
    up_acc = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(tl.cdiv(H, BK)):
        k = block * BK + kk
        x = tl.load(X + rows[:, None] * H + k[None, :],
                    mask=(rows[:, None] < B) & (k[None, :] < H), other=0)
        wg = tl.load(WG + cols[:, None] * H + k[None, :],
                     mask=(cols[:, None] < I) & (k[None, :] < H), other=0)
        wu = tl.load(WU + cols[:, None] * H + k[None, :],
                     mask=(cols[:, None] < I) & (k[None, :] < H), other=0)
        gate_acc = tl.dot(x, tl.trans(wg), gate_acc)
        up_acc = tl.dot(x, tl.trans(wu), up_acc)
    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    up = up_acc.to(tl.bfloat16).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    product = (activated * up).to(tl.bfloat16)
    tl.store(OUT + rows[:, None] * I + cols[None, :], product,
             mask=(rows[:, None] < B) & (cols[None, :] < I))


def fused_gate_up_swiglu(x: torch.Tensor, gate_weight: torch.Tensor,
                         up_weight: torch.Tensor) -> torch.Tensor:
    """Decode BF16 [B,1,H] and separate BF16 [I,H] weights to [B,1,I]."""
    if x.ndim != 3 or x.shape[1] != 1 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("input must be contiguous BF16 [B,1,H]")
    batch, _, hidden = x.shape
    if not 1 <= batch <= 16:
        raise ValueError("fused kernel is limited to batch 1..16")
    intermediate = gate_weight.shape[0]
    if (gate_weight.shape != (intermediate, hidden)
            or up_weight.shape != gate_weight.shape
            or gate_weight.dtype != torch.bfloat16 or up_weight.dtype != torch.bfloat16
            or not gate_weight.is_contiguous() or not up_weight.is_contiguous()):
        raise ValueError("gate/up weights must be separate contiguous BF16 [I,H]")
    if not (x.is_cuda and gate_weight.is_cuda and up_weight.is_cuda):
        raise ValueError("input and weights must be CUDA tensors")
    out = torch.empty((batch, 1, intermediate), device=x.device, dtype=x.dtype)
    _gate_up_swiglu[(triton.cdiv(batch, 16), triton.cdiv(intermediate, 64))](
        x, gate_weight, up_weight, out,
        batch, hidden, intermediate, 16, 64, 64,
        num_warps=4, num_stages=2,
    )
    return out
