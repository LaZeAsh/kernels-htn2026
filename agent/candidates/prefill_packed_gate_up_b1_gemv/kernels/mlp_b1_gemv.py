"""B1 decode: original BF16 gate/up GEMV and SwiGLU in one launch."""

import torch
import triton
import triton.language as tl

from kernels.compiler_diag import log_compiler_once


@triton.autotune(
    configs=[
        triton.Config({"ROWS": rows}, num_warps=warps, num_stages=2)
        for rows in (1, 2, 4) for warps in (4, 8)
    ],
    key=["B"], warmup=10, rep=50, use_cuda_graph=False,
)
@triton.jit
def _gemv_swiglu(X, GW, UW, PRODUCT, B: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, 4096)
    x = tl.load(X + k, k < 2560, 0).to(tl.float32)
    offsets = row[:, None] * 2560 + k[None, :]
    mask = (row[:, None] < 9728) & (k[None, :] < 2560)
    gate_weight = tl.load(GW + offsets, mask, 0).to(tl.float32)
    gate = tl.sum(gate_weight * x[None, :], 1).to(tl.bfloat16)
    up_weight = tl.load(UW + offsets, mask, 0).to(tl.float32)
    up = tl.sum(up_weight * x[None, :], 1).to(tl.bfloat16)
    gf = gate.to(tl.float32)
    activated = (gf / (1.0 + tl.exp(-gf))).to(tl.bfloat16)
    product = (activated.to(tl.float32) * up.to(tl.float32)).to(tl.bfloat16)
    tl.store(PRODUCT + row, product, row < 9728)


def b1_gemv_swiglu(x, gate_weight, up_weight):
    if (x.shape != (1, 1, 2560) or x.dtype != torch.bfloat16
            or not x.is_contiguous() or gate_weight.shape != (9728, 2560)
            or up_weight.shape != gate_weight.shape
            or gate_weight.dtype != x.dtype or up_weight.dtype != x.dtype
            or not gate_weight.is_contiguous() or not up_weight.is_contiguous()):
        raise ValueError("invalid B1 decode MLP x or original weights")
    product = torch.empty((1, 1, 9728), dtype=x.dtype, device=x.device)
    compiled = _gemv_swiglu[(lambda meta: (triton.cdiv(9728, meta["ROWS"]),))](
        x, gate_weight, up_weight, product, 1, enable_fp_fusion=False,
    )
    log_compiler_once("b1_mlp_gemv", compiled, _gemv_swiglu.best_config)
    return product
