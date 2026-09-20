"""B1 decode down GEMV and BF16 residual addition in one launch."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"ROWS": rows}, num_warps=warps, num_stages=2)
        for rows in (1, 2, 4) for warps in (4, 8)
    ],
    key=["B"], warmup=10, rep=50, use_cuda_graph=False,
)
@triton.jit
def _down_add(X, W, RESIDUAL, OUTPUT, B: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    lane = tl.arange(0, 2048)
    accumulator = tl.full((ROWS,), 0.0, tl.float32)
    # Five K blocks cover all 9728 BF16 input values. This bounds each CTA's
    # live weight tile while retaining an FP32 accumulator until the final cast.
    for block in range(5):
        k = block * 2048 + lane
        x = tl.load(X + k, k < 9728, 0).to(tl.float32)
        weight = tl.load(
            W + row[:, None] * 9728 + k[None, :],
            (row[:, None] < 2560) & (k[None, :] < 9728), 0,
        ).to(tl.float32)
        accumulator += tl.sum(weight * x[None, :], 1)
    projection = accumulator.to(tl.bfloat16)
    residual = tl.load(RESIDUAL + row, row < 2560, 0).to(tl.float32)
    summed = (projection.to(tl.float32) + residual).to(tl.bfloat16)
    tl.store(OUTPUT + row, summed, row < 2560)


def down_b1_gemv_add(x, weight, residual):
    if (x.shape != (1, 1, 9728) or x.dtype != torch.bfloat16
            or residual.shape != (1, 1, 2560) or residual.dtype != x.dtype
            or weight.shape != (2560, 9728) or weight.dtype != x.dtype
            or not x.is_contiguous() or not weight.is_contiguous()
            or not residual.is_contiguous()):
        raise ValueError("invalid B1 down GEMV input, weight, or residual")
    output = torch.empty_like(residual)
    _down_add[(lambda meta: (triton.cdiv(2560, meta["ROWS"]),))](
        x, weight, residual, output, 1, enable_fp_fusion=False,
    )
    return output
