"""B2–4 decode gate/up GEMV with shared original BF16 weight loads."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"ROWS": rows}, num_warps=warps, num_stages=2)
        for rows in (1, 2) for warps in (4, 8)
    ],
    key=["B"], warmup=10, rep=50, use_cuda_graph=False,
)
@triton.jit
def _gemv_swiglu_batched(X, GW, UW, PRODUCT,
                         B: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, 4096)
    batch_lanes = tl.arange(0, 4)
    offsets = row[:, None] * 2560 + k[None, :]
    mask = (row[:, None] < 9728) & (k[None, :] < 2560)
    gate_weight = tl.load(GW + offsets, mask, 0).to(tl.float32)
    gates = tl.full((4, ROWS), 0, tl.bfloat16)
    for b in tl.static_range(0, B):
        x = tl.load(X + b * 2560 + k, k < 2560, 0).to(tl.float32)
        gate = tl.sum(gate_weight * x[None, :], 1).to(tl.bfloat16)
        gates = tl.where(batch_lanes[:, None] == b, gate[None, :], gates)
    up_weight = tl.load(UW + offsets, mask, 0).to(tl.float32)
    ups = tl.full((4, ROWS), 0, tl.bfloat16)
    for b in tl.static_range(0, B):
        x = tl.load(X + b * 2560 + k, k < 2560, 0).to(tl.float32)
        up = tl.sum(up_weight * x[None, :], 1).to(tl.bfloat16)
        ups = tl.where(batch_lanes[:, None] == b, up[None, :], ups)
    gf = gates.to(tl.float32)
    activated = (gf / (1.0 + tl.exp(-gf))).to(tl.bfloat16)
    product = (activated.to(tl.float32) * ups.to(tl.float32)).to(tl.bfloat16)
    tl.store(PRODUCT + batch_lanes[:, None] * 9728 + row[None, :],
             product, (batch_lanes[:, None] < B) & (row[None, :] < 9728))


def b2_4_gemv_swiglu(x, gate_weight, up_weight):
    batch = x.shape[0]
    if (batch < 2 or batch > 4 or x.shape != (batch, 1, 2560)
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or gate_weight.shape != (9728, 2560)
            or up_weight.shape != gate_weight.shape
            or gate_weight.dtype != x.dtype or up_weight.dtype != x.dtype
            or not gate_weight.is_contiguous() or not up_weight.is_contiguous()):
        raise ValueError("invalid B2–4 decode MLP x or original weights")
    product = torch.empty((batch, 1, 9728), dtype=x.dtype, device=x.device)
    _gemv_swiglu_batched[(lambda meta: (triton.cdiv(9728, meta["ROWS"]),))](
        x, gate_weight, up_weight, product, batch, enable_fp_fusion=False,
    )
    return product
