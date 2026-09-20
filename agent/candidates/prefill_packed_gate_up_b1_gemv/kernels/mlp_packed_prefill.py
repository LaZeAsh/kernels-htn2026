"""SwiGLU on contiguous [gate, up] prefill projection rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_swiglu(PACKED, PRODUCT,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    valid = col < 9728
    gate = tl.load(PACKED + row * 19456 + col, valid, 0).to(tl.float32)
    up = tl.load(PACKED + row * 19456 + 9728 + col, valid, 0).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    result = (activated.to(tl.float32) * up).to(tl.bfloat16)
    tl.store(PRODUCT + row * 9728 + col, result, valid)


def packed_prefill_swiglu(packed):
    if (packed.ndim != 3 or packed.shape[-1] != 19456
            or packed.dtype != torch.bfloat16 or not packed.is_contiguous()):
        raise ValueError("packed prefill gate/up must be contiguous BF16 [B,T,19456]")
    batch, length, _ = packed.shape
    product = torch.empty((batch, length, 9728),
                          dtype=packed.dtype, device=packed.device)
    rows = batch * length
    block = 1024 if rows > 32 else 256
    _packed_swiglu[(rows, triton.cdiv(9728, block))](
        packed, product, block, num_warps=4, enable_fp_fusion=False,
    )
    return product
