"""Decode residual addition and next RMSNorm with Qwen3 BF16 boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_norm(RESIDUAL, BRANCH, GAIN, OUT_RESIDUAL, OUT_NORM,
              EPS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    offset = row * WIDTH + col
    residual = tl.load(RESIDUAL + offset, mask, 0).to(tl.float32)
    branch = tl.load(BRANCH + offset, mask, 0).to(tl.float32)
    # PyTorch's BF16 residual addition rounds before the RMSNorm reduction.
    summed = (residual + branch).to(tl.bfloat16)
    tl.store(OUT_RESIDUAL + offset, summed, mask)
    value = summed.to(tl.float32)
    variance = tl.sum(value * value, 0) / WIDTH
    normalized = (value * tl.rsqrt(variance + EPS)).to(tl.bfloat16)
    gain = tl.load(GAIN + col, mask, 0).to(tl.float32)
    tl.store(OUT_NORM + offset,
             (normalized.to(tl.float32) * gain).to(tl.bfloat16), mask)


def add_norm(residual, branch, gain, eps):
    """Return distinct BF16 (residual+branch, RMSNorm) tensors.

    Inputs must be contiguous BF16 [B,1,2560] CUDA tensors; gain is BF16
    [2560]. Outputs own their storage and do not alias either input. Graph
    capture retains the allocations, providing stable replay addresses.
    """
    if (residual.shape != branch.shape or residual.ndim != 3
            or residual.shape[1:] != (1, 2560)
            or residual.dtype != torch.bfloat16 or branch.dtype != torch.bfloat16
            or not residual.is_cuda or not branch.is_cuda
            or not residual.is_contiguous() or not branch.is_contiguous()
            or gain.shape != (2560,) or gain.dtype != torch.bfloat16
            or not gain.is_cuda or not gain.is_contiguous()):
        raise ValueError("add_norm expects contiguous CUDA BF16 [B,1,2560]")
    out_residual = torch.empty_like(residual)
    out_norm = torch.empty_like(residual)
    _add_norm[(residual.shape[0],)](
        residual, branch, gain, out_residual, out_norm,
        eps, 2560, 4096, num_warps=8, enable_fp_fusion=False,
    )
    return out_residual, out_norm
