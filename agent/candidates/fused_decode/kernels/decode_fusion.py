"""Decode-only Q/K RMSNorm + RoPE + KV write and BF16 SwiGLU."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope(x, gain, cos, sin, eps: tl.constexpr):
    d = tl.arange(0, 128)
    f = x.to(tl.float32)
    variance = tl.sum(f * f, 0) / 128.0
    normalized = (f * tl.rsqrt(variance + eps)).to(tl.bfloat16)
    weighted = (normalized.to(tl.float32) * gain.to(tl.float32)).to(tl.bfloat16)
    halves = tl.trans(tl.reshape(weighted, (2, 64)))
    first, second = tl.split(halves)
    rotated = tl.reshape(tl.trans(tl.join(-second, first)), (128,))
    left = (weighted.to(tl.float32) * cos.to(tl.float32)).to(tl.bfloat16)
    right = (rotated.to(tl.float32) * sin.to(tl.float32)).to(tl.bfloat16)
    return (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _qk_norm_rope_write(QKV, QGAIN, KGAIN, COS, SIN, POS, KC, VC, QOUT,
                        CAPACITY: tl.constexpr, QEPS: tl.constexpr, KEPS: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 128)
    cos = tl.load(COS + d)
    sin = tl.load(SIN + d)
    q = tl.load(QKV + b * 6144 + h * 128 + d)
    q_gain = tl.load(QGAIN + d)
    q_rot = _norm_rope(q, q_gain, cos, sin, QEPS)
    tl.store(QOUT + (b * 32 + h) * 128 + d, q_rot)
    if h < 8:
        position = tl.load(POS)
        k = tl.load(QKV + b * 6144 + 4096 + h * 128 + d)
        v = tl.load(QKV + b * 6144 + 5120 + h * 128 + d)
        k_gain = tl.load(KGAIN + d)
        k_rot = _norm_rope(k, k_gain, cos, sin, KEPS)
        offset = ((b * 8 + h) * CAPACITY + position) * 128 + d
        tl.store(KC + offset, k_rot)
        tl.store(VC + offset, v)


@triton.jit
def _swiglu(GATEUP, OUT, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    d = block * BLOCK + tl.arange(0, BLOCK)
    valid = d < WIDTH
    gate = tl.load(GATEUP + row * 2 * WIDTH + d, mask=valid, other=0).to(tl.float32)
    up = tl.load(GATEUP + row * 2 * WIDTH + WIDTH + d, mask=valid, other=0)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    product = (activated.to(tl.float32) * up.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + row * WIDTH + d, product, mask=valid)


def qk_norm_rope_cache(qkv, q_gain, k_gain, cos, sin, position,
                       k_cache, v_cache, q_eps, k_eps):
    """Consume contiguous BF16 QKV [B,1,6144], return Q [B,32,1,128].

    Writes K/V at the device scalar position in contiguous fixed cache.
    Every native BF16 normalization and RoPE rounding boundary is retained.
    """
    batch = qkv.shape[0]
    capacity = k_cache.shape[2]
    if qkv.shape != (batch, 1, 6144) or qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
        raise ValueError("QKV must be contiguous BF16 [B,1,6144]")
    if (k_cache.shape != (batch, 8, capacity, 128)
            or v_cache.shape != k_cache.shape
            or not k_cache.is_contiguous() or not v_cache.is_contiguous()):
        raise ValueError("K/V cache must be contiguous [B,8,C,128]")
    if q_gain.shape != (128,) or k_gain.shape != (128,):
        raise ValueError("Q/K norm gains must have 128 elements")
    if cos.shape[-1] != 128 or sin.shape[-1] != 128 or not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("RoPE cosine/sine must be contiguous with 128 columns")
    if position.shape != (1,) or position.dtype != torch.int64 or not position.is_cuda:
        raise ValueError("position must be one CUDA int64 value")
    out = torch.empty((batch, 32, 1, 128), device=qkv.device, dtype=torch.bfloat16)
    _qk_norm_rope_write[(batch, 32)](
        qkv, q_gain, k_gain, cos, sin, position, k_cache, v_cache, out,
        capacity, q_eps, k_eps, num_warps=4, enable_fp_fusion=False,
    )
    return out


def swiglu(gate_up: torch.Tensor, width: int) -> torch.Tensor:
    """Decode BF16 [B,1,2*width] to BF16 [B,1,width]."""
    if gate_up.ndim != 3 or gate_up.shape[-2:] != (1, 2 * width):
        raise ValueError("gate/up must have shape [B,1,2*width]")
    if gate_up.dtype != torch.bfloat16 or not gate_up.is_contiguous():
        raise ValueError("gate/up must be contiguous BF16")
    out = torch.empty((*gate_up.shape[:-1], width), device=gate_up.device, dtype=gate_up.dtype)
    _swiglu[(gate_up.shape[0], triton.cdiv(width, 256))](
        gate_up, out, width, 256, num_warps=4, enable_fp_fusion=False,
    )
    return out
