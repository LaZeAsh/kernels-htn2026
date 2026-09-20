"""Q/K RMSNorm + RoPE + KV write and BF16 SwiGLU."""

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
def _qk_norm_rope_write(Q, K, V, QGAIN, KGAIN, COS, SIN, POS, KC, VC, QOUT,
                        CAPACITY: tl.constexpr, QEPS: tl.constexpr, KEPS: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 128)
    cos = tl.load(COS + d)
    sin = tl.load(SIN + d)
    q = tl.load(Q + b * 4096 + h * 128 + d)
    q_gain = tl.load(QGAIN + d)
    q_rot = _norm_rope(q, q_gain, cos, sin, QEPS)
    tl.store(QOUT + (b * 32 + h) * 128 + d, q_rot)
    if h < 8:
        position = tl.load(POS)
        k = tl.load(K + b * 1024 + h * 128 + d)
        v = tl.load(V + b * 1024 + h * 128 + d)
        k_gain = tl.load(KGAIN + d)
        k_rot = _norm_rope(k, k_gain, cos, sin, KEPS)
        offset = ((b * CAPACITY + position) * 8 + h) * 128 + d
        tl.store(KC + offset, k_rot)
        tl.store(VC + offset, v)


@triton.jit
def _prefill_qk_norm_rope_write(Q, K, V, QGAIN, KGAIN, COS, SIN,
                                KC, VC, QOUT, LENGTH: tl.constexpr,
                                CAPACITY: tl.constexpr, ANGLE_BATCH: tl.constexpr,
                                QEPS: tl.constexpr,
                                KEPS: tl.constexpr):
    token = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    d = tl.arange(0, 128)
    # rotary_emb usually returns [1,T,128] for shared position_ids.
    angle_offset = (((b if ANGLE_BATCH > 1 else 0) * LENGTH) + token) * 128 + d
    cos = tl.load(COS + angle_offset)
    sin = tl.load(SIN + angle_offset)
    q = tl.load(Q + (b * LENGTH + token) * 4096 + h * 128 + d)
    q_gain = tl.load(QGAIN + d)
    q_rot = _norm_rope(q, q_gain, cos, sin, QEPS)
    tl.store(QOUT + ((b * 32 + h) * LENGTH + token) * 128 + d, q_rot)
    if h < 8:
        k = tl.load(K + (b * LENGTH + token) * 1024 + h * 128 + d)
        v = tl.load(V + (b * LENGTH + token) * 1024 + h * 128 + d)
        k_gain = tl.load(KGAIN + d)
        k_rot = _norm_rope(k, k_gain, cos, sin, KEPS)
        cache_offset = ((b * CAPACITY + token) * 8 + h) * 128 + d
        tl.store(KC + cache_offset, k_rot)
        tl.store(VC + cache_offset, v)


@triton.jit
def _swiglu(GATE, UP, OUT, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    d = block * BLOCK + tl.arange(0, BLOCK)
    valid = d < WIDTH
    gate = tl.load(GATE + row * WIDTH + d, mask=valid, other=0).to(tl.float32)
    up = tl.load(UP + row * WIDTH + d, mask=valid, other=0)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    product = (activated.to(tl.float32) * up.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + row * WIDTH + d, product, mask=valid)


def qk_norm_rope_cache(q, k, v, q_gain, k_gain, cos, sin, position,
                       k_cache, v_cache, q_eps, k_eps):
    """Consume separate native BF16 projections, return Q [B,32,1,128].

    Writes K/V at the device scalar position in time-major fixed cache.
    Every native BF16 normalization and RoPE rounding boundary is retained.
    """
    batch = q.shape[0]
    capacity = k_cache.shape[2]
    for tensor, width in ((q, 4096), (k, 1024), (v, 1024)):
        if tensor.shape != (batch, 1, width) or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("Q/K/V must be separate contiguous BF16 native projections")
    if (k_cache.shape != (batch, 8, capacity, 128)
            or v_cache.shape != k_cache.shape
            or k_cache.stride() != (capacity * 8 * 128, 128, 8 * 128, 1)
            or v_cache.stride() != k_cache.stride()):
        raise ValueError("K/V cache must be time-major [B,8,C,128] views")
    if q_gain.shape != (128,) or k_gain.shape != (128,):
        raise ValueError("Q/K norm gains must have 128 elements")
    if cos.shape[-1] != 128 or sin.shape[-1] != 128 or not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("RoPE cosine/sine must be contiguous with 128 columns")
    if position.shape != (1,) or position.dtype != torch.int64 or not position.is_cuda:
        raise ValueError("position must be one CUDA int64 value")
    out = torch.empty((batch, 32, 1, 128), device=q.device, dtype=torch.bfloat16)
    _qk_norm_rope_write[(batch, 32)](
        q, k, v, q_gain, k_gain, cos, sin, position, k_cache, v_cache, out,
        capacity, q_eps, k_eps, num_warps=4, enable_fp_fusion=False,
    )
    return out


def prefill_qk_norm_rope_cache(q, k, v, q_gain, k_gain, cos, sin,
                               k_cache, v_cache, q_eps, k_eps):
    """Normalize/rotate native BF16 projections, write cache, return Q.

    Q/K/V are contiguous [B,T,4096/1024]. Cos/sin are contiguous
    [1 or B,T,128]. Cache is time-major [B,8,C,128], with C >= T. Only the
    initialized prefix [:T] may be exposed to causal SDPA.
    """
    batch, length, width = q.shape
    capacity = k_cache.shape[2]
    if (width != 4096 or length < 1 or capacity < length
            or q.dtype != torch.bfloat16 or not q.is_contiguous()
            or k.shape != (batch, length, 1024)
            or v.shape != k.shape or k.dtype != q.dtype or v.dtype != q.dtype
            or not k.is_contiguous() or not v.is_contiguous()
            or cos.shape not in ((1, length, 128), (batch, length, 128))
            or sin.shape != cos.shape
            or not cos.is_contiguous() or not sin.is_contiguous()
            or k_cache.shape != (batch, 8, capacity, 128)
            or v_cache.shape != k_cache.shape
            or k_cache.stride() != (capacity * 8 * 128, 128, 8 * 128, 1)
            or v_cache.stride() != k_cache.stride()
            or q_gain.shape != (128,) or k_gain.shape != (128,)):
        raise ValueError("invalid prefill Q/K/V, RoPE, gain or cache layout")
    out = torch.empty((batch, 32, length, 128), device=q.device, dtype=q.dtype)
    _prefill_qk_norm_rope_write[(length, batch, 32)](
        q, k, v, q_gain, k_gain, cos, sin, k_cache, v_cache, out,
        length, capacity, cos.shape[0], q_eps, k_eps,
        num_warps=4, enable_fp_fusion=False,
    )
    return out


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Separate BF16 [B,T,I] projections to BF16 [B,T,I]."""
    if gate.ndim != 3 or gate.shape != up.shape:
        raise ValueError("gate/up must each have shape [B,T,I]")
    if gate.dtype != torch.bfloat16 or up.dtype != gate.dtype or not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("gate/up must be contiguous BF16")
    width = gate.shape[-1]
    out = torch.empty_like(gate)
    rows = gate.shape[0] * gate.shape[1]
    block = 1024 if rows > 32 else 256
    _swiglu[(rows, triton.cdiv(width, block))](
        gate, up, out, width, block, num_warps=4, enable_fp_fusion=False,
    )
    return out
