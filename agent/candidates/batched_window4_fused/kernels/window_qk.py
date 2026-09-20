"""Four-token Q/K norm, RoPE and speculative KV writes at device position."""

import torch
import triton
import triton.language as tl

from kernels.decode_fusion import _norm_rope


@triton.jit
def _window_write(Q, K, V, QGAIN, KGAIN, COS, SIN, POS, KC, VC, QOUT,
                  C: tl.constexpr, PS: tl.constexpr,
                  CS0: tl.constexpr, CS1: tl.constexpr, CS2: tl.constexpr,
                  SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr,
                  QEPS: tl.constexpr, KEPS: tl.constexpr):
    b = tl.program_id(0) // 4
    token = tl.program_id(0) % 4
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    cos = tl.load(COS + b * CS0 + token * CS1 + d * CS2)
    sin = tl.load(SIN + b * SS0 + token * SS1 + d * SS2)
    q = tl.load(Q + b * 4 * 4096 + token * 4096 + head * 128 + d)
    qrot = _norm_rope(q, tl.load(QGAIN + d), cos, sin, QEPS)
    tl.store(QOUT + ((b * 32 + head) * 4 + token) * 128 + d, qrot)
    if head < 8:
        offset = b * 4 * 1024 + token * 1024 + head * 128 + d
        k = tl.load(K + offset)
        v = tl.load(V + offset)
        krot = _norm_rope(k, tl.load(KGAIN + d), cos, sin, KEPS)
        position = tl.load(POS + b * PS) + token
        cache_offset = ((b * 8 + head) * C + position) * 128 + d
        tl.store(KC + cache_offset, krot)
        tl.store(VC + cache_offset, v)


def window_qk_cache(q, k, v, q_gain, k_gain, cos, sin, position,
                    k_cache, v_cache, q_eps, k_eps):
    """B<=4/T4 contiguous BF16 Q/K/V; write each sequence at t..t+3."""
    batch = q.shape[0]
    if (batch < 1 or batch > 4 or q.shape != (batch, 4, 4096)
            or k.shape != (batch, 4, 1024)
            or v.shape != k.shape or cos.shape != (batch, 4, 128)
            or sin.shape != cos.shape or q.dtype != torch.bfloat16
            or k.dtype != q.dtype or v.dtype != q.dtype
            or not q.is_contiguous() or not k.is_contiguous()
            or not v.is_contiguous() or position.shape != (batch,)
            or position.dtype != torch.int64
            or k_cache.shape != v_cache.shape or k_cache.shape[:2] != (batch, 8)
            or k_cache.shape[-1] != 128
            or not k_cache.is_contiguous() or not v_cache.is_contiguous()):
        raise ValueError("invalid four-token Q/K/cache layout")
    out = torch.empty((batch, 32, 4, 128), device=q.device, dtype=q.dtype)
    _window_write[(batch * 4, 32)](
        q, k, v, q_gain, k_gain, cos, sin, position,
        k_cache, v_cache, out, k_cache.shape[2], position.stride(0),
        *cos.stride(), *sin.stride(), q_eps, k_eps,
        num_warps=4, enable_fp_fusion=False,
    )
    return out
