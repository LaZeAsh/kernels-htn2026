"""Four-token Q/K norm, RoPE and speculative KV writes at device position."""

import torch
import triton
import triton.language as tl

from kernels.decode_fusion import _norm_rope


@triton.jit
def _window_write(Q, K, V, QGAIN, KGAIN, COS, SIN, POS, KC, VC, QOUT,
                  C: tl.constexpr, QEPS: tl.constexpr, KEPS: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    cos = tl.load(COS + token * 128 + d)
    sin = tl.load(SIN + token * 128 + d)
    q = tl.load(Q + token * 4096 + head * 128 + d)
    qrot = _norm_rope(q, tl.load(QGAIN + d), cos, sin, QEPS)
    tl.store(QOUT + (head * 4 + token) * 128 + d, qrot)
    if head < 8:
        k = tl.load(K + token * 1024 + head * 128 + d)
        v = tl.load(V + token * 1024 + head * 128 + d)
        krot = _norm_rope(k, tl.load(KGAIN + d), cos, sin, KEPS)
        position = tl.load(POS) + token
        offset = (head * C + position) * 128 + d
        tl.store(KC + offset, krot)
        tl.store(VC + offset, v)


def window_qk_cache(q, k, v, q_gain, k_gain, cos, sin, position,
                    k_cache, v_cache, q_eps, k_eps):
    """B1/T4 contiguous BF16 Q/K/V; write cache [1,8,C,128] at t..t+3."""
    if (q.shape != (1, 4, 4096) or k.shape != (1, 4, 1024)
            or v.shape != k.shape or cos.shape != (1, 4, 128)
            or sin.shape != cos.shape or q.dtype != torch.bfloat16
            or k.dtype != q.dtype or v.dtype != q.dtype
            or not q.is_contiguous() or not k.is_contiguous()
            or not v.is_contiguous() or not cos.is_contiguous()
            or not sin.is_contiguous() or position.shape != (1,)
            or position.dtype != torch.int64
            or k_cache.shape != v_cache.shape or k_cache.shape[:2] != (1, 8)
            or k_cache.shape[-1] != 128
            or not k_cache.is_contiguous() or not v_cache.is_contiguous()):
        raise ValueError("invalid four-token Q/K/cache layout")
    out = torch.empty((1, 32, 4, 128), device=q.device, dtype=q.dtype)
    _window_write[(4, 32)](
        q, k, v, q_gain, k_gain, cos, sin, position,
        k_cache, v_cache, out, k_cache.shape[2], q_eps, k_eps,
        num_warps=4, enable_fp_fusion=False,
    )
    return out
