"""Experimental BF16 Q/K/V split-K decode, retaining original weights."""

import torch
import triton
import triton.language as tl

from kernels.decode_fusion import _norm_rope


@triton.jit
def _project(X, QW, KW, VW, PART, B: tl.constexpr):
    tile = tl.program_id(0)  # 32 Q, 8 K, 8 V head tiles
    split = tl.program_id(1)
    # BM64 may allow Hopper MMA v3 lowering; only B<=16 rows are valid.
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 128)
    kk = tl.arange(0, 64)
    acc = tl.full((64, 128), 0, tl.float32)
    if tile < 32:
        weight = QW
        head = tile
    elif tile < 40:
        weight = KW
        head = tile - 32
    else:
        weight = VW
        head = tile - 40
    for block in range(10):
        k = split * 640 + block * 64 + kk
        x = tl.load(X + rows[:, None] * 2560 + k[None, :],
                    rows[:, None] < B, 0)
        w = tl.load(weight + (head * 128 + cols[:, None]) * 2560 + k[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    offsets = ((rows[:, None] * 48 + tile) * 4 + split) * 128 + cols[None, :]
    tl.store(PART + offsets, acc, rows[:, None] < B)


@triton.jit
def _reduce_rope(PART, QGAIN, KGAIN, COS, SIN, POS, KC, VC, QOUT,
                 CAPACITY: tl.constexpr, QEPS: tl.constexpr, KEPS: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 128)
    base = (b * 48 + h) * 4 * 128 + d
    q = (tl.load(PART + base) + tl.load(PART + base + 128)
         + tl.load(PART + base + 256) + tl.load(PART + base + 384)).to(tl.bfloat16)
    cos = tl.load(COS + d)
    sin = tl.load(SIN + d)
    qrot = _norm_rope(q, tl.load(QGAIN + d), cos, sin, QEPS)
    tl.store(QOUT + (b * 32 + h) * 128 + d, qrot)
    if h < 8:
        kb = (b * 48 + 32 + h) * 4 * 128 + d
        vb = (b * 48 + 40 + h) * 4 * 128 + d
        k = (tl.load(PART + kb) + tl.load(PART + kb + 128)
             + tl.load(PART + kb + 256) + tl.load(PART + kb + 384)).to(tl.bfloat16)
        v = (tl.load(PART + vb) + tl.load(PART + vb + 128)
             + tl.load(PART + vb + 256) + tl.load(PART + vb + 384)).to(tl.bfloat16)
        krot = _norm_rope(k, tl.load(KGAIN + d), cos, sin, KEPS)
        position = tl.load(POS)
        cache_offset = ((b * 8 + h) * CAPACITY + position) * 128 + d
        tl.store(KC + cache_offset, krot)
        tl.store(VC + cache_offset, v)


def project_norm_rope_cache(x, qw, kw, vw, qgain, kgain, cos, sin,
                            position, kc, vc, qeps, keps):
    """Decode only: x [B,1,2560], original BF16 weights, B <= 16."""
    batch = x.shape[0]
    if (x.shape != (batch, 1, 2560) or batch > 16 or batch < 1
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or qw.shape != (4096, 2560) or kw.shape != (1024, 2560)
            or vw.shape != (1024, 2560)
            or not qw.is_contiguous() or not kw.is_contiguous()
            or not vw.is_contiguous()
            or kc.shape != vc.shape or kc.shape[:2] != (batch, 8)
            or kc.shape[-1] != 128 or not kc.is_contiguous()
            or not vc.is_contiguous()):
        raise ValueError("invalid split QKV input, weights or cache")
    partial = torch.empty((batch, 48, 4, 128), dtype=torch.float32, device=x.device)
    qout = torch.empty((batch, 32, 1, 128), dtype=x.dtype, device=x.device)
    _project[(48, 4)](x, qw, kw, vw, partial, batch, num_warps=4)
    _reduce_rope[(batch, 32)](
        partial, qgain, kgain, cos, sin, position, kc, vc, qout,
        kc.shape[2], qeps, keps, num_warps=4, enable_fp_fusion=False,
    )
    return qout
