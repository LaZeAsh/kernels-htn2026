"""Lossless BF16 gate/up storage with exact bit reconstruction on decode."""

import torch
import triton
import triton.language as tl


I = 9728
H = 2560
BLOCKS = H // 64
_CODEC_SELFTESTED = False


@triton.jit
def _weight_block(SM, DELTA, BASE, ORIGINAL, rows, k, block,
                  N: tl.constexpr, HIDDEN: tl.constexpr,
                  K_BLOCKS: tl.constexpr):
    base = tl.load(BASE + rows[:, None] * K_BLOCKS + block,
                   mask=rows[:, None] < N, other=0).to(tl.uint16)
    compressed = (base != 0) & (rows[:, None] < N)
    byte_offset = (rows[:, None] * K_BLOCKS + block) * 64 + k[None, :]
    sm = tl.load(SM + byte_offset, mask=compressed, other=0).to(tl.uint16)
    nibble_offset = (rows[:, None] * K_BLOCKS + block) * 32 + k[None, :] // 2
    packed = tl.load(DELTA + nibble_offset,
                     mask=compressed, other=0).to(tl.uint16)
    delta = tl.where((k[None, :] & 1) == 0, packed & 15, packed >> 4)
    bits = ((sm & 128) << 8) | ((base + delta) << 7) | (sm & 127)
    decoded = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    original = tl.load(ORIGINAL + rows[:, None] * HIDDEN + block * 64 + k[None, :],
                       mask=(rows[:, None] < N) & ~compressed, other=0)
    return tl.where(compressed, decoded, original)


@triton.jit
def _validate_decode(SM, DELTA, BASE, ORIGINAL, OUT,
                     N: tl.constexpr, HIDDEN: tl.constexpr,
                     K_BLOCKS: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    block = tl.program_id(1)
    k = tl.arange(0, 64)
    weights = _weight_block(SM, DELTA, BASE, ORIGINAL, rows, k, block,
                            N, HIDDEN, K_BLOCKS)
    tl.store(OUT + rows[:, None] * HIDDEN + block * 64 + k[None, :],
             weights, mask=rows[:, None] < N)


@triton.jit
def _gate_up_swiglu(X, GSM, GDELTA, GBASE, GW,
                    USM, UDELTA, UBASE, UW, OUT, B: tl.constexpr,
                    N: tl.constexpr, HIDDEN: tl.constexpr,
                    K_BLOCKS: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 64)
    gate = tl.full((16, 64), 0.0, tl.float32)
    up = tl.full((16, 64), 0.0, tl.float32)
    for block in range(K_BLOCKS):
        x = tl.load(X + rows[:, None] * HIDDEN + block * 64 + k[None, :],
                    mask=rows[:, None] < B, other=0)
        gw = _weight_block(GSM, GDELTA, GBASE, GW, cols, k, block,
                           N, HIDDEN, K_BLOCKS)
        uw = _weight_block(USM, UDELTA, UBASE, UW, cols, k, block,
                           N, HIDDEN, K_BLOCKS)
        gate = tl.dot(x, tl.trans(gw), gate)
        up = tl.dot(x, tl.trans(uw), up)
    gate_bf16 = gate.to(tl.bfloat16).to(tl.float32)
    up_bf16 = up.to(tl.bfloat16).to(tl.float32)
    activated = (gate_bf16 / (1.0 + tl.exp(-gate_bf16))).to(tl.bfloat16)
    product = (activated.to(tl.float32) * up_bf16).to(tl.bfloat16)
    tl.store(OUT + rows[:, None] * N + cols[None, :], product,
             mask=rows[:, None] < B)


def _encode_validate_impl(weight):
    """Encode arbitrary contiguous BF16 [N,K], K divisible by 64."""
    if (weight.ndim != 2 or weight.shape[1] % 64 != 0
            or weight.dtype != torch.bfloat16 or not weight.is_cuda
            or not weight.is_contiguous()):
        raise ValueError("expected contiguous CUDA BF16 [N,K64multiple]")
    n, hidden = weight.shape
    blocks = hidden // 64
    with torch.no_grad():
        bits = weight.view(torch.int16).to(torch.int32) & 0xffff
        parts = bits.reshape(n, blocks, 64)
        exponent = (parts >> 7) & 255
        minimum = exponent.amin(dim=-1)
        maximum = exponent.amax(dim=-1)
        can_pack = (minimum >= 1) & (maximum <= 254) & (maximum - minimum <= 15)
        base = torch.where(can_pack, minimum, 0).to(torch.uint8).contiguous()
        sm = ((((parts >> 8) & 128) | (parts & 127))
              .to(torch.uint8).contiguous())
        delta = (exponent - minimum.unsqueeze(-1)).clamp(0, 15).to(torch.uint8)
        packed = (delta[..., 0::2] | (delta[..., 1::2] << 4)).contiguous()
        del bits, parts, exponent, minimum, maximum, can_pack, delta
        decoded = torch.empty_like(weight)
        _validate_decode[(triton.cdiv(n, 16), blocks)](
            sm, packed, base, weight, decoded, n, hidden, blocks, num_warps=4,
        )
        if not torch.equal(decoded.view(torch.int16), weight.view(torch.int16)):
            raise ValueError("lossless BF16 MLP codec bitwise validation failed")
        del decoded
    return sm, packed, base


def encode_validate(weight):
    """Self-test all BF16 bit patterns, then validate each model weight bit."""
    global _CODEC_SELFTESTED
    if not _CODEC_SELFTESTED:
        index = torch.arange(65536, device=weight.device, dtype=torch.int32)
        sign = (index >> 4) & 1
        exponent = ((index >> 12) << 4) | (index & 15)
        mantissa = (((index >> 6) & 63) << 1) | ((index >> 5) & 1)
        bits = (sign << 15) | (exponent << 7) | mantissa
        patterns = bits.to(torch.int16).view(torch.bfloat16).reshape(1024, 64)
        _encode_validate_impl(patterns)
        _CODEC_SELFTESTED = True
    if weight.shape != (I, H):
        raise ValueError("model MLP weight must be BF16 [9728,2560]")
    return _encode_validate_impl(weight)


def lossless_swiglu(x, gate_weight, up_weight, gate_codec, up_codec):
    """Decode exact BF16 gate/up blocks inside unsplit GEMM, B 1..16."""
    batch = x.shape[0]
    if (batch < 1 or batch > 16 or x.shape != (batch, 1, H)
            or x.dtype != torch.bfloat16 or not x.is_contiguous()):
        raise ValueError("expected contiguous BF16 decode input")
    out = torch.empty((batch, 1, I), device=x.device, dtype=x.dtype)
    _gate_up_swiglu[(triton.cdiv(I, 64),)](
        x, *gate_codec, gate_weight, *up_codec, up_weight, out, batch,
        I, H, BLOCKS,
        num_warps=4, num_stages=2,
    )
    return out
