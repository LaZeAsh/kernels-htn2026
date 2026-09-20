"""Lossless BF16 gate/up codec shared by validation and split-K GEMM."""

import torch
import triton
import triton.language as tl


I = 9728
H = 2560
BLOCKS = H // 64
_CODEC_SELFTESTED = False


@triton.jit
def _weight_block(SM, DELTA, BASE, ORIGINAL, rows, global_k,
                  N: tl.constexpr, HIDDEN: tl.constexpr,
                  K_BLOCKS: tl.constexpr):
    block = global_k // 64
    local_k = global_k % 64
    base = tl.load(BASE + rows[:, None] * K_BLOCKS + block[None, :],
                   mask=rows[:, None] < N, other=0).to(tl.uint16)
    compressed = (base != 0) & (rows[:, None] < N)
    byte_offset = (rows[:, None] * K_BLOCKS + block[None, :]) * 64 + local_k[None, :]
    sm = tl.load(SM + byte_offset, mask=compressed, other=0).to(tl.uint16)
    nibble_offset = ((rows[:, None] * K_BLOCKS + block[None, :]) * 32
                     + local_k[None, :] // 2)
    packed = tl.load(DELTA + nibble_offset,
                     mask=compressed, other=0).to(tl.uint16)
    delta = tl.where((local_k[None, :] & 1) == 0, packed & 15, packed >> 4)
    bits = ((sm & 128) << 8) | ((base + delta) << 7) | (sm & 127)
    decoded = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    original = tl.load(ORIGINAL + rows[:, None] * HIDDEN + global_k[None, :],
                       mask=(rows[:, None] < N) & ~compressed, other=0)
    return tl.where(compressed, decoded, original)


@triton.jit
def _validate_decode(SM, DELTA, BASE, ORIGINAL, OUT,
                     N: tl.constexpr, HIDDEN: tl.constexpr,
                     K_BLOCKS: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    weights = _weight_block(SM, DELTA, BASE, ORIGINAL, rows, k,
                            N, HIDDEN, K_BLOCKS)
    tl.store(OUT + rows[:, None] * HIDDEN + k[None, :],
             weights, mask=rows[:, None] < N)


def _encode_validate_impl(weight):
    """Encode contiguous BF16 [N,K], K divisible by 128."""
    if (weight.ndim != 2 or weight.shape[1] % 128 != 0
            or weight.dtype != torch.bfloat16 or not weight.is_cuda
            or not weight.is_contiguous()):
        raise ValueError("expected contiguous CUDA BF16 [N,K128multiple]")
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
        for bk in (64, 128):
            decoded = torch.empty_like(weight)
            _validate_decode[(triton.cdiv(n, 16), triton.cdiv(hidden, bk))](
                sm, packed, base, weight, decoded,
                n, hidden, blocks, bk, num_warps=4,
            )
            if not torch.equal(decoded.view(torch.int16), weight.view(torch.int16)):
                raise ValueError(f"lossless BF16 codec validation failed at BK={bk}")
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
        patterns = bits.to(torch.int16).view(torch.bfloat16).reshape(512, 128)
        _encode_validate_impl(patterns)
        _CODEC_SELFTESTED = True
    if weight.shape != (I, H):
        raise ValueError("model MLP weight must be BF16 [9728,2560]")
    return _encode_validate_impl(weight)
