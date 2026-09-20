"""Native packed varlen FlashAttention over the time-major KV cache."""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_used_k(POS, USED, B: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.arange(0, BLOCK)
    length = tl.load(POS).to(tl.int32) + 1
    tl.store(USED + b, length, b < B)


def set_used_k(used_k, position):
    """Update the device-side valid prefix once before every decode layer loop."""
    batch = used_k.numel()
    if (used_k.shape != (batch,) or used_k.dtype != torch.int32
            or not used_k.is_cuda or not used_k.is_contiguous()
            or position.shape != (1,) or position.dtype != torch.int64
            or not position.is_cuda):
        raise ValueError("invalid varlen decode length buffers")
    _write_used_k[(1,)](position, used_k, batch,
                          triton.next_power_of_2(batch), num_warps=4)


def flash_varlen_decode(query, cache, layer_idx, scaling):
    """Q [B,32,1,128] and physical K/V [B,C,8,128] -> [B,32,128]."""
    batch = query.shape[0]
    key = cache.keys_packed[layer_idx]
    value = cache.values_packed[layer_idx]
    capacity = cache.capacity
    if (query.shape != (batch, 32, 1, 128)
            or query.dtype != torch.bfloat16 or not query.is_contiguous()
            or key.shape != (batch, capacity, 8, 128)
            or value.shape != key.shape or not key.is_contiguous()
            or not value.is_contiguous() or key.dtype != query.dtype
            or value.dtype != query.dtype
            or cache.cu_q.shape != (batch + 1,)
            or cache.cu_k.shape != (batch + 1,)
            or cache.used_k.shape != (batch,)):
        raise ValueError("invalid packed varlen FlashAttention buffers")
    packed_query = query.transpose(1, 2).reshape(batch, 32, 128).contiguous()
    output = torch.ops.aten._flash_attention_forward.default(
        packed_query,
        key.view(batch * capacity, 8, 128),
        value.view(batch * capacity, 8, 128),
        cache.cu_q, cache.cu_k,
        1, capacity, 0.0, False, False,
        scale=scaling, seqused_k=cache.used_k,
    )[0]
    return output
