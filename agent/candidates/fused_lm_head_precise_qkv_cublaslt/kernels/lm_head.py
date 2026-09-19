"""Decode full-vocabulary BF16 LM head and deterministic argmax."""

import torch
import triton
import triton.language as tl


VOCAB = 151936
HIDDEN = 2560
MAX_TILES = triton.cdiv(VOCAB, 64)


@triton.autotune(
    configs=[
        triton.Config({"BN": bn, "BK": bk}, num_warps=4, num_stages=2)
        for bn in (64, 128) for bk in (64, 128)
    ],
    key=["B"], warmup=10, rep=50, use_cuda_graph=False,
)
@triton.jit
def _project_max(X, W, PMAX, PIDX, B: tl.constexpr,
                 V: tl.constexpr, H: tl.constexpr, STRIDE: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    rows = tl.arange(0, 16)
    cols = tile * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((16, BN), 0.0, tl.float32)
    for block in range(H // BK):
        k = block * BK + kk
        x = tl.load(X + rows[:, None] * H + k[None, :],
                    rows[:, None] < B, 0)
        w = tl.load(W + cols[:, None] * H + k[None, :],
                    cols[:, None] < V, 0)
        acc = tl.dot(x, tl.trans(w), acc)
    # Native BF16 linear rounds each vocabulary logit before greedy choice.
    logits = tl.where(cols[None, :] < V,
                      acc.to(tl.bfloat16).to(tl.float32), -float("inf"))
    local_max = tl.max(logits, 1)
    local_index = tl.min(tl.where(logits == local_max[:, None],
                                  tl.broadcast_to(cols[None, :], (16, BN)),
                                  2147483647), 1)
    tl.store(PMAX + rows * STRIDE + tile, local_max, rows < B)
    tl.store(PIDX + rows * STRIDE + tile, local_index, rows < B)


@triton.jit
def _reduce_argmax(PMAX, PIDX, OUT, TILES: tl.constexpr,
                   STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    valid = tile < TILES
    local_max = tl.load(PMAX + row * STRIDE + tile,
                        valid, -float("inf"))
    local_index = tl.load(PIDX + row * STRIDE + tile,
                          valid, 2147483647)
    maximum = tl.max(local_max, 0)
    index = tl.min(tl.where(local_max == maximum, local_index, 2147483647), 0)
    tl.store(OUT + row, index.to(tl.int64))


def fused_lm_argmax(x, weight):
    """Full original BF16 [V,2560] head; returns int64 [B,1]."""
    batch = x.shape[0]
    if (batch < 1 or batch > 16 or x.shape != (batch, 1, HIDDEN)
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or weight.shape != (VOCAB, HIDDEN) or weight.dtype != x.dtype
            or not weight.is_contiguous()):
        raise ValueError("invalid full-vocabulary decode head")
    partial_max = torch.empty((batch, MAX_TILES), dtype=torch.float32, device=x.device)
    partial_idx = torch.empty((batch, MAX_TILES), dtype=torch.int32, device=x.device)
    result = torch.empty((batch, 1), dtype=torch.int64, device=x.device)
    _project_max[(lambda meta: (triton.cdiv(VOCAB, meta["BN"]),))](
        x, weight, partial_max, partial_idx, batch,
        VOCAB, HIDDEN, MAX_TILES,
    )
    selected_bn = _project_max.best_config.kwargs["BN"]
    tiles = triton.cdiv(VOCAB, selected_bn)
    _reduce_argmax[(batch,)](
        partial_max, partial_idx, result, tiles, MAX_TILES,
        triton.next_power_of_2(tiles), num_warps=4,
    )
    return result
