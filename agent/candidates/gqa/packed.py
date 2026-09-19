"""Pack Qwen3 projection weights while preserving native post-GEMM operations."""

import types

import torch
import torch.nn.functional as F
from kernels.gqa import gqa_decode
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
)


def _attention_forward(
    self, hidden_states, position_embeddings, attention_mask,
    past_key_value=None, cache_position=None, **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    qkv = F.linear(hidden_states, self.qkv_weight)
    q_width = self.config.num_attention_heads * self.head_dim
    kv_width = self.config.num_key_value_heads * self.head_dim
    q, k, v = qkv.split((q_width, kv_width, kv_width), dim=-1)
    query_states = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    if (input_shape[-1] == 1 and past_key_value is not None
            and not past_key_value.prefill_mode):
        attn_output = gqa_decode(
            query_states, key_states, value_states, cache_position, self.scaling
        )
        attn_weights = None
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window,
            **kwargs,
        )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def _mlp_forward(self, x):
    gate, up = F.linear(x, self.gate_up_weight).split(self.intermediate_size, -1)
    return self.down_proj(self.act_fn(gate) * up)


def pack_layer(layer):
    """Pack BF16 parameters once, then remove separate source projections."""
    attn = layer.self_attn
    attn.qkv_weight = torch.nn.Parameter(
        torch.cat((attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight), 0),
        requires_grad=False,
    )
    del attn.q_proj, attn.k_proj, attn.v_proj
    attn.forward = types.MethodType(_attention_forward, attn)

    mlp = layer.mlp
    mlp.gate_up_weight = torch.nn.Parameter(
        torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight), 0),
        requires_grad=False,
    )
    del mlp.gate_proj, mlp.up_proj
    mlp.forward = types.MethodType(_mlp_forward, mlp)
