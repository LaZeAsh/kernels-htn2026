"""Native Qwen3 projections and prefill; direct full-context GQA on decode."""

import types

from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
)

from kernels.grouped_tc import grouped_tc_decode


def _attention_forward(
    self, hidden_states, position_embeddings, attention_mask,
    past_key_value=None, cache_position=None, **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
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
        attn_output = grouped_tc_decode(
            query_states, key_states, value_states, cache_position, self.scaling,
            past_key_value.prefill_length,
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


def install_direct_gqa(layer):
    """Change only the attention algorithm for cached one-token decode."""
    attention = layer.self_attn
    attention.forward = types.MethodType(_attention_forward, attention)
