"""Native projections with fused prefill and direct full-context GQA decode."""

import types

from kernels.decode_fusion import qk_norm_rope_cache, prefill_qk_norm_rope_cache, swiglu
from kernels.qkv_split import project_norm_rope_cache
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
    cos, sin = position_embeddings
    if (input_shape[-1] == 1 and past_key_value is not None
            and not past_key_value.prefill_mode):
        if hidden_states.shape[0] <= 16:
            query_states = project_norm_rope_cache(
                hidden_states, self.q_proj.weight, self.k_proj.weight,
                self.v_proj.weight, self.q_norm.weight, self.k_norm.weight,
                cos, sin, cache_position,
                past_key_value.keys[self.layer_idx],
                past_key_value.values[self.layer_idx],
                self.q_norm.variance_epsilon, self.k_norm.variance_epsilon,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            query_states = qk_norm_rope_cache(
                q, k, v, self.q_norm.weight, self.k_norm.weight,
                cos, sin, cache_position,
                past_key_value.keys[self.layer_idx],
                past_key_value.values[self.layer_idx],
                self.q_norm.variance_epsilon, self.k_norm.variance_epsilon,
            )
        attn_output = grouped_tc_decode(
            query_states, past_key_value.keys[self.layer_idx],
            past_key_value.values[self.layer_idx], cache_position, self.scaling,
            past_key_value.prefill_length,
        )
        attn_weights = None
    elif past_key_value is not None and past_key_value.prefill_mode:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        query_states = prefill_qk_norm_rope_cache(
            q, k, v, self.q_norm.weight, self.k_norm.weight,
            cos, sin, past_key_value.keys[self.layer_idx],
            past_key_value.values[self.layer_idx],
            self.q_norm.variance_epsilon, self.k_norm.variance_epsilon,
        )
        length = input_shape[-1]
        if self.layer_idx == 0:
            past_key_value.prefill_length = length
        key_states = past_key_value.keys[self.layer_idx][:, :, :length, :]
        value_states = past_key_value.values[self.layer_idx][:, :, :length, :]
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window,
            **kwargs,
        )
    else:
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
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
    """Install decode fusion while retaining native projection modules."""
    attention = layer.self_attn
    attention.forward = types.MethodType(_attention_forward, attention)
    mlp = layer.mlp
    mlp.decode_mode = False
    mlp.forward = types.MethodType(_mlp_forward, mlp)


def _mlp_forward(self, x):
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    product = swiglu(gate, up)
    return self.down_proj(product)
