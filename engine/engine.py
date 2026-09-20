"""Experimental fixed-cache Qwen3 engine with a captured decode step.

Copy this directory's contents to submission root only after public validation.
"""

import torch
from transformers import AutoModelForCausalLM

from kernels.rmsnorm import rms_norm
from kernels.residual_norm import add_norm
from attention import install_direct_gqa


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight
        self.variance_epsilon = reference.variance_epsilon

    def forward(self, x):
        return rms_norm(x, self.weight, self.variance_epsilon)


class FixedCache:
    """Layer KV storage. Prefill exposes its prefix; decode exposes capacity.

    The caller supplies an explicit additive mask for decode, so no unfilled
    slot affects attention. The cache is overwritten from position zero on
    every generation; no prompt data is reused.
    """

    def __init__(self, layers, batch, heads, capacity, head_dim):
        self.capacity = capacity
        self.keys = [torch.zeros((batch, heads, capacity, head_dim),
                                 device="cuda:0", dtype=torch.bfloat16)
                     for _ in range(layers)]
        self.values = [torch.zeros_like(k) for k in self.keys]
        self.prefill_length = 0
        self.prefill_mode = True

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k, v = self.keys[layer_idx], self.values[layer_idx]
        length = key_states.shape[-2]
        if self.prefill_mode:
            k[:, :, :length, :].copy_(key_states)
            v[:, :, :length, :].copy_(value_states)
            if layer_idx == 0:
                self.prefill_length = length
            return k[:, :, :length, :], v[:, :, :length, :]
        position = cache_kwargs["cache_position"].reshape(1)
        k.index_copy_(2, position, key_states)
        v.index_copy_(2, position, value_states)
        return k, v

    def get_seq_length(self, layer_idx=0):
        return self.prefill_length


@torch.inference_mode()
def _forward_last(model, token_ids, cache, positions, attention_mask):
    base = model.model
    hidden = base.embed_tokens(token_ids)
    position_ids = positions.unsqueeze(0)
    position_embeddings = base.rotary_emb(hidden, position_ids)
    for layer in base.layers[:-1]:
        hidden = layer(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=positions,
            position_embeddings=position_embeddings,
        )[0]
    final_layer = base.layers[-1]
    attention_output = final_layer.self_attn(
        final_layer.input_layernorm(hidden),
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_value=cache,
        cache_position=positions,
    )[0]
    # Final-layer Q/K/V for every prompt token has already filled the cache.
    # Only the last token's post-attention MLP can affect the returned logits.
    residual = hidden[:, -1:, :].contiguous()
    attention_last = attention_output[:, -1:, :].contiguous()
    after_attention, mlp_input = add_norm(
        residual, attention_last,
        final_layer.post_attention_layernorm.weight,
        final_layer.post_attention_layernorm.variance_epsilon,
    )
    mlp_output = final_layer.mlp(mlp_input)
    _, normalized = add_norm(
        after_attention, mlp_output,
        base.norm.weight, base.norm.variance_epsilon,
    )
    return model.lm_head(normalized).argmax(-1)


@torch.inference_mode()
def _decode_last(model, token_ids, cache, positions):
    """Single-token path carries each layer's residual and normalized input."""
    base = model.model
    residual = base.embed_tokens(token_ids)
    position_ids = positions.unsqueeze(0)
    position_embeddings = base.rotary_emb(residual, position_ids)
    normalized = base.layers[0].input_layernorm(residual)
    layers = base.layers
    for index, layer in enumerate(layers):
        attention_output = layer.self_attn(
            normalized,
            position_embeddings=position_embeddings,
            attention_mask=None,
            past_key_value=cache,
            cache_position=positions,
        )[0]
        after_attention, mlp_input = add_norm(
            residual, attention_output,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.variance_epsilon,
        )
        mlp_output = layer.mlp(mlp_input)
        if index + 1 < len(layers):
            next_norm = layers[index + 1].input_layernorm
        else:
            next_norm = base.norm
        residual, normalized = add_norm(
            after_attention, mlp_output,
            next_norm.weight, next_norm.variance_epsilon,
        )
    return model.lm_head(normalized).argmax(-1)


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True,
        ).eval().to("cuda:0"))
        base = self.model.model
        base.norm = FusedRMSNorm(base.norm)
        for layer in base.layers:
            layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
            layer.self_attn.q_norm = FusedRMSNorm(layer.self_attn.q_norm)
            layer.self_attn.k_norm = FusedRMSNorm(layer.self_attn.k_norm)
            install_direct_gqa(layer)
        self._graph_shape = None
        self._cache = None
        self._graph = None

    def _prepare(self, batch, prompt_length, output_length):
        shape = (batch, prompt_length, output_length)
        if shape == self._graph_shape:
            return
        config = self.model.config
        heads = config.num_key_value_heads
        head_dim = self.model.model.layers[0].self_attn.k_proj.out_features // heads
        self._cache = FixedCache(
            len(self.model.model.layers), batch, heads,
            prompt_length + output_length, head_dim,
        )
        self._input = torch.empty((batch, 1), device="cuda:0", dtype=torch.int64)
        self._position = torch.empty((1,), device="cuda:0", dtype=torch.int64)
        self._graph_shape = shape
        self._graph = None

    def _decode(self):
        return _decode_last(self.model, self._input, self._cache, self._position)

    def _capture(self, token, position):
        previous_blas = torch.backends.cuda.preferred_blas_library()
        try:
            torch.backends.cuda.preferred_blas_library("cublaslt")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._input.copy_(token)
                self._position.fill_(position)
                self._decode()  # warm kernels and allocator on the capture stream
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self._decode()
            torch.cuda.current_stream().wait_stream(stream)
        finally:
            torch.backends.cuda.preferred_blas_library(previous_blas)
        self._graph = graph
        self._graph_output = output

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        if max_new_tokens <= 0:
            return
        batch, prompt_length = len(input_ids), len(input_ids[0])
        self._prepare(batch, prompt_length, max_new_tokens)
        prompt = torch.tensor(input_ids, device="cuda:0", dtype=torch.int64)
        positions = torch.arange(prompt_length, device="cuda:0")
        with torch.inference_mode():
            self._cache.prefill_mode = True
            for layer in self.model.model.layers:
                layer.mlp.decode_mode = False
            current = _forward_last(self.model, prompt, self._cache, positions, None)
            self._cache.prefill_mode = False
            for layer in self.model.model.layers:
                layer.mlp.decode_mode = True
            yield current[:, 0].tolist()
            if max_new_tokens == 1:
                return
            if self._graph is None:
                self._capture(current, prompt_length)
            for position in range(prompt_length, prompt_length + max_new_tokens - 1):
                self._input.copy_(current)
                self._position.fill_(position)
                self._graph.replay()
                current = self._graph_output
                yield current[:, 0].tolist()
