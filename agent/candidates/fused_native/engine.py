"""Qwen3 greedy engine with fused norms and direct decoder-layer dispatch."""

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from kernels.rmsnorm import rms_norm


class FusedRMSNorm(torch.nn.Module):
    """Keep the reference gain and BF16 rounding boundary."""

    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight
        self.variance_epsilon = reference.variance_epsilon

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


@torch.inference_mode()
def _forward_last_logits(model, input_ids, cache, first_position):
    base = model.model
    hidden_states = base.embed_tokens(input_ids)
    length = input_ids.shape[1]
    cache_position = torch.arange(
        first_position, first_position + length, device=input_ids.device
    )
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = base.rotary_emb(hidden_states, position_ids)

    for layer in base.layers:
        hidden_states = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]

    hidden_states = base.norm(hidden_states)
    return model.lm_head(hidden_states[:, -1:, :])


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )

        base = self.model.model
        base.norm = FusedRMSNorm(base.norm)
        for layer in base.layers:
            layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = FusedRMSNorm(
                layer.post_attention_layernorm
            )
            layer.self_attn.q_norm = FusedRMSNorm(layer.self_attn.q_norm)
            layer.self_attn.k_norm = FusedRMSNorm(layer.self_attn.k_norm)

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        current = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
        cache = DynamicCache()
        position = 0
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = _forward_last_logits(self.model, current, cache, position)
                position += current.shape[1]
                current = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                yield current[:, 0].tolist()
