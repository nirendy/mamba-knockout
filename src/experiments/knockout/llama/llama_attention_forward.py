from typing import Callable, Iterable, Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import (
    Cache,
    FlashAttentionKwargs,
    LlamaAttention,
    Unpack,
    apply_rotary_pos_emb,
)

from src.experiments.knockout.llama.sdpa_attention import sdpa_attention_forward


def llama_attention_forward(
    module: LlamaAttention,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    knockout_mask: Optional[Iterable[Tuple[int, int]]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # , Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, module.layer_idx, cache_kwargs)

    # attention_interface: Callable = eager_attention_forward
    # if module.config._attn_implementation != "eager":
    #     if module.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
    #         logger.warning_once(
    #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
    #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
    #         )
    #     else:
    #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attention_interface: Callable = sdpa_attention_forward

    attn_output, attn_weights = attention_interface(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=module.scaling,
        is_causal=True,
        knockout_mask=knockout_mask,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = module.o_proj(attn_output)
    return attn_output, attn_weights
