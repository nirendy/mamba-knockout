from typing import Iterable, Optional, Tuple

import torch
from torch import LongTensor, Tensor, nn
from transformers.models.llama.modeling_llama import (
    Cache,
    FlashAttentionKwargs,
    LlamaAttention,
    Unpack,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
)

# from src.experiments.knockout.llama.llama_attention_forward import llama_attention_forward

T_LLAMA_ATTN = LlamaAttention | MistralAttention | Qwen2Attention


class LlamaAttentionKnockout(nn.Module):
    def __init__(
        self,
        inner: T_LLAMA_ATTN,
        knockout_mask: Optional[Iterable[tuple[int, int]]] = None,
    ):
        super().__init__()
        self.inner = inner
        self.knockout_mask = knockout_mask

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Call the original forward method
        # return llama_attention_forward(
        #     self.inner,
        #     hidden_states,
        #     position_embeddings,
        #     attention_mask=attention_mask,
        #     past_key_value=past_key_value,
        #     cache_position=cache_position,
        #     knockout_mask=self.knockout_mask,
        #     **kwargs,
        # )

        t = hidden_states.shape[1]
        knockout_mask = torch.ones([1, 1, t, t], dtype=torch.bool, device=hidden_states.device)

        if self.knockout_mask is not None:
            for q, k in self.knockout_mask:
                knockout_mask[:, :, q, k] = False

            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :t, :t]
                attention_mask = attention_mask.logical_and(knockout_mask)
            else:
                attention_mask = knockout_mask

        out = self.inner(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        return out
