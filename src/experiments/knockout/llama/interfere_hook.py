from typing import Iterable, Optional, Tuple

from torch import FloatTensor, Tensor, nn

from src.core.types import KnockoutMode
from src.experiments.knockout.llama.llama_attention_forward import llama_attention_forward


class InterfereHook:
    def __init__(
        self,
        layer: int | str | nn.Module,
        knockout_type: KnockoutMode,
        feature_mask: Optional[FloatTensor | Tensor] = None,
    ):
        self.counter = 0
        self.layer = layer
        self.knockout_type = knockout_type
        self.knockout_indices: Iterable[int] = []
        self.affected_outputs: Iterable[int] = []
        self.feature_mask = feature_mask

    def hook(self, module: nn.Module, inp: Tuple[Tensor, ...], out: Tuple[Tensor, ...]) -> Optional[Tensor]:
        """
        module - the actual layer
        inp - previous layer ssm state (in mamba 1 - this is X)
        out - the current layer output ssm state (in mamba 1 / 2 - this is Y)
        """
        # TODO make the knockout more efficient - maybe replace the module.forward with a custom forward

        knockout_mask = [(k, q) for k in self.knockout_indices for q in self.affected_outputs]

        if self.feature_mask is not None:
            self.feature_mask = self.feature_mask.to(inp[0].device)
        curr_out, _ = llama_attention_forward(
            module=module,
            hidden_states=inp[0],
            position_embeddings=inp[1],
            attention_mask=inp[2] if len(inp) > 2 else None,
            # past_key_value = inp[3] if len(inp) > 3 else None,
            # cache_position = inp[4] if len(inp) > 4 else None,
            knockout_mask=knockout_mask,
            # **kwargs: Unpack[FlashAttentionKwargs],
        )
        return curr_out

    def __call__(self, module: nn.Module, inp: Tuple[Tensor, ...], out: Tuple[Tensor, ...]) -> Optional[Tensor]:
        return self.hook(module, inp, out)

    def __str__(self):
        return (
            f"InterfereHook for layer {self.layer} "
            f"with knockout type {self.knockout_type}, "
            f"knockout indices {self.knockout_indices}, "
            f"affected outputs {self.affected_outputs}, "
            f"and feature mask {self.feature_mask}"
        )
