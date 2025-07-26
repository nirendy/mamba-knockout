from typing import Iterable, Optional

from torch import FloatTensor, Tensor, nn

from src.core.types import KnockoutMode
from src.experiments.knockout.mamba.mamba1.falcon_variant import (
    slow_forward_for_ssm_materializing_knockout_falcon,
)
from src.experiments.knockout.mamba.mamba1.original_variant import (
    slow_forward_for_ssm_materializing_knockout,
)


class SSMInterfereHook:
    def __init__(
        self,
        layer: int | str | nn.Module,
        knockout_type: KnockoutMode,
        is_falcon: bool,
        feature_mask: Optional[FloatTensor | Tensor] = None,
    ):
        self.counter = 0
        self.layer = layer
        self.is_falcon = is_falcon
        self.knockout_type = knockout_type
        self.knockout_indices: Iterable[int] = []
        self.affected_outputs: Iterable[int] = []
        self.feature_mask = feature_mask

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        """
        module - the actual layer
        inp - previous layer ssm state (in mamba 1 - this is X)
        out - the current layer output ssm state (in mamba 1 / 2 - this is Y)
        """
        # TODO make the knockout more efficient - maybe replace the module.forward with a custom forward
        slow_forward = (
            slow_forward_for_ssm_materializing_knockout_falcon
            if self.is_falcon
            else slow_forward_for_ssm_materializing_knockout
        )
        if self.feature_mask is not None:
            self.feature_mask = self.feature_mask.to(inp[0].device)
        curr_out = slow_forward(
            module,
            inp[0],
            knockout_indices=self.knockout_indices,
            affected_outputs=self.affected_outputs,
            knockout_mode=self.knockout_type,
            knockout_feature_mask=self.feature_mask,
        )
        return curr_out

    def __call__(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        return self.hook(module, inp, out)

    def __str__(self):
        return (
            f"SSMInterfereHook for layer {self.layer} "
            f"with knockout type {self.knockout_type}, "
            f"knockout indices {self.knockout_indices}, "
            f"affected outputs {self.affected_outputs}, "
            f"and feature mask {self.feature_mask}"
        )
