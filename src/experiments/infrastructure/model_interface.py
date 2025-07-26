from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, Union, assert_never, cast

import torch
import torch.nn.functional as F
import transformers.models
from torch import Tensor

from src.core.consts import is_falcon
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    FeatureCategory,
    KnockoutMode,
    TDevice,
    TGP2Model,
    TLayerIndex,
    TLlamaModel,
    TMamba1Model,
    TMamba2Model,
    TModelSize,
    TTokenizer,
)
from src.experiments.infrastructure.setup_models import get_tokenizer_and_model
from src.experiments.knockout.gpt.gpt2 import gpt2_knockout_utils
from src.experiments.knockout.llama.llama_attn import (
    LlamaAttention,
    LlamaAttentionKnockout,
    MistralAttention,
    Qwen2Attention,
)
from src.experiments.knockout.mamba.mamba1.helpers.ssm_interfere import SSMInterfereHook
from src.experiments.knockout.mamba.mamba2.minimal_mamba2 import Mamba2


class ModelInterface(ABC):
    """Abstract interface for language models with attention knockout capability."""

    def __init__(
        self,
        model_arch: MODEL_ARCH,
        model_size: TModelSize,
        device: Optional[TDevice] = None,
        tokenizer: Optional[TTokenizer] = None,
    ):
        """Initialize the model with given size and device."""
        # x = get_tokenizer_and_model()
        self.tokenizer, self.model = get_tokenizer_and_model(model_arch, model_size, device)
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.device: TDevice = cast(TDevice, self.model.device)

    def setup(self, layers: Optional[Iterable[TLayerIndex]] = None):
        self.model.eval()

    @abstractmethod
    def generate_logits(
        self,
        input_ids: torch.Tensor,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]],
        feature_category: FeatureCategory,
    ) -> torch.Tensor:
        """
        Generate logits for the input sequence with optional attention masking.

        Args:
            input_ids: Input token IDs
            attention: Whether to use attention mechanism
            num_to_masks: Dict mapping layer numbers to list of (idx1, idx2) tuples,
                        where idx1 won't get information from idx2

        Returns:
            Tuple of (next_token_logits, all_logits)
        """
        pass

    @abstractmethod
    def n_layers(self) -> int:
        pass


class MambaInterface(ModelInterface):
    model: Union[TMamba1Model, TMamba2Model]

    @property
    def backbone(self):
        return self.model.backbone

    @property
    def layers(self) -> torch.nn.ModuleList:
        layers = self.backbone.layers
        assert isinstance(layers, torch.nn.ModuleList)
        return layers

    @abstractmethod
    def get_layer_moi(self, layer_i: int) -> torch.nn.Module:
        # "mixer of interest" - moi
        pass

    def n_layers(self):
        return len(self.layers)


class Mamba1Interface(MambaInterface):
    model: TMamba1Model  # type:ignore[valid-type]

    def __init__(
        self,
        model_size: TModelSize,
        device: Optional[TDevice] = None,
        tokenizer: Optional[TTokenizer] = None,
        is_falcon: bool = False,
    ):
        super().__init__(MODEL_ARCH.MAMBA1, model_size, device, tokenizer)

        self.hooks: list[SSMInterfereHook] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.is_falcon = is_falcon
        if is_falcon:
            print("using falcon")
        else:
            print("not using falcon")

        self.knockout_mode = KnockoutMode.ZERO_ATTENTION
        self.feature_masks = {}

    def setup(self, layers: Optional[Iterable[TLayerIndex]] = None):
        super().setup(layers)

        for handle in self.handles:
            handle.remove()

        # Assert that no hooks are left
        for m in self.model.modules():
            assert len(list(m._forward_hooks.items())) == 0

        self.handles = []
        self.hooks = []

        if layers is not None:
            # set up hooks
            for i in range(len(self.model.backbone.layers)):
                if i in layers:
                    self.hooks.append(SSMInterfereHook(i, self.knockout_mode, is_falcon=self.is_falcon))
                    self.handles.append(self.get_layer_moi(i).register_forward_hook(self.hooks[-1]))  # type: ignore

    def get_layer_moi(
        self, layer_i: int
    ) -> (
        transformers.models.mamba.modeling_mamba.MambaMixer
        | transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaMixer
    ):
        # "mixer of interest" - moi
        if isinstance(self.model, transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaForCausalLM):
            moi = self.model.backbone.layers[layer_i].mixer
            assert isinstance(moi, transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaMixer)
        else:
            moi = self.model.backbone.layers[layer_i].mixer
            assert isinstance(moi, transformers.models.mamba.modeling_mamba.MambaMixer)
        return moi

    def _get_feature_mask(self, layer: torch.nn.Module, feature_category: FeatureCategory) -> Tensor:
        assert isinstance(layer.A_log, torch.Tensor)

        if feature_category == FeatureCategory.ALL:
            if layer not in self.feature_masks:
                self.feature_masks[layer] = torch.zeros(layer.A_log.shape[0]).to(layer.A_log.device)
            return self.feature_masks[layer]

        # if feature_category == FeatureCategory.NONE:
        #     return torch.ones(layer.A_log.shape[0])

        decay_matrices = torch.exp(-torch.exp(layer.A_log))
        n_ssms = decay_matrices.shape[0]

        # get the norms
        norms = torch.norm(decay_matrices, p=1, dim=1)

        sorted_indices = torch.argsort(norms, descending=(feature_category == FeatureCategory.SLOW_DECAY))
        mask = torch.zeros_like(norms, dtype=torch.bool)
        mask[sorted_indices[: n_ssms // 3]] = True
        return mask

    def generate_logits(
        self,
        input_ids: Tensor,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        feature_category: FeatureCategory = FeatureCategory.ALL,
    ) -> torch.Tensor:
        if num_to_masks is not None:
            source_indices = []
            target_indices = []

            for layer, hook in zip(num_to_masks, self.hooks):
                source_indices = [num_to_masks[layer][i][1] for i in range(len(num_to_masks[layer]))]
                target_indices = [num_to_masks[layer][i][0] for i in range(len(num_to_masks[layer]))]

                hook.knockout_indices = source_indices
                hook.affected_outputs = target_indices
                hook.feature_mask = self._get_feature_mask(self.get_layer_moi(layer), feature_category)

        with torch.no_grad():
            out = self.model(input_ids)

        logits = out.logits
        probs = F.softmax(logits, dim=-1)

        return probs[:, -1, :].detach().cpu().numpy()  # type: ignore


class LlamaInterface(ModelInterface):
    model: TLlamaModel

    def __init__(
        self,
        model_arch: MODEL_ARCH,
        model_size: TModelSize,
        device: Optional[TDevice] = None,
        tokenizer: Optional[TTokenizer] = None,
    ):
        super().__init__(model_arch, model_size, device, tokenizer)

        self.knockouts: list[LlamaAttentionKnockout] = []
        self.handles: list[torch.nn.Module] = []
        self.knockout_mode = KnockoutMode.ZERO_ATTENTION

    def setup(self, layers: Optional[Iterable[TLayerIndex]] = None):
        super().setup(layers)

        for handle in self.handles:
            handle.self_attn = handle.self_attn.inner  # type: ignore

        # Assert that no hooks are left
        for m in self.model.modules():
            assert len(list(m._forward_hooks.items())) == 0
            assert not isinstance(m, LlamaAttentionKnockout)

        self.handles = []
        self.knockouts = []

        if layers is not None:
            # set up hooks
            for i in range(len(self.model.model.layers)):
                if i in layers:
                    # "mixer of interest" - moi
                    moi = self.model.model.layers[i].self_attn
                    assert (
                        isinstance(moi, LlamaAttention)
                        or isinstance(moi, MistralAttention)
                        or isinstance(moi, Qwen2Attention)
                    )
                    knockout = LlamaAttentionKnockout(moi)
                    knockout.eval()
                    knockout.to(next(moi.parameters()).device)
                    self.model.model.layers[i].self_attn = knockout

                    self.knockouts.append(knockout)
                    self.handles.append(self.model.model.layers[i])

    def generate_logits(
        self,
        input_ids: Tensor,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        feature_category: FeatureCategory = FeatureCategory.ALL,
    ) -> torch.Tensor:
        if num_to_masks is not None:
            for layer, hook in zip(num_to_masks, self.knockouts):
                hook.knockout_mask = num_to_masks[layer]

        with torch.no_grad():
            out = self.model(input_ids)

        logits = out.logits
        # print(logits, logits.max(dim=-1))
        probs = F.softmax(logits, dim=-1)
        probs = probs[:, -1, :].detach().cpu().numpy()  # type: ignore

        return probs  # type: ignore

    def n_layers(self) -> int:
        return len(self.model.model.layers)


class Mamba2Interface(MambaInterface):
    model: TMamba2Model  # type:ignore[valid-type]

    def __init__(
        self,
        model_size: TModelSize,
        device: Optional[torch.device] = None,
        tokenizer: Optional[TTokenizer] = None,
    ):
        super().__init__(MODEL_ARCH.MAMBA2, model_size, device, tokenizer)
        self.feature_masks = {}

    def _get_feature_mask(self, layer: torch.nn.Module, feature_category: FeatureCategory) -> Tensor:
        assert isinstance(layer.A_log, torch.Tensor)

        if feature_category == FeatureCategory.ALL:
            if layer not in self.feature_masks:
                self.feature_masks[layer] = torch.zeros(layer.A_log.shape[0]).to(layer.A_log.device)
            return self.feature_masks[layer]

        # if feature_category == FeatureCategory.NONE:
        #     return torch.ones(layer.A_log.shape[0])

        decay_matrices = torch.exp(-torch.exp(layer.A_log)).unsqueeze(-1)
        n_ssms = decay_matrices.shape[0]

        # get the norms
        norms = torch.norm(decay_matrices, p=1, dim=1)

        sorted_indices = torch.argsort(norms, descending=(feature_category == FeatureCategory.SLOW_DECAY))
        mask = torch.zeros_like(norms, dtype=torch.bool)
        mask[sorted_indices[: n_ssms // 3]] = True
        return mask

    def generate_logits(
        self,
        input_ids: Tensor,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        feature_category: FeatureCategory = FeatureCategory.ALL,
    ) -> torch.Tensor:
        self.setup(num_to_masks)

        feature_masks = {}
        if num_to_masks is not None:
            for layer in num_to_masks:
                feature_masks[layer] = self._get_feature_mask(self.get_layer_moi(layer), feature_category).to(
                    self.model.device
                )

        with torch.no_grad():
            out = self.model.generate_single(
                input_ids=input_ids,  # type: ignore
                max_new_length=input_ids.shape[1] + 1,
                temperature=1.0,
                top_k=0,
                top_p=1,
                num_to_masks=num_to_masks,
                attention=((num_to_masks is not None) or (feature_category != FeatureCategory.ALL)),
                feature_mask=feature_masks,
            )

        return out[-1].detach().cpu().numpy()  # type: ignore

    def get_layer_moi(self, layer_i: int) -> Mamba2:
        # "mixer of interest" - moi
        moi = self.layers[layer_i].mixer
        assert isinstance(moi, Mamba2)
        return moi


class GPT2Interface(ModelInterface):
    model: TGP2Model

    def __init__(
        self,
        model_size: TModelSize,
        device: Optional[torch.device] = None,
        tokenizer: Optional[TTokenizer] = None,
    ):
        super().__init__(MODEL_ARCH.GPT2, model_size, device, tokenizer)

        self.knockout_mode = KnockoutMode.ZERO_ATTENTION

    def _trace_with_attn_block(
        self,
        model,
        inp,
        from_to_index_per_layer,  # A list of (source index, target index) to block
    ):
        with torch.no_grad():
            # set hooks
            block_attn_hooks = gpt2_knockout_utils.set_block_attn_hooks(model, from_to_index_per_layer)

            # get prediction
            outputs_exp = model(**inp)

            # remove hooks
            gpt2_knockout_utils.remove_wrapper(model, block_attn_hooks)

        probs = torch.softmax(outputs_exp.logits[:, -1, :], dim=-1)

        return probs

    def generate_logits(
        self,
        input_ids: Tensor,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        feature_category: FeatureCategory = FeatureCategory.ALL,
    ) -> torch.Tensor:
        assert feature_category == FeatureCategory.ALL, "GPT2 does not support feature category"
        assert input_ids.shape[0] == 1
        num_to_masks = num_to_masks or {}
        max_len = input_ids.shape[1]
        attention_mask = [[1] * max_len]
        inp = dict(
            input_ids=input_ids.to(self.model.device),
            attention_mask=torch.tensor(attention_mask).to(self.model.device),
        )

        probs = self._trace_with_attn_block(self.model, inp, num_to_masks)

        return probs.detach().cpu().numpy()  # type: ignore

    def n_layers(self) -> int:
        return len(self.model.transformer.h)


MODEL_INTERFACES_CACHE: dict[MODEL_ARCH_AND_SIZE, ModelInterface] = {}


def get_model_interface(
    model_arch_and_size: MODEL_ARCH_AND_SIZE, device: Optional[torch.device] = None
) -> ModelInterface:
    key = model_arch_and_size
    if key in MODEL_INTERFACES_CACHE:
        return MODEL_INTERFACES_CACHE[key]

    model_interface: Optional[ModelInterface] = None
    match model_arch_and_size.arch:
        case MODEL_ARCH.MAMBA2:
            model_interface = Mamba2Interface(model_arch_and_size.size, device)
        case MODEL_ARCH.MAMBA1:
            model_interface = Mamba1Interface(
                model_arch_and_size.size, device, is_falcon=is_falcon(model_arch_and_size.size)
            )
        case MODEL_ARCH.GPT2:
            model_interface = GPT2Interface(model_arch_and_size.size, device)
        # TO ADD AN ARCH:
        case (
            MODEL_ARCH.LLAMA2
            | MODEL_ARCH.LLAMA3_2
            | MODEL_ARCH.LLAMA3
            | MODEL_ARCH.MISTRAL0_1
            | MODEL_ARCH.MISTRAL0_3
            | MODEL_ARCH.QWEN2
            | MODEL_ARCH.QWEN2_5
        ):
            # For LLAMA3, we use the same model interface as LLAMA2
            # because the architecture is the same.
            # Same goes for Mistral
            model_interface = LlamaInterface(model_arch_and_size.arch, model_arch_and_size.size, device)

        case _:
            assert_never(model_arch_and_size.arch)

    MODEL_INTERFACES_CACHE[key] = model_interface
    return model_interface
