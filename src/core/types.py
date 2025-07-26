from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, NewType, Sequence, TypeAlias, TypedDict, Union, assert_never

import pandas as pd
import torch
from jaxtyping import Float
from torch import Tensor

# TO ADD AN ARCH:
if TYPE_CHECKING:
    from transformers import (
        FalconMambaForCausalLM,
        GPT2LMHeadModel,
        LlamaForCausalLM,
        MambaForCausalLM,
        MistralForCausalLM,
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        Qwen2ForCausalLM,
    )

    import src.experiments.knockout.mamba.mamba2.minimal_mamba2 as minimal_mamba2

    TTokenizer: TypeAlias = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    TMamba1Model: TypeAlias = Union[MambaForCausalLM, FalconMambaForCausalLM]
    # TO ADD AN ARCH:
    TLlamaModel: TypeAlias = LlamaForCausalLM | MistralForCausalLM | Qwen2ForCausalLM
    # TMistralModel: TypeAlias = MistralForCausalLM
    TGP2Model: TypeAlias = GPT2LMHeadModel
    TMamba2Model: TypeAlias = minimal_mamba2.Mamba2LMHeadModel
    TModel: TypeAlias = Union[TMamba1Model, TGP2Model, TMamba2Model, PreTrainedModel, TLlamaModel]
else:
    TTokenizer = Any
    TMamba1Model = Any
    TGP2Model = Any
    TMamba2Model = Any
    TLlamaModel = Any
    # TMistralModel = Any
    TModel = Any


class SPLIT(StrEnum):
    TRAIN1 = "train1"
    TRAIN2 = "train2"
    TRAIN3 = "train3"
    TRAIN4 = "train4"
    TRAIN5 = "train5"
    TEST = "test"


ALL_SPLITS_LITERAL = "all"

TSplitChoise = Union[SPLIT, Sequence[SPLIT], Literal["all"]]


class MODEL_ARCH(StrEnum):
    # TO ADD AN ARCH:
    MAMBA1 = "mamba1"
    MAMBA2 = "mamba2"
    GPT2 = "gpt2"
    MISTRAL0_1 = "mistral0.1"
    MISTRAL0_3 = "mistral0.3"
    QWEN2 = "qwen2"
    QWEN2_5 = "qwen2.5"
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    LLAMA3_2 = "llama3.2"

    # TO ADD AN ARCH:
    @property
    def model_title(self) -> str:
        match self:
            case MODEL_ARCH.MAMBA1:
                return "Mamba1"
            case MODEL_ARCH.MAMBA2:
                return "Mamba2"
            case MODEL_ARCH.LLAMA2:
                return "Llama2"
            case MODEL_ARCH.LLAMA3_2:
                return "Llama3.2"
            case MODEL_ARCH.GPT2:
                return "GPT2"
            case MODEL_ARCH.LLAMA3:
                return "Llama3"
            case MODEL_ARCH.MISTRAL0_1:
                return "Mistral 0.1"
            case MODEL_ARCH.MISTRAL0_3:
                return "Mistral 0.3"
            case MODEL_ARCH.QWEN2:
                return "Qwen2"
            case MODEL_ARCH.QWEN2_5:
                return "Qwen2.5"
        assert_never(self.value)


class MODEL_SIZE_CAT(Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    HUGE = 3


TLayerIndex: TypeAlias = int
TTokenIndex: TypeAlias = int
TBatchSize = NewType("TBatchSize", int)
TModelID = NewType("TModelID", str)
TPlotID = NewType("TPlotID", str)
TPresetID = str
TDatasetID = NewType("TDatasetID", str)
TCodeVersionName = NewType("TCodeVersionName", str)
TModelSize = NewType("TModelSize", str)
TWindowSize = NewType("TWindowSize", int)
TWindowStartIndex = NewType("TWindowStartIndex", TLayerIndex)
TWindow = NewType("TWindow", list[TLayerIndex])
TPromptData = NewType("TPromptData", pd.DataFrame)
TPromptDataFlat = NewType("TPromptDataFlat", pd.DataFrame)
TNum2Mask = NewType("TNum2Mask", dict[TLayerIndex, list[tuple[TTokenIndex, TTokenIndex]]])
TPromptOriginalIndex = NewType("TPromptOriginalIndex", int)
TRowPosition = NewType("TRowPosition", int)
TDevice = Union[torch.device, str]
TErrorMessage = NewType("TErrorMessage", str)


class TokenType(StrEnum):
    first = "first"
    last = "last"
    subject = "subject"
    relation = "relation"
    context = "context"
    all = "all"
    relation_minus_last = "relation_minus_last"


TSSMState = Float[Tensor, "batch hidden_size ssm_dim"]
TSSMInput = Float[Tensor, "batch hidden_size seq_len"]
TSSM_A = Float[Tensor, "batch hidden_size seq_len ssm_dim"]
TSSM_B = Float[Tensor, "batch hidden_size seq_len ssm_dim"]
TSSM_Bu = Float[Tensor, "batch hidden_size seq_len ssm_dim"]
TSSM_C = Float[Tensor, "batch seq_len ssm_dim"]


class KnockoutMode(StrEnum):
    ZERO_ATTENTION = "ZERO_ATTENTION"
    ZERO_DELTA = "ZERO_DELTA"
    IGNORE_CONTEXT = "IGNORE_CONTEXT"
    ONLY_CONTEXT = "ONLY_CONTEXT"
    IGNORE_LAYER = "IGNORE_LAYER"
    IGNORE_SSM = "IGNORE_SSM"
    INCREASE_DELTA = "INCREASE_DELTA"


class FeatureCategory(StrEnum):
    ALL = "ALL"
    FAST_DECAY = "FAST_DECAY"
    SLOW_DECAY = "SLOW_DECAY"
    # NONE = "NONE"


class TInfoFlowWindowValue(TypedDict):
    hit: list[bool]
    true_probs: list[float]
    diffs: list[float]
    original_idx: list[TPromptOriginalIndex]


TInfoFlowOutputJSONOutput = dict[str, TInfoFlowWindowValue]
TInfoFlowOutput = dict[TLayerIndex, TInfoFlowWindowValue]
IHeatmap = pd.DataFrame


class MODEL_ARCH_AND_SIZE(NamedTuple):
    arch: MODEL_ARCH
    size: TModelSize

    @property
    def model_name(self) -> str:
        return f"{self.arch}-{self.size}"


class TLineStyle(StrEnum):
    solid = "-"
    dash = "--"
    dot = ":"
    dashdot = "-."
    dashdotdot = "-.-"
    dashdotdotdot = "-.-."
