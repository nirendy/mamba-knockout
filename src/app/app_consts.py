from enum import Enum, auto
from typing import Literal, Union, assert_never

from src.app.texts import (
    DATA_REQUIREMENTS_TEXTS,
    FINAL_PLOTS_TEXTS,
    HEATMAP_TEXTS,
    HOME_TEXTS,
    INFO_FLOW_ANALYSIS_TEXTS,
    MAMBA_ANALYSIS_TEXTS,
    PROMPT_FILTERATION_PRESETS_TEXTS,
    PROMPTS_COMPARISON_TEXTS,
    RESULTS_BANK_TEXTS,
)
from src.core.consts import ALL_VARIANT_PARAMETERS, GRAPHS_ORDER, model_and_size_to_slurm_gpu_type
from src.core.names import COLS, SummarizedDataFulfilledReqsCols
from src.core.types import MODEL_ARCH_AND_SIZE, TCodeVersionName, TWindowSize
from src.utils.infra.slurm import SLURM_GPU_TYPE
from src.utils.streamlit.helpers.session_keys import SessionKeyDescriptor, SessionKeysBase

# region Global App constants


class GLOBAL_APP_CONSTS:
    DEFAULT_CODE_VERSION = TCodeVersionName("v1")
    DEFAULT_WINDOW_SIZE = TWindowSize(9)
    DEFAULT_SEED = 42
    MODELS_COMBINATIONS = list(GRAPHS_ORDER.keys())
    PROMPT_RELATED_COLUMNS = [
        COLS.COUNTER_FACT.PROMPT,
        COLS.COUNTER_FACT.TARGET_TRUE,
        COLS.COUNTER_FACT.TARGET_FALSE,
        COLS.COUNTER_FACT.SUBJECT,
        COLS.COUNTER_FACT.TARGET_FALSE_ID,
        COLS.COUNTER_FACT.RELATION,
    ]

    MODEL_EVALS_COLUMNS = [
        COLS.ORIGINAL_IDX,
        COLS.COUNTER_FACT.PROMPT,
        COLS.COUNTER_FACT.TARGET_TRUE,
        COLS.EVALUATE_MODEL.TARGET_PROBS,
        COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUT_CONFIDENCE,
        COLS.EVALUATE_MODEL.MODEL_CORRECT,
        COLS.EVALUATE_MODEL.MODEL_OUTPUT,
        COLS.EVALUATE_MODEL.TARGET_RANK,
        COLS.EVALUATE_MODEL.MODEL_GENERATION,
        COLS.EVALUATE_MODEL.TARGET_TOKENS,
        COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUTS,
    ]


class _AppSessionKeys(SessionKeysBase["_AppSessionKeys"]):
    # Each descriptor creates a SessionKey with the class name prefix
    code_version = SessionKeyDescriptor[TCodeVersionName](GLOBAL_APP_CONSTS.DEFAULT_CODE_VERSION)
    _selected_gpu = SessionKeyDescriptor[Union[SLURM_GPU_TYPE, Literal["smart"]]]("smart")
    window_size = SessionKeyDescriptor[TWindowSize](GLOBAL_APP_CONSTS.DEFAULT_WINDOW_SIZE)

    def get_selected_gpu(self, model_arch_and_size: MODEL_ARCH_AND_SIZE) -> SLURM_GPU_TYPE:
        selected_gpu = self._selected_gpu.value
        if selected_gpu == "smart":
            return model_and_size_to_slurm_gpu_type(model_arch_and_size)
        return selected_gpu


AppSessionKeys = _AppSessionKeys()


# region Data Requirements


class DataReqConsts:
    # Data Requirements filter columns
    DATA_REQS_FILTER_COLUMNS = [SummarizedDataFulfilledReqsCols.AvailableOptions, *ALL_VARIANT_PARAMETERS]


# endregion


# region Heatmap Creation


class HeatmapConsts:
    MINIMUM_COMBINATIONS_FOR_FILTERING = 30


# endregion


class PAGE_ORDER(Enum):
    HOME = auto()
    RESULTS_BANK = auto()
    DATA_REQUIREMENTS = auto()
    HEATMAP = auto()
    INFO_FLOW_ANALYSIS = auto()
    FINAL_PLOTS = auto()
    PROMPTS_COMPARISON = auto()
    MAMBA_ANALYSIS = auto()
    PROMPT_FILTERATION_PRESETS = auto()

    @property
    def page_details(self) -> tuple[str, str]:
        match self:
            case PAGE_ORDER.HOME:
                return (HOME_TEXTS.title, HOME_TEXTS.icon)
            case PAGE_ORDER.HEATMAP:
                return (HEATMAP_TEXTS.title, HEATMAP_TEXTS.icon)
            case PAGE_ORDER.RESULTS_BANK:
                return (RESULTS_BANK_TEXTS.title, RESULTS_BANK_TEXTS.icon)
            case PAGE_ORDER.DATA_REQUIREMENTS:
                return (DATA_REQUIREMENTS_TEXTS.title, DATA_REQUIREMENTS_TEXTS.icon)
            case PAGE_ORDER.FINAL_PLOTS:
                return (FINAL_PLOTS_TEXTS.title, FINAL_PLOTS_TEXTS.icon)
            case PAGE_ORDER.INFO_FLOW_ANALYSIS:
                return (INFO_FLOW_ANALYSIS_TEXTS.title, INFO_FLOW_ANALYSIS_TEXTS.icon)
            case PAGE_ORDER.PROMPTS_COMPARISON:
                return (PROMPTS_COMPARISON_TEXTS.title, PROMPTS_COMPARISON_TEXTS.icon)
            case PAGE_ORDER.MAMBA_ANALYSIS:
                return (MAMBA_ANALYSIS_TEXTS.title, MAMBA_ANALYSIS_TEXTS.icon)
            case PAGE_ORDER.PROMPT_FILTERATION_PRESETS:
                return (PROMPT_FILTERATION_PRESETS_TEXTS.title, PROMPT_FILTERATION_PRESETS_TEXTS.icon)
            case _:
                assert_never(self)

    @property
    def icon(self) -> str:
        return self.page_details[1]

    @property
    def title(self) -> str:
        return self.page_details[0]
