from enum import StrEnum
from typing import Literal, Sequence, Union

from src.utils.types_utils import literal_guard, str_enum_values


class ToClassifyNames(StrEnum):
    dataset_name = "dataset_name"
    code_version = "code_version"
    prompt_filteration = "prompt_filteration"


class DatasetName(StrEnum):
    counter_fact = "counter_fact"


class RunnerParamName(StrEnum):
    variant_params = "variant_params"
    input_params = "input_params"
    metadata_params = "metadata_params"


class BaseVariantParamName(StrEnum):
    experiment_name = "experiment_name"
    model_arch = "model_arch"
    model_size = "model_size"


class WindowedVariantParam(StrEnum):
    window_size = "window_size"


class EvaluateVariantParam(StrEnum):
    pass


class InfoFlowVariantParam(StrEnum):
    source = "source"
    feature_category = "feature_category"
    target = "target"


class HeatmapVariantParam(StrEnum):
    pass


VARIANT_PARAM_NAME = Union[
    BaseVariantParamName, WindowedVariantParam, EvaluateVariantParam, InfoFlowVariantParam, HeatmapVariantParam
]


class ExperimentName(StrEnum):
    evaluate_model = "evaluate_model"
    info_flow = "info_flow"
    heatmap = "heatmap"
    full_pipeline = "full_pipeline"

    @staticmethod
    def get_variant_cols(experiment_name: "ExperimentName") -> Sequence[VARIANT_PARAM_NAME]:
        base_cols: list[VARIANT_PARAM_NAME] = str_enum_values(BaseVariantParamName)
        match experiment_name:
            case ExperimentName.info_flow:
                return base_cols + str_enum_values(WindowedVariantParam) + str_enum_values(InfoFlowVariantParam)
            case ExperimentName.heatmap:
                return base_cols + str_enum_values(WindowedVariantParam) + str_enum_values(HeatmapVariantParam)
            case ExperimentName.evaluate_model:
                return base_cols + str_enum_values(EvaluateVariantParam)
            case _:
                raise ValueError(f"Experiment name {experiment_name} is not implemented")


class COLS:
    # Preprocessing
    ORIGINAL_IDX: Literal["original_idx"] = "original_idx"
    SPLIT: Literal["split"] = "split"

    # Counter Fact
    class COUNTER_FACT(StrEnum):
        PROMPT = "prompt"
        TARGET_TRUE = "target_true"
        RELATION = "relation"
        SUBJECT = "subject"
        TARGET_FALSE = "target_false"
        RELATION_PREFIX = "relation_prefix"
        RELATION_SUFFIX = "relation_suffix"
        TARGET_TRUE_ID = "target_true_id"
        TARGET_FALSE_ID = "target_false_id"
        RELATION_ID = "relation_id"

    # Evaluate Model
    class EVALUATE_MODEL(StrEnum):
        TARGET_PROBS = "target_probs"
        MODEL_TOP_OUTPUT_CONFIDENCE = "model_top_output_confidence"
        MODEL_CORRECT = "model_correct"
        MODEL_OUTPUT = "model_output"
        TARGET_RANK = "target_rank"
        MODEL_TOP_OUTPUTS = "model_top_outputs"
        MODEL_GENERATION = "model_generation"
        TARGET_TOKENS = "target_tokens"

    # Info Flow
    class INFO_FLOW(StrEnum):
        HIT = "hit"
        TRUE_PROBS = "true_probs"
        DIFFS = "diffs"


class EvaluateModelMetricName(StrEnum):
    target_probs = COLS.EVALUATE_MODEL.TARGET_PROBS
    model_top_output_confidence = COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUT_CONFIDENCE
    model_correct = COLS.EVALUATE_MODEL.MODEL_CORRECT
    model_output = COLS.EVALUATE_MODEL.MODEL_OUTPUT
    target_rank = COLS.EVALUATE_MODEL.TARGET_RANK
    model_top_outputs = COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUTS
    model_generation = COLS.EVALUATE_MODEL.MODEL_GENERATION


class InfoFlowMetricName:
    hit: Literal["hit"] = literal_guard(COLS.INFO_FLOW.HIT, "hit")
    diffs: Literal["diffs"] = literal_guard(COLS.INFO_FLOW.DIFFS, "diffs")
    true_probs: Literal["true_probs"] = literal_guard(COLS.INFO_FLOW.TRUE_PROBS, "true_probs")


class ResultBankParamNames(StrEnum):
    code_version = ToClassifyNames.code_version
    path = "path"


class HeatmapCols:
    PROMPT_COUNT = "Prompt Count"
    SELECTED_PROMPT = "Selected Prompt"
    MODEL_CORRECT = "Model Correct"


class FinalPlotsPlanOrientation(StrEnum):
    grids = "grids"
    rows = "rows"
    cols = "cols"
    lines = "lines"


class SummarizedDataFulfilledReqsCols:
    AvailableOptions = "Available Options"
    Options = "Options"
    Key = "Key"
    filters_requested = "filters_requested"


class ModelCombinationCols(StrEnum):
    correct_models = "correct_models"
    incorrect_models = "incorrect_models"
    prompts = "prompts"
    chosen_prompt = "chosen_prompt"


class SlurmStatus(StrEnum):
    NOT_SUBMITTED = "NOT_SUBMITTED"
    RUNNING = "RUNNING"
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def scheduled(self) -> bool:
        return self in [self.PENDING, self.RUNNING]


class RunningHistoryCols(StrEnum):
    run_id = "run_id"
    git_commit_hash = "git_commit_hash"


class PAGE_ORDER(StrEnum):
    HOME = "home"
    RESULTS_BANK = "results_bank"
    DATA_REQUIREMENTS = "data_requirements"
    HEATMAP_CREATION = "heatmap_creation"
    INFO_FLOW_ANALYSIS = "info_flow_analysis"
    FINAL_PLOTS = "final_plots"
    PROMPTS_COMPARISON = "prompts_comparison"
    MAMBA_ANALYSIS = "mamba_analysis"
    PROMPT_FILTERATION_PRESETS = "prompt_filteration_presets"
