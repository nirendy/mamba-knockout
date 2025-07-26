import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Sequence, assert_never

from src.core.names import (
    COLS,
    VARIANT_PARAM_NAME,
    BaseVariantParamName,
    DatasetName,
    ExperimentName,
    InfoFlowVariantParam,
    ToClassifyNames,
    WindowedVariantParam,
)
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    MODEL_SIZE_CAT,
    FeatureCategory,
    TCodeVersionName,
    TDatasetID,
    TLineStyle,
    TModelID,
    TModelSize,
    TokenType,
    TWindowSize,
)
from src.utils.infra.output_path import OutputKey
from src.utils.infra.slurm import SLURM_GPU_TYPE
from src.utils.types_utils import str_enum_values, subset_dict_by_keys

prev_umask = os.umask(0o002)  # Set umask to 0o002


# print(f"Previous umask: {prev_umask:03o}")


class C_ACTIVE_USERS(StrEnum):
    nirendy = "nirendy"
    idangrosbard = "idangrosbard"
    other = "other"


ACTIVE_USER = C_ACTIVE_USERS.other
if env_user := os.environ.get("USER"):
    ACTIVE_USER = C_ACTIVE_USERS(env_user)


@dataclass
class PathsConfig:
    """Configuration for project paths that can be easily mocked."""

    PROJECT_DIR: Path

    def __hash__(self) -> int:
        return hash(str(self.PROJECT_DIR))

    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_DIR / "data"

    @property
    def RAW_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def PREPROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "preprocessed"

    def dataset_dir(self, dataset_name: DatasetName) -> Path:
        return self.PREPROCESSED_DATA_DIR / dataset_name

    @property
    def DATA_SHARED_DIR(self) -> Path:
        return self.PROJECT_DIR / "shared"

    @property
    def RUNS_DIR(self) -> Path:
        return self.PROJECT_DIR / "runs"

    @property
    def TENSORBOARD_DIR(self) -> Path:
        return self.PROJECT_DIR / "tensorboard"

    @property
    def RESULTS_DIR(self) -> Path:
        return self.PROJECT_DIR / "results"

    @property
    def OUTPUT_DIR(self) -> Path:
        return self.PROJECT_DIR / "output"

    @property
    def SLURM_DIR(self) -> Path:
        return self.PROJECT_DIR / "slurm"

    @property
    def FINAL_PLOTS_DIR(self) -> Path:
        return self.PROJECT_DIR / "final_plots"

    def get_slurm_job_log_folder(self, job_name: str, job_id: str) -> Path:
        return self.SLURM_DIR / job_name / f"{job_id}"

    def get_slurm_job_submission_file_path(self, job_name: str, job_id: str) -> Path:
        return self.get_slurm_job_log_folder(job_name, job_id) / "experiment_variation_base_path"


# Global instance
PATHS = PathsConfig(PROJECT_DIR=Path(__file__).parent.parent.parent.resolve())


@dataclass
class RunnerPaths:
    variation_base_path: Path

    @property
    def running_history_path(self) -> Path:
        return self.variation_base_path / "running_history"

    @property
    def plots_path(self) -> Path:
        return self.variation_base_path / "plots"

    @property
    def outputs_path(self) -> Path:
        return self.variation_base_path / "outputs"

    @property
    def slurm_logs_path(self) -> Path:
        return self.variation_base_path / "slurm_logs"

    @property
    def intermediate_outputs(self) -> Path:
        return self.variation_base_path / "intermediate_outputs"

    def running_history_json_path(self, run_id: str) -> Path:
        return self.running_history_path / f"{run_id}.json"

    def slurm_log_folder(self, job_id: str) -> Path:
        return self.slurm_logs_path / f"{job_id}"


class ENV_VARS:
    MASTER_PORT = "MASTER_PORT"
    MASTER_ADDR = "MASTER_ADDR"


class FORMATS:
    TIME_WITH_MICROSECONDS = "%Y-%m-%d_%H-%M-%S_%f"
    LOGGER_FORMAT = "%(asctime)s - %(message)s"


class DDP:
    MASTER_PORT = os.environ.get(ENV_VARS.MASTER_PORT, "12355")
    MASTER_ADDR = "localhost"
    BACKEND = "nccl"
    SHUFFLE = True
    DROP_LAST = True
    NUM_WORKERS = 0


# TO ADD AN ARCH:
MODEL_SIZES_PER_ARCH_TO_MODEL_ID: dict[MODEL_ARCH, dict[TModelSize, TModelID]] = {
    MODEL_ARCH.MAMBA1: {
        TModelSize("130M"): TModelID("state-spaces/mamba-130M-hf"),
        TModelSize("1.4B"): TModelID("state-spaces/mamba-1.4B-hf"),
        TModelSize("2.8B"): TModelID("state-spaces/mamba-2.8B-hf"),
        TModelSize("7B"): TModelID("TRI-ML/mamba-7b-rw"),
        TModelSize("7B-falcon"): TModelID("tiiuae/falcon-mamba-7b"),
        TModelSize("7B-falcon-base"): TModelID("tiiuae/Falcon3-Mamba-7B-Base"),
    },
    MODEL_ARCH.MAMBA2: {
        TModelSize("130M"): TModelID("state-spaces/mamba2-130M"),
        TModelSize("1.3B"): TModelID("state-spaces/mamba2-1.3b"),
        TModelSize("2.7B"): TModelID("state-spaces/mamba2-2.7B"),
        # TModelSize("8B"): TModelID("nvidia/mamba2-8b-3t-4k"),
    },
    MODEL_ARCH.LLAMA2: {
        TModelSize("7B"): TModelID("meta-llama/Llama-2-7b-hf"),
        TModelSize("13B"): TModelID("meta-llama/Llama-2-13b-hf"),
    },
    MODEL_ARCH.LLAMA3: {
        TModelSize("8B"): TModelID("meta-llama/Meta-Llama-3-8B"),
    },
    MODEL_ARCH.LLAMA3_2: {
        TModelSize("1B"): TModelID("meta-llama/Llama-3.2-1B"),
        TModelSize("3B"): TModelID("meta-llama/Llama-3.2-3B"),
    },
    MODEL_ARCH.MISTRAL0_1: {
        TModelSize("7B"): TModelID("mistralai/Mistral-7B-v0.1"),
    },
    MODEL_ARCH.MISTRAL0_3: {
        TModelSize("7B"): TModelID("mistralai/Mistral-7B-v0.3"),
    },
    MODEL_ARCH.QWEN2: {
        TModelSize("0.5B"): TModelID("Qwen/Qwen2-0.5B"),
        TModelSize("1.5B"): TModelID("Qwen/Qwen2-1.5B"),
        TModelSize("7B"): TModelID("Qwen/Qwen2-7B"),
    },
    MODEL_ARCH.QWEN2_5: {
        TModelSize("0.5B"): TModelID("Qwen/Qwen2.5-0.5B"),
        TModelSize("1.5B"): TModelID("Qwen/Qwen2.5-1.5B"),
        TModelSize("3B"): TModelID("Qwen/Qwen2.5-3B"),
        TModelSize("7B"): TModelID("Qwen/Qwen2.5-7B"),
        TModelSize("14B"): TModelID("Qwen/Qwen2.5-14B"),
    },
    MODEL_ARCH.GPT2: {
        TModelSize("124M"): TModelID("openai-community/gpt2"),
        TModelSize("355M"): TModelID("openai-community/gpt2-medium"),
        TModelSize("774M"): TModelID("openai-community/gpt2-large"),
        TModelSize("1.5B"): TModelID("openai-community/gpt2-xl"),
    },
}


class ARCH_FAMILY(StrEnum):
    MAMBA1 = "mamba1"
    MAMBA2 = "mamba2"
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    GPT2 = "gpt2"
    MAMBA = "mamba"
    TRANSFORMER = "transformer"


MODEL_FAMILY_TO_ARCH: dict[ARCH_FAMILY, list[MODEL_ARCH]] = {
    ARCH_FAMILY.MAMBA1: [MODEL_ARCH.MAMBA1],
    ARCH_FAMILY.MAMBA2: [MODEL_ARCH.MAMBA2],
    ARCH_FAMILY.LLAMA: [MODEL_ARCH.LLAMA2, MODEL_ARCH.LLAMA3, MODEL_ARCH.LLAMA3_2],
    ARCH_FAMILY.MISTRAL: [MODEL_ARCH.MISTRAL0_1, MODEL_ARCH.MISTRAL0_3],
    ARCH_FAMILY.QWEN: [MODEL_ARCH.QWEN2, MODEL_ARCH.QWEN2_5],
    ARCH_FAMILY.GPT2: [MODEL_ARCH.GPT2],
    ARCH_FAMILY.MAMBA: [MODEL_ARCH.MAMBA1, MODEL_ARCH.MAMBA2],
    ARCH_FAMILY.TRANSFORMER: [
        MODEL_ARCH.LLAMA2,
        MODEL_ARCH.LLAMA3,
        MODEL_ARCH.LLAMA3_2,
        MODEL_ARCH.GPT2,
        MODEL_ARCH.MISTRAL0_1,
        MODEL_ARCH.MISTRAL0_3,
    ],
}

GRAPHS_ORDER: dict[MODEL_ARCH_AND_SIZE, MODEL_SIZE_CAT] = {
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.GPT2, "124M"): MODEL_SIZE_CAT.SMALL,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("130M")): MODEL_SIZE_CAT.SMALL,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA2, TModelSize("130M")): MODEL_SIZE_CAT.SMALL,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.GPT2, TModelSize("355M")): MODEL_SIZE_CAT.SMALL,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.GPT2, TModelSize("774M")): MODEL_SIZE_CAT.MEDIUM,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.LLAMA3_2, TModelSize("1B")): MODEL_SIZE_CAT.MEDIUM,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("1.4B")): MODEL_SIZE_CAT.MEDIUM,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA2, TModelSize("1.3B")): MODEL_SIZE_CAT.MEDIUM,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.GPT2, TModelSize("1.5B")): MODEL_SIZE_CAT.LARGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("2.8B")): MODEL_SIZE_CAT.LARGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA2, TModelSize("2.7B")): MODEL_SIZE_CAT.LARGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.LLAMA3_2, TModelSize("3B")): MODEL_SIZE_CAT.LARGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("7B")): MODEL_SIZE_CAT.HUGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("7B-falcon")): MODEL_SIZE_CAT.HUGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("7B-falcon-base")): MODEL_SIZE_CAT.HUGE,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.LLAMA2, TModelSize("7B")): MODEL_SIZE_CAT.HUGE,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.LLAMA2, TModelSize("13B")): MODEL_SIZE_CAT.HUGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MISTRAL0_1, TModelSize("7B")): MODEL_SIZE_CAT.HUGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MISTRAL0_3, TModelSize("7B")): MODEL_SIZE_CAT.HUGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.LLAMA3, TModelSize("8B")): MODEL_SIZE_CAT.HUGE,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.QWEN2, TModelSize("0.5B")): MODEL_SIZE_CAT.SMALL,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.QWEN2, TModelSize("1.5B")): MODEL_SIZE_CAT.MEDIUM,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.QWEN2, TModelSize("7B")): MODEL_SIZE_CAT.HUGE,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.QWEN2_5, TModelSize("0.5B")): MODEL_SIZE_CAT.SMALL,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.QWEN2_5, TModelSize("1.5B")): MODEL_SIZE_CAT.MEDIUM,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.QWEN2_5, TModelSize("3B")): MODEL_SIZE_CAT.LARGE,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.QWEN2_5, TModelSize("7B")): MODEL_SIZE_CAT.HUGE,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.QWEN2_5, TModelSize("14B")): MODEL_SIZE_CAT.HUGE,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.LLAMA2, TModelSize("13B")): MODEL_SIZE_CAT.HUGE,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA2, "8B"): MODEL_SIZE_CAT.HUGE,
}


ALL_IMPORTANT_MODELS: dict[MODEL_ARCH_AND_SIZE, MODEL_SIZE_CAT] = subset_dict_by_keys(
    GRAPHS_ORDER,
    [
        # MODEL_ARCH_AND_SIZE(MODEL_ARCH.GPT2, TModelSize("774M")),
        # MODEL_ARCH_AND_SIZE(MODEL_ARCH.LLAMA3_2, TModelSize("1B")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("1.4B")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA2, TModelSize("1.3B")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.GPT2, TModelSize("1.5B")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("2.8B")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA2, TModelSize("2.7B")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.LLAMA3_2, TModelSize("3B")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("7B")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("7B-falcon")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("7B-falcon-base")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.MISTRAL0_1, TModelSize("7B")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.MISTRAL0_3, TModelSize("7B")),
        MODEL_ARCH_AND_SIZE(MODEL_ARCH.LLAMA3, TModelSize("8B")),
        # MODEL_ARCH_AND_SIZE(MODEL_ARCH.QWEN2, TModelSize("0.5B")),
        # MODEL_ARCH_AND_SIZE(MODEL_ARCH.QWEN2_5, TModelSize("0.5B")),
        # MODEL_ARCH_AND_SIZE(MODEL_ARCH.QWEN2_5, TModelSize("7B")),
    ],
)


def get_model_by_cat_size(cat_size: MODEL_SIZE_CAT) -> list[MODEL_ARCH_AND_SIZE]:
    return [
        model_arch_and_size
        for model_arch_and_size, model_size_cat in GRAPHS_ORDER.items()
        if model_size_cat == cat_size
    ]


def reverse_model_id(model_id: TModelID) -> MODEL_ARCH_AND_SIZE:
    for model_arch_and_size in GRAPHS_ORDER.keys():
        for model_id_prefix in ["", "state-spaces/", "tiiuae/"]:
            if (
                MODEL_SIZES_PER_ARCH_TO_MODEL_ID[model_arch_and_size.arch][model_arch_and_size.size]
                == f"{model_id_prefix}{model_id}"
            ):
                return model_arch_and_size
    raise ValueError(f"Model id {model_id} not found in MODEL_SIZES_PER_ARCH_TO_MODEL_ID")


def model_and_size_to_slurm_gpu_type(
    model_arch_and_size: MODEL_ARCH_AND_SIZE,
) -> SLURM_GPU_TYPE:
    model_cat_size = GRAPHS_ORDER[model_arch_and_size]
    match model_cat_size:
        case MODEL_SIZE_CAT.SMALL | MODEL_SIZE_CAT.MEDIUM:
            return SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN
        case MODEL_SIZE_CAT.LARGE | MODEL_SIZE_CAT.HUGE:
            match ACTIVE_USER:
                case C_ACTIVE_USERS.nirendy:
                    return SLURM_GPU_TYPE.L40S
                case C_ACTIVE_USERS.idangrosbard:
                    return SLURM_GPU_TYPE.H100
                case _:
                    raise NotImplementedError(f"No SLURM GPU type for user {ACTIVE_USER}")
        case _:
            assert_never(model_cat_size)


def is_llama(model_arch: MODEL_ARCH) -> bool:
    return model_arch in [
        MODEL_ARCH.LLAMA2,
        MODEL_ARCH.LLAMA3,
        MODEL_ARCH.LLAMA3_2,
        MODEL_ARCH.MISTRAL0_1,
        MODEL_ARCH.MISTRAL0_3,
        MODEL_ARCH.QWEN2,
        MODEL_ARCH.QWEN2_5,
    ]


def is_mamba_arch(model_arch: MODEL_ARCH) -> bool:
    return model_arch in [MODEL_ARCH.MAMBA1, MODEL_ARCH.MAMBA2]


def is_falcon(model_size: str) -> bool:
    return "falcon" in model_size


DATASETS_IDS: dict[DatasetName, TDatasetID] = {DatasetName.counter_fact: TDatasetID("NeelNanda/counterfact-tracing")}  # type: ignore

COUNTER_FACT_2_KNOWN1000_COL_CONV = {
    COLS.COUNTER_FACT.TARGET_TRUE: "attribute",
}

TOKEN_TYPE_COLORS: dict[TokenType, str] = {
    TokenType.first: "#0000FF",  # blue
    TokenType.last: "#D2691E",  # orange
    TokenType.subject: "#008000",  # green
    TokenType.relation: "#800080",  # purple
    TokenType.context: "#FF0000",  # red
    TokenType.all: "#000000",  # black
    TokenType.relation_minus_last: "#800000",  # maroon
}

TOKEN_TYPE_LINE_STYLES: dict[FeatureCategory, TLineStyle] = {
    # Options: "-", ":", "--", "-.", "-.-", "-.-."
    FeatureCategory.ALL: TLineStyle.solid,
    FeatureCategory.FAST_DECAY: TLineStyle.dashdot,
    FeatureCategory.SLOW_DECAY: TLineStyle.dash,
}

CONVERT_TO_PLOTLY_LINE_STYLE = {
    **{line_style: line_style.name for line_style in TLineStyle},
}


def format_params_for_title(params: dict) -> str:
    """Format parameters for title display in a consistent order."""

    parts_remaining = set(params.keys())
    ordered_parts = []
    for param in BaseVariantParamName:
        if param in parts_remaining:
            parts_remaining.remove(param)
            match param:
                case BaseVariantParamName.experiment_name | ToClassifyNames.code_version:
                    ordered_parts.append(params[param])
                case BaseVariantParamName.model_arch:
                    if BaseVariantParamName.model_size in parts_remaining:
                        ordered_parts.append(
                            f"{params[BaseVariantParamName.model_arch]} {params[BaseVariantParamName.model_size]}"
                        )
                        parts_remaining.remove(BaseVariantParamName.model_size)
                    else:
                        ordered_parts.append(params[BaseVariantParamName.model_arch])
                case WindowedVariantParam.window_size:
                    ordered_parts.append(f"ws={params[WindowedVariantParam.window_size]}")
                case InfoFlowVariantParam.source:
                    base_str = f"From {params[InfoFlowVariantParam.source]}"
                    if InfoFlowVariantParam.feature_category in parts_remaining:
                        if params[InfoFlowVariantParam.feature_category] is not None:
                            base_str = f"{base_str} - {params[InfoFlowVariantParam.feature_category]}"
                        parts_remaining.remove(InfoFlowVariantParam.feature_category)
                    ordered_parts.append(base_str)
                case _:
                    ordered_parts.append(f"{param}={params[param]}")

    for param in parts_remaining:
        ordered_parts.append(f"{param}={params[param]}")

    return " | ".join(ordered_parts)


class BASE_OUTPUT_KEYS:
    MODEL_ARCH = OutputKey[MODEL_ARCH]("model_arch", key_display_name="arch=")
    MODEL_SIZE = OutputKey[TModelSize]("model_size", key_display_name="size=")
    CODE_VERSION = OutputKey[TCodeVersionName]("code_version", key_display_name="v=")
    EXPERIMENT_NAME = OutputKey[ExperimentName]("experiment_name", key_display_name="")
    DATASET_NAME = OutputKey[DatasetName]("dataset_name", key_display_name="ds=")
    WINDOW_SIZE = OutputKey[TWindowSize]("window_size", key_display_name="ws=")


ALL_VARIANT_PARAMETERS: Sequence[VARIANT_PARAM_NAME] = (
    str_enum_values(BaseVariantParamName)
    + str_enum_values(WindowedVariantParam)
    + str_enum_values(InfoFlowVariantParam)
)

DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION = TCodeVersionName("v1")
DEFAULT_MODEL_CORRECT_DATASET_NAME = DatasetName.counter_fact
