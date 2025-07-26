"""Baseline builder for test pipeline experiments."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyrallis
import pytest
from datasets import DatasetDict

from src.analysis.experiment_results.helpers import serialize_result_bank
from src.analysis.experiment_results.results_bank import get_experiment_results_bank
from src.analysis.prompt_filterations import Correctness, ModelCorrectPromptFilteration
from src.core.consts import PathsConfig
from src.core.names import COLS
from src.core.types import (
    ALL_SPLITS_LITERAL,
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    SPLIT,
    FeatureCategory,
    TCodeVersionName,
    TModelSize,
    TokenType,
    TPromptOriginalIndex,
    TWindowSize,
)
from src.data_ingestion.datasets.download_dataset import DatasetName, load_splitted_counter_fact
from src.experiments.infrastructure.base_prompt_filteration import SelectivePromptFilteration
from src.experiments.infrastructure.base_runner import InputParams, MetadataParams
from src.experiments.runners.full_pipeline import FullPipelineParams, FullPipelineRunner

# Test configuration constants
BASELINES_DIR = Path(__file__).parent / "baselines"
TEST_BASE_PATH = BASELINES_DIR / "full_pipeline"

# Test model configurations - only test with subset
TEST_MODEL_CONFIGS = [
    (MODEL_ARCH.MAMBA1, "130M"),
    (MODEL_ARCH.MAMBA2, "130M"),
    (MODEL_ARCH.GPT2, "355M"),
    (MODEL_ARCH.LLAMA3_2, "1B"),
    (MODEL_ARCH.QWEN2, "0.5B"),
    (MODEL_ARCH.QWEN2_5, "0.5B"),
]

# Original IDs for filtering dataset to subset
ORIGINAL_IDS = cast(
    dict[SPLIT, list[TPromptOriginalIndex]],
    {
        SPLIT.TRAIN1: [
            53,
            59,
            74,
            90,
            93,
        ],
        SPLIT.TRAIN2: [
            10594,
            6410,
            140,
            148,
            159,
            182,
        ],
    },
)

# Hardcoded code paths for monkey patching
PATHS_PROJECT_DIR_PATH = "src.core.consts.PATHS.PROJECT_DIR"
INFO_FLOW_FORWARD_EVAL_PATH = "src.experiments.runners.info_flow.forward_eval"
INFO_FLOW_PRINT_INTERVAL_PATH = "src.experiments.runners.info_flow.PRINT_INTERVAL"
GET_COMMIT_HASH_PATH = "src.experiments.infrastructure.base_runner.get_git_commit_hash"
CREATE_RUN_ID_PATH = "src.experiments.infrastructure.base_runner.create_run_id"


def get_test_full_pipeline_config(
    code_version_name: str, model_arch: MODEL_ARCH, model_size: str, with_plotting: bool
) -> FullPipelineRunner:
    """Get base test configuration for full pipeline."""
    return FullPipelineRunner(
        variant_params=FullPipelineParams(
            model_arch=model_arch,
            model_size=TModelSize(model_size),
            knockout_map={
                TokenType.last: [
                    (TokenType.last, FeatureCategory.ALL),
                    (TokenType.subject, FeatureCategory.SLOW_DECAY),
                    (TokenType.subject, FeatureCategory.FAST_DECAY),
                    (TokenType.first, FeatureCategory.ALL),
                    (TokenType.subject, FeatureCategory.ALL),
                    (TokenType.relation, FeatureCategory.ALL),
                ],
                TokenType.subject: [
                    (TokenType.context, FeatureCategory.ALL),
                    (TokenType.subject, FeatureCategory.ALL),
                ],
                TokenType.relation: [
                    (TokenType.context, FeatureCategory.ALL),
                    (TokenType.subject, FeatureCategory.ALL),
                    (TokenType.relation, FeatureCategory.ALL),
                ],
            },
            info_flow_window_size=TWindowSize(15),
            heatmap_window_size=TWindowSize(15),
            heatmap_prompts=SelectivePromptFilteration(prompt_ids=tuple(ORIGINAL_IDS[SPLIT.TRAIN1])),
            with_plotting=with_plotting,
            enforce_no_missing_outputs=True,
            with_generation=True,
        ),
        input_params=InputParams(
            dataset_name=DatasetName.counter_fact,
            filteration=ModelCorrectPromptFilteration(
                dataset_name=DatasetName.counter_fact,
                model_arch_and_size=MODEL_ARCH_AND_SIZE(model_arch, TModelSize(model_size)),
                correctness=Correctness.correct,
                code_version=TCodeVersionName(code_version_name),
            ),
        ),
        metadata_params=MetadataParams(
            code_version=TCodeVersionName(code_version_name),
            with_slurm=False,
        ),
    )


def get_test_full_pipeline_config_per_model_arch(
    code_version_name: str, model_arch: MODEL_ARCH, model_size: str, with_plotting: bool
) -> FullPipelineRunner:
    """Get configuration for specific model architecture - shared by baseline and tests."""
    if model_arch in [
        MODEL_ARCH.MAMBA1,
        MODEL_ARCH.MAMBA2,
        MODEL_ARCH.GPT2,
        MODEL_ARCH.LLAMA3_2,
    ]:
        return get_test_full_pipeline_config(code_version_name, model_arch, model_size, with_plotting)
    elif model_arch in [MODEL_ARCH.QWEN2_5, MODEL_ARCH.QWEN2]:
        config = get_test_full_pipeline_config(code_version_name, model_arch, model_size, with_plotting)
        return config.modify(
            variant_params=config.variant_params.modify(
                knockout_map={
                    TokenType.last: [
                        (TokenType.last, FeatureCategory.ALL),
                        (TokenType.first, FeatureCategory.ALL),
                        (TokenType.subject, FeatureCategory.ALL),
                    ],
                },
                info_flow_window_size=TWindowSize(3),
                heatmap_window_size=TWindowSize(3),
            )
        )
    else:
        raise ValueError(f"Model architecture {model_arch} is not supported")


class BaselineBuilder:
    """Generates baseline test data exactly as the current script does."""

    def __init__(self, test_base_path: Path):
        self.test_base_path = test_base_path

    def clean_and_generate_base_test_data(self) -> None:
        """Clean test directory and generate base test data filtered by ORIGINAL_IDS."""
        test_paths = PathsConfig(PROJECT_DIR=self.test_base_path)

        # Clean test base path
        if self.test_base_path.exists():
            shutil.rmtree(self.test_base_path)
        self.test_base_path.mkdir(parents=True, exist_ok=True)

        # Get sample of real data - only load and filter by ORIGINAL_IDS subset
        dataset = {
            split: load_splitted_counter_fact(
                ALL_SPLITS_LITERAL,
                align_to_known=False,
            ).filter(lambda x: x[COLS.ORIGINAL_IDX] in original_ids)
            for split, original_ids in ORIGINAL_IDS.items()
        }

        # Save filtered dataset to disk
        DatasetDict(dataset).save_to_disk(test_paths.dataset_dir(DatasetName.counter_fact) / "splitted")

    def run_baseline_experiments(self, normalizing_outputs: bool = True, with_plotting: bool = True) -> None:
        """Run all experiments using TEST_MODEL_CONFIGS to generate baseline results."""
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(PATHS_PROJECT_DIR_PATH, self.test_base_path)
            mp.setattr(INFO_FLOW_PRINT_INTERVAL_PATH, 1)
            if normalizing_outputs:
                mp.setattr(GET_COMMIT_HASH_PATH, lambda *args, **kwargs: "test_commit_hash")
                mp.setattr(CREATE_RUN_ID_PATH, lambda *args, **kwargs: "test_run_id")

            # Run experiments for all test model configurations
            for model_arch, model_size in TEST_MODEL_CONFIGS:
                config = get_test_full_pipeline_config_per_model_arch(
                    code_version_name="test_baseline",
                    model_arch=model_arch,
                    model_size=model_size,
                    with_plotting=with_plotting,
                )

                config.compute_dependencies(rec_depth=-1)
                config.run(with_dependencies=True)

            if normalizing_outputs:
                (self.test_base_path / "serialized_results.json").write_text(
                    serialize_result_bank(get_experiment_results_bank())
                )
            print(f"Baseline updated at: {self.test_base_path}")

    def build_baseline(self, params: "CreateBaselineParams") -> None:
        """Main method to build complete baseline."""
        if not params.resume:
            self.clean_and_generate_base_test_data()
        self.run_baseline_experiments(params.normalizing_outputs, params.with_plotting)

    def get_config_for_model(
        self, model_arch: MODEL_ARCH, model_size: str, with_plotting: bool = True
    ) -> FullPipelineRunner:
        """Get configuration for specific model - used by both baseline and tests."""
        return get_test_full_pipeline_config_per_model_arch(
            code_version_name="test_baseline",
            model_arch=model_arch,
            model_size=model_size,
            with_plotting=with_plotting,
        )


@dataclass
class CreateBaselineParams:
    """Parameters for baseline creation."""

    resume: bool = False
    normalizing_outputs: bool = True
    with_plotting: bool = True


@pyrallis.wrap()  # type: ignore
def main(params: CreateBaselineParams):
    """Generate baseline test data - maintains same CLI interface as original script."""
    builder = BaselineBuilder(TEST_BASE_PATH)
    builder.build_baseline(params)


if __name__ == "__main__":
    main()  # type: ignore
