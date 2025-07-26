from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Optional

from src.core.consts import DEFAULT_MODEL_CORRECT_DATASET_NAME, DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION
from src.core.names import COLS, DatasetName
from src.core.types import (
    ALL_SPLITS_LITERAL,
    MODEL_ARCH_AND_SIZE,
    TCodeVersionName,
    TPromptOriginalIndex,
    TSplitChoise,
)
from src.data_ingestion.datasets.download_dataset import get_prompt_ids
from src.experiments.infrastructure.base_prompt_filteration import BasePromptFilteration, LogicalPromptFilteration
from src.experiments.infrastructure.base_runner import (
    BaseRunner,
    InputParams,
    MetadataParams,
    TDependencies,
)
from src.experiments.runners.evaluate_model import EvaluateModelParams, EvaluateModelRunner
from src.experiments.runners.heatmap import HeatmapRunner


@dataclass(frozen=True)
class AllPromptFilteration(BasePromptFilteration):
    dataset_name: DatasetName = DatasetName.counter_fact
    split: TSplitChoise = ALL_SPLITS_LITERAL

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:  # type: ignore
        return get_prompt_ids(self.dataset_name, self.split)

    def get_dependencies(self) -> TDependencies:
        return {}

    def display_name(self) -> str:
        return f"{self.dataset_name} {self.split}"


@dataclass(frozen=True)
class AnyExistingCompletePromptFilteration(BasePromptFilteration):
    def _get_base_runner_from_context(self) -> BaseRunner:
        assert self._context is not None
        assert isinstance(self._context, BaseRunner)
        return self._context

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        base_runner = self._get_base_runner_from_context()
        from src.experiments.runners.info_flow import InfoFlowRunner

        if isinstance(base_runner, EvaluateModelRunner):
            return base_runner.get_outputs()[COLS.ORIGINAL_IDX].tolist()
        elif isinstance(base_runner, InfoFlowRunner):
            return list(base_runner.output_file.get_computed_prompt_idx())
        elif isinstance(base_runner, HeatmapRunner):
            return list(base_runner.output_hdf5_path.get_existing_prompt_idx())
        else:
            raise NotImplementedError(f"Prompt filteration for {base_runner.__class__.__name__} not implemented")

    def get_dependencies(self) -> TDependencies:
        return {}

    def display_name(self) -> str:
        return "Any Existing (Complete Only)"


class Correctness(StrEnum):
    correct = "correct"
    top_5_correct = "top_5_correct"
    top_3_correct = "top_3_correct"
    top_2_to_5_correct = "top_2_to_5_correct"


@dataclass(frozen=True)
class ModelCorrectPromptFilteration(BasePromptFilteration):
    dataset_name: DatasetName
    model_arch_and_size: Optional[MODEL_ARCH_AND_SIZE]
    correctness: Correctness
    code_version: TCodeVersionName

    def _get_model_arch_and_size_from_context(self) -> MODEL_ARCH_AND_SIZE:
        if self.model_arch_and_size is not None:
            return self.model_arch_and_size
        assert self.has_context
        if isinstance(self._context, BaseRunner):
            return self._context.variant_params.model_arch_and_size

        raise NotImplementedError(f"Prompt filteration for {self._context.__class__.__name__} not implemented")

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:  # type: ignore
        df = self.get_dependencies()["evaluate_model"].get_outputs()

        match self.correctness:
            case Correctness.correct:
                df = df[df[COLS.EVALUATE_MODEL.MODEL_CORRECT]]
            case Correctness.top_5_correct:
                df = df[df[COLS.EVALUATE_MODEL.TARGET_RANK] <= 5]
            case Correctness.top_3_correct:
                df = df[df[COLS.EVALUATE_MODEL.TARGET_RANK] <= 3]
            case Correctness.top_2_to_5_correct:
                df = df[(2 <= df[COLS.EVALUATE_MODEL.TARGET_RANK]) & (df[COLS.EVALUATE_MODEL.TARGET_RANK] <= 5)]
            case _:
                raise NotImplementedError(f"Correctness {self.correctness} not implemented")

        return df[COLS.ORIGINAL_IDX].tolist()

    def get_dependencies(self):
        if not (self.model_arch_and_size is not None or self.has_context):
            return {}

        model_arch_and_size = self._get_model_arch_and_size_from_context()
        return {
            "evaluate_model": EvaluateModelRunner(
                variant_params=EvaluateModelParams(
                    model_arch=model_arch_and_size.arch,
                    model_size=model_arch_and_size.size,
                ),
                input_params=InputParams(
                    filteration=AllPromptFilteration(dataset_name=self.dataset_name),
                ),
                metadata_params=MetadataParams(
                    code_version=self.code_version,
                ),
            ),
        }

    def is_computed(self) -> bool:
        assert self._get_model_arch_and_size_from_context()
        return self.get_dependencies()["evaluate_model"].is_computed()

    def display_name(self) -> str:
        return f"Model {self.correctness.value} on {self.model_arch_and_size}"


def get_shared_models_correctness_prompt_filteration(
    model_arch_and_sizes: Iterable[MODEL_ARCH_AND_SIZE],
    correctness: Correctness,
    code_version: TCodeVersionName = DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION,
    dataset_name: DatasetName = DEFAULT_MODEL_CORRECT_DATASET_NAME,
):
    # Create the base filteration
    # Create a list of all model filterations
    model_filterations = []
    for model_arch_and_size in model_arch_and_sizes:
        model_filter = ModelCorrectPromptFilteration(
            dataset_name=dataset_name,
            model_arch_and_size=model_arch_and_size,
            correctness=correctness,
            code_version=code_version,
        )
        model_filterations.append(model_filter)

    return LogicalPromptFilteration.create_and(model_filterations)
