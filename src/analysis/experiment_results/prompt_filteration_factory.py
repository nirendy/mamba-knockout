from __future__ import annotations

from abc import ABCMeta, abstractmethod
from enum import StrEnum
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from src.analysis.prompt_filterations import (
    AnyExistingCompletePromptFilteration,
    Correctness,
    ModelCorrectPromptFilteration,
)
from src.core.consts import (
    ALL_IMPORTANT_MODELS,
    DEFAULT_MODEL_CORRECT_DATASET_NAME,
    DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION,
)
from src.core.types import MODEL_ARCH_AND_SIZE, TPresetID
from src.data_ingestion.data_defs.data_defs import PromptFilterationsPresets
from src.experiments.infrastructure.base_prompt_filteration import BasePromptFilteration, LogicalPromptFilteration


class FilterationSource(StrEnum):
    """Enum for the source of a filteration."""

    preset = "preset"
    current_model = "current_model"
    context_models = "context_models"
    all_important_models = "all_important_models"
    existing_prompts = "existing_prompts"


# Base class
class PromptFilterationFactory(BaseModel, metaclass=ABCMeta):
    """Base class for prompt filteration factories.

    This class defines the interface for creating prompt filterations. Subclasses
    implement specific types of filterations (preset-based, model correctness, etc).
    """

    # Type field (discriminator for serialization/deserialization)
    type: str

    # Optional flag for combining with existing prompts
    combine_with_existing: bool = False
    model_config = ConfigDict(frozen=True)

    def get_filteration(self, context_model_arch_and_sizes: List[MODEL_ARCH_AND_SIZE]) -> BasePromptFilteration:
        """Get filteration, potentially combined with existing prompts."""
        base_filteration = self._get_base_filteration(context_model_arch_and_sizes)

        if self.combine_with_existing:
            base_filteration = base_filteration & AnyExistingCompletePromptFilteration()

        return base_filteration

    @abstractmethod
    def _get_base_filteration(self, context_model_arch_and_sizes: List[MODEL_ARCH_AND_SIZE]) -> BasePromptFilteration:
        """Get base filteration, to be implemented by subclasses."""
        raise NotImplementedError

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Get display name, to be implemented by subclasses."""
        raise NotImplementedError


# Preset factory
class PresetFilterationFactory(PromptFilterationFactory):
    """Factory for creating prompt filterations from presets."""

    type: Literal[FilterationSource.preset]  # type:ignore[assignment]
    preset_id: TPresetID

    def _get_base_filteration(self, context_model_arch_and_sizes: List[MODEL_ARCH_AND_SIZE]) -> BasePromptFilteration:
        presets = PromptFilterationsPresets.load()
        if self.preset_id in presets:
            return presets[self.preset_id]
        else:
            raise ValueError(f"Unknown preset: {self.preset_id}")

    @property
    def display_name(self) -> str:
        return f"Preset: {self.preset_id}"


# Abstract base for model correctness filterations
class ModelCorrectnessFilterationFactory(PromptFilterationFactory):
    """Base class for factories that filter based on model correctness."""

    correctness: Correctness

    @property
    def display_name(self) -> str:
        base_name = self.type.replace("_", " ")
        if self.correctness:
            base_name = f"{base_name} - {self.correctness}"
        if self.combine_with_existing:
            return f"{base_name} (existing only)"
        return base_name


# Current model factory
class CurrentModelFilterationFactory(ModelCorrectnessFilterationFactory):
    """Factory for creating filterations based on the current model's correctness."""

    type: Literal[FilterationSource.current_model]  # type:ignore[assignment]

    def _get_base_filteration(self, context_model_arch_and_sizes: List[MODEL_ARCH_AND_SIZE]) -> BasePromptFilteration:
        return ModelCorrectPromptFilteration(
            dataset_name=DEFAULT_MODEL_CORRECT_DATASET_NAME,
            model_arch_and_size=None,  # Will be contextualized
            correctness=self.correctness,
            code_version=DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION,
        )


# Context models factory
class ContextModelsFilterationFactory(ModelCorrectnessFilterationFactory):
    """Factory for creating filterations based on correctness across context models."""

    type: Literal[FilterationSource.context_models]  # type:ignore[assignment]

    def _get_base_filteration(self, context_model_arch_and_sizes: List[MODEL_ARCH_AND_SIZE]) -> BasePromptFilteration:
        return LogicalPromptFilteration.create_and(
            [
                ModelCorrectPromptFilteration(
                    dataset_name=DEFAULT_MODEL_CORRECT_DATASET_NAME,
                    model_arch_and_size=model_arch_and_size,
                    correctness=self.correctness,
                    code_version=DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION,
                )
                for model_arch_and_size in context_model_arch_and_sizes
            ]
        )


# All important models factory
class AllImportantModelsFilterationFactory(ModelCorrectnessFilterationFactory):
    """Factory for creating filterations based on correctness across all important models."""

    type: Literal[FilterationSource.all_important_models]  # type:ignore[assignment]

    def _get_base_filteration(self, context_model_arch_and_sizes: List[MODEL_ARCH_AND_SIZE]) -> BasePromptFilteration:
        return LogicalPromptFilteration.create_and(
            [
                ModelCorrectPromptFilteration(
                    dataset_name=DEFAULT_MODEL_CORRECT_DATASET_NAME,
                    model_arch_and_size=model_arch_and_size,
                    correctness=self.correctness,
                    code_version=DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION,
                )
                for model_arch_and_size in ALL_IMPORTANT_MODELS
            ]
        )


# Existing prompts factory
class ExistingPromptsFilterationFactory(PromptFilterationFactory):
    """Factory for creating filterations that select only existing/computed prompts."""

    type: Literal[FilterationSource.existing_prompts]  # type:ignore[assignment]

    def _get_base_filteration(self, context_model_arch_and_sizes: List[MODEL_ARCH_AND_SIZE]) -> BasePromptFilteration:
        return AnyExistingCompletePromptFilteration()

    @property
    def display_name(self) -> str:
        return "Existing prompts only"


# Union type for type hints
PromptFilterationFactoryUnion = Annotated[
    Union[
        PresetFilterationFactory,
        CurrentModelFilterationFactory,
        ContextModelsFilterationFactory,
        AllImportantModelsFilterationFactory,
        ExistingPromptsFilterationFactory,
    ],
    Field(discriminator="type"),
]
