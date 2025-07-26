from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Generic, Iterator, Literal, Sequence, TypeVar, Union, final

from src.analysis.experiment_results.prompt_filteration_factory import (
    AllImportantModelsFilterationFactory,
    ContextModelsFilterationFactory,
    CurrentModelFilterationFactory,
    ExistingPromptsFilterationFactory,
    FilterationSource,
    PresetFilterationFactory,
    PromptFilterationFactory,
    PromptFilterationFactoryUnion,
)
from src.analysis.prompt_filterations import (
    AllPromptFilteration,
    Correctness,
)
from src.core.consts import (
    ARCH_FAMILY,
    GRAPHS_ORDER,
    MODEL_FAMILY_TO_ARCH,
)
from src.core.names import (
    VARIANT_PARAM_NAME,
    BaseVariantParamName,
    InfoFlowVariantParam,
    ToClassifyNames,
    WindowedVariantParam,
)
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    FeatureCategory,
    TModelSize,
    TokenType,
    TPromptOriginalIndex,
    TWindowSize,
)
from src.experiments.infrastructure.base_prompt_filteration import BasePromptFilteration, SelectivePromptFilteration
from src.experiments.infrastructure.base_runner import BaseVariantParams
from src.utils.types_utils import str_enum_values

_T = TypeVar("_T")


PossibleHPDTypes = Union[
    MODEL_ARCH_AND_SIZE,
    MODEL_ARCH,
    TModelSize,
    TokenType,
    FeatureCategory,
    TWindowSize,
    TPromptOriginalIndex,
    PromptFilterationFactoryUnion,
]

PossibleDerivedHPDTypes = Union[MODEL_ARCH, TModelSize, TokenType, FeatureCategory, TWindowSize, BasePromptFilteration]


class VirtualExperimentHyperParams(StrEnum):
    model_arch_and_size = "model_arch_and_size"
    filteration_factory = "filteration_factory"
    prompt_idx = "prompt_idx"
    model_family = "model_family"


TExperimentHyperParams = Union[
    VARIANT_PARAM_NAME,
    VirtualExperimentHyperParams,
]

T_VARIANT_PARAM_AND_FILTERATION = Union[VARIANT_PARAM_NAME, Literal[ToClassifyNames.prompt_filteration]]


class HyperParamDefinition(ABC, Generic[_T]):
    @abstractmethod
    def get_options(self) -> Sequence[_T]:
        pass

    @abstractmethod
    def get_display_name(self, option: _T) -> str:
        pass

    def default_fix_value(self) -> _T:
        raise NotImplementedError(f"Default fix value not implemented for {self.__class__.__name__}")

    @abstractmethod
    def derived_variants_params(
        self,
    ) -> Sequence[T_VARIANT_PARAM_AND_FILTERATION]:
        pass

    @abstractmethod
    def get_derived_hpds(self, option: _T) -> dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]:
        pass

    def _expand_value(self, option: _T) -> Iterator[dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]:
        yield self.get_derived_hpds(option)

    @final
    def expand_values(self, values: list[_T]) -> Iterator[dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]:
        for value in values:
            for expanded_value in self._expand_value(value):
                yield expanded_value

    def get_line_id_from_runner(self, runner_variant_params: BaseVariantParams) -> str:
        """
        Generates a line ID string for a plot, based on the provided runner's variant parameters.
        This method is called when this HPD instance is the one defining the 'lines' orientation.
        'plot_plan' is provided for context if needed.
        """
        derived_params = self.derived_variants_params()
        if derived_params:
            derived_vp_name = derived_params[0]
            attr_name = derived_vp_name.value

            actual_derived_value = getattr(runner_variant_params, attr_name)
            assert derived_vp_name != ToClassifyNames.prompt_filteration

            hpd_for_derived_display = get_hyper_param_definition(derived_vp_name)
            return hpd_for_derived_display.get_display_name(actual_derived_value)
        else:
            raise NotImplementedError(
                f"HPD {self.__class__.__name__} used for lines but derives no variant params "
                f"and does not override get_line_id_from_runner. Runner params: {runner_variant_params}"
            )


_T_BaseVariant = TypeVar("_T_BaseVariant", bound=PossibleDerivedHPDTypes)


class BaseVariantHPD(HyperParamDefinition[_T_BaseVariant]):
    def get_display_name(self, option: _T_BaseVariant) -> str:
        return str(option)

    @abstractmethod
    def derived_variants_params(self) -> Sequence[T_VARIANT_PARAM_AND_FILTERATION]:
        pass

    def get_derived_hpds(self, option: _T_BaseVariant) -> dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]:
        derived_params = self.derived_variants_params()
        assert len(derived_params) == 1, (
            f"BaseVariantHPD.get_derived_hpds expects exactly one derived param, but got {derived_params}"
        )
        derived_param = derived_params[0]
        assert derived_param != ToClassifyNames.prompt_filteration

        return {
            derived_param: option,
        }


# region Variant HPDs


class ModelArchHPD(BaseVariantHPD[MODEL_ARCH]):
    def get_options(self):
        return str_enum_values(MODEL_ARCH)

    def get_display_name(self, option):
        return option

    def derived_variants_params(self) -> Sequence[T_VARIANT_PARAM_AND_FILTERATION]:
        return [BaseVariantParamName.model_arch]


class ModelSizeHPD(BaseVariantHPD[TModelSize]):
    def get_options(self):
        return list({size: size for _, size in GRAPHS_ORDER.keys()}.keys())

    def get_display_name(self, option):
        return option

    def derived_variants_params(self) -> Sequence[T_VARIANT_PARAM_AND_FILTERATION]:
        return [BaseVariantParamName.model_size]


class SourceHPD(BaseVariantHPD[TokenType]):
    def get_options(self):
        return str_enum_values(TokenType)

    def get_display_name(self, option: TokenType) -> str:
        return option

    def derived_variants_params(self) -> Sequence[T_VARIANT_PARAM_AND_FILTERATION]:
        return [InfoFlowVariantParam.source]


class TargetHPD(BaseVariantHPD[TokenType]):
    def get_options(self):
        return str_enum_values(TokenType)

    def get_display_name(self, option: TokenType) -> str:
        return option

    def default_fix_value(self) -> TokenType:
        return TokenType.last

    def derived_variants_params(self) -> Sequence[T_VARIANT_PARAM_AND_FILTERATION]:
        return [InfoFlowVariantParam.target]


class FeatureCategoryHPD(BaseVariantHPD[FeatureCategory]):
    def get_options(self):
        return str_enum_values(FeatureCategory)

    def get_display_name(self, option: FeatureCategory) -> str:
        return str(option)

    def default_fix_value(self) -> FeatureCategory:
        return FeatureCategory.ALL

    def derived_variants_params(self) -> Sequence[T_VARIANT_PARAM_AND_FILTERATION]:
        return [InfoFlowVariantParam.feature_category]


class WindowSizeHPD(BaseVariantHPD[TWindowSize]):
    def get_options(self):
        return list([TWindowSize(i) for i in range(1, 20)])

    def get_display_name(self, option: TWindowSize) -> str:
        return f"{option}"

    def default_fix_value(self) -> TWindowSize:
        return TWindowSize(9)

    def derived_variants_params(self) -> Sequence[T_VARIANT_PARAM_AND_FILTERATION]:
        return [WindowedVariantParam.window_size]


# endregion

# region Virtual HPDs


class ModelArchAndSizeHPD(HyperParamDefinition[MODEL_ARCH_AND_SIZE]):
    def get_options(self) -> Sequence[MODEL_ARCH_AND_SIZE]:
        return list(GRAPHS_ORDER.keys())

    def get_display_name(self, option: MODEL_ARCH_AND_SIZE) -> str:
        return option.model_name

    def derived_variants_params(self) -> Sequence[T_VARIANT_PARAM_AND_FILTERATION]:
        return [BaseVariantParamName.model_arch, BaseVariantParamName.model_size]

    def get_derived_hpds(self, option: MODEL_ARCH_AND_SIZE) -> dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]:
        return {
            BaseVariantParamName.model_arch: option.arch,
            BaseVariantParamName.model_size: option.size,
        }

    def get_line_id_from_runner(self, runner_variant_params: BaseVariantParams) -> str:
        model_arch = runner_variant_params.model_arch
        model_size = runner_variant_params.model_size
        arch_and_size_tuple = MODEL_ARCH_AND_SIZE(model_arch, model_size)
        return self.get_display_name(arch_and_size_tuple)


class ModelFamilyHPD(HyperParamDefinition[ARCH_FAMILY]):
    def get_options(self) -> Sequence[ARCH_FAMILY]:
        return str_enum_values(ARCH_FAMILY)

    def get_display_name(self, option: ARCH_FAMILY) -> str:
        return option

    def derived_variants_params(self) -> Sequence[T_VARIANT_PARAM_AND_FILTERATION]:
        return [BaseVariantParamName.model_arch]

    def get_derived_hpds(self, option: ARCH_FAMILY) -> dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]:
        raise RuntimeError("Should not be called")

    def _expand_value(self, option: ARCH_FAMILY) -> Iterator[dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]:
        for arch in MODEL_FAMILY_TO_ARCH[option]:
            yield {BaseVariantParamName.model_arch: arch}


class PromptFilterationHPD(HyperParamDefinition[_T]):
    def get_derived_hpds(self, option):
        raise NotImplementedError("Only get_derived_hpd_with_context should be called")

    def derived_variants_params(self) -> Sequence[T_VARIANT_PARAM_AND_FILTERATION]:
        return [ToClassifyNames.prompt_filteration]

    @abstractmethod
    def get_derived_hpd_with_context(
        self, option: _T, context_model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
    ) -> tuple[BasePromptFilteration, dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]: ...


class PromptIdxHPD(PromptFilterationHPD[TPromptOriginalIndex]):
    def get_options(self):
        return AllPromptFilteration().get_prompt_ids()

    def get_display_name(self, option: TPromptOriginalIndex) -> str:
        return f"{option}"

    def get_derived_hpd_with_context(
        self, option: TPromptOriginalIndex, context_model_arch_and_sizes
    ) -> tuple[BasePromptFilteration, dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]:
        return SelectivePromptFilteration(prompt_ids=(option,)), {}


class FilterationFactoryHPD(PromptFilterationHPD[PromptFilterationFactory]):
    def get_options(self) -> Sequence[PromptFilterationFactory]:
        from src.data_ingestion.data_defs.data_defs import PromptFilterationsPresets

        options = []

        # Add preset options
        presets = PromptFilterationsPresets.load()
        for preset_id in presets:
            options.append(PresetFilterationFactory(type=FilterationSource.preset, preset_id=preset_id))

        # Add model correctness options
        for correctness in Correctness:
            # Current model
            options.append(
                CurrentModelFilterationFactory(
                    type=FilterationSource.current_model, correctness=correctness, combine_with_existing=False
                )
            )
            options.append(
                CurrentModelFilterationFactory(
                    type=FilterationSource.current_model, correctness=correctness, combine_with_existing=True
                )
            )

            # Context models
            options.append(
                ContextModelsFilterationFactory(
                    type=FilterationSource.context_models, correctness=correctness, combine_with_existing=False
                )
            )
            options.append(
                ContextModelsFilterationFactory(
                    type=FilterationSource.context_models, correctness=correctness, combine_with_existing=True
                )
            )

            # All important models
            options.append(
                AllImportantModelsFilterationFactory(
                    type=FilterationSource.all_important_models, correctness=correctness, combine_with_existing=False
                )
            )
            options.append(
                AllImportantModelsFilterationFactory(
                    type=FilterationSource.all_important_models, correctness=correctness, combine_with_existing=True
                )
            )

        # Add existing prompts option
        options.append(ExistingPromptsFilterationFactory(type=FilterationSource.existing_prompts))

        return options

    def get_display_name(self, option: PromptFilterationFactory) -> str:
        return option.display_name

    def get_derived_hpd_with_context(
        self, option: PromptFilterationFactory, context_model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
    ) -> tuple[BasePromptFilteration, dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]:
        return option.get_filteration(context_model_arch_and_sizes), {}


# endregion

# region Experiment Hyper Params


def get_hyper_param_definition(option: TExperimentHyperParams) -> HyperParamDefinition:
    match option:
        case VirtualExperimentHyperParams.model_arch_and_size:
            return ModelArchAndSizeHPD()
        case BaseVariantParamName.model_arch:
            return ModelArchHPD()
        case BaseVariantParamName.model_size:
            return ModelSizeHPD()
        case InfoFlowVariantParam.source:
            return SourceHPD()
        case InfoFlowVariantParam.target:
            return TargetHPD()
        case InfoFlowVariantParam.feature_category:
            return FeatureCategoryHPD()
        case WindowedVariantParam.window_size:
            return WindowSizeHPD()
        case VirtualExperimentHyperParams.prompt_idx:
            return PromptIdxHPD()
        case VirtualExperimentHyperParams.filteration_factory:
            return FilterationFactoryHPD()
        case VirtualExperimentHyperParams.model_family:
            return ModelFamilyHPD()
        case _:
            raise ValueError(f"Unsupported variation option: {option}")
