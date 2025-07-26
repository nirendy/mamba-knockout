from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.analysis.experiment_results.helpers import init_variant_params_from_values
from src.analysis.experiment_results.hyper_param_definition import (
    HyperParamDefinition,
    PossibleDerivedHPDTypes,
    PossibleHPDTypes,
    PromptFilterationHPD,
    TExperimentHyperParams,
    VirtualExperimentHyperParams,
    get_hyper_param_definition,
)
from src.analysis.plots.heatmaps import HeatmapPlotConfig
from src.analysis.plots.image_combiner import ImageGridParams
from src.analysis.plots.info_flow_confidence import InfoFlowPlotConfig
from src.core.consts import GRAPHS_ORDER
from src.core.names import (
    VARIANT_PARAM_NAME,
    BaseVariantParamName,
    ExperimentName,
    FinalPlotsPlanOrientation,
    ToClassifyNames,
)
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    TModelSize,
    TPlotID,
)
from src.data_ingestion.data_defs.data_defs import (
    DataReqiermentCollection,
    DataReqs,
    PlotPlans,
    ResultBank,
)
from src.utils.types_utils import str_enum_values


def get_experiment_orientations(
    experiment_name: ExperimentName,
) -> list[FinalPlotsPlanOrientation]:
    """Get the relevant parameters for a specific experiment type."""
    if experiment_name == ExperimentName.info_flow:
        return [
            FinalPlotsPlanOrientation.lines,
            FinalPlotsPlanOrientation.rows,
            FinalPlotsPlanOrientation.cols,
            FinalPlotsPlanOrientation.grids,
        ]
    elif experiment_name == ExperimentName.heatmap:
        return [
            FinalPlotsPlanOrientation.rows,
            FinalPlotsPlanOrientation.cols,
            FinalPlotsPlanOrientation.grids,
        ]
    else:
        return list(FinalPlotsPlanOrientation)


T_HPD_OPTION = VARIANT_PARAM_NAME | Literal[ToClassifyNames.prompt_filteration]


def get_experiment_hyper_param_hyper_param(
    experiment_name: ExperimentName,
) -> list[TExperimentHyperParams]:
    """Get the relevant parameters for a specific experiment type."""
    return [
        VirtualExperimentHyperParams.model_arch_and_size,
        *experiment_name.get_variant_cols(experiment_name),
    ]


@dataclass(frozen=True)
class Cell:
    grids: Optional[PossibleHPDTypes] = None
    rows: Optional[PossibleHPDTypes] = None
    cols: Optional[PossibleHPDTypes] = None

    @classmethod
    def from_orientation_combination(
        cls, orientation_combination: dict[FinalPlotsPlanOrientation, PossibleHPDTypes | None]
    ) -> Cell:
        """Create a Cell from an orientation combination dictionary."""
        return cls(
            grids=orientation_combination.get(FinalPlotsPlanOrientation.grids),
            rows=orientation_combination.get(FinalPlotsPlanOrientation.rows),
            cols=orientation_combination.get(FinalPlotsPlanOrientation.cols),
        )

    def get_field_display_name(self, field: str, plot_plan: PlotPlan) -> Optional[str]:
        """Get the display name for a field value using the plot plan's parameter definition."""
        value = getattr(self, field)
        if value is None:
            return None

        param_config = plot_plan.get_param_config_by_orientation(FinalPlotsPlanOrientation[field])
        if param_config is None:
            return str(value)

        return get_hyper_param_definition(param_config.param).get_display_name(value)

    def to_dict(self) -> dict[str, PossibleHPDTypes]:
        """Convert cell to dictionary for data requirements."""
        return {FinalPlotsPlanOrientation[field]: getattr(self, field) for field in ["grids", "rows", "cols"]}

    def get_display_name(self, plot_plan: PlotPlan) -> str:
        """Get the display name for the cell."""
        display_names = {
            field: self.get_field_display_name(field, plot_plan) or "None" for field in ["grids", "rows", "cols"]
        }

        # Create a unique identifier for the cell
        return "_".join(f"{value}" for key, value in display_names.items()).replace(" ", "_")


class ParamConfig(BaseModel):
    """Configuration for a hyperparameter in the plot plan."""

    param: TExperimentHyperParams
    orientation: Optional[FinalPlotsPlanOrientation] = None
    values: List[PossibleHPDTypes] = Field(default_factory=list)

    @property
    def fixed_value(self) -> PossibleHPDTypes:
        """Return the first value if this is a fixed parameter (orientation is None and exactly one value)."""
        assert self.is_fixed()
        return self.values[0]

    def get_param_def(self) -> HyperParamDefinition:
        """Get the HyperParamDefinition for this parameter."""
        return get_hyper_param_definition(self.param)

    @model_validator(mode="after")  # type: ignore
    def validate_param_values(self) -> "ParamConfig":
        """Validate that values match the expected type for the param."""
        # Skip validation during initialization
        # If orientation is None, values must have exactly one item or be empty
        if self.orientation is None and len(self.values) > 1:
            raise ValueError("Fixed parameters (orientation=None) must have exactly one or zero values")

        param_def = self.get_param_def()

        # Validate values if provided
        if self.values:
            for value in self.values:
                # Try to get the display name to ensure the value is valid for this param
                try:
                    param_def.get_display_name(value)
                except Exception as e:
                    raise ValueError(f"Invalid value {value} for parameter {self.param}: {str(e)}")

        return self

    def is_fixed(self) -> bool:
        """Check if this parameter is fixed (not variable across an orientation)."""
        return self.orientation is None and len(self.values) == 1

    def is_variable(self) -> bool:
        """Check if this parameter varies across an orientation."""
        return self.orientation is not None

    def get_values(self) -> List[dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]:
        """Get the values for this parameter, either specified or from the result bank."""
        return [self.get_param_def().get_derived_hpds(value) for value in self.values]

    def derived_variant_params(self) -> list[dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]:
        """Get the derived variant parameters for this parameter."""
        return [self.get_param_def().get_derived_hpds(value) for value in self.values]


class PlotPlan(BaseModel):
    """A plan for plotting experiment results in a grid layout."""

    model_config = ConfigDict(validate_assignment=True)

    plot_id: TPlotID
    title: str = Field(default="")
    description: str = Field(default="")
    order: int
    observation: str = Field(default="")
    notes: str = Field(default="")
    is_appendix: bool = Field(default=False)

    # Use proper type annotation for params
    experiment_name: ExperimentName
    params: List[ParamConfig] = Field(default_factory=list)

    # Plot configuration
    cell_plot_config: InfoFlowPlotConfig | HeatmapPlotConfig = Field(default_factory=dict)
    combine_plot_config: ImageGridParams = Field(default_factory=ImageGridParams)

    @model_validator(mode="after")  # type: ignore
    def validate_param_configs(self) -> PlotPlan:
        """Validate the parameter configurations."""
        # Skip validation for empty models or during initialization
        if not self.params:
            return self

        # Check for duplicate parameters with the same orientation
        orientation_to_param: Dict[FinalPlotsPlanOrientation, TExperimentHyperParams] = {}
        for config in self.params:
            if config.orientation is not None:
                if config.orientation in orientation_to_param:
                    raise ValueError(
                        f"Duplicate orientation {config.orientation} for parameters "
                        f"{orientation_to_param[config.orientation]} and {config.param}"
                    )
                orientation_to_param[config.orientation] = config.param

        produces_model_arch = any(
            BaseVariantParamName.model_arch in config.get_param_def().derived_variants_params()
            for config in self.params
        )
        produces_model_size = any(
            BaseVariantParamName.model_size in config.get_param_def().derived_variants_params()
            for config in self.params
        )
        assert produces_model_arch and produces_model_size
        return self

    def get_param_config(self, param: TExperimentHyperParams) -> Optional[ParamConfig]:
        """Get the configuration for a specific parameter."""
        for config in self.params:
            if config.param == param:
                return config
        return None

    def get_param_config_by_orientation(self, orientation: FinalPlotsPlanOrientation) -> Optional[ParamConfig]:
        """Get the parameter configuration associated with a specific orientation."""
        for config in self.params:
            if config.orientation == orientation:
                return config
        return None

    def get_orientation_value_hpd(self, orientation: FinalPlotsPlanOrientation) -> Optional[HyperParamDefinition]:
        """Get the HyperParamDefinition for a specific orientation."""
        config = self.get_param_config_by_orientation(orientation)
        if config is None:
            return None
        return get_hyper_param_definition(config.param)

    def get_options_for_orientation(
        self, orientation: FinalPlotsPlanOrientation, result_bank: ResultBank
    ) -> list[dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]] | tuple[None]:
        """Get the options for a specific orientation."""
        config = self.get_param_config_by_orientation(orientation)
        if config is None:
            return (None,)
        return list(config.get_param_def().expand_values(config.values))

    def get_summary(self) -> Dict[FinalPlotsPlanOrientation, list[str]]:
        """Get a summary of the plot structure."""
        result: Dict[FinalPlotsPlanOrientation, list[str]] = {}
        for orientation in str_enum_values(FinalPlotsPlanOrientation):
            config = self.get_param_config_by_orientation(orientation)
            if config is None:
                result[orientation] = []
                continue

            param_def = get_hyper_param_definition(config.param)
            result[orientation] = [param_def.get_display_name(option) for option in config.values or []]

        return result

    def get_data_requirements_per_cell(self, result_bank: ResultBank) -> dict[Cell, DataReqs]:
        """Generate data requirements for this plot plan based on the result bank."""
        data_reqs_per_cell: defaultdict[Cell, DataReqiermentCollection] = defaultdict(DataReqiermentCollection)
        experiment_orientations = get_experiment_orientations(self.experiment_name)

        # Prepare iterables for original value combinations that define cells
        param_configs_by_orientation = {pc.orientation: pc for pc in self.params if pc.orientation is not None}

        iterables_for_cell_definition_product: list[Sequence[Optional[PossibleHPDTypes]]] = []
        active_orientations_for_cell_definition: list[FinalPlotsPlanOrientation] = []

        # Global context for prompt filteration, derived once for the plot plan
        prompt_filter_context = self.derive_model_arch_and_sizes_context()

        for orientation in experiment_orientations:
            if orientation in param_configs_by_orientation:
                config = param_configs_by_orientation[orientation]
                if config.values:  # Ensure there are values to iterate over
                    iterables_for_cell_definition_product.append(config.values)
                    active_orientations_for_cell_definition.append(orientation)
                else:  # Parameter configured for orientation but no values, effectively empty set for this orientation
                    iterables_for_cell_definition_product.append([None])  # Add a placeholder for product
                    active_orientations_for_cell_definition.append(orientation)  # Still track it
            else:
                # This orientation is not actively varied by a param in this plot plan
                iterables_for_cell_definition_product.append([None])
                active_orientations_for_cell_definition.append(orientation)

        original_value_combinations = product(*iterables_for_cell_definition_product)
        for original_value_combo_tuple in original_value_combinations:
            orientation_to_original_value: Dict[FinalPlotsPlanOrientation, Optional[PossibleHPDTypes]] = {}
            for i, orientation in enumerate(experiment_orientations):
                # Use the value from original_value_combo_tuple corresponding to this orientation
                # This mapping needs to be careful if not all experiment_orientations are in param_configs_by_orientation # noqa: E501
                # The construction of iterables_for_cell_definition_product and original_value_combo_tuple ensures direct mapping  # noqa: E501
                orientation_to_original_value[orientation] = original_value_combo_tuple[i]

            cell = Cell.from_orientation_combination(orientation_to_original_value)

            # For this cell, generate all specific DataReqs by expanding parameters
            param_expansion_lists: list[list[dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]] = []
            prompt_filteration_configs: list[tuple[PromptFilterationHPD, PossibleHPDTypes]] = []

            for param_config in self.params:
                hpd = param_config.get_param_def()
                current_value_for_param: Optional[PossibleHPDTypes] = None

                if param_config.is_fixed():
                    current_value_for_param = param_config.fixed_value
                elif param_config.orientation is not None:  # Variable parameter
                    current_value_for_param = orientation_to_original_value.get(param_config.orientation)
                assert current_value_for_param is not None

                if isinstance(hpd, PromptFilterationHPD):
                    # We are sure current_value_for_param is not None here due to prior checks or it's a config error
                    prompt_filteration_configs.append((hpd, cast(PossibleHPDTypes, current_value_for_param)))
                else:
                    expanded_dicts = list(hpd.expand_values([cast(PossibleHPDTypes, current_value_for_param)]))
                    if expanded_dicts:  # Only add if expansion yields something
                        param_expansion_lists.append(expanded_dicts)

            # Ensure there's exactly one prompt filteration config.
            # The PlotPlan validator should ensure only one PromptFilterationHPD is configured.
            # Here we ensure it was found and has a value for the current cell context.
            assert len(prompt_filteration_configs) == 1
            pf_hpd_instance, pf_original_value = prompt_filteration_configs[0]

            prompt_filteration_derived_value, more_values_from_prompt_filter = (
                pf_hpd_instance.get_derived_hpd_with_context(pf_original_value, prompt_filter_context)
            )

            for expanded_param_combination_tuple in product(*param_expansion_lists):
                data_req_params: dict[VARIANT_PARAM_NAME, Any] = {
                    BaseVariantParamName.experiment_name: self.experiment_name,
                }
                data_req_params.update(more_values_from_prompt_filter)  # Add values from prompt filter first

                has_mismatch = False
                for expanded_dict in expanded_param_combination_tuple:
                    for k, v in expanded_dict.items():
                        if k in data_req_params and data_req_params[k] != v:
                            if v != data_req_params[k]:
                                has_mismatch = True
                                break
                        data_req_params[k] = v
                if has_mismatch:
                    continue

                model_arch = data_req_params.get(BaseVariantParamName.model_arch)
                model_size = data_req_params.get(BaseVariantParamName.model_size)

                if model_arch is not None and model_size is not None:
                    current_mas = MODEL_ARCH_AND_SIZE(
                        cast(MODEL_ARCH, model_arch),
                        cast(TModelSize, model_size),
                    )
                    if current_mas not in GRAPHS_ORDER:
                        continue
                else:  # If model_arch or model_size is not in data_req_params, cannot check GRAPHS_ORDER
                    # This implies an incomplete data_req_param set for this combination.
                    # It might be valid if the experiment doesn't depend on arch/size (unlikely for plots).
                    # For safety, if we can't determine arch/size, and GRAPHS_ORDER is important, skip.
                    # However, validation should ensure arch/size are always derivable if needed.
                    # If an experiment type *can* operate without arch/size, this continue might be too strict.
                    # For now, assume they are needed for GRAPHS_ORDER check.
                    if self.experiment_name in [
                        ExperimentName.info_flow,
                        ExperimentName.heatmap,
                    ]:  # These typically need arch/size
                        continue

                data_reqs_per_cell[cell].add_data_req(
                    init_variant_params_from_values(data_req_params), prompt_filteration_derived_value
                )

        return {
            cell: DataReqs.from_data_reqs_collection(data_reqs)
            for cell, data_reqs in data_reqs_per_cell.items()
            if data_reqs
        }

    def get_data_requirements(self, result_bank: ResultBank) -> DataReqs:
        """Generate aggregated data requirements for the entire plot plan."""
        data_reqs_per_cell = self.get_data_requirements_per_cell(result_bank)
        data_reqs_collection = DataReqiermentCollection()

        for data_reqs_per_cell in data_reqs_per_cell.values():
            for data_req, prompt_filteration in data_reqs_per_cell.items():
                data_reqs_collection.add_data_req(data_req, prompt_filteration)

        return DataReqs.from_data_reqs_collection(data_reqs_collection)

    def derive_model_arch_and_sizes_context(self) -> list[MODEL_ARCH_AND_SIZE]:
        """Derive context model architectures and sizes based on all params in the plot plan."""
        # Check if model_arch_and_size is directly configured
        # This HPD, if present, is the authoritative source for (arch,size) context.
        # TODO: need to handle all type of hpds that handle model_size or model_arch
        mas_param_config = self.get_param_config(VirtualExperimentHyperParams.model_arch_and_size)
        if mas_param_config:
            hpd = mas_param_config.get_param_def()
            all_model_arch_and_sizes: set[MODEL_ARCH_AND_SIZE] = set()

            current_values = mas_param_config.values
            if not current_values and mas_param_config.is_fixed():  # Handle fixed value
                current_values = [mas_param_config.fixed_value]

            if current_values:
                for value_option in current_values:
                    # expand_values expects a list of options, here value_option is a single option
                    for expanded_dict in hpd.expand_values([value_option]):
                        arch = expanded_dict.get(BaseVariantParamName.model_arch)
                        size = expanded_dict.get(BaseVariantParamName.model_size)
                        if arch is not None and size is not None:
                            all_model_arch_and_sizes.add(
                                MODEL_ARCH_AND_SIZE(cast(MODEL_ARCH, arch), cast(TModelSize, size))
                            )
            return [ma for ma in list(all_model_arch_and_sizes) if ma in GRAPHS_ORDER]

        # If model_arch_and_size HPD is not used, derive from separate model_arch and model_size contributors
        potential_model_archs: set[MODEL_ARCH] = set()
        potential_model_sizes: set[TModelSize] = set()

        for config in self.params:
            hpd = config.get_param_def()

            current_values_for_param = config.values
            if not current_values_for_param and config.is_fixed():
                current_values_for_param = [config.fixed_value]

            if not current_values_for_param:
                continue

            # Check if this HPD contributes to model_arch
            if BaseVariantParamName.model_arch in hpd.derived_variants_params():
                for val_option in current_values_for_param:
                    for expanded_dict in hpd.expand_values([val_option]):
                        if BaseVariantParamName.model_arch in expanded_dict:
                            potential_model_archs.add(cast(MODEL_ARCH, expanded_dict[BaseVariantParamName.model_arch]))

            # Check if this HPD contributes to model_size
            if BaseVariantParamName.model_size in hpd.derived_variants_params():
                for val_option in current_values_for_param:
                    for expanded_dict in hpd.expand_values([val_option]):
                        if BaseVariantParamName.model_size in expanded_dict:
                            potential_model_sizes.add(cast(TModelSize, expanded_dict[BaseVariantParamName.model_size]))

        # If no specific arch or size params found, but validation requires them, it's an issue for validator.
        # Here, if sets are empty, product will be empty.

        combined_mas = {
            MODEL_ARCH_AND_SIZE(m, s) for m, s in product(list(potential_model_archs), list(potential_model_sizes))
        }
        return [ma for ma in list(combined_mas) if ma in GRAPHS_ORDER]

    def get_option_display_names_for_orientation(self, orientation: FinalPlotsPlanOrientation) -> list[str]:
        """Get display names for options for an orientation (compatibility method)."""
        config = self.get_param_config_by_orientation(orientation)
        if config is None:
            return []

        param_def = get_hyper_param_definition(config.param)
        return [param_def.get_display_name(option) for option in config.values or []]

    def get_options_for_param(self, orientation: FinalPlotsPlanOrientation) -> List[PossibleHPDTypes]:
        """Get the selected options for a parameter (compatibility method)."""
        config = self.get_param_config_by_orientation(orientation)
        if config is None:
            return []
        return config.values or []

    def save(self) -> None:
        """Save the parameter configuration."""
        PlotPlans.save_plot_plan(self)
