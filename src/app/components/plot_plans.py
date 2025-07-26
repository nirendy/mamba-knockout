# Purpose: Provide components for managing plot plans in the final plots page
# High Level Outline:
# 1. Plot plan management components (add, edit, delete)
# 2. Plot plan display components
# 3. Plot plan execution components
# Outline Issues:
# - Consider adding batch operations for plot plans
# - Add visualization preview for plot plans
# Outline Compatibility Issues:
# - New file, outline will be implemented

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, TypedDict, Union, cast

import streamlit as st
import streamlit_antd_components as sac

from src.analysis.experiment_results.hyper_param_definition import (
    PossibleHPDTypes,
    TExperimentHyperParams,
    VirtualExperimentHyperParams,
    get_hyper_param_definition,
)
from src.analysis.experiment_results.plot_plan import (
    ParamConfig,
    PlotPlan,
    get_experiment_orientations,
)
from src.analysis.experiment_results.prompt_filteration_factory import PromptFilterationFactory
from src.analysis.plots.image_combiner import ImageGridParams
from src.app.components.prompt_filter import SelectFilterationFactoryComponent
from src.app.texts import FINAL_PLOTS_TEXTS
from src.core.names import (
    BaseVariantParamName,
    ExperimentName,
    FinalPlotsPlanOrientation,
    ToClassifyNames,
)
from src.core.types import TPlotID
from src.data_ingestion.data_defs.data_defs import PlotPlans
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.streamlit.helpers.session_keys import SessionKey
from src.utils.types_utils import get_enum_or_literal_options, str_enum_values

# Session keys for plot plans


class PlotPlanSelector(StreamlitComponent[None]):
    """Component for selecting a plot plan from the list of available plans."""

    def __init__(
        self,
        plot_plans: PlotPlans,
        selected_plot_id_sk: SessionKey[TPlotID],
        new_label: TPlotID,
    ):
        self.plot_plans = plot_plans
        self.selected_plot_id_sk = selected_plot_id_sk
        self.new_plot_id = new_label

    def render(self):
        if not self.plot_plans.is_plan_exists(self.selected_plot_id_sk.value):
            self.selected_plot_id_sk.value = self.new_plot_id

        # Group plans by appendix/main
        main_plans = [p for p in self.plot_plans.values() if not p.is_appendix]
        appendix_plans = [p for p in self.plot_plans.values() if p.is_appendix]

        # Create menu items
        menu_items: List[Union[str, dict, sac.MenuItem]] = []

        if self.new_plot_id:
            menu_items.append(sac.MenuItem(self.new_plot_id, icon="plus-circle"))

        if main_plans:
            menu_items.append(sac.MenuItem("Main Plots", icon="graph-up", disabled=True))
            for plan in main_plans:
                menu_items.append(
                    sac.MenuItem(
                        plan.plot_id,
                        icon="file-earmark-bar-graph",
                        tag=plan.experiment_name.name,
                    )
                )

        if appendix_plans:
            menu_items.append(sac.MenuItem("Appendix Plots", icon="journal-code", disabled=True))
            for plan in appendix_plans:
                menu_items.append(
                    sac.MenuItem(
                        plan.plot_id,
                        icon="file-earmark-bar-graph",
                        tag=plan.experiment_name.name,
                    )
                )

        sac.menu(
            items=menu_items,
            size="xs",
            format_func=lambda x: self.plot_plans.get_plan(x).plot_id if self.plot_plans.is_plan_exists(x) else x,
            key=self.selected_plot_id_sk.key_for_component,
            return_index=False,
        )


@dataclass
class PlotPlanDetailsSummary(StreamlitComponent[None]):
    """Component for displaying the details of a selected plot plan."""

    plot_plans: PlotPlans
    selected_plan_id: Optional[TPlotID]

    def render(self) -> None:
        if not self.selected_plan_id:
            st.info(FINAL_PLOTS_TEXTS.no_plan_selected)
            return

        plan = self.plot_plans.get_plan(self.selected_plan_id)

        col1, col2 = st.columns([3, 5])
        with col1:
            st.dataframe(
                {
                    col: str(getattr(plan, col))
                    for col in [
                        "plot_id",
                        "title",
                        "description",
                        "experiment_name",
                        "is_appendix",
                        "order",
                    ]
                },
                use_container_width=True,
                column_config={
                    "description": st.column_config.TextColumn(
                        width="medium",
                    ),
                },
            )

        class SummaryRow(TypedDict):
            orientation: FinalPlotsPlanOrientation
            param: TExperimentHyperParams
            options_count: int
            options: list[str]

        configuration_data: List[SummaryRow] = []
        for orientation in str_enum_values(FinalPlotsPlanOrientation):
            if orientation == FinalPlotsPlanOrientation.lines and plan.experiment_name != ExperimentName.info_flow:
                continue

            # Get param configuration directly
            param_config = plan.get_param_config_by_orientation(orientation)
            if param_config:
                param = param_config.param
                options = param_config.values
                variation_option = get_hyper_param_definition(param)
                if not options:
                    # get all options from result bank
                    options = variation_option.get_options()
                configuration_data.append(
                    {
                        "orientation": orientation,
                        "param": param,
                        "options_count": len(options),
                        "options": [variation_option.get_display_name(option) for option in options],
                    }
                )

        with col2:
            st.dataframe(configuration_data)

        # Display summary
        summary = plan.get_summary()
        grid_total = 1
        grid_structure_text_parts = []
        for orientation in [
            FinalPlotsPlanOrientation.rows,
            FinalPlotsPlanOrientation.cols,
            FinalPlotsPlanOrientation.grids,
        ]:
            size = max(len(summary[orientation]), 1)
            grid_structure_text_parts.append(f"{size} {orientation.value}")
            grid_total *= size

        with col2:
            st.dataframe(
                {
                    FINAL_PLOTS_TEXTS.total_plots_title: (
                        f"{grid_total} = ({FINAL_PLOTS_TEXTS.grid_structure(grid_structure_text_parts)})"
                    ),
                    **(
                        {
                            FINAL_PLOTS_TEXTS.lines_per_plot_title: str(
                                max(len(summary[FinalPlotsPlanOrientation.lines]), 1)
                            ),
                        }
                        if plan.experiment_name == ExperimentName.info_flow
                        else {}
                    ),
                },
                use_container_width=True,
            )


@dataclass
class PlotPlanEditor(StreamlitComponent[Optional[PlotPlan]]):
    """Component for editing or creating a plot plan."""

    plot_plans: PlotPlans
    plan_id: Optional[TPlotID]

    @property
    def is_new(self) -> bool:
        return self.plan_id is None

    def _get_options_for_param(self, param_type: Optional[TExperimentHyperParams]) -> List[Any]:
        """Get available options for a parameter type."""
        if not param_type:
            return []

        return list(get_hyper_param_definition(param_type).get_options())

    def _get_display_names_for_options(
        self, options: List[Any], param_type: Optional[TExperimentHyperParams]
    ) -> List[str]:
        """Get display names for options."""
        if not param_type or not options:
            return []

        variation_option = get_hyper_param_definition(param_type)
        return [variation_option.get_display_name(option) for option in options]

    def _display_option_selector(
        self,
        param_type: FinalPlotsPlanOrientation,
        existing_plan: Optional[PlotPlan],
        experiment_name: ExperimentName,
        param_value: Optional[TExperimentHyperParams],
    ) -> Tuple[List[Any], bool]:
        """Display a multi-select for parameter options."""
        if not param_value:
            return [], False

        # Check if this parameter is relevant for the experiment type
        relevant_params = get_experiment_orientations(experiment_name)
        if param_type not in relevant_params:
            return [], False

        # Get available options
        available_options = self._get_options_for_param(param_value)
        if not available_options:
            return [], False

        # Get display names for options
        display_names = self._get_display_names_for_options(available_options, param_value)
        options_map = {name: option for name, option in zip(display_names, available_options)}

        # Get currently selected options
        selected_options = []
        if existing_plan:
            param_config = existing_plan.get_param_config_by_orientation(param_type)
            if param_config:
                selected_options = param_config.values
                selected_display_names = self._get_display_names_for_options(selected_options, param_value)
            else:
                selected_display_names = []
        else:
            selected_display_names = []

        # Display multi-select
        st.markdown(f"**Select {param_type.value.capitalize()} Options:**")
        selected_names = st.multiselect(
            f"Available {param_value} options",
            options=display_names,
            default=selected_display_names,
            help=f"Select specific {param_value} values to include in the plot",
            key=f"multiselect_{param_type}",
        )

        # Convert selected names back to actual options
        selected = [options_map[name] for name in selected_names]

        return selected, True

    def render(self) -> Optional[PlotPlan]:
        # Get the existing plan if editing
        if self.plan_id:
            existing_plan = self.plot_plans.get_plan(self.plan_id)
        else:
            existing_plan = PlotPlan(
                plot_id=TPlotID(""),
                experiment_name=ExperimentName.info_flow,
                is_appendix=False,
                order=0,
                combine_plot_config=ImageGridParams(),
            )

        # Form for editing/creating a plot plan
        st.subheader("New Plot Plan" if self.is_new else "Edit Plot Plan")

        # Create data dictionaries to collect form values
        orientation_data = {}
        fixed_values_data: dict[TExperimentHyperParams, PossibleHPDTypes] = {}

        for i, col in enumerate(st.columns([2, 5, 1, 1])):
            with col:
                if i == 0:
                    existing_plan.plot_id = TPlotID(
                        st.text_input(
                            "Plot ID",
                            value=existing_plan.plot_id,
                            help="Path where the plot will be saved",
                            disabled=not self.is_new,
                        )
                    )
                elif i == 1:
                    existing_plan.experiment_name = ExperimentName(
                        st.selectbox(
                            "Experiment",
                            options=[exp.name for exp in ExperimentName],
                            index=list(ExperimentName).index(existing_plan.experiment_name),
                            help="Experiment type for the plot",
                        )
                    )

                elif i == 2:
                    existing_plan.order = st.number_input(
                        "Order",
                        value=existing_plan.order,
                        help="Order of the plot plan",
                    )
                elif i == 3:
                    existing_plan.is_appendix = st.checkbox(
                        "Appendix",
                        value=existing_plan.is_appendix,
                        help="Whether this plot should be included in the appendix",
                    )

        # Get all available hyperparameters
        hyperparams_options = get_enum_or_literal_options(TExperimentHyperParams)
        missing_cols: set[TExperimentHyperParams | Literal[ToClassifyNames.prompt_filteration]] = set(
            ExperimentName.get_variant_cols(existing_plan.experiment_name)
        )
        missing_cols.remove(BaseVariantParamName.experiment_name)
        missing_cols.add(ToClassifyNames.prompt_filteration)

        # Get experiment-specific parameters
        orientations = get_experiment_orientations(existing_plan.experiment_name)

        NONE_STR = "None"
        # Parameter selection
        for i, col in enumerate(st.columns(len(orientations))):
            with col:
                orientation = orientations[i]
                param_name = orientation.value
                current_index = 0
                if current_value := existing_plan.get_param_config_by_orientation(orientation):
                    current_index = hyperparams_options.index(current_value.param.name) + 1
                _orientation_input = st.selectbox(
                    orientation.value.capitalize(),
                    options=[NONE_STR] + hyperparams_options,
                    index=current_index,
                    help="Parameter to vary across rows",
                    key=f"select_{param_name}",
                )
                orientation_input = (
                    None if _orientation_input == NONE_STR else cast(TExperimentHyperParams, _orientation_input)
                )

                # Store orientation parameters
                if orientation_input:
                    orientation_data[orientation] = orientation_input
                    hpd = get_hyper_param_definition(orientation_input)
                    missing_cols -= set(hpd.derived_variants_params())
                    selected_options, has_options = self._display_option_selector(
                        orientation, existing_plan, existing_plan.experiment_name, orientation_input
                    )
                    if has_options:
                        # Store selected options for this orientation
                        orientation_data[f"{orientation}_options"] = selected_options

        missing_no_default_cols = []

        for col_name, st_col in zip(missing_cols, st.columns(len(missing_cols))):
            if col_name == ToClassifyNames.prompt_filteration:
                if existing_plan.experiment_name == ExperimentName.info_flow:
                    col_name = VirtualExperimentHyperParams.filteration_factory
                elif existing_plan.experiment_name == ExperimentName.heatmap:
                    col_name = VirtualExperimentHyperParams.prompt_idx
                else:
                    raise NotImplementedError(f"Experiment {existing_plan.experiment_name} not implemented")
            col_name = cast(TExperimentHyperParams, col_name)
            hpd_col = cast(TExperimentHyperParams, col_name)
            hpd = get_hyper_param_definition(hpd_col)
            param_config = existing_plan.get_param_config(hpd_col)
            original_value = param_config and param_config.fixed_value

            if col_name == VirtualExperimentHyperParams.filteration_factory:
                assert original_value is None or isinstance(original_value, PromptFilterationFactory)
                fixed_values_data[col_name] = SelectFilterationFactoryComponent(
                    key=f"select_{col_name}",
                    default_filteration_factory=original_value,
                ).render()
                continue

            options = hpd.get_options()
            try:
                index = options.index(original_value or hpd.default_fix_value())
            except NotImplementedError:
                missing_no_default_cols.append(col_name)
                continue
            with st_col:
                # Store fixed values
                fixed_values_data[hpd_col] = st.selectbox(
                    col_name.capitalize(),
                    options=options,
                    index=index,
                    key=f"select_{col_name}",
                )

        if missing_no_default_cols:
            st.error(f"Missing column: {missing_no_default_cols}")
            return

        # Submit button
        submit_button = st.button("Save Plot Plan")

        # Display option selectors outside the form
        if not submit_button:
            # Display summary
            if any(existing_plan.get_param_config_by_orientation(orientation) for orientation in orientations):
                st.subheader("Plot Summary")

                rows_config = existing_plan.get_param_config_by_orientation(FinalPlotsPlanOrientation.rows)
                cols_config = existing_plan.get_param_config_by_orientation(FinalPlotsPlanOrientation.cols)
                grids_config = existing_plan.get_param_config_by_orientation(FinalPlotsPlanOrientation.grids)
                lines_config = existing_plan.get_param_config_by_orientation(FinalPlotsPlanOrientation.lines)

                rows_count = len(rows_config.values) if rows_config else 1
                cols_count = len(cols_config.values) if cols_config else 1
                grids_count = len(grids_config.values) if grids_config else 1
                lines_count = len(lines_config.values) if lines_config else 1

                total_plots = rows_count * cols_count * grids_count

                st.markdown(f"**Total plots:** {total_plots}")
                st.markdown(f"**Grid structure:** {rows_count} rows × {cols_count} columns × {grids_count} grids")
                if existing_plan.experiment_name == ExperimentName.info_flow:
                    st.markdown(f"**Lines per plot:** {lines_count}")

        if submit_button:
            # Clear existing params to rebuild them
            existing_plan.params = []

            # Add orientation parameters
            for orientation, param in orientation_data.items():
                if isinstance(orientation, FinalPlotsPlanOrientation):
                    existing_plan.params.append(
                        ParamConfig(
                            param=param,
                            orientation=orientation,
                            values=orientation_data.get(f"{orientation}_options", []),
                        )
                    )

            # Add fixed values
            for param, value in fixed_values_data.items():
                existing_plan.params.append(
                    ParamConfig(
                        param=param,
                        orientation=None,
                        values=[value],
                    )
                )

            return existing_plan

        return None
