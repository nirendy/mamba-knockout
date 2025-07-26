# Purpose: Provide components for generating plots based on plot plans
# High Level Outline:
# 1. Plot generation components for different plot types
# 2. Utility functions for loading and processing data
# 3. Plot rendering and saving functionality
# Outline Issues:
# - Consider adding more customization options for plots
# - Add support for interactive plots
# Outline Compatibility Issues:
# - New file, outline will be implemented
import contextlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Optional, cast

import matplotlib.figure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import streamlit_antd_components as sac
from more_itertools import unique_everseen
from PIL import Image
from pydantic import BaseModel

from src.analysis.experiment_results.helpers import get_model_evaluations
from src.analysis.experiment_results.hyper_param_definition import FilterationFactoryHPD
from src.analysis.experiment_results.plot_plan import Cell, PlotPlan
from src.analysis.experiment_results.prompt_filteration_factory import PromptFilterationFactory
from src.analysis.plots.heatmaps import HeatmapPlotConfig, simple_diff_fixed
from src.analysis.plots.image_combiner import ImageGridParams, LegendItem, combine_image_grid
from src.analysis.plots.info_flow_confidence import (
    InfoFlowPlotConfig,
    create_confidence_plot,
)
from src.app.texts import FINAL_PLOTS_TEXTS
from src.core.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID
from src.core.names import (
    ExperimentName,
    FinalPlotsPlanOrientation,
    SummarizedDataFulfilledReqsCols,
)
from src.core.types import MODEL_ARCH_AND_SIZE, TInfoFlowOutput, TLineStyle, TPromptData
from src.data_ingestion.data_defs.data_defs import (
    DataReqs,
    PlotPlans,
    ResultBank,
)
from src.data_ingestion.helpers.logits_utils import decode_tokens, get_prompt_row_index
from src.experiments.infrastructure.base_prompt_filteration import SamplePromptFilteration
from src.experiments.infrastructure.base_runner import BaseRunner
from src.experiments.infrastructure.setup_models import get_tokenizer
from src.experiments.runners.heatmap import HeatmapRunner
from src.experiments.runners.info_flow import InfoFlowRunner
from src.utils.infra.image_utils import save_at_dpi
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.streamlit.st_pydantic_v2.input import pydantic_ui
from src.utils.types_utils import class_values


@st.cache_data(hash_funcs={DataReqs: hash, ResultBank: hash})
def _cache_get_runners(data_reqs: DataReqs, result_bank: ResultBank) -> list[BaseRunner]:
    return list(data_reqs.to_fulfilled_reqs(result_bank).choose_latest_fulfilled().get_config().values())


@dataclass
class GridLayout:
    """Handles the organization and rendering of plots in a grid layout."""

    plot_plan: PlotPlan
    cells: list[Cell]
    data_reqs_per_cell: dict[Cell, DataReqs]
    plot_generator: "PlotGenerator"

    @property
    def row_values(self) -> Iterator[Any]:
        return unique_everseen([cell.rows for cell in self.cells])

    @property
    def col_values(self) -> Iterator[Any]:
        return unique_everseen([cell.cols for cell in self.cells])

    def get_cell_at(self, row_value: Any, col_value: Any) -> Optional[Cell]:
        """Get cell at the specified position."""
        return next((cell for cell in self.cells if cell.rows == row_value and cell.cols == col_value), None)

    def get_labels(self) -> tuple[list[str], list[str]]:
        """Get row and column labels."""
        row_labels = []
        col_labels = []

        # Get row labels
        for row_value in self.row_values:
            if row_value is not None:
                row_cell = next(cell for cell in self.cells if cell.rows == row_value)
                row_labels.append(row_cell.get_field_display_name("rows", self.plot_plan))

        # Get column labels
        for col_value in self.col_values:
            if col_value is not None:
                col_cell = next(cell for cell in self.cells if cell.cols == col_value)
                col_labels.append(col_cell.get_field_display_name("cols", self.plot_plan))

        return row_labels, col_labels

    def _get_legend_items_per_row(self) -> dict[Any, list[LegendItem]]:
        """Get legend items for each row separately."""
        legend_per_row: dict[Any, list[LegendItem]] = {}

        for row_value in self.row_values:
            # Collect all DataReqs for cells in this row
            row_data_reqs = []
            for col_value in self.col_values:
                cell = self.get_cell_at(row_value, col_value)
                if cell and cell in self.data_reqs_per_cell:
                    row_data_reqs.append(self.data_reqs_per_cell[cell])

            # Get legend items for this row
            if row_data_reqs:
                legend_per_row[row_value] = self.plot_generator._get_legend_items(relevant_data_reqs_list=row_data_reqs)
            else:
                legend_per_row[row_value] = []

        return legend_per_row

    def _legend_items_to_comparable(self, items: list[LegendItem]) -> frozenset[tuple[str, str, str]]:
        """Convert legend items to a comparable format for equality checking."""
        return frozenset((item.label, item.color, item.linestyle) for item in items)

    def _legend_items_differ_by_row(self) -> bool:
        """Check if legend items differ between rows."""
        legend_per_row = self._get_legend_items_per_row()
        if len(legend_per_row) <= 1:
            return False

        # Convert all to comparable format
        comparable_legends = [self._legend_items_to_comparable(items) for items in legend_per_row.values()]

        # Check if all are the same (if set has more than 1 element, they differ)
        return len(set(comparable_legends)) > 1

    def render_combined(self, recreate_plots: bool, grid_params: ImageGridParams):
        """Render all plots combined into a single image."""
        image_grid: list[list[Optional[Path]]] = []
        row_labels, col_labels = self.get_labels()  # noqa: F841

        # Generate all plots and collect their paths
        for row_value in self.row_values:
            row_images: list[Optional[Path]] = []
            for col_value in self.col_values:
                cell = self.get_cell_at(row_value, col_value)
                if cell:
                    img_path = self.plot_generator._plot_cell(
                        self.data_reqs_per_cell[cell],
                        cell,
                        recreate_plots,
                        show_plot=False,
                        show_button=False,
                    )
                    row_images.append(img_path)
                else:
                    row_images.append(None)
            image_grid.append(row_images)

        # Filter out None values from image grid
        filtered_grid = [[path for path in row if path is not None] for row in image_grid]
        filtered_grid = [row for row in filtered_grid if row]  # Remove empty rows

        # Prepare grid-specific DataReqs for legend items
        grid_specific_data_reqs = [
            self.data_reqs_per_cell[cell] for cell in self.cells if cell in self.data_reqs_per_cell
        ]

        # Check if legend items differ by row
        if self._legend_items_differ_by_row():
            # Use row-specific legends
            legend_per_row = self._get_legend_items_per_row()
            # Convert row values to row indices for combine_image_grid
            legend_items = {
                row_idx: legend_per_row[row_value]
                for row_idx, row_value in enumerate(self.row_values)
                if row_value in legend_per_row
            }

        else:
            # Use single legend for entire grid
            legend_items = self.plot_generator._get_legend_items(relevant_data_reqs_list=grid_specific_data_reqs)

        combined_image = combine_image_grid(
            filtered_grid,
            grid_params,
            legend_items=legend_items,
            col_labels=col_labels,
            row_labels=row_labels,
        )

        if combined_image:
            st.image(combined_image, width=combined_image.width)

        return combined_image

    def render_separate(self, recreate_plots: bool, show_button: bool) -> None:
        """Render plots in separate Streamlit columns."""
        has_row_labels = any(cell.rows is not None for cell in self.cells)
        has_col_labels = any(cell.cols is not None for cell in self.cells)
        row_labels, col_labels = self.get_labels()

        columns_count = ([0.5] if has_row_labels else []) + [1] * len(col_labels)

        progress_bar = st.progress(0, text="Generating plots...")
        total_plots = len(self.cells)
        total_plots_completed = 0
        # Create rows
        for i, row_value in enumerate(self.row_values):
            st_cols = st.columns(columns_count)

            # Show column headers if needed
            if i == 0 and has_col_labels:
                for col_name, col_col in zip((["Row"] if has_row_labels else []) + col_labels, st_cols):
                    col_col.write(f"**{col_name}**")

            # Add row label if needed
            start_col = 0
            if has_row_labels:
                with st_cols[0]:
                    st.write(f"**{row_labels[i]}**")
                start_col = 1

            # Add plots
            for col_value, col_col in zip(self.col_values, st_cols[start_col:]):
                cell = self.get_cell_at(row_value, col_value)
                if cell:
                    with col_col:
                        progress_bar.progress(
                            total_plots_completed / total_plots,
                            text=f"Generating plots... {total_plots_completed}/{total_plots}",
                        )
                        self.plot_generator._plot_cell(
                            self.data_reqs_per_cell[cell], cell, recreate_plots, show_button=show_button
                        )
                        total_plots_completed += 1
                        progress_bar.progress(
                            total_plots_completed / total_plots,
                            text=f"Generating plots... {total_plots_completed}/{total_plots}",
                        )
        progress_bar.empty()


class Tabs:
    PLOT_INDIVIDUAL = "Plot Individually"
    PLOT_COMBINED = "Plot Combined"
    CUSTOMIZE_PLOT = "Customize Plot"


@dataclass
class PlotGenerator(StreamlitComponent[Optional[str]]):
    """Component for generating plots based on plot plans."""

    plot_plan: PlotPlan
    result_bank: ResultBank

    def _get_cell_cache_path(self, grid_name: Any, row_name: Any, col_name: Any) -> Path:
        """Generate a unique cache path for a cell's plot."""
        cache_dir = PlotPlans.get_cache_dir(self.plot_plan.plot_id)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique identifier for the cell
        cell_id = f"{grid_name}_{row_name}_{col_name}".replace(" ", "_")
        return cache_dir / f"{cell_id}.png"

    def _get_combined_plot_cache_path(self, grid_name: Any) -> Path:
        """Generate a unique cache path for the combined plot."""
        cache_dir = PlotPlans.get_plot_plan_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        grid_hpd = self.plot_plan.get_orientation_value_hpd(FinalPlotsPlanOrientation.grids)
        if grid_hpd is not None:
            grid_display_name = grid_hpd.get_display_name(grid_name)
        else:
            grid_display_name = grid_name
        return cache_dir / f"{self.plot_plan.plot_id}_{grid_display_name}.png"

    def _get_legend_items(self, relevant_data_reqs_list: Optional[list[DataReqs]] = None) -> list[LegendItem]:
        plot_config = self._get_config_for_experiment_name(self.plot_plan.experiment_name, None)
        legend_items = []
        if isinstance(plot_config, InfoFlowPlotConfig):
            relevant_line_ids: Optional[set[str]] = None
            if relevant_data_reqs_list:
                relevant_line_ids = set()
                lines_param_config = self.plot_plan.get_param_config_by_orientation(FinalPlotsPlanOrientation.lines)
                assert lines_param_config is not None
                original_lines_hpd = lines_param_config.get_param_def()

                for data_reqs in relevant_data_reqs_list:
                    runners = _cache_get_runners(data_reqs, self.result_bank)
                    for runner_instance in runners:
                        if isinstance(runner_instance, InfoFlowRunner):
                            line_id_str = original_lines_hpd.get_line_id_from_runner(runner_instance.variant_params)
                            relevant_line_ids.add(line_id_str)

            for line_id, color in plot_config.custom_colors.items():
                if relevant_line_ids is None or line_id in relevant_line_ids:
                    legend_items.append(
                        LegendItem(
                            label=plot_config.custom_line_labels.get(line_id, line_id),
                            color=color.as_hex(),
                            linestyle=plot_config.custom_line_styles.get(line_id, TLineStyle.solid.value),
                        )
                    )
        return legend_items

    def _plot_data_reqs(self, data_reqs: DataReqs, cell_plot_config: dict[str, Any]):
        runners = _cache_get_runners(data_reqs, self.result_bank)

        fig = None
        match self.plot_plan.experiment_name:
            case ExperimentName.info_flow:
                # Convert dict to InfoFlowPlotConfig
                config = self._get_config_for_experiment_name(self.plot_plan.experiment_name, cell_plot_config)
                assert isinstance(config, InfoFlowPlotConfig)
                fig = self._generate_cell_knockout(runners, config)
            case ExperimentName.heatmap:
                assert len(runners) == 1
                runner = runners[0]
                assert isinstance(runner, HeatmapRunner)
                prompt_idx = runner.input_params.filteration.get_prompt_ids()
                assert len(prompt_idx) == 1
                prompt_id = prompt_idx[0]
                model_arch_and_size = MODEL_ARCH_AND_SIZE(
                    runner.variant_params.model_arch, runner.variant_params.model_size
                )
                data = cast(
                    TPromptData,
                    get_model_evaluations(runner.metadata_params.code_version, [model_arch_and_size])[
                        model_arch_and_size
                    ],
                )
                tokenizer = get_tokenizer(runner.variant_params.model_arch, runner.variant_params.model_size)
                model_id = MODEL_SIZES_PER_ARCH_TO_MODEL_ID[runner.variant_params.model_arch][
                    runner.variant_params.model_size
                ]
                prob_mat = runner.get_outputs()[prompt_id]
                prompt = get_prompt_row_index(data, prompt_id)
                input_ids = prompt.input_ids(tokenizer, "cpu")
                toks = cast(list[str], decode_tokens(tokenizer, input_ids[0]))
                for i, tok in enumerate(toks):
                    if input_ids[0][i] == tokenizer.bos_token_id:
                        toks[i] = "<BOS>"
                    if input_ids[0][i] == tokenizer.eos_token_id:
                        toks[i] = "<EOS>"
                last_tok = toks[-1]
                toks[-1] = toks[-1] + "*"

                # Convert dict to HeatmapPlotConfig
                config = self._get_config_for_experiment_name(self.plot_plan.experiment_name, cell_plot_config)
                assert isinstance(config, HeatmapPlotConfig)

                fig, _ = simple_diff_fixed(
                    prob_mat=prob_mat,
                    model_id=model_id,
                    window_size=runner.variant_params.window_size,
                    last_tok=last_tok,
                    base_prob=prompt.base_prob,
                    target_rank=prompt.target_rank,
                    true_word=prompt.true_word,
                    toks=toks,
                    config=config,
                )

            case _:
                raise ValueError(f"Unknown experiment name: {self.plot_plan.experiment_name}")

        return fig

    def _save_plot_plan(self):
        PlotPlans.save_plot_plan(self.plot_plan)

    def _plot_cell(
        self,
        data_reqs: DataReqs,
        cell: Cell,
        recreate: bool = False,
        with_plotly: bool = False,
        show_plot: bool = True,
        show_button: bool = True,
    ) -> Path:
        """Plot a single cell with caching."""
        cache_path = PlotPlans.get_cell_cache_path(self.plot_plan, cell)
        recreate = recreate or (show_button and st.button(f"Recreate_{cache_path.name}"))
        if not recreate and cache_path.exists():
            # Load and display cached plot if needed
            if show_plot:
                st.image(str(cache_path))
            return cache_path

        fig = self._plot_data_reqs(data_reqs, self.plot_plan.cell_plot_config.model_dump())
        if fig is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(fig, go.Figure):
                fig.write_image(str(cache_path), scale=4)
            elif isinstance(fig, matplotlib.figure.Figure):
                fig.savefig(str(cache_path), dpi=600)
                plt.close(fig)
            else:
                st.warning(
                    f"Plot for {cache_path.name} was of unexpected type {type(fig)}"
                    "and could not be saved as a known image type."
                )

            # Display the plot if needed
            if show_plot:
                if with_plotly:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.image(str(cache_path))

        return cache_path

    def _generate_cell_knockout(self, runners: list[BaseRunner], cell_plot_config: InfoFlowPlotConfig):
        """Generate knockout plot for a single cell."""
        data: dict[str, TInfoFlowOutput] = {}

        lines_param_config = self.plot_plan.get_param_config_by_orientation(FinalPlotsPlanOrientation.lines)
        assert lines_param_config is not None
        original_lines_hpd = lines_param_config.get_param_def()

        for i, runner_instance in enumerate(runners):
            assert isinstance(runner_instance, InfoFlowRunner), f"Expected InfoFlowRunner, got {type(runner_instance)}"

            if isinstance(original_lines_hpd, FilterationFactoryHPD):
                cur_prompt_filteration = lines_param_config.values[i]
                assert isinstance(cur_prompt_filteration, PromptFilterationFactory)
                line_id_str = original_lines_hpd.get_display_name(cur_prompt_filteration)
            else:
                line_id_str = original_lines_hpd.get_line_id_from_runner(runner_instance.variant_params)
            data[line_id_str] = runner_instance.get_outputs()

        fig = create_confidence_plot(
            lines=data,
            confidence_level=cell_plot_config.confidence_level,
            config=cell_plot_config,
        )

        return fig

    def _get_model_for_experiment_name(self, experiment_name: ExperimentName):
        """Get the appropriate configuration model based on experiment name."""
        if experiment_name == ExperimentName.info_flow:
            return InfoFlowPlotConfig.specify_config(
                self.plot_plan.get_option_display_names_for_orientation(FinalPlotsPlanOrientation.lines)
            )
        elif experiment_name == ExperimentName.heatmap:
            return HeatmapPlotConfig
        else:
            raise ValueError(f"Experiment name {experiment_name} is not implemented")

    def _get_config_for_experiment_name(
        self, experiment_name: ExperimentName, config: Optional[BaseModel | dict[str, Any]]
    ):
        """Get the appropriate configuration model based on experiment name."""
        if config is None:
            config = self.plot_plan.cell_plot_config
        if isinstance(config, BaseModel):
            config = config.model_dump()
        return self._get_model_for_experiment_name(experiment_name).model_validate(config)

    def render(self) -> Optional[str]:
        """Generate and display a plot based on the plot plan."""
        st.subheader(f"{FINAL_PLOTS_TEXTS.generating_plot(self.plot_plan.title)}")

        # Check if we have all the required data
        data_reqs = self.plot_plan.get_data_requirements(self.result_bank)
        summarized_fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank).summarize()
        reqs = summarized_fulfilled_reqs.to_data_reqs().to_rows()

        missing_reqs = [
            req
            for summary, req in zip(summarized_fulfilled_reqs, reqs)
            if summary[SummarizedDataFulfilledReqsCols.AvailableOptions] == 0
        ]

        if missing_reqs:
            st.error(f"Missing data for {len(missing_reqs)} requirements. Please run the missing requirements first.")
            return None

        # Generate the plot based on the plot type
        data_reqs_per_cell = self.plot_plan.get_data_requirements_per_cell(self.result_bank)

        # Add checkbox for plot recreation
        tab = sac.tabs([sac.TabsItem(label=tab_name) for tab_name in class_values(Tabs)])
        combine_plots = tab == Tabs.PLOT_COMBINED

        if tab == Tabs.CUSTOMIZE_PLOT:
            # Initialize configs if they don't exist
            # Show customization UI
            st.write("### Cell Plot Configuration")
            save_configuration = st.button("Save Configuration")
            config_model = self._get_model_for_experiment_name(self.plot_plan.experiment_name)
            with st.sidebar:
                with st.expander("Cell Plot Configuration"):
                    cell_config_dict = pydantic_ui(
                        key=f"cell_config_{self.plot_plan.plot_id}",
                        model=config_model.model_validate(self.plot_plan.cell_plot_config.model_dump()),
                    ).model_dump()

            data_reqs_cells = list(data_reqs_per_cell.keys())
            # Show preview of first cell
            if data_reqs_per_cell:
                select_col, sample_col = st.columns(2)
                with select_col:
                    indcies = st.multiselect(
                        "Select cell to preview",
                        range(len(data_reqs_cells)),
                        default=[0],
                        format_func=lambda i: data_reqs_cells[i].get_display_name(self.plot_plan),
                    )

                for i in indcies:
                    cell_to_show = data_reqs_cells[i]

                    runner = next(
                        iter(data_reqs_per_cell[cell_to_show].to_fulfilled_reqs(self.result_bank).get_config().values())
                    )
                    prompt_filteration = runner.input_params.filteration
                    prompt_ids = prompt_filteration.get_prompt_ids()

                    if self.plot_plan.experiment_name == ExperimentName.info_flow:
                        with sample_col:
                            sample_results_count = st.slider(
                                "Sample results count",
                                min_value=50,
                                max_value=len(prompt_ids),
                                key=f"sample_results_count_{cell_to_show.get_display_name(self.plot_plan)}",
                                value=50,
                            )

                        data_req = DataReqs(
                            {
                                runner: SamplePromptFilteration(
                                    base_prompt_filteration=prompt_filteration,
                                    sample_size=sample_results_count,
                                    seed=42,
                                )
                                for runner, prompt_filteration in data_reqs_per_cell[cell_to_show].items()
                            }
                        )
                    else:
                        data_req = data_reqs_per_cell[cell_to_show]

                    fig = self._plot_data_reqs(data_req, cell_config_dict)
                    # st.pyplot(fig, use_container_width=False)
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    st.image(buf)

            # Add save button
            if save_configuration:
                self.plot_plan.cell_plot_config = cell_config_dict
                self._save_plot_plan()
                st.toast("Configuration saved successfully!")

        else:
            recreate_plots = tab == Tabs.PLOT_INDIVIDUAL and st.button("Recreate all plots")
            show_recreate_button = tab == Tabs.PLOT_INDIVIDUAL and st.checkbox("Show recreate buttons", value=False)
            save_combined_plot = tab == Tabs.PLOT_COMBINED and st.checkbox("Save Combined Plot", value=True)
            save_configuration = tab == Tabs.PLOT_COMBINED and st.button("Save Configuration")
            grid_progress_bar = st.progress(0, text="Generating Grids...")

            # Group cells by grid
            cells_by_grid: dict[Any, list[Cell]] = {}
            for cell in data_reqs_per_cell:
                cells_by_grid.setdefault(cell.grids, []).append(cell)

            # Create tabs for different plot views
            if len(cells_by_grid) == 1 and None in cells_by_grid:
                tabs = [contextlib.nullcontext()]
                grid_names = [None]
            else:
                grid_names = self.plot_plan.get_options_for_param(FinalPlotsPlanOrientation.grids)
                grid_display_names = self.plot_plan.get_option_display_names_for_orientation(
                    FinalPlotsPlanOrientation.grids
                )
                tabs = st.tabs(grid_display_names)

            maybe_combined_image = None
            grid_params = None
            if combine_plots:
                image_path = self._plot_cell(
                    data_reqs_per_cell[cells_by_grid[grid_names[0]][0]],
                    cells_by_grid[grid_names[0]][0],
                    recreate_plots,
                    show_plot=False,
                    show_button=False,
                )
                image = Image.open(image_path)

                specified_image_grid_params = ImageGridParams.specify_config(
                    rows=self.plot_plan.get_option_display_names_for_orientation(FinalPlotsPlanOrientation.rows),
                    columns=self.plot_plan.get_option_display_names_for_orientation(FinalPlotsPlanOrientation.cols),
                    image=image,
                )
                # Get configuration from plot_plan
                combine_config = specified_image_grid_params.model_validate(
                    self.plot_plan.combine_plot_config.model_dump()
                )

                with st.sidebar.expander("Combined Plot Configuration"):
                    grid_params = specified_image_grid_params.model_validate(
                        pydantic_ui(
                            key=f"combine_config_{self.plot_plan.plot_id}",
                            model=combine_config,
                        )
                    )

                    # Add save button
                    if save_configuration:
                        self.plot_plan.combine_plot_config = grid_params
                        self._save_plot_plan()
                        st.toast("Configuration saved successfully!")
            # Render each grid
            total_grid_plots = len(grid_names)
            total_grid_plots_completed = 0
            for grid_name, tab in zip(grid_names, tabs):
                grid_layout = GridLayout(
                    plot_plan=self.plot_plan,
                    cells=cells_by_grid[grid_name],
                    data_reqs_per_cell=data_reqs_per_cell,
                    plot_generator=self,
                )
                with tab:
                    if combine_plots:
                        assert grid_params is not None
                        maybe_combined_image = grid_layout.render_combined(recreate_plots, grid_params)
                    else:
                        grid_layout.render_separate(recreate_plots, show_button=show_recreate_button)
                    total_grid_plots_completed += 1
                    grid_progress_bar.progress(
                        total_grid_plots_completed,
                        text=f"Generating plots... {total_grid_plots_completed}/{total_grid_plots}",
                    )

                if maybe_combined_image and save_combined_plot:
                    path = self._get_combined_plot_cache_path(grid_name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_at_dpi(maybe_combined_image, str(path))
                    st.toast("Combined plot saved successfully!")
            grid_progress_bar.empty()
