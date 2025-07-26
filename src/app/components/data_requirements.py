import json
from dataclasses import asdict
from typing import cast

import pandas as pd
import rich
import rich.errors
import rich.traceback
import streamlit as st
from rich.console import Console
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode

from src.analysis.experiment_results.helpers import init_runner_from_params
from src.app.app_consts import (
    GLOBAL_APP_CONSTS,
    AppSessionKeys,
    DataReqConsts,
)
from src.app.components.inputs import select_gpu_type, select_window_size
from src.app.components.result_bank import ShowRunnerStatus
from src.core.names import HeatmapCols, SlurmStatus, SummarizedDataFulfilledReqsCols
from src.core.types import MODEL_ARCH_AND_SIZE, TCodeVersionName, TPromptOriginalIndex, TWindowSize
from src.data_ingestion.data_defs.data_defs import DataReqs, SummarizedDataFulfilledReqs
from src.experiments.infrastructure.base_prompt_filteration import SelectivePromptFilteration
from src.experiments.infrastructure.base_runner import InputParams, MetadataParams
from src.experiments.runners.evaluate_model import EvaluateModelRunner
from src.experiments.runners.heatmap import HeatmapParams, HeatmapRunner
from src.experiments.runners.info_flow import InfoFlowRunner
from src.utils.streamlit.components.aagrid import SelectionMode, base_grid_builder, set_aagrid_apply_default_filters
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.types_utils import ommit_none, select_indexes_from_list

console = Console()


class RequirementsDisplay(StreamlitComponent):
    def __init__(
        self,
        summarized_data_fulfilled_reqs: SummarizedDataFulfilledReqs,
        selection_mode: SelectionMode,
        height: int | None = None,
        key: str = "data_requirements",
        hide_columns: list[str] = [],
    ):
        self.summarized_data_fulfilled_reqs = summarized_data_fulfilled_reqs
        self.height = height
        self.key = key
        self.selection_mode = selection_mode
        self.hide_columns = hide_columns

    def render(self) -> DataReqs | None:
        original_df = self.summarized_data_fulfilled_reqs.to_df()
        data_reqs_df = original_df[DataReqConsts.DATA_REQS_FILTER_COLUMNS]

        df, grid_builder = base_grid_builder(data_reqs_df, self.selection_mode, hide_columns=self.hide_columns)
        for col in DataReqConsts.DATA_REQS_FILTER_COLUMNS:
            grid_builder.configure_column(col, type=["textColumn"])
        grid_options = grid_builder.build()
        set_aagrid_apply_default_filters(
            grid_builder,
            {SummarizedDataFulfilledReqsCols.AvailableOptions: ["0"]},
        )
        # Display the table
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            height=cast(int, self.height),  # allow None
            fit_columns_on_grid_load=True,
            floatingFilter=True,
            key=self.key,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED,
            allow_unsafe_jscode=True,
        )
        if self.selection_mode == SelectionMode.DISABLED:
            return None

        if grid_response["selected_data"] is None or len(grid_response["selected_data"]) == 0:
            st.warning("No requirements selected")
            return None
        return DataReqs(
            dict(
                select_indexes_from_list(
                    self.summarized_data_fulfilled_reqs.to_data_reqs().to_rows(),
                    [int(i) for i in grid_response["selected_data"].index],
                )
            )
        )


class RequirementExecution(StreamlitComponent):
    def __init__(self, data_reqs_to_run: DataReqs, key: str = "requirement_execution"):
        self.data_reqs_to_run = data_reqs_to_run
        self.key = key

    def render(self):
        # Add SLURM configuration in sidebar
        if len(self.data_reqs_to_run) == 1:
            with_slurm = st.checkbox("Run with SLURM", value=False)
        else:
            with_slurm = True

        # Show count of selected requirements

        # SLURM configuration
        param_cols = st.columns(4, vertical_alignment="center")
        status_cols = st.columns([3, 2])

        with param_cols[0]:
            AppSessionKeys.code_version.create_input_widget()

        with param_cols[1]:
            if with_slurm:
                select_gpu_type()
        with param_cols[2]:
            skip_scheduled = st.checkbox("Skip scheduled jobs", value=True)

        table_data = []
        configs = []
        configs_to_run = []
        for i, (req, filteration) in enumerate(self.data_reqs_to_run.items()):
            config = init_runner_from_params(
                req,
                InputParams(filteration=filteration),
                MetadataParams(
                    code_version=AppSessionKeys.code_version.value,
                    with_slurm=with_slurm,
                    slurm_gpu_type=AppSessionKeys.get_selected_gpu(req.model_arch_and_size),
                ),
            )
            configs.append(config)
            requested_prompts = set(filteration.contextualize(config).get_prompt_ids())

            if isinstance(config, HeatmapRunner):
                remaining_prompts = set(config.get_remaining_prompt_original_indices())
                computed_prompts = requested_prompts - remaining_prompts
                banned_prompts = None
            elif isinstance(config, EvaluateModelRunner):
                remaining_prompts = set()
                computed_prompts = requested_prompts - remaining_prompts
                banned_prompts = None
            elif isinstance(config, InfoFlowRunner):
                computed_prompts = config.output_file.get_computed_prompt_idx()
                banned_prompts = len(config.output_file.get_banned_prompt_indices())
                remaining_prompts = requested_prompts - computed_prompts
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")

            status = None
            if remaining_prompts:
                if skip_scheduled:
                    status = config.slurm_job_folder.get_latest_slurm_job_status()
                    if not isinstance(status, SlurmStatus) or not status.scheduled():
                        configs_to_run.append(config)
                else:
                    configs_to_run.append(config)

            table_data.append(
                ommit_none(
                    {
                        "#": i,
                        "Experiment": req.experiment_name,
                        "Model": req.model_arch_and_size.model_name,
                        "Status": status,
                        "Requested Prompts": len(requested_prompts),
                        "Computed Prompts": len(computed_prompts),
                        "Missing Prompts": len(requested_prompts - computed_prompts),
                        "Banned Prompts": banned_prompts,
                        "GPU": AppSessionKeys.get_selected_gpu(req.model_arch_and_size),
                    }
                )
            )

        df = pd.DataFrame(table_data)
        singular_columns = list(df.columns[df.nunique() == 1])
        if len(df) == 1:
            singular_columns = singular_columns[2:]
        df, grid_builder = base_grid_builder(
            df,
            selection_mode=SelectionMode.SINGLE,
            hide_columns=singular_columns,
        )
        grid_options = grid_builder.build()
        set_aagrid_apply_default_filters(
            grid_builder,
            {SummarizedDataFulfilledReqsCols.AvailableOptions: ["0"]},
        )
        with status_cols[0]:
            # Display the table
            if singular_columns:
                st.write(" | ".join([f"{col} = {df[col].iloc[0]}" for col in singular_columns]))
            grid_response = AgGrid(
                df,
                gridOptions=grid_options,
                height=cast(int, 400),  # allow None
                fit_columns_on_grid_load=True,
                floatingFilter=True,
                key=self.key,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                data_return_mode=DataReturnMode.FILTERED,
                allow_unsafe_jscode=True,
            )
        selected_rows = grid_response["selected_data"]
        if selected_rows is not None and len(selected_rows) > 0:
            with status_cols[1]:
                ShowRunnerStatus(configs[int(grid_response.selected_rows_id[0])]).render()

        # Run button
        total_to_run = len(configs_to_run)
        if total_to_run > 0 and param_cols[3].button(f"🚀 Run {total_to_run} Selected Requirements"):
            with status_cols[0]:
                st.info(f"Preparing to run {total_to_run} requirements...")

                success_count = 0
                failed_count = 0
                skipped_count = 0
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Get all rows from filtered_df that match selected requirements
                last_error = None
                new_scheduled = 0
                for config in configs_to_run:
                    try:
                        # Get config and set running parameters
                        if skip_scheduled:
                            status = config.slurm_job_folder.get_latest_slurm_job_status()
                            if isinstance(status, SlurmStatus) and status.scheduled():
                                skipped_count += 1
                                continue
                        # Run the configuration
                        config.run(with_dependencies=False)
                        new_scheduled += 1
                        success_count += 1

                    except Exception as e:
                        st.error(
                            f"Failed to run requirement: {str(json.dumps(asdict(config.variant_params), indent=4))}"
                        )
                        st.exception(e)
                        failed_count += 1
                        last_error = e
                        console.print(
                            rich.traceback.Traceback.from_exception(
                                exc_type=type(e), exc_value=e, traceback=e.__traceback__
                            )
                        )

                    # Update progress
                    progress_bar.progress(min((new_scheduled + failed_count) / total_to_run, 1))
                    status_text.text(
                        " | ".join(
                            [
                                f"Processed: {new_scheduled}/{total_to_run}",
                                f"Success: {success_count}",
                                f"Failed: {failed_count}",
                                f"Skipped: {skipped_count}",
                            ]
                        )
                    )

                if success_count > 0:
                    st.success(f"Successfully submitted {success_count} requirements to run")
                if failed_count > 0:
                    raise Exception(f"Failed to submit {failed_count} requirements") from last_error


def get_models_remaining_prompts(
    model_combinations: list[MODEL_ARCH_AND_SIZE],
    window_size: TWindowSize,
    code_version: TCodeVersionName,
    prompt_original_indices: list[TPromptOriginalIndex],
) -> dict[MODEL_ARCH_AND_SIZE, HeatmapRunner]:
    """Get the remaining prompts for each model."""
    res = {}
    for model_arch, model_size in model_combinations:
        config = HeatmapRunner(
            variant_params=HeatmapParams(
                model_arch=model_arch,
                model_size=model_size,
                window_size=window_size,
            ),
            input_params=InputParams(
                filteration=SelectivePromptFilteration(
                    prompt_ids=tuple(prompt_original_indices),
                ),
            ),
            metadata_params=MetadataParams(
                code_version=code_version,
            ),
        )
        if config.get_remaining_prompt_original_indices():
            res[MODEL_ARCH_AND_SIZE(model_arch, model_size)] = config
    return res


class HeatmapGenerationComponent(StreamlitComponent):
    def __init__(self, filtered_df: pd.DataFrame):
        self.filtered_df = filtered_df

    def render(self):
        # Show count of selected prompts
        # SLURM configuration
        select_window_size()

        prompt_original_indices = [TPromptOriginalIndex(int(x)) for x in self.filtered_df[HeatmapCols.SELECTED_PROMPT]]

        RequirementExecution(
            DataReqs(
                {
                    HeatmapParams(
                        model_arch=model_arch,
                        model_size=model_size,
                        window_size=AppSessionKeys.window_size.value,
                    ): SelectivePromptFilteration(prompt_ids=tuple(prompt_original_indices))
                    for model_arch, model_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS
                }
            )
        ).render()
