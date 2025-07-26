from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, TypeVar

import streamlit as st
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode

from src.app.data_store import load_results_bank
from src.core.consts import ALL_IMPORTANT_MODELS
from src.core.names import BaseVariantParamName, WindowedVariantParam
from src.data_ingestion.data_defs.data_defs import EvaluateModelResults, ResultBank
from src.experiments.infrastructure.base_runner import BaseRunner
from src.utils.streamlit.components.aagrid import (
    SelectionMode,
    base_grid_builder,
    set_aagrid_apply_default_filters,
)
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.streamlit.helpers.session_keys import SessionKeyDescriptor, SessionKeysBase

T_RESULT_BANK_TYPE = TypeVar("T_RESULT_BANK_TYPE", bound=ResultBank)


class ShowResultsBank(StreamlitComponent[T_RESULT_BANK_TYPE]):
    def __init__(
        self,
        results_bank: T_RESULT_BANK_TYPE,
        selection_mode: SelectionMode = SelectionMode.DISABLED,
        height: int = 1000,
        key: str = "results_bank",
        filters: Optional[dict[str, list]] = None,
        hide_columns: list[str] = [],
        hide_singular_columns: bool = False,
        pre_select_all_rows: bool = False,
    ):
        super().__init__()
        self.results_bank = results_bank
        self.selection_mode = selection_mode
        self.height = height
        self.key = key
        self.filters = filters
        self.hide_columns = [self.results_bank._KEY] + hide_columns
        self.hide_singular_columns = hide_singular_columns
        self.pre_select_all_rows = pre_select_all_rows

    def render(self) -> T_RESULT_BANK_TYPE:
        df = self.results_bank.to_experiment_results_df()
        df, grid_builder = base_grid_builder(
            df,
            self.selection_mode,
            hide_columns=self.hide_columns,
            hide_singular_columns=self.hide_singular_columns,
            pre_selected_rows=(list(map(str, df.index)) if self.pre_select_all_rows else None),
        )
        if self.filters is not None:
            set_aagrid_apply_default_filters(
                grid_builder,
                self.filters,
            )
        for col in [WindowedVariantParam.window_size]:
            if col not in self.hide_columns:
                grid_builder.configure_column(col, type=["textColumn"])

        grid_options = grid_builder.build()

        # Display the table
        grid_response = AgGrid(
            data=df,
            gridOptions=grid_options,
            height=self.height,
            fit_columns_on_grid_load=True,
            floatingFilter=True,
            key=self.key,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED,
            allow_unsafe_jscode=True,
        )

        if grid_response.grid_state is None and self.pre_select_all_rows:
            return self.results_bank

        return self.results_bank.from_experiment_results_df(grid_response.selected_rows)


def select_model_evaluations(
    base_results: Optional[EvaluateModelResults] = None,
    key: str = "select_model_evaluations",
    pre_select_all_rows: bool = False,
) -> EvaluateModelResults:
    if base_results is None:
        base_results = load_results_bank.call_and_render().to_evaluate_model_results()
    result_bank = ShowResultsBank(
        base_results,
        selection_mode=SelectionMode.MULTIPLE,
        height=300,
        hide_singular_columns=True,
        key=key,
        filters={
            BaseVariantParamName.model_arch: [
                model_arch_and_size.arch for model_arch_and_size in ALL_IMPORTANT_MODELS.keys()
            ],
            BaseVariantParamName.model_size: [
                model_arch_and_size.size for model_arch_and_size in ALL_IMPORTANT_MODELS.keys()
            ],
        },
        pre_select_all_rows=pre_select_all_rows,
    ).render()

    return result_bank


class ShowRunnerStatus(StreamlitComponent):
    class ShowRunnerStatusSks(SessionKeysBase["ShowRunnerStatusSks"]):
        first_line_count = SessionKeyDescriptor[int](100)
        last_line_count = SessionKeyDescriptor[int](200)
        wrap_log = SessionKeyDescriptor[bool](False)

    @st.dialog(title="File Output", width="large")
    def show_job_dialog(
        self,
        file_path: Path,
        title: str,
    ):
        st.code(file_path, wrap_lines=True)

        st.title(title)

        if Path(file_path).exists():
            for i, col in enumerate(st.columns(3)):
                with col:
                    if i == 0:
                        st.number_input(
                            "First N lines",
                            min_value=10,
                            max_value=1000,
                            key=self.sks.first_line_count.key,
                        )
                    elif i == 1:
                        st.number_input(
                            "Last N lines",
                            min_value=10,
                            max_value=1000,
                            key=self.sks.last_line_count.key,
                        )
                    else:
                        st.checkbox("Wrap log", key=self.sks.wrap_log.key)

            lines = Path(file_path).read_text().split("\n")
            total_lines = len(lines)
            max_display_lines = self.sks.first_line_count.value + self.sks.last_line_count.value

            if total_lines > max_display_lines:
                first_n = lines[: self.sks.first_line_count.value]
                last_n = lines[-self.sks.last_line_count.value :]
                skipped_lines = total_lines - max_display_lines
                formatted_lines = [
                    *first_n,
                    f"...Skipped {skipped_lines} lines...",
                    *last_n,
                ]
                st.code("\n".join(formatted_lines), wrap_lines=self.sks.wrap_log.value)
            else:
                st.code("\n".join(lines), wrap_lines=self.sks.wrap_log.value)

        else:
            st.error("File not found or could not be read")
            if st.button("Close Error"):
                st.rerun()

    def __init__(self, runner: BaseRunner):
        super().__init__()
        self.runner = runner
        self.sks = self.ShowRunnerStatusSks()

    def render(self):
        st.code(self.runner.variation_paths.variation_base_path, wrap_lines=True)
        st.expander(expanded=False, label="Params").write(asdict(self.runner))
        # Computation status

        all_slurm_jobs = self.runner.slurm_job_folder.get_all_slurm_jobs()
        all_slurm_jobs = sorted(all_slurm_jobs, key=lambda x: -int(x.job_id))

        with st.expander(expanded=False, label=f"{len(all_slurm_jobs)} Slurm Runs"):
            for slurm_job in all_slurm_jobs:
                st.markdown(f"**Job ID:** {slurm_job.job_id} - **Status:** {slurm_job.get_slurm_status()}")

                cols = st.columns(2)
                with cols[0]:
                    if st.button("Show Job Output", key=f"show_job_output_{slurm_job.job_id}"):
                        self.show_job_dialog(
                            slurm_job.slurm_job_output_path,
                            f"Job Output - {slurm_job.job_id}",
                        )

                with cols[1]:
                    if st.button("Show Job Error", key=f"show_job_error_{slurm_job.job_id}"):
                        # Use Streamlit's experimental dialog
                        self.show_job_dialog(
                            slurm_job.slurm_job_error_path,
                            f"Job Error - {slurm_job.job_id}",
                        )
        # Show uncomputed dependencies if any
        if not self.runner.dependencies_are_computed():
            with st.expander("**Uncomputed Dependencies** ❌"):
                uncomputed = self.runner.uncomputed_dependencies()

                def render_dependencies(deps, level=0):
                    for key, dep in deps.items():
                        indent = "&nbsp;" * (4 * level)
                        if isinstance(dep, BaseRunner):
                            st.markdown(
                                f"{indent}• {key}: {dep.experiment_name} ({dep.variant_params.model_arch_and_size})",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(f"{indent}• {key}:", unsafe_allow_html=True)
                            render_dependencies(dep, level + 1)

                render_dependencies(uncomputed)

                if st.button("Compute Dependencies"):
                    with st.spinner("Computing..."):
                        try:
                            self.runner.compute_dependencies(1)
                            st.success("Computation completed or submitted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during computation: {e}")
