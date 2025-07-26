# Purpose: Create and manage heatmaps for model analysis with filtering and batch processing capabilities
# High Level Outline:
# 1. Page setup and configuration
# 2. Model combinations analysis
# 3. Prompt filtering and selection
# 4. Heatmap generation and execution
# Outline Issues:
# - Add comparison view for multiple heatmaps
# - Consider adding heatmap export functionality
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly


import streamlit as st
import streamlit_antd_components as sac

from src.app.app_consts import (
    GLOBAL_APP_CONSTS,
    AppSessionKeys,
)
from src.app.components.data_requirements import HeatmapGenerationComponent
from src.app.components.multi_plots import HeatmapPlotGenerationComponent
from src.app.components.prompt_filter import PromptSelectionForCombinationComponent, ShowModelCombinations
from src.app.components.result_bank import select_model_evaluations
from src.app.data_store import (
    load_model_combinations_prompts,
    load_model_evaluations_dict,
)
from src.app.texts import COMMON_TEXTS, HEATMAP_TEXTS
from src.utils.streamlit.helpers.component import StreamlitPage


class HeatmapCreationPage(StreamlitPage):
    def render(self):
        # region Data Loading
        clear_deps_expander = st.sidebar.expander("Clear Dependencies", expanded=False)
        sidebar_expander = st.sidebar.expander("Configuration", expanded=True)
        with sidebar_expander:
            seed = st.number_input("Seed", value=GLOBAL_APP_CONSTS.DEFAULT_SEED, min_value=0, max_value=1000000, step=1)

        with st.expander("Select Models"):
            selected_model_evaluations = select_model_evaluations(key="heatmap_select_model_evaluations")

        with st.spinner(COMMON_TEXTS.LOADING("data"), show_time=True):
            # Get combinations data
            model_evaluations = load_model_evaluations_dict(AppSessionKeys.code_version.value)
            representative_model_evaluations = next(iter(model_evaluations.values()))
            # Get combinations using selected models
            model_combinations_prompts = load_model_combinations_prompts(
                AppSessionKeys.code_version.value, selected_model_evaluations.model_arch_and_sizes, seed
            )

        with clear_deps_expander:
            load_model_evaluations_dict.render()
            load_model_combinations_prompts.render()
        # endregion

        filtered_df, selected_combination_row = ShowModelCombinations(
            model_combinations_prompts, representative_model_evaluations, sidebar_expander
        ).render()

        if selected_combination_row is None:
            st.write(HEATMAP_TEXTS.NO_SELECTED_COMBINATION)

        tab = sac.tabs(
            [
                sac.TabsItem(label=HEATMAP_TEXTS.run_selected_prompts_button(len(filtered_df))),
                sac.TabsItem(label=HEATMAP_TEXTS.TAB_SELECT_COMBINATION, disabled=selected_combination_row is None),
                sac.TabsItem(
                    label=HEATMAP_TEXTS.TAB_HEATMAP_PLOTS_GENERATION, disabled=selected_combination_row is None
                ),
            ]
        )

        if selected_combination_row is not None:
            combination_row = model_combinations_prompts[selected_combination_row]
            if tab == HEATMAP_TEXTS.TAB_SELECT_COMBINATION:
                PromptSelectionForCombinationComponent(
                    combination_row,
                    representative_model_evaluations,
                    model_combinations_prompts,
                ).render()
            elif tab == HEATMAP_TEXTS.TAB_HEATMAP_PLOTS_GENERATION:
                prompt_idx = combination_row.chosen_prompt
                if prompt_idx is not None:
                    HeatmapPlotGenerationComponent(prompt_idx).render()

        if tab == HEATMAP_TEXTS.run_selected_prompts_button(len(filtered_df)):
            # Add SLURM configuration in sidebar
            HeatmapGenerationComponent(filtered_df).render()


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_icon=HEATMAP_TEXTS.icon, page_title=HEATMAP_TEXTS.title)
    st.header(HEATMAP_TEXTS.MODEL_COMBINATIONS_HEADER)

    HeatmapCreationPage().render()
