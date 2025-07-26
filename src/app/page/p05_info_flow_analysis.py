"""
# Purpose: Display and analyze info flow requirements with interactive visualization capabilities
# High Level Outline:
# 1. Page setup and configuration
# 2. Load and display latest fulfilled info flow requirements
# 3. Interactive selection of multiple info flow data using RequirementsDisplay
# 4. Analysis of selected info flow data with visualizations using InfoFlowAnalysisComponent
# Outline Issues:
# - Add more interactive visualization options
# - Consider adding batch analysis capabilities
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly
"""

import streamlit as st

from src.analysis.prompt_filterations import (
    AnyExistingCompletePromptFilteration,
)
from src.app.components.info_flow import InfoFlowAnalysisComponent
from src.app.components.prompt_filter import (
    FilterPromptsComponent,
)
from src.app.components.result_bank import SelectionMode, ShowResultsBank
from src.app.data_store import load_prompts, load_results_bank
from src.app.texts import INFO_FLOW_ANALYSIS_TEXTS
from src.core.consts import GRAPHS_ORDER
from src.core.names import COLS, BaseVariantParamName, InfoFlowMetricName, InfoFlowVariantParam, WindowedVariantParam
from src.core.types import (
    MODEL_SIZE_CAT,
    FeatureCategory,
    TInfoFlowWindowValue,
    TPromptOriginalIndex,
)
from src.experiments.infrastructure.base_prompt_filteration import LogicalPromptFilteration
from src.utils.streamlit.helpers.component import StreamlitPage
from src.utils.types_utils import (
    get_list_indexes_of_set_values,
    select_indexes_from_list,
)


def select_indexes_from_window_values(
    window_values: TInfoFlowWindowValue, prompt_ids: list[TPromptOriginalIndex]
) -> TInfoFlowWindowValue:
    indexes = get_list_indexes_of_set_values(window_values[COLS.ORIGINAL_IDX], set(prompt_ids))
    return {
        InfoFlowMetricName.hit: select_indexes_from_list(window_values[InfoFlowMetricName.hit], indexes),
        InfoFlowMetricName.true_probs: select_indexes_from_list(window_values[InfoFlowMetricName.true_probs], indexes),
        InfoFlowMetricName.diffs: select_indexes_from_list(window_values[InfoFlowMetricName.diffs], indexes),
        COLS.ORIGINAL_IDX: prompt_ids,
    }


class InfoFlowAnalysisPage(StreamlitPage):
    def render(self):
        with st.expander("Choose Info Flows"):
            results_bank = load_results_bank.call_and_render().to_info_flow_results()
            result_bank = ShowResultsBank(
                results_bank,
                selection_mode=SelectionMode.MULTIPLE,  # Changed to MULTIPLE
                height=300,
                filters={
                    # ResultBankParamNames.code_version: [GLOBAL_APP_CONSTS.DEFAULT_CODE_VERSION],
                    BaseVariantParamName.model_size: [
                        model_arch_and_size.size
                        for model_arch_and_size, size_cat in GRAPHS_ORDER.items()
                        if size_cat.value > MODEL_SIZE_CAT.MEDIUM.value
                    ],
                    WindowedVariantParam.window_size: ["9"],
                    InfoFlowVariantParam.target: ["last"],
                    InfoFlowVariantParam.source: ["subject"],
                    BaseVariantParamName.model_arch: ["gpt2"],
                    InfoFlowVariantParam.feature_category: [FeatureCategory.ALL],
                },
                hide_columns=[
                    BaseVariantParamName.experiment_name,
                ],
                key="info_flow_results_bank3",
            ).render()

        if result_bank.is_empty():
            st.warning(INFO_FLOW_ANALYSIS_TEXTS.no_requirements)
            return

        with st.expander(f"Prompt Filteration for {len(result_bank)} info flows"):
            prompt_filteration = FilterPromptsComponent(
                key="info_flow_analysis_prompt_filteration",
                base_prompt_filteration=LogicalPromptFilteration.create_and(
                    [AnyExistingCompletePromptFilteration().contextualize(runner) for runner in result_bank]
                ),
            ).render()

        prompts = load_prompts()
        if prompts.filter_by_prompt_filteration(prompt_filteration).empty:
            st.warning(INFO_FLOW_ANALYSIS_TEXTS.no_common_indices)
            return

        with st.sidebar:
            max_layer = result_bank.max_layer()
            min_layer = result_bank.min_layer()
            layers_range = st.slider(
                INFO_FLOW_ANALYSIS_TEXTS.layers_range,
                value=(min_layer, max_layer),
                min_value=min_layer,
                max_value=max_layer,
                step=1,
            )

        # Use the sampled prompt IDs to filter the result bank
        result_bank = result_bank.subset_layers(layers_range).set_prompt_filteration(prompt_filteration)

        # Show analysis if we have data
        if result_bank:
            InfoFlowAnalysisComponent(result_bank).render()


if __name__ == "__main__":
    st.set_page_config(
        page_title=INFO_FLOW_ANALYSIS_TEXTS.title,
        page_icon=INFO_FLOW_ANALYSIS_TEXTS.icon,
        layout="wide",
    )
    st.title(f"{INFO_FLOW_ANALYSIS_TEXTS.title} {INFO_FLOW_ANALYSIS_TEXTS.icon}")

    InfoFlowAnalysisPage().render()
