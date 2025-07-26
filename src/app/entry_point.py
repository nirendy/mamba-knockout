from typing import assert_never

import streamlit as st

from src.app.app_consts import PAGE_ORDER
from src.app.page import (
    p01_home,
    p02_results_bank,
    p03_data_requirements,
    p04_heatmap_creation,
    p05_info_flow_analysis,
    p06_final_plots,
    p07_prompts_comparison,
    p08_mamba_analysis,
    p09_prompt_filteration_presets,
)
from src.utils.streamlit.helpers.component import StreamlitPage
from src.utils.streamlit.helpers.session_keys import mark_finished_global_refresh

st.set_page_config(layout="wide")


def get_page(page_order: PAGE_ORDER) -> StreamlitPage:
    match page_order:
        case PAGE_ORDER.HOME:
            return p01_home.HomePage()
        case PAGE_ORDER.RESULTS_BANK:
            return p02_results_bank.ResultsBankPage()
        case PAGE_ORDER.DATA_REQUIREMENTS:
            return p03_data_requirements.DataRequirementsPage()
        case PAGE_ORDER.HEATMAP:
            return p04_heatmap_creation.HeatmapCreationPage()
        case PAGE_ORDER.INFO_FLOW_ANALYSIS:
            return p05_info_flow_analysis.InfoFlowAnalysisPage()
        case PAGE_ORDER.FINAL_PLOTS:
            return p06_final_plots.FinalPlotsPage()
        case PAGE_ORDER.PROMPTS_COMPARISON:
            return p07_prompts_comparison.PromptsComparisonPage()
        case PAGE_ORDER.MAMBA_ANALYSIS:
            return p08_mamba_analysis.MambaAnalysisPage()
        case PAGE_ORDER.PROMPT_FILTERATION_PRESETS:
            return p09_prompt_filteration_presets.PromptFilterationPresetsPage()
        case _:
            assert_never(page_order)


pg = st.navigation(
    {
        "Pages": [
            st.Page(
                page=(
                    get_page(page)
                    # .profile_render
                    .render
                ),
                title=page.title,
                icon=page.icon,
                url_path=page.name.lower(),
            )
            for page in PAGE_ORDER
        ]
    },
)

pg.run()

mark_finished_global_refresh()
