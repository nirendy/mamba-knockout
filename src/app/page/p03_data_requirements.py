# Purpose: Manage and display data requirements for experiments with filtering and execution capabilities
# High Level Outline:
# 1. Page setup and configuration
# 2. Initialize session state and load data
# 3. Display and manage requirements with filters
# 4. Handle requirement selection and execution
# Outline Issues:
# - Consider adding batch operations for requirements
# - Add progress tracking for running requirements
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly

import streamlit as st

from src.app.components.data_requirements import RequirementExecution, RequirementsDisplay
from src.app.data_store import load_fulfilled_reqs_df
from src.app.texts import DATA_REQUIREMENTS_TEXTS
from src.utils.streamlit.components.aagrid import SelectionMode
from src.utils.streamlit.helpers.component import StreamlitPage


class DataRequirementsPage(StreamlitPage):
    def render(self):
        df = load_fulfilled_reqs_df.call_and_render()

        # Display requirements
        data_reqs_to_run = RequirementsDisplay(
            df,
            height=400,
            selection_mode=SelectionMode.MULTIPLE,
            hide_columns=[],
        ).render()

        if data_reqs_to_run is not None:
            # Handle requirement execution
            RequirementExecution(data_reqs_to_run).render()


if __name__ == "__main__":
    st.set_page_config(page_title=DATA_REQUIREMENTS_TEXTS.title, page_icon=DATA_REQUIREMENTS_TEXTS.icon, layout="wide")
    st.title(f"{DATA_REQUIREMENTS_TEXTS.title} {DATA_REQUIREMENTS_TEXTS.icon}")

    DataRequirementsPage().render()
