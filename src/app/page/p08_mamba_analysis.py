import streamlit as st
import streamlit_antd_components as sac

from src.app.components.model_analysis import (
    AMatrixAnalysisComponent,
    FeatureDynamicsComponent,
    select_mamba_model,
)
from src.app.texts import MAMBA_ANALYSIS_TEXTS
from src.utils.streamlit.helpers.component import StreamlitPage


class MambaAnalysisPage(StreamlitPage):
    def render(self):
        st.title(f"{MAMBA_ANALYSIS_TEXTS.title} {MAMBA_ANALYSIS_TEXTS.icon}")

        # Model selection
        selected_model = select_mamba_model()

        if not selected_model:
            st.warning("Please select a Mamba model to analyze.")
            return

        # Create tabs for different analyses
        tabs = sac.tabs(
            [
                sac.TabsItem(label="A Matrix Analysis", icon="graph-up"),
                sac.TabsItem(label="Feature Dynamics", icon="activity"),
            ]
        )

        # Render the appropriate component based on selected tab
        if tabs == "A Matrix Analysis":
            AMatrixAnalysisComponent(selected_model).render()
        elif tabs == "Feature Dynamics":
            FeatureDynamicsComponent(selected_model).render()


if __name__ == "mamba_analysis":
    st.set_page_config(
        page_title=MAMBA_ANALYSIS_TEXTS.title,
        page_icon=MAMBA_ANALYSIS_TEXTS.icon,
        layout="wide",
    )

    MambaAnalysisPage().render()
