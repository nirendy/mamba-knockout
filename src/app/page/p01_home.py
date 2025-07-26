import streamlit as st

from src.app.app_consts import AppSessionKeys
from src.app.components.inputs import select_gpu_type, select_window_size
from src.app.texts import HOME_TEXTS
from src.utils.streamlit.helpers.component import StreamlitPage


class HomePage(StreamlitPage):
    def render(self):
        # Create navigation
        st.sidebar.success("Select a page above to explore different analyses.")

        # Global variables
        if st.button("Reset App"):
            st.session_state.clear()

        select_gpu_type()
        AppSessionKeys.code_version.create_input_widget("CodeVersion")
        select_window_size()

        # TreeComponentDemo().render()


if __name__ == "home":
    st.set_page_config(page_title=HOME_TEXTS.title, page_icon=HOME_TEXTS.icon, layout="wide")
    st.title(f"{HOME_TEXTS.title} {HOME_TEXTS.icon}")

    HomePage().render()
