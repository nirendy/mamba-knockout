from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import streamlit as st

OutputType = TypeVar("OutputType")


class StreamlitComponent(ABC, Generic[OutputType]):
    @abstractmethod
    def render(self) -> OutputType:
        pass

    def profile_render(self):
        from wfork_streamlit_profiler import Profiler

        with Profiler():
            try:
                self.render()
            finally:
                st.toast("Rendering complete, generating profile...")


class StreamlitPage(StreamlitComponent[None]):
    pass
