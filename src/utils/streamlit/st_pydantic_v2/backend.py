from __future__ import annotations

# mypy: ignore-errors
import datetime as _dt
from typing import (
    Any,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    cast,
)

import streamlit as st
from PIL import Image
from streamlit.delta_generator import DeltaGenerator

from src.utils.streamlit.ui_pydantic_v2.extra_types import XYWH_SIDES, Box


class BackendProtocol(Protocol):
    def text_input(self, label: str, key: str, **kw) -> str | None: ...

    def text_area(self, label: str, key: str, **kw) -> str | None: ...

    def number_input(self, label: str, key: str, **kw) -> Union[int, float]: ...

    def checkbox(self, label: str, key: str, **kw) -> bool: ...

    def button(self, label: str, key: Optional[str] = None, **kw) -> bool: ...

    def selectbox(self, label: str, options: Sequence[str], key: str, **kw) -> Optional[str]: ...

    def multiselect(self, label: str, options: Sequence[str], key: str, **kw) -> List[str]: ...

    def date_input(self, label: str, key: str, **kw) -> _dt.date: ...

    def color_picker(self, label: str, key: str, **kw) -> str: ...

    def subheader(self, txt: str) -> None: ...

    def markdown(self, txt: str) -> None: ...

    def info(self, txt: str) -> None: ...

    def warning(self, txt: str) -> None: ...

    def columns(self, spec: Sequence[int]) -> List[Any]: ...

    def expander(self, label: str, **kw) -> BackendProtocol: ...

    def form(self, key: str, clear_on_submit: bool = False) -> Any: ...

    def form_submit_button(self, label: str) -> bool: ...

    def text(self, txt: str) -> None: ...

    def toast(self, txt: str) -> None: ...

    def container(self, **kw) -> BackendProtocol: ...

    def cropper(self, img: Image.Image, key: str, **kw) -> None: ...

    def __getattr__(self, item):
        # This is a fallback for unknown attributes
        pass


class StreamlitBackend(BackendProtocol):  # Implementation of BackendProtocol
    def __init__(self, container: Optional[DeltaGenerator] = None):
        self.dg = container or st

    # input components
    def text_input(self, label: str, key: str, **kw) -> str | None:
        return self.dg.text_input(label, key=key, **kw)

    def text_area(self, label: str, key: str, **kw) -> str | None:
        return self.dg.text_area(label, key=key, **kw)

    def number_input(self, label: str, key: str, **kw) -> Union[int, float]:
        return self.dg.number_input(label, key=key, **kw)

    def checkbox(self, label: str, key: str, **kw) -> bool:
        return self.dg.checkbox(label, key=key, **kw)

    def button(self, label: str, key: Optional[str] = None, **kw) -> bool:
        return self.dg.button(label, key=key, **kw)

    def selectbox(self, label: str, options: Sequence[str], key: str, **kw) -> Optional[str]:
        return self.dg.selectbox(label, options, key=key, **kw)

    def multiselect(self, label: str, options: Sequence[str], key: str, **kw) -> List[str]:
        return self.dg.multiselect(label, options, key=key, **kw)

    def date_input(self, label: str, key: str, **kw) -> _dt.date:
        return self.dg.date_input(label, key=key, **kw)

    def color_picker(self, label: str, key: str, **kw) -> str:
        return self.dg.color_picker(label, key=key, **kw)

    def slider(self, label: str, min_value: float, max_value: float, key: str, **kw) -> float:
        return self.dg.slider(label, min_value=min_value, max_value=max_value, key=key, **kw)

    # containers + other components
    def subheader(self, txt: str) -> None:
        self.dg.subheader(txt)

    def markdown(self, txt: str) -> None:
        self.dg.markdown(txt)

    def info(self, txt: str) -> None:
        self.dg.info(txt)

    def warning(self, txt: str) -> None:
        self.dg.warning(txt)

    def columns(self, spec: Sequence[int]) -> List[BackendProtocol]:
        return [StreamlitBackend(col) for col in self.dg.columns(spec)]

    def expander(self, label: str, **kw) -> BackendProtocol:
        return StreamlitBackend(self.dg.expander(label, **kw))

    def container(self, **kw) -> BackendProtocol:
        return StreamlitBackend(self.dg.container(**kw))

    def form(self, key: str, clear_on_submit: bool = False) -> Any:
        return self.dg.form(key, clear_on_submit=clear_on_submit)

    def form_submit_button(self, label: str) -> bool:
        return self.dg.form_submit_button(label)

    def text(self, txt: str) -> None:
        self.dg.text(txt)

    def toast(self, txt: str) -> None:
        self.dg.toast(txt)

    def cropper(self, img: Image.Image, key: str, with_debug: bool = True, **kw) -> None:
        """Render a streamlit-cropper component."""

        def write_debug(*args):
            if with_debug:
                st.write(*args)

        try:
            from streamlit_cropper import st_cropper

        except ImportError:
            self.warning("streamlit-cropper is not installed. Run 'pip install streamlit-cropper'.")
            return

        @st.dialog(title="Interactive Crop", width="large")
        def _crop_dialog(_, key: str, **kw):
            # Need to resize to fit the screen, but need to keep it in the same aspect ratio
            MAX_WIDTH = 700
            resized_image = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)))

            box = Box.from_xywh(
                number_type="percentage",
                **{side: st.session_state[f"{key}.{side}"] for side in XYWH_SIDES},
            )
            box = box.to_unit(to_unit_type="absolute", dimensions=resized_image.size)

            box = st_cropper(
                cast(Any, resized_image),
                default_coords=box.lrtb,
                return_type="box",
                key=f"{key}_cropper",
                should_resize_image=False,
                **kw,
            )
            assert isinstance(box, dict)

            def show_box(box: Box):
                write_debug(",".join(f"{side}: {box.get_side(side)}" for side in XYWH_SIDES))

            box = Box(
                number_type="absolute",
                left=box["left"],
                top=box["top"],
                right=box["left"] + box["width"],
                bottom=box["top"] + box["height"],
            )

            write_debug("original", img.size)
            write_debug("resized", resized_image.size)
            show_box(box)
            box = box.to_unit(to_unit_type="percentage", dimensions=resized_image.size)
            show_box(box)
            show_box(box.to_unit(to_unit_type="absolute", dimensions=resized_image.size))

            # If save button is clicked, update the session state
            if st.button("Save", key=f"{key}_save"):
                for side in XYWH_SIDES:
                    st.session_state[f"{key}.{side}"] = getattr(box, side)
                st.rerun()  # Rerun to update the UI

        if self.dg.button("Open Crop Dialog", key=f"{key}_open_crop_dialog"):
            _crop_dialog(img, key, **kw)

    def __getattr__(self, item):
        try:
            return getattr(self.dg, item)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{item}'")
