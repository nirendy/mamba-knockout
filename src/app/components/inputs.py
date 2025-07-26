from enum import StrEnum
from typing import NamedTuple, Optional, Type, TypeVar

import streamlit as st

from src.app.app_consts import AppSessionKeys
from src.app.texts import AppGlobalText
from src.core.types import TokenType
from src.experiments.runners.heatmap import HEATMAP_PLOT_FUNCS
from src.utils.infra.slurm import SLURM_GPU_TYPE
from src.utils.streamlit.helpers.session_keys import SessionKey

T = TypeVar("T", bound=StrEnum)


def select_enum(label: str, enum_class: Type[T], session_key: SessionKey[T]) -> T:
    """Display a selection widget for a StrEnum.

    Args:
        label: The label to display for the widget
        enum_class: The StrEnum class to select from
        session_key: The SessionKey to store the selection in
        default: The default value to select
    Returns:
        The selected value(s) from the enum, or None if none is selected
    """
    return st.selectbox(
        label,
        options=enum_class,
        key=session_key.key_for_component,
    )


def select_token_type(session_key: SessionKey[TokenType], label: Optional[str] = None):
    """Specialized function for selecting a TokenType"""
    if label is None:
        label = session_key.key
    select_enum(label, TokenType, session_key)


def select_gpu_type():
    options = ["smart"] + [value for value in SLURM_GPU_TYPE]
    st.selectbox(
        AppGlobalText.gpu_type,
        options=options,
        key=AppSessionKeys._selected_gpu.key,
    )


def select_window_size():
    options = [1, 3, 5, 7, 9, 12, 15]
    st.selectbox(
        AppGlobalText.window_size,
        options=options,
        key=AppSessionKeys.window_size.key_for_component,
        index=options.index(AppSessionKeys.window_size.value),
    )


class HeatmapPlotsParams(NamedTuple):
    plot_name: HEATMAP_PLOT_FUNCS


def choose_heatmap_parms():
    return HeatmapPlotsParams(
        plot_name=st.selectbox(
            "Plot Name",
            options=list(HEATMAP_PLOT_FUNCS),
            index=0,
        )
    )
