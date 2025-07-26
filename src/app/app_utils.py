from pathlib import Path
from typing import Optional, TypeVar

import pandas as pd
import streamlit as st
from streamlit.elements.arrow import DataframeState

from src.app.app_consts import HeatmapConsts
from src.core.consts import PATHS
from src.core.names import HeatmapCols
from src.utils.file_system import fast_relative_to

T = TypeVar("T")


def format_path_for_display(path: Path | str | None, allow_slow: bool = False) -> str:
    """Format a path for display in the UI.

    Args:
        path: Path to format

    Returns:
        Formatted path string
    """
    if path is None:
        return ""

    if isinstance(path, Path):
        try:
            return str(fast_relative_to(path, PATHS.PROJECT_DIR, allow_slow=allow_slow))
        except ValueError:
            return str(path)

    return format_path_for_display(Path(path))


def reverse_format_path_for_display(path: Path) -> Path:
    """Reverse format a path for display in the UI.

    Args:
        path: Path to reverse format

    Returns:
        Reversed formatted path
    """
    return PATHS.PROJECT_DIR / path


def filter_combinations(df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    df = df.copy()

    modification_container = st.container()

    model_name_filters = []
    for model_name in model_names:
        model_name_filters.append((model_name, True))
        model_name_filters.append((f"{model_name}", False))

    def format_func(option: str | tuple[str, bool]) -> str:
        if isinstance(option, tuple):
            return f"{option[0]} - {'correct ✅' if option[1] else 'incorrect ❌'}"
        else:
            assert option == HeatmapCols.PROMPT_COUNT
            return "Prompt Count"

    with modification_container:
        to_filter_columns = st.multiselect(
            "Filter dataframe on",
            [HeatmapCols.PROMPT_COUNT, *model_name_filters],
            format_func=format_func,
            default=[HeatmapCols.PROMPT_COUNT],
        )

        # Existing column filtering logic
        for column in to_filter_columns:
            if column == HeatmapCols.PROMPT_COUNT:
                _min = int(df[column].min())
                _max = int(df[column].max())
                user_num_input = st.number_input(
                    "Minimum prompt count",
                    min_value=_min,
                    max_value=_max,
                    value=max(_min, HeatmapConsts.MINIMUM_COMBINATIONS_FOR_FILTERING),
                    step=1,
                )
                df = df[df[column] >= user_num_input]
            else:
                assert isinstance(column, tuple)
                model_name, is_correct = column
                if is_correct:
                    df = df[df[model_name] == "✅"]
                else:
                    df = df[df[model_name] == "❌"]

        # New minimum correct models filter
        min_correct = st.number_input(
            "Minimum correct models",
            min_value=0,
            max_value=len(model_names),
            value=0,
            help="Show only combinations with at least this many correct models",
        )
        if min_correct > 0:
            # Count number of ✅ in model columns for each row
            df = df[df[model_names].apply(lambda row: (row == "✅").sum(), axis=1) >= min_correct]

    return df


def get_steamlit_dataframe_selected_row(selected_row: Optional[DataframeState]) -> Optional[int]:
    if selected_row is not None:
        assert "selection" in selected_row
        assert "rows" in selected_row["selection"]
        if len(selected_row["selection"]["rows"]) == 1:
            return selected_row["selection"]["rows"][0]
