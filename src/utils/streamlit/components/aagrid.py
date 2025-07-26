import json
from enum import StrEnum

import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder, JsCode


def _validate_grid_builder_on_first_data_rendered_not_set(grid_builder: GridOptionsBuilder):
    assert getattr(grid_builder, "_GridOptionsBuilder__grid_options").get("onFirstDataRendered") is None, (
        "onFirstDataRendered already set"
    )


def set_aagrid_apply_default_filters(
    grid_builder: GridOptionsBuilder, filter_defaults: dict[str, list], with_st_code: bool = False
):
    """
    Applies default filters to AG Grid using JavaScript.

    Args:
        grid_builder: The AG GridOptionsBuilder instance.
        filter_defaults: A dictionary where keys are column names and values are the default filter values.

    Notice you have to have allow_unsafe_jscode=True in the AgGrid call.
    """

    # Convert Python dictionary to JavaScript object format
    filter_model_js = json.dumps(
        {
            col: {
                "filterType": "set",
                "values": values,
            }
            for col, values in filter_defaults.items()
        }
    )
    code = f"""
    function (params) {{
        params.api.setFilterModel({filter_model_js});
        params.api.onFilterChanged();
    }}
    """

    if with_st_code:
        st.code(filter_model_js)

    # JavaScript function to set the filter on first render
    onFirstDataRendered = JsCode(code)
    _validate_grid_builder_on_first_data_rendered_not_set(grid_builder)
    # Apply the generated JavaScript code to AG Grid
    grid_builder.configure_grid_options(onFirstDataRendered=onFirstDataRendered.js_code)


class SelectionMode(StrEnum):
    SINGLE = "single"
    MULTIPLE = "multiple"
    DISABLED = "disabled"


# https://www.ag-grid.com/javascript-data-grid/column-sizing/#auto-sizing-columns
class FIT_STRATEGY(StrEnum):
    FIT_CELL_CONTENTS = "fitCellContents"
    FIT_CONTENTS = "fitGridWidth"


def base_grid_builder(
    df: pd.DataFrame,
    selection_mode: SelectionMode,
    hide_columns: list[str],
    fit_strategy: FIT_STRATEGY = FIT_STRATEGY.FIT_CELL_CONTENTS,
    hide_singular_columns: bool = False,
    pre_selected_rows: list[str] | None = None,
) -> tuple[pd.DataFrame, GridOptionsBuilder]:
    if hide_singular_columns:
        hide_columns = hide_columns + list(df.columns[df.nunique() <= 1])

    # if the first column is hidden, we need to reorder the columns
    if selection_mode != SelectionMode.DISABLED and df.columns[0] in hide_columns:
        df = df[[*df.columns[1:], df.columns[0]]]
    grid_builder = GridOptionsBuilder.from_dataframe(df)
    grid_builder.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=100)
    if selection_mode != SelectionMode.DISABLED:
        grid_builder.configure_selection(
            selection_mode=selection_mode,
            use_checkbox=True,
            header_checkbox=True,
            pre_selected_rows=pre_selected_rows or [],
        )
    grid_builder.configure_default_column(filter=True, floatingFilter=True)
    grid_builder.configure_side_bar()
    for col in hide_columns:
        grid_builder.configure_column(col, hide=True)

    grid_builder.configure_grid_options(autoSizeStrategy={"type": fit_strategy, "skipHeader": True})

    return df, grid_builder


def set_pre_selected_rows(grid_builder: GridOptionsBuilder, pre_selected_rows: list[str] | None = None):
    if pre_selected_rows:
        grid_builder.configure_selection(pre_selected_rows=pre_selected_rows)
        code = (
            """
        function (params) {
            """
            + f"index = {pre_selected_rows[0]};"
            + """
            const gridApi = params.api;
            window.parent.aggrid_api = gridApi;
            const pageSize = gridApi.paginationGetPageSize();
            const targetPage = Math.floor(index / pageSize);
            const currentPage = gridApi.paginationGetCurrentPage();
            if (targetPage !== currentPage) {
                gridApi.paginationGoToPage(targetPage);
                gridApi.dispatchEvent({ type: "selectionChanged" });
            }
            setTimeout(() => {
                params.api.ensureIndexVisible(index, 'middle');
                gridApi.dispatchEvent({ type: "selectionChanged" });
            }, 100);
        }
        """
        )
        onFirstDataRendered = JsCode(code)
        _validate_grid_builder_on_first_data_rendered_not_set(grid_builder)

        grid_builder.configure_grid_options(onFirstDataRendered=onFirstDataRendered.js_code)
