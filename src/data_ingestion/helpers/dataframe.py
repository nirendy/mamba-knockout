from typing import Optional, cast

import pandas as pd

from src.core.types import TPromptOriginalIndex, TRowPosition


def validate_one_selected_row_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    return df.iloc[0]


def index_to_row_position(df: pd.DataFrame, index_idx: TPromptOriginalIndex) -> TRowPosition:
    return TRowPosition(cast(int, df.index.get_loc(index_idx)))
