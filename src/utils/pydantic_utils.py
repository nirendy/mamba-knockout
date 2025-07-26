from typing import Any, Literal


def create_literal_value(values: list[str]) -> Any:
    # Will raise error if values is empty
    if len(values) == 1:
        return Literal[values[0]]  # type: ignore
    return Literal[*values]  # type: ignore
