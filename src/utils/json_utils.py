from __future__ import annotations

from typing import Any, Iterable, Mapping

type GeneralJsonObject = Mapping[Any, GeneralJsonObject] | Iterable[GeneralJsonObject] | int | float | str | bool | None
type SanitizedJsonObject = dict[str, SanitizedJsonObject] | list[SanitizedJsonObject] | int | float | str | bool | None


def sanitize(obj: GeneralJsonObject) -> SanitizedJsonObject:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable):
        return [sanitize(x) for x in obj]
    else:
        return obj
