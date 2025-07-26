import contextlib
import json
from abc import ABC
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from enum import Enum, StrEnum
from typing import (
    Any,
    Callable,
    ContextManager,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import pyrallis


def class_values(cls: Type) -> list[str]:
    if issubclass(cls, StrEnum):
        return [member.value for member in cls]  # Handle StrEnum
    return [value for key, value in vars(cls).items() if not key.startswith("__")]


_T_STR_ENUM = TypeVar("_T_STR_ENUM", bound=StrEnum)


def str_enum_values(cls: Type[_T_STR_ENUM]) -> list[_T_STR_ENUM]:
    return cast(list[_T_STR_ENUM], class_values(cls))


def get_enum_or_literal_options(typ: Any) -> list[str]:
    origin = get_origin(typ)
    args = get_args(typ)

    if origin is Literal:
        return [str(a) for a in args]

    elif isinstance(typ, type) and issubclass(typ, Enum):
        return [e.name if isinstance(e, StrEnum) else e.name for e in typ]

    elif origin is Union:
        values = []
        for arg in args:
            values += get_enum_or_literal_options(arg)
        return values

    return []


def init_str_enum_from_value(cls: Type[_T_STR_ENUM], value: str) -> _T_STR_ENUM:
    assert value in str_enum_values(cls)
    return cast(_T_STR_ENUM, value)


_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")


def select_indexes_from_list(lst: list[_T], indexes: list[int]) -> list[_T]:
    return [lst[i] for i in indexes]


def get_list_indexes_of_set_values(lst: list[_T], values: set[_T]) -> list[int]:
    return [i for i, v in enumerate(lst) if v in values]


def subset_dict_by_keys(d: dict[_K, _V], keys: list[_K]) -> dict[_K, _V]:
    return {k: v for k, v in d.items() if k in keys}


def get_dict_keys_by_condition(d: dict[_K, _V], condition: Callable[[_K, _V], bool]) -> list[_K]:
    return [k for k, v in d.items() if condition(k, v)]


def subset_dict_by_condition(d: dict[_K, _V], condition: Callable[[_K, _V], bool]) -> dict[_K, _V]:
    return subset_dict_by_keys(d, get_dict_keys_by_condition(d, condition))


def first_dict_value(d: dict[Any, _T]) -> _T:
    return next(iter(d.values()))


def first_dict_key(d: dict[Any, _T]) -> Any:
    return next(iter(d.keys()))


def ommit_none(d: dict[Any, Optional[Any]]) -> dict[Any, Any]:
    return {k: v for k, v in d.items() if v is not None}


def ommit_unique_values(dict_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_values = defaultdict(set)
    for d in dict_list:
        for k, v in d.items():
            if v is not None and not isinstance(v, (list, tuple, set)):
                unique_values[k].add(v)

    return [{k: v for k, v in d.items() if v not in unique_values[k]} for d in dict_list]


def conditional_context_manager(use_ctx: bool, ctx: ContextManager[None]) -> ContextManager[None]:
    """
    Returns the given context manager if use_ctx is True, otherwise returns a dummy context.

    :param use_ctx: Boolean flag to determine whether to use the actual context manager.
    :param ctx: The actual context manager to use if use_ctx is True.
    :return: The appropriate context manager (either ctx or nullcontext).
    """
    return ctx if use_ctx else contextlib.nullcontext()


_ATTRIBUTE_TYPE = TypeVar("_ATTRIBUTE_TYPE")


def create_mutable_field(
    default_factory: Callable[[], _ATTRIBUTE_TYPE],
) -> _ATTRIBUTE_TYPE:
    # Pyralis need mutable fields to be defined with field but it's typing is not complete.
    # This is a fix to make it work.
    return cast(
        _ATTRIBUTE_TYPE,
        pyrallis.field(default_factory=default_factory, is_mutable=True),  # type:ignore[return-value,valid-type]
    )


_T_LITERAL = TypeVar("_T_LITERAL")


def literal_guard(value: Any, expected: _T_LITERAL) -> _T_LITERAL:
    assert value == expected, f"Expected literal {expected!r}, got {value!r}"
    return value
    # return cast(_T_LITERAL, value)


def json_dumps_dataclass(obj: Any, **kwargs) -> str:
    def dataclass_json_encoder(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    return json.dumps(obj, default=dataclass_json_encoder, **kwargs)


def format_dict(d: dict[Any, Any], sep: str = " | ") -> str:
    return sep.join([f"{k}: {v}" for k, v in d.items()])


def compare_dicts(dict1, dict2):
    """
    Compares two dictionaries and returns a dictionary of differences.

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.

    Returns:
        A dictionary containing the differences between the dictionaries.
        The dictionary has the following keys:
            'only_in_dict1': Keys present only in dict1.
            'only_in_dict2': Keys present only in dict2.
            'different_values': Keys with different values in the two dicts.
    """

    only_in_dict1 = []
    only_in_dict2 = []
    different_values = {}
    is_same = True

    for key, value1 in dict1.items():
        if key not in dict2:
            only_in_dict1.append(key)
            is_same = False
        else:
            value2 = dict2[key]
            if value1 != value2:
                different_values[key] = (value1, value2)
                is_same = False

    for key in dict2:
        if key not in dict1:
            only_in_dict2.append(key)
            is_same = False

    return is_same, {
        "only_in_dict1": only_in_dict1,
        "only_in_dict2": only_in_dict2,
        "different_values": different_values,
    }


@dataclass(frozen=True)
class BaseParams(ABC):
    def modify(
        self,
        **kwargs,
    ):
        return replace(self, **kwargs)

    def modify_ommit_none(self, **kwargs) -> "BaseParams":
        return self.modify(**ommit_none(kwargs))
