"""
TODO: This module works well enough for our need, but need to be improved

"""

from __future__ import annotations

import importlib
import inspect
import json
from abc import ABC
from collections import defaultdict, namedtuple
from enum import Enum, StrEnum
from typing import (
    Any,
    ClassVar,
    ForwardRef,
    Iterable,
    Optional,
    Self,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from devtools import debug
from pydantic import BaseModel
from pydantic_extra_types.color import Color


class _JSONAbleMarkers:
    TYPE_MARKER = "__JSONABLE_type__"
    MODULE_MARKER = "__JSONABLE_module__"
    VALUE_MARKER = "__JSONABLE_value__"
    METADATA_MARKER = "__JSONABLE_metadata__"


class METADATA_TYPES(StrEnum):
    DATACLASS = "__JSONABLE_dataclass__"
    NAMEDTUPLE = "__JSONABLE_namedtuple__"
    ENUM = "__JSONABLE_enum__"
    TUPLE = "__JSONABLE_tuple__"
    SET = "__JSONABLE_set__"


# Helper functions to detect types
def _is_namedtuple_instance(obj):
    return (
        isinstance(obj, tuple)
        and hasattr(obj, "_fields")
        and hasattr(obj, "_asdict")
        and isinstance(getattr(obj, "_fields"), tuple)
    )


def _is_dataclass_instance(obj):
    return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)


def is_namedtuple_class(cls):
    return inspect.isclass(cls) and hasattr(cls, "_fields") and isinstance(getattr(cls, "_fields", None), tuple)


def _resolve_typevars(tp: Any, mapping: dict[TypeVar, Any]) -> Any:
    if isinstance(tp, TypeVar):
        return _resolve_typevars(mapping.get(tp, tp), mapping)

    origin = get_origin(tp)
    if origin is None:  # not a parametrised generic
        return tp

    args = tuple(_resolve_typevars(a, mapping) for a in get_args(tp))

    # --- single‑arg generics (ClassVar, Optional, list, …) -------------
    if len(args) == 1:
        return origin[args[0]]
    # --- multi‑arg generics (dict, tuple, Callable[..., T], …) ----------
    return origin[args]


def resolved_type_hints(cls: type) -> dict[str, Any]:
    hints: dict[str, Any] = get_type_hints(cls, include_extras=True)

    mapping: dict[TypeVar, Any] = {}
    for c in cls.__mro__:
        for base in getattr(c, "__orig_bases__", ()):
            origin = get_origin(base)
            if origin is None:
                continue
            params = getattr(origin, "__parameters__", ())
            mapping.update(zip(params, get_args(base)))

    return {name: _resolve_typevars(tp, mapping) for name, tp in hints.items()}


class JSONAble(ABC):
    _registry: ClassVar[dict[str, Type[JSONAble]]] = {}
    _old_registries: ClassVar[dict[str, list[Type[JSONAble]]]] = defaultdict(list)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ in cls._registry:
            cls._old_registries[cls.__name__].append(cls._registry[cls.__name__])
            del cls._registry[cls.__name__]
            raise ValueError(f"Serializable class {cls.__name__} already registered")
        cls._registry[cls.__name__] = cls

    def to_jsonable_dict(self) -> dict:
        # Special handling for dataclass-based JSONAble objects

        def _rec_to_dict(obj, type_hint: Optional[Type]):
            # handle Union types
            if get_origin(type_hint) is Union:
                args: tuple[Any, ...] = get_args(type_hint)
                for arg in args:
                    try:
                        return _rec_to_dict(obj, arg)
                    except Exception:
                        continue
                raise ValueError(f"Failed to serialize Union type: {type_hint}")
            if isinstance(obj, JSONAble):
                return {
                    _JSONAbleMarkers.TYPE_MARKER: obj.__class__.__name__,
                    _JSONAbleMarkers.VALUE_MARKER: {
                        k: _rec_to_dict(v, type_hint=type_hint and type_hint.__annotations__.get(k))
                        for k, v in obj.__dict__.items()
                    },
                }
            elif _is_dataclass_instance(obj):
                # Handle dataclasses that are not Serializable instances
                data = {
                    k: _rec_to_dict(v, type_hint=type_hint and type_hint.__annotations__.get(k))
                    for k, v in obj.__dict__.items()
                }
                if type_hint is None:
                    return {
                        _JSONAbleMarkers.METADATA_MARKER: METADATA_TYPES.DATACLASS,
                        _JSONAbleMarkers.MODULE_MARKER: obj.__class__.__module__,
                        _JSONAbleMarkers.TYPE_MARKER: obj.__class__.__name__,
                        _JSONAbleMarkers.VALUE_MARKER: data,
                    }
                else:
                    return data
            elif _is_namedtuple_instance(obj):
                # Handle named tuples
                data = [
                    _rec_to_dict(getattr(obj, field), type_hint=type_hint and type_hint.__annotations__.get(field))
                    for field in obj._fields
                ]
                if type_hint is None:
                    return {
                        _JSONAbleMarkers.METADATA_MARKER: METADATA_TYPES.NAMEDTUPLE,
                        _JSONAbleMarkers.MODULE_MARKER: obj.__class__.__module__,
                        _JSONAbleMarkers.TYPE_MARKER: obj.__class__.__name__,
                        _JSONAbleMarkers.VALUE_MARKER: data,
                    }
                else:
                    return data
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                if isinstance(obj, dict):
                    _, value_type = None, None
                    if type_hint is not None:
                        args = get_args(type_hint)
                        if args and len(args) > 1:
                            # TODO: handle key as well
                            _, value_type = args[0], args[1]
                    data = {k: _rec_to_dict(v, type_hint=value_type) for k, v in obj.items()}
                    return data
                elif isinstance(obj, BaseModel):
                    return _rec_to_dict(obj.model_dump(), type_hint=None)
                elif type_hint is not None:
                    origin = get_origin(type_hint)
                    args = get_args(type_hint)
                    item_type = None
                    if origin is not None and len(args) > 0:
                        item_type = args[0]
                    return [_rec_to_dict(item, type_hint=item_type) for item in obj]
                # Handle tuples without type hints
                elif isinstance(obj, tuple):
                    return {
                        _JSONAbleMarkers.METADATA_MARKER: METADATA_TYPES.TUPLE,
                        _JSONAbleMarkers.VALUE_MARKER: [
                            _rec_to_dict(item, type_hint=None) for i, item in enumerate(obj)
                        ],
                    }

                # Handle sets without type hints
                elif isinstance(obj, set):
                    return {
                        _JSONAbleMarkers.METADATA_MARKER: METADATA_TYPES.SET,
                        _JSONAbleMarkers.VALUE_MARKER: [_rec_to_dict(item, None) for item in obj],
                    }

                return [_rec_to_dict(item, type_hint=None) for item in obj]
            elif isinstance(obj, Enum):
                if type_hint is None:
                    return {
                        _JSONAbleMarkers.METADATA_MARKER: METADATA_TYPES.ENUM,
                        _JSONAbleMarkers.MODULE_MARKER: obj.__class__.__module__,
                        _JSONAbleMarkers.TYPE_MARKER: obj.__class__.__name__,
                        _JSONAbleMarkers.VALUE_MARKER: obj.value,
                    }
                else:
                    return obj.value
            elif type(obj) is Color:
                return str(obj)
            return obj

        return cast(dict, _rec_to_dict(self, type_hint=self.__class__))

    def to_jsonable_json(self, **kwargs) -> str:
        """Serialize the object to a JSON string"""
        return json.dumps(self.to_jsonable_dict(), **kwargs)

    @classmethod
    def from_jsonable_dict(cls, data: dict):
        def _init_cls_from_typehint(obj, type_hint: Type):
            # Use type hint to deserialize
            origin: Type = get_origin(type_hint)  # type: ignore

            # Handle Optional types
            if origin is Union and type(None) in get_args(type_hint):
                args = get_args(type_hint)
                if len(args) > 0:
                    # Find non-None type in Optional[T]
                    inner_types = [arg for arg in args if arg is not type(None)]
                    if inner_types and obj is not None:
                        return _rec_from_dict(obj, inner_types[0])
                    return None

            # Handle Union types
            if origin is Union:
                args = get_args(type_hint)
                # Try each type in the union
                for arg_type in args:
                    try:
                        return _rec_from_dict(obj, arg_type)
                    except Exception:
                        continue
                # If all fail, return as is
                return obj

            # Check if it's an Enum by checking if it's a class and subclass of Enum
            if inspect.isclass(type_hint) and issubclass(type_hint, Enum):
                if isinstance(obj, str):
                    return type_hint(obj)
                else:
                    assert isinstance(obj, dict)
                    assert _JSONAbleMarkers.VALUE_MARKER in obj
                    assert obj[_JSONAbleMarkers.TYPE_MARKER] == type_hint.__name__
                    return type_hint(obj[_JSONAbleMarkers.VALUE_MARKER])
            # Check if it's a namedtuple
            elif is_namedtuple_class(type_hint):
                # Convert list back to tuple for namedtuple
                assert isinstance(obj, list)
                type_hints = get_type_hints(type_hint)
                return type_hint(*[_rec_from_dict(item, type_hint=type_hints.get(i)) for i, item in enumerate(obj)])  # type: ignore
            elif origin is dict:
                key_type, value_type = None, None
                if type_hint is not None:
                    args = get_args(type_hint)
                    if args and len(args) > 1:
                        key_type, value_type = args[0], args[1]
                return {_rec_from_dict(k, key_type): _rec_from_dict(v, value_type) for k, v in obj.items()}
            elif inspect.isclass(origin) and issubclass(origin, Iterable):  # type: ignore
                args = get_args(type_hint)
                if args[-1] is ...:
                    item_types = [args[0]] * len(obj["values"])
                else:
                    item_types = args
                return [_rec_from_dict(item, type_hint=item_types[i]) for i, item in enumerate(obj["values"])]
            elif type_hint == Any:
                return obj
            else:
                # Check if it's a dataclass
                new_obj = type_hint.__new__(type_hint)  # type: ignore

                type_hints = {}
                if inspect.isclass(type_hint) and (
                    hasattr(type_hint, "__dataclass_fields__") or issubclass(type_hint, BaseModel)
                ):
                    # Process fields with their type hints
                    type_hints = resolved_type_hints(type_hint)
                elif type_hint is not None:
                    type_hints = type_hint.__annotations__

                new_obj.__dict__.update({k: _rec_from_dict(v, type_hint=type_hints.get(k)) for k, v in obj.items()})

                if hasattr(new_obj, "__post_init__"):
                    getattr(new_obj, "__post_init__")()
                return new_obj

        def _rec_from_dict(obj, type_hint: Optional[Type]):
            # handle Union types
            if type_hint is not None and get_origin(type_hint) is Union:
                args = get_args(type_hint)
                any_type = [True]
                if isinstance(obj, dict) and _JSONAbleMarkers.TYPE_MARKER in obj:
                    any_type.insert(0, False)
                for is_any_type in any_type:
                    options = []
                    for arg in args:
                        if is_any_type or arg.__name__ == obj[_JSONAbleMarkers.TYPE_MARKER]:
                            try:
                                options.append(_rec_from_dict(obj, arg))
                            except Exception:
                                continue
                    if options:
                        if len(options) > 1:
                            try:
                                options = list(set(options))
                            except Exception:
                                pass
                        assert len(options) == 1, f"Failed to deserialize Union type. {debug(obj, type_hint, options)}"
                        return options[0]
                raise ValueError(f"Failed to deserialize Union type. {debug(obj, type_hint)}")

            if isinstance(obj, dict):
                if type_hint is not None:
                    if (
                        _JSONAbleMarkers.TYPE_MARKER in obj
                        and _JSONAbleMarkers.VALUE_MARKER in obj
                        and obj[_JSONAbleMarkers.TYPE_MARKER] in cls._registry
                    ):
                        assert issubclass(cls._registry[obj[_JSONAbleMarkers.TYPE_MARKER]], type_hint)
                        return _init_cls_from_typehint(
                            obj[_JSONAbleMarkers.VALUE_MARKER], cls._registry[obj[_JSONAbleMarkers.TYPE_MARKER]]
                        )
                    else:
                        if isinstance(type_hint, ForwardRef):
                            module = importlib.import_module(obj[_JSONAbleMarkers.MODULE_MARKER])
                            type_hint = type_hint._evaluate(globalns=module.__dict__, localns={}, recursive_guard=set())
                            obj = obj[_JSONAbleMarkers.VALUE_MARKER]
                            _rec_from_dict(obj=obj, type_hint=type_hint)
                        elif _JSONAbleMarkers.METADATA_MARKER in obj:
                            if _JSONAbleMarkers.TYPE_MARKER in obj:
                                assert obj[_JSONAbleMarkers.TYPE_MARKER] == type_hint.__name__
                            obj = obj[_JSONAbleMarkers.VALUE_MARKER]
                            return _rec_from_dict(obj=obj, type_hint=type_hint)
                        return _init_cls_from_typehint(obj=obj, type_hint=type_hint)
                elif _JSONAbleMarkers.METADATA_MARKER in obj:
                    if _JSONAbleMarkers.TYPE_MARKER in obj:
                        if obj[_JSONAbleMarkers.TYPE_MARKER] in cls._registry:
                            # This is a nested Serializable object
                            serializable_cls = cls._registry[obj[_JSONAbleMarkers.TYPE_MARKER]]

                            # Get type hints for the class
                            type_hints = get_type_hints(serializable_cls)

                            # Recursively process attributes with type hints
                            processed_params = {}
                            for k, v in obj[_JSONAbleMarkers.VALUE_MARKER].items():
                                type_hint = type_hints.get(k)
                                processed_params[k] = _rec_from_dict(v, type_hint)

                            # Special handling for dataclass-based JSONAble objects
                            if hasattr(serializable_cls, "__dataclass_fields__"):
                                return serializable_cls(**processed_params)
                            else:
                                new_obj = serializable_cls.__new__(serializable_cls)
                                new_obj.__dict__.update(processed_params)
                                if hasattr(new_obj, "__post_init__"):
                                    getattr(new_obj, "__post_init__")()
                                return new_obj
                        elif obj[_JSONAbleMarkers.METADATA_MARKER] == METADATA_TYPES.ENUM:
                            # Import the enum class dynamically
                            module = importlib.import_module(obj[_JSONAbleMarkers.MODULE_MARKER])
                            enum_cls = getattr(module, obj[_JSONAbleMarkers.TYPE_MARKER])
                            return enum_cls(obj[_JSONAbleMarkers.VALUE_MARKER])
                        elif obj[_JSONAbleMarkers.METADATA_MARKER] == METADATA_TYPES.DATACLASS:
                            # Import the dataclass dynamically
                            module = importlib.import_module(obj[_JSONAbleMarkers.MODULE_MARKER])
                            dataclass_cls = getattr(module, obj[_JSONAbleMarkers.TYPE_MARKER])
                            type_hints = get_type_hints(dataclass_cls)
                            # Recursively process the dataclass fields
                            processed_data = {
                                k: _rec_from_dict(v, type_hint=type_hints.get(k))
                                for k, v in obj[_JSONAbleMarkers.VALUE_MARKER].items()
                            }
                            return dataclass_cls(**processed_data)
                        elif obj[_JSONAbleMarkers.METADATA_MARKER] == METADATA_TYPES.NAMEDTUPLE:
                            # Import the namedtuple class dynamically or recreate it
                            # First try to import the existing namedtuple class
                            try:
                                module = importlib.import_module(obj[_JSONAbleMarkers.MODULE_MARKER])
                                nt_cls = getattr(module, obj[_JSONAbleMarkers.TYPE_MARKER])
                            except (ImportError, AttributeError):
                                # If not found, recreate the namedtuple class
                                nt_cls = namedtuple(
                                    obj[_JSONAbleMarkers.TYPE_MARKER], obj[_JSONAbleMarkers.VALUE_MARKER]
                                )

                            # Recursively process the values
                            processed_values = [
                                _rec_from_dict(v, type_hint=nt_cls.__annotations__.get(i))  # type: ignore
                                for i, v in enumerate(obj[_JSONAbleMarkers.VALUE_MARKER])
                            ]
                            return nt_cls(*processed_values)
                        else:
                            raise NotImplementedError(f"Unknown pattern: {debug(obj, type_hint)}")
                    elif obj[_JSONAbleMarkers.METADATA_MARKER] == METADATA_TYPES.TUPLE:
                        item_types = None
                        if type_hint is not None:
                            origin = get_origin(type_hint)
                            args = get_args(type_hint)
                            if args[-1] is ...:
                                item_types = [args[0]] * len(obj[_JSONAbleMarkers.VALUE_MARKER])
                            else:
                                item_types = args
                        # Handle special marker for tuples
                        return tuple(
                            _rec_from_dict(v, type_hint=item_types and item_types[i])  # type: ignore
                            for i, v in enumerate(obj[_JSONAbleMarkers.VALUE_MARKER])
                        )
                    elif obj[_JSONAbleMarkers.METADATA_MARKER] == METADATA_TYPES.SET:
                        # Handle special marker for sets
                        return set(_rec_from_dict(v, type_hint=set) for v in obj[_JSONAbleMarkers.VALUE_MARKER])
                else:
                    # Regular dictionary without type hints
                    return {k: _rec_from_dict(v, type_hint=type_hint and type_hint.get(k)) for k, v in obj.items()}
                # Check for special markers
            elif isinstance(obj, list):
                item_type = None
                if type_hint is not None:
                    origin = get_origin(type_hint)
                    args = get_args(type_hint)
                    if origin is not None and len(args) > 0:
                        item_type = args[0]

                # Convert to the appropriate container type
                result = [_rec_from_dict(item, item_type) for item in obj]

                # Handle type hint conversion
                if type_hint is not None:
                    origin = get_origin(type_hint)
                    if origin is tuple:
                        return tuple(result)
                    elif origin is set:
                        return set(result)
                    if is_namedtuple_class(type_hint):
                        return _init_cls_from_typehint(result, type_hint)
                return result
            # check if it's an enum
            if type_hint is not None and inspect.isclass(type_hint) and issubclass(type_hint, Enum):
                return _init_cls_from_typehint(obj, type_hint)
            # For primitive values, return as is
            return obj

        if _JSONAbleMarkers.TYPE_MARKER in data:
            target_cls = cls._registry[data[_JSONAbleMarkers.TYPE_MARKER]]
        else:
            target_cls = cls

            if target_cls == JSONAble:
                target_cls = None
        # Process parameters recursively
        deserialized = _rec_from_dict(data, type_hint=target_cls)

        return cast(Self, deserialized)

    @classmethod
    def from_jsonable_json(cls, json_str: str):
        """Deserialize the object from a JSON string"""
        parsed = json.loads(json_str)
        return cls.from_jsonable_dict(parsed)
