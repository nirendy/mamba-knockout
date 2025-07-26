from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from enum import Enum, StrEnum
from functools import singledispatch
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo
from pydantic.type_adapter import TypeAdapter
from pydantic_extra_types.color import Color

from src.utils.streamlit.helpers.allow_nested_expanders import decorator_allow_nested_st_elements
from src.utils.streamlit.st_pydantic_v2.backend import BackendProtocol, StreamlitBackend
from src.utils.streamlit.ui_pydantic_v2.extra_types import PercentageCrop

# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────


class SpecialFieldKeys:
    column_group = "pui_column_group"  # key to group fields in columns
    separator = "pui_separator"  # key for field that renders as a separator
    expander = "pui_expander"  # key for field that renders contents in an expander
    hide = "pui_hide"  # key to hide a field
    component = "pui_component"  # key for custom component override
    kwargs = "pui_kwargs"  # key for component kwargs


class ComponentKeys:
    PREFIX_STATE_DATA = "pui_data_"  # where we store draft input in st.session_state
    PREFIX_WIDGET_KEY = "pui_w_"  # widget key prefix to avoid collisions


@dataclass
class DictComponentKeys:
    """
    Manages consistent key naming for dictionary components.

    This ensures all session state and widget keys follow the same pattern
    and are properly prefixed with the component's widget key.
    """

    widget_key: str  # The base widget key for the component

    @property
    def state_key(self) -> str:
        """Key for storing the dictionary state in session state."""
        return f"{self.widget_key}_state"

    @property
    def original_key(self) -> str:
        """Key for storing the original dictionary value."""
        return f"{self.widget_key}_original"

    def add_button_key(self) -> str:
        """Key for the Add button."""
        return f"{self.widget_key}_add"

    def reset_button_key(self) -> str:
        """Key for the Reset button."""
        return f"{self.widget_key}_reset"

    def key_input_key(self, item_key: Any) -> str:
        """Key for a key input widget for a specific dictionary item."""
        # Ensure item_key is a string and not None
        safe_key = str(item_key) if item_key is not None else "none"
        return f"{self.widget_key}_{safe_key}_k"

    def remove_button_key(self, item_key: Any) -> str:
        """Key for the remove button for a specific dictionary item."""
        # Ensure item_key is a string and not None
        safe_key = str(item_key) if item_key is not None else "none"
        return f"{self.widget_key}_{safe_key}_rm"

    def value_path_key(self, item_key: Any) -> str:
        """Key for the value path of a specific dictionary item."""
        # Ensure item_key is a string and not None
        safe_key = str(item_key) if item_key is not None else "none"
        return f"{self.widget_key}_{safe_key}_v"


T = TypeVar("T", bound=BaseModel)


# ────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────
def is_literal_type(type_obj: Any) -> bool:
    """Check if type_obj is a Literal type."""
    origin = get_origin(type_obj)
    return origin is Literal


def get_literal_values(type_obj: Any) -> List[Any]:
    """Get the allowed values for a Literal type."""
    if not is_literal_type(type_obj):
        return []
    return list(get_args(type_obj))


def has_single_literal_value(type_obj: Any) -> bool:
    """Check if the type has exactly one possible literal value."""
    if not is_literal_type(type_obj):
        return False
    return len(get_literal_values(type_obj)) == 1


def get_single_literal_value(type_obj: Any) -> Any:
    """Get the single literal value if there's only one."""
    if not has_single_literal_value(type_obj):
        return None
    return get_literal_values(type_obj)[0]


def snake_to_title(s: str) -> str:
    """Convert a snake case string to a title case string."""
    return " ".join(word.capitalize() for word in s.split("_"))


def separator() -> Any:
    """Create a field that renders as a separator."""
    return Field(None, json_schema_extra={SpecialFieldKeys.separator: True})


def expander(title: str) -> Any:
    """Create a field that renders its contents in an expander."""
    return Field(None, json_schema_extra={SpecialFieldKeys.expander: title})


def _default_value_for_type(typ: Any) -> Any:
    """Create an appropriate default value for a given type.

    Args:
        typ: The type to create a default value for

    Returns:
        An appropriate default value (empty string for strings, 0 for numbers, etc.)
    """
    if typ is str:
        return ""
    elif typ is int:
        return 0
    elif typ is float:
        return 0.0
    elif typ is bool:
        return False
    elif isinstance(typ, type) and issubclass(typ, Enum):
        # Return the first enum value
        for e in typ:
            return e.name
        return None
    elif is_literal_type(typ):
        # Return the first literal value if available
        values = get_literal_values(typ)
        return values[0] if values else None
    elif isinstance(typ, type) and issubclass(typ, BaseModel):
        # Try to create a default instance
        try:
            return typ()
        except Exception:
            return None
    # For any other type, return empty string as a safe default
    return ""


# ────────────────────────────────────────────────────────────
# Rendering context
# ────────────────────────────────────────────────────────────
class RenderCtx:
    def __init__(self, backend: BackendProtocol, path: str, field_name: str, field_info: FieldInfo):
        self.backend = backend
        self.path = path
        self.field_name = field_name
        self.field_info = field_info

    @property
    def widget_key(self) -> str:
        return f"{self.path}.{ComponentKeys.PREFIX_WIDGET_KEY}"

    @property
    def label(self) -> str:
        return self.field_info.title or snake_to_title(self.field_name)

    @property
    def extra_kwargs(self) -> Dict[str, Any]:
        """Extract kwargs for widget from field_info."""
        extra = self.get_json_schema_extra() or {}
        kwargs = extra.get(SpecialFieldKeys.kwargs, {})
        if hasattr(self.field_info, "description") and self.field_info.description:
            kwargs["help"] = self.field_info.description
        return kwargs

    def get_json_schema_extra(self) -> Dict[str, Any]:
        """Safely get json_schema_extra accounting for it being a function or dict."""
        if not hasattr(self.field_info, "json_schema_extra"):
            return {}

        extra = self.field_info.json_schema_extra
        if extra is None:
            return {}

        if callable(extra):
            # If it's a function, create a new dict and call the function with it
            # This mirrors Pydantic's behavior
            result = {}
            extra(result)
            return result

        # Otherwise assume it's a dict-like object
        return extra

    def get_string_constraint(self, name: str, default=None) -> Any:
        """Get a string constraint from field_info safely"""
        # Try to get constraint from metadata dict
        extra = self.get_json_schema_extra()
        if extra and name in extra:
            return extra.get(name)

        # Try direct attribute access
        if hasattr(self.field_info, name):
            return getattr(self.field_info, name)

        return default

    def get_json_schema_annotation(self):
        annotatated_type = Annotated[self.field_info.annotation, self.field_info]
        return TypeAdapter(annotatated_type).json_schema()

    def get_enum_values_from_field_info(self) -> list[str] | None:
        schema = self.get_json_schema_annotation()

        # If it's an enum or list of enums, the enum is either in 'enum' or in 'items'
        if "enum" in schema:
            return schema["enum"]
        elif "items" in schema:
            ref = schema["items"].get("$ref")
            if ref and "$defs" in schema:
                def_key = ref.split("/")[-1]
                return schema["$defs"].get(def_key, {}).get("enum")
            elif "enum" in schema["items"]:
                return schema["items"]["enum"]

        return None


# ────────────────────────────────────────────────────────────
# singledispatch registry for per‑type widgets
# ────────────────────────────────────────────────────────────
@singledispatch
def render_field(typ: Any, ctx: RenderCtx, init_val: Any) -> Any:  # Change return type from None to Any
    ctx.backend.warning(f"No renderer for type {typ} (path {ctx.path})")
    return None


@render_field.register
def _(typ: str, ctx: RenderCtx, init_val: Any) -> str | None:
    # Use get_string_constraint to safely get max_length
    if ctx.widget_key not in st.session_state:
        st.session_state[ctx.widget_key] = init_val
    max_len = ctx.get_string_constraint("max_length", 0)
    if max_len and max_len > 120:
        return ctx.backend.text_area(ctx.label, key=ctx.widget_key, **ctx.extra_kwargs)
    return ctx.backend.text_input(ctx.label, key=ctx.widget_key, **ctx.extra_kwargs)


@render_field.register
def _(typ: int | float, ctx: RenderCtx, init_val: Any) -> int | float:
    if ctx.widget_key not in st.session_state:
        st.session_state[ctx.widget_key] = init_val
    return ctx.backend.number_input(ctx.label, key=ctx.widget_key, **ctx.extra_kwargs)


@render_field.register
def _(typ: bool, ctx: RenderCtx, init_val: Any) -> bool:
    if ctx.widget_key not in st.session_state:
        st.session_state[ctx.widget_key] = init_val
    result = ctx.backend.checkbox(ctx.label, key=ctx.widget_key, **ctx.extra_kwargs)
    return bool(result) if result is not None else False


@render_field.register
def _(typ: _dt.date | _dt.datetime, ctx: RenderCtx, init_val: Any) -> _dt.date | _dt.datetime:
    if ctx.widget_key not in st.session_state:
        st.session_state[ctx.widget_key] = init_val
    result = ctx.backend.date_input(ctx.label, key=ctx.widget_key, **ctx.extra_kwargs)
    return result if result is not None else _dt.date.today()


@render_field.register
def _(typ: Color, ctx: RenderCtx, init_val: Any) -> Color:
    # Convert the Color object to a hex string or use default
    default_color = "#000000"

    if init_val:
        try:
            if hasattr(init_val, "as_hex") and callable(getattr(init_val, "as_hex")):
                hex_value = init_val.as_hex()
            else:
                hex_value = str(init_val)
        except Exception:
            hex_value = default_color
    else:
        hex_value = default_color

    # Use Streamlit's color picker
    result = ctx.backend.color_picker(ctx.label, key=ctx.widget_key, value=hex_value, **ctx.extra_kwargs)

    # Convert the result back to a Color object
    if result is not None:
        try:
            return Color(result)
        except Exception:
            return Color(default_color)

    return Color(default_color)  # Default to black if None


@render_field.register(Enum | StrEnum)
def _(typ: Type[Enum] | Type[StrEnum], ctx: RenderCtx, init_val: Any) -> Enum | StrEnum:
    if init_val is not None and ctx.widget_key not in st.session_state:
        if isinstance(init_val, typ):
            st.session_state[ctx.widget_key] = init_val.name
        else:
            st.session_state[ctx.widget_key] = init_val
    options = [e.name for e in typ]

    name = ctx.backend.selectbox(
        ctx.label, format_func=lambda x: typ[x], options=options, key=ctx.widget_key, **ctx.extra_kwargs
    )
    assert name is not None
    return typ[name]


@render_field.register(BaseModel)
def _(typ: Type[BaseModel], ctx: RenderCtx, init_val: Any):
    return _render_model(typ, ctx.backend, ctx.path, init_val)


@render_field.register
def _(typ: list | tuple, ctx: RenderCtx, init_val: Any) -> list | tuple:
    backend = ctx.backend
    elem_type = get_args(ctx.field_info.annotation)[0] if get_args(ctx.field_info.annotation) else Any
    items: list | tuple = list(init_val or [])

    enum_items = ctx.get_enum_values_from_field_info()

    if enum_items:
        # Use multiselect
        selected = backend.multiselect(
            ctx.label, options=enum_items, key=ctx.widget_key, default=init_val, **ctx.extra_kwargs
        )

        return selected

    # Default handling for other list types
    backend.subheader(ctx.label)
    for i, current_val in enumerate(list(items)):
        # Get columns as a list
        backend_cols = backend.columns([8, 1])

        item_path = f"{ctx.path}.{i}"
        # Create a new Field with the correct type
        item_field = Field(default=None)
        item_field.annotation = elem_type
        item_ctx = RenderCtx(backend_cols[0], item_path, f"{ctx.field_name}_{i}", item_field)
        items[i] = render_field.dispatch(elem_type)(elem_type, item_ctx, current_val)  # type: ignore[arg-type]

        if backend_cols[1].button("×", key=f"{ctx.widget_key}_rm_{i}"):
            # Use a safe method to remove item
            if i < len(items):
                items.pop(i)
                break

    if backend.button("Add", key=f"{ctx.widget_key}_add"):
        items.append(None)
    return items


# Dict editor --------------------------------------------------------------
@render_field.register(dict)
def _(typ, ctx: RenderCtx, init_val: Any) -> Dict[Any, Any]:
    backend = ctx.backend
    key_type, val_type = get_args(ctx.field_info.annotation) if get_args(ctx.field_info.annotation) else (str, Any)

    # Use DictComponentKeys to manage keys consistently
    keys = DictComponentKeys(widget_key=ctx.widget_key)

    # Initialize session state if not already done
    if keys.original_key not in st.session_state:
        st.session_state[keys.original_key] = dict(init_val or {})

    # Initialize data from session state if available, otherwise use init_val
    if keys.state_key in st.session_state:
        data = dict(st.session_state[keys.state_key])
    else:
        data = dict(init_val or {})
        # Initialize session state with the data
        st.session_state[keys.state_key] = data.copy()

    backend.subheader(ctx.label)
    rows_container_backend = backend.container()

    # Add action buttons in a row (Add and Reset)
    button_cols = backend.columns([1, 1, 2])
    add_button = button_cols[0].button("Add", key=keys.add_button_key())
    reset_button = button_cols[1].button("Reset", key=keys.reset_button_key())

    # Handle Reset button
    if reset_button:
        # Reset to original dictionary value
        data = dict(st.session_state[keys.original_key])
        # Update session state
        st.session_state[keys.state_key] = data.copy()
        # Force UI update
        st.rerun()

    # Handle Add button
    if add_button:
        # Choose an appropriate default key for the new entry
        default_key = None
        if isinstance(key_type, type) and issubclass(key_type, Enum):
            # For Enum types, use the first enum value
            for e in key_type:
                default_key = e.name
                break
        elif is_literal_type(key_type):
            # For Literal types, find an unused value from the available options
            literal_options = get_literal_values(key_type)
            for option in literal_options:
                if option not in data:
                    default_key = option
                    break
                else:
                    # If all options are used, don't add a new key
                    backend.warning("All available keys are already in use.")
                    # Update session state before returning
                    st.session_state[keys.state_key] = data.copy()
                    return data
        else:
            # For regular string keys
            default_key = "key"

        if default_key is not None:
            # Initialize with appropriate default value based on type
            # Ensure string values are always empty strings, not None
            if val_type is str:
                data[default_key] = ""
            else:
                data[default_key] = _default_value_for_type(val_type)

            # Update session state immediately
            st.session_state[keys.state_key] = data.copy()
            # Force UI update
            st.rerun()

    # Render existing items
    for k in list(data.keys()):
        # Get columns as a list
        backend_cols = rows_container_backend.columns([4, 5, 1])

        if isinstance(key_type, type) and issubclass(key_type, Enum):
            enum_options = [e.name for e in key_type]
            try:
                current_index = enum_options.index(k)
            except (ValueError, TypeError):
                current_index = 0
            new_key = backend_cols[0].selectbox(
                "key",
                options=enum_options,
                key=keys.key_input_key(k),
                index=current_index,
                **ctx.extra_kwargs,
            )
        elif is_literal_type(key_type):
            # Handle Literal typed keys
            literal_options = get_literal_values(key_type)
            try:
                current_index = literal_options.index(k)
            except (ValueError, TypeError):
                current_index = 0
            new_key = backend_cols[0].selectbox(
                "key",
                options=literal_options,
                key=keys.key_input_key(k),
                index=current_index,
                **ctx.extra_kwargs,
            )
        else:
            new_key = backend_cols[0].text_input("key", key=keys.key_input_key(k), value=str(k))

        # Create a new Field with the correct type
        val_field = Field(default=None)
        val_field.annotation = val_type
        # Use consistent key for the value path
        val_path_key = keys.value_path_key(k)
        val_ctx = RenderCtx(backend_cols[1], val_path_key, f"{ctx.field_name}_{k}", val_field)
        new_val = render_field.dispatch(val_type)(val_type, val_ctx, data[k])  # type: ignore[arg-type]

        # Put the X button back in the same row
        if backend_cols[2].button("×", key=keys.remove_button_key(k)):
            data.pop(k)
            # Update session state immediately
            st.session_state[keys.state_key] = data.copy()
            # Force UI update
            st.rerun()
            continue

        if new_key != k:
            data[new_key] = data.pop(k)
            # Update session state immediately
            st.session_state[keys.state_key] = data.copy()

        data[new_key] = new_val
        # Update session state immediately
        st.session_state[keys.state_key] = data.copy()
    backend.markdown("---")

    # Final update to session state before returning
    st.session_state[keys.state_key] = data.copy()
    return data


# Union handling (replacing the problematic code) --------------------------
# Remove the problematic @render_field.register(tuple(...)) line
def handle_union(typ, ctx: RenderCtx, init_val):
    """Handle Union types (including Optional)"""
    options = get_args(typ)
    if not options:
        return None

    labels = [t.__name__ if hasattr(t, "__name__") else str(t) for t in options]
    sel_idx = 0
    if init_val is not None:
        for i, t in enumerate(options):
            if isinstance(init_val, t):
                sel_idx = i
                break

    chosen = ctx.backend.selectbox(f"{ctx.label} – type", labels, key=f"{ctx.widget_key}_union", index=sel_idx)
    if chosen is None and labels:
        chosen = labels[0]

    if chosen not in labels:
        chosen = labels[0] if labels else ""

    chosen_type = options[labels.index(chosen)] if chosen in labels else options[0]
    # Create a new Field with the correct type
    sub_field = Field(default=None)
    sub_field.annotation = chosen_type
    field_ctx_sub = RenderCtx(ctx.backend, ctx.path, ctx.field_name, sub_field)
    return render_field.dispatch(chosen_type)(chosen_type, field_ctx_sub, init_val)


# Literal type handling ----------------------------------------------------
def handle_literal(typ, ctx: RenderCtx, init_val):
    """Handle Literal types with multiple options"""
    options = get_literal_values(typ)
    if not options:
        return None

    # Convert all options to strings for display
    str_options = [str(opt) for opt in options]

    if ctx.widget_key not in st.session_state:
        st.session_state[ctx.widget_key] = init_val

    # Use selectbox for the user to choose
    chosen_str = ctx.backend.selectbox(ctx.label, options=str_options, key=ctx.widget_key)

    # Convert back to the original type
    if chosen_str is None and str_options:
        chosen_str = str_options[0]

    chosen_idx = str_options.index(chosen_str) if chosen_str in str_options else 0
    chosen_value = options[chosen_idx]

    # Handle specific types
    if isinstance(chosen_value, bool):
        return bool(chosen_value)
    elif isinstance(chosen_value, int):
        return int(chosen_value)
    elif isinstance(chosen_value, float):
        return float(chosen_value)

    return chosen_value


@render_field.register(PercentageCrop)
def _(typ: Type[PercentageCrop], ctx: RenderCtx, init_val: dict) -> PercentageCrop:
    """Render a Crop field with streamlit-cropper if an image is available."""
    for side in ["left", "top", "width", "height"]:
        if f"{ctx.path}.{side}" not in st.session_state:
            st.session_state[f"{ctx.path}.{side}"] = init_val[side]

    # If an image is available, render the cropper
    if (image := ctx.get_json_schema_extra().get("image")) is not None:
        ctx.backend.cropper(image, key=ctx.path, **ctx.extra_kwargs)

    pass
    # Render regular fields for the crop values
    for side, col in zip(["left", "top", "width", "height"], ctx.backend.columns([1] * 4)):
        col.number_input(side, min_value=0.0, max_value=100.0, format="%.2f", key=f"{ctx.path}.{side}")

    return PercentageCrop.model_validate(
        {side: st.session_state[f"{ctx.path}.{side}"] for side in ["left", "top", "width", "height"]}
    )


# ────────────────────────────────────────────────────────────
# Core recursive renderer
# ────────────────────────────────────────────────────────────


def _render_model(
    model_cls: Type[BaseModel],
    backend: BackendProtocol,
    path_prefix: str,
    init_values: Dict[str, Any],
) -> Dict[str, Any]:
    data: Dict[str, Any] = {}

    # Collect fields by column group
    column_groups: Dict[
        str, tuple[BackendProtocol, List[Tuple[str, Any, Any]]]
    ] = {}  # group_name -> [(backend, [(fname, finfo, init_val)])]

    for fname, finfo in model_cls.model_fields.items():
        # Create context object
        field_ctx = RenderCtx(backend, f"{path_prefix}.{fname}", fname, finfo)

        # Get field schema extra information safely
        extra = field_ctx.get_json_schema_extra() or {}

        # hidden field? ------------------------------------------------------
        if extra.get(SpecialFieldKeys.hide):
            continue

        # Get initial value
        init_val = init_values.get(fname)

        # Skip rendering and use the only possible value for literal constants
        if has_single_literal_value(finfo.annotation):
            # For fields with only one possible value, just use that value directly
            data[fname] = get_single_literal_value(finfo.annotation)
            continue

        # Check for special field types

        # Separator field?
        if extra.get(SpecialFieldKeys.separator):
            backend.markdown("---")
            continue

        # Expander field?
        new_backend = (
            backend.expander(str(expander_title))
            if (expander_title := extra.get(SpecialFieldKeys.expander))
            else backend
        )

        # Column group field?
        column_group = extra.get(SpecialFieldKeys.column_group)
        if column_group:
            if column_group not in column_groups:
                column_groups[column_group] = (new_backend.container(), [])
            column_groups[column_group][1].append((fname, finfo, init_val))
            continue

        # Regular field - render as before
        # custom component override? ---------------------------------------
        custom_component = extra.get(SpecialFieldKeys.component)
        if callable(custom_component):
            data[fname] = custom_component(new_backend, finfo, init_val)
        else:
            anno = finfo.annotation
            origin = get_origin(anno)

            # Handle specific type patterns
            if origin is Union:
                # discriminated union / simple union
                data[fname] = handle_union(anno, field_ctx, init_val)
            elif anno is not None and isinstance(anno, type) and issubclass(anno, BaseModel):
                # Recursively render the sub-model inside the expander
                assert init_val is not None
                data[fname] = _render_model(anno, new_backend, f"{path_prefix}.{fname}", init_val)
            # elif is_literal_type(anno):
            #     # Literal type with multiple options
            #     data[fname] = handle_literal(anno, field_ctx, init_val)
            else:
                # Other types
                origin_or_anno = origin or anno
                handler = render_field.dispatch(origin_or_anno)
                data[fname] = handler(origin_or_anno, field_ctx, init_val)

    # Process column groups
    for group_name, (new_backend, fields) in column_groups.items():
        # Create columns
        if not group_name.startswith("_"):
            new_backend.subheader(snake_to_title(group_name))
        backend_cols = new_backend.columns([1] * len(fields))

        # Render each field in its own column
        for i, (fname, finfo, init_val) in enumerate(fields):
            field_ctx = RenderCtx(backend_cols[i], f"{path_prefix}.{fname}" if path_prefix else fname, fname, finfo)

            # Render field using dispatch method
            anno = finfo.annotation
            origin = get_origin(anno)

            # Use the same rendering logic as regular fields
            if origin is Union:
                data[fname] = handle_union(anno, field_ctx, init_val)
            elif is_literal_type(anno):
                data[fname] = handle_literal(anno, field_ctx, init_val)
            else:
                origin_or_anno = origin or anno
                try:
                    handler = render_field.dispatch(origin_or_anno)
                    data[fname] = handler(origin_or_anno, field_ctx, init_val)
                except (AttributeError, TypeError):
                    # Fallback
                    if origin_or_anno is not None and hasattr(origin_or_anno, "__origin__"):
                        container_type = origin_or_anno.__origin__
                        if container_type is list or container_type is tuple:
                            data[fname] = render_field.dispatch(list)(list, field_ctx, init_val)
                        elif container_type is dict:
                            data[fname] = render_field.dispatch(dict)(dict, field_ctx, init_val)
                        else:
                            data[fname] = render_field.dispatch(str)(
                                str, field_ctx, str(init_val) if init_val is not None else ""
                            )
                    else:
                        data[fname] = render_field.dispatch(str)(
                            str, field_ctx, str(init_val) if init_val is not None else ""
                        )

    return data


# ────────────────────────────────────────────────────────────
# Public façade
# ────────────────────────────────────────────────────────────


@decorator_allow_nested_st_elements
def pydantic_ui(
    key: str,
    model: Union[Type[T], T],
    *,
    backend: Optional[BackendProtocol] = None,
    use_form: bool = False,
    submit_label: str = "Submit",
    clear_on_submit: bool = False,
) -> T:  # noqa: E501
    """Render UI, validate and return model instance (or None on validation error)."""

    # Ensure backend is never None
    backend_impl = StreamlitBackend() if backend is None else backend
    bucket = ComponentKeys.PREFIX_STATE_DATA + key
    reset_to_default = st.button("Reset", key=f"{key}_reset")

    # Create a model class and an adapter for validation
    if isinstance(model, BaseModel):
        instance_val = model
        model_cls = model.__class__
    else:
        instance_val = None
        model_cls = model

    adapter = TypeAdapter(model_cls)

    if reset_to_default:
        for k in st.session_state:
            if k.startswith(bucket):
                del st.session_state[k]

    # Initialize with saved or instance data if available
    if bucket in st.session_state:
        try:
            instance_val = adapter.validate_python(st.session_state[bucket])
        except ValidationError:
            backend_impl.toast("Invalid saved data")

    if instance_val is None:
        backend_impl.toast("Creating a new instance")
        instance_val = model_cls()

    init_values = instance_val.model_dump()  # mode=json?
    st.session_state[bucket] = init_values

    # Handle the form context
    # Create form context if use_form is True
    if use_form:
        with st.form(key=key, clear_on_submit=clear_on_submit):
            raw_data = _render_model(model_cls, backend_impl, bucket, init_values)
            submitted = st.form_submit_button(submit_label)

            if submitted:
                try:
                    validated = adapter.validate_python(raw_data)
                    st.session_state[bucket] = validated.model_dump(mode="json")
                    return validated  # type: ignore[return-value]
                except ValidationError as exc:
                    # Show error but return the original instance
                    backend_impl.warning(str(exc))
                    return instance_val
            # Return the instance if form not submitted
            return instance_val
    else:
        # No form - simpler case
        raw_data = _render_model(model_cls, backend_impl, bucket, init_values)

        try:
            validated = adapter.validate_python(raw_data)
            st.session_state[bucket] = validated.model_dump(mode="json")
            return validated  # type: ignore[return-value]
        except ValidationError as exc:
            # Show validation error and return the original instance
            backend_impl.toast("Invalid data, returning the original instance")
            backend_impl.warning(str(exc))
            return instance_val  # type: ignore[return-value]
