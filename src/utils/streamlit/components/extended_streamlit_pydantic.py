from enum import Enum
from typing import Annotated, Any, Callable, Dict, Literal, Type, TypeVar, cast, get_args, get_origin

import streamlit as st
from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color
from streamlit_pydantic import schema_utils
from streamlit_pydantic.ui_renderer import GroupOptionalFieldsStrategy, InputUI

from src.utils.pydantic_utils import create_literal_value
from src.utils.types_utils import conditional_context_manager

# Also, need to fix a bug in the library:
# see  https://github.com/lukasmasuch/streamlit-pydantic/issues/69


T = TypeVar("T")


def get_dict_key_literal_values(model: BaseModel, field_name: str):
    annotation = model.model_fields[field_name].annotation
    if annotation:
        args = get_args(annotation)[0]
        if get_origin(args) == Literal:
            return get_args(args)
    return None


def annotate_dict_with_literal_values(keys: list[str], value_type: type, default_factory: Callable[[], dict] = dict):
    result_type = dict[
        create_literal_value(keys) if len(keys) > 0 else str,
        value_type,
    ]

    def f_json_schema_extra(schema: dict):
        schema["maxItems"] = len(keys)
        # schema["minItems"] = len(value)
        if len(keys) == 0:
            schema["readOnly"] = True
        pass

    return Annotated[
        result_type,
        Field(
            default_factory=default_factory,
            json_schema_extra=f_json_schema_extra,
        ),
    ]


class ExtendedSchemaUtils:
    """Extension of schema utils with additional type checks."""

    @staticmethod
    def is_multi_enum_property_extended(property: Dict, references: Dict) -> bool:
        """Check if property is a list of enums, extending the original check."""
        # First check original implementation
        if schema_utils.is_multi_enum_property(property, references):
            return True

        if property.get("type") != "array":
            return False

        if property.get("items", {}).get("$ref") is None:
            return False

        try:
            # Get the reference item
            reference = schema_utils.resolve_reference(property["items"]["$ref"], references)
            # Check if it's an enum
            return bool(reference.get("enum"))
        except Exception:
            return False


class CustomInputUI(InputUI):
    """Extended version of InputUI that supports custom type renderers."""

    # Registry to store custom type renderers
    _custom_type_renderers: Dict[Type, Callable] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema_properties = self._input_class.model_json_schema(by_alias=True).get("properties", {})
        self._schema_references = self._input_class.model_json_schema(by_alias=True).get("$defs", {})

    @classmethod
    def register_type_renderer(cls, type_cls: Type[T], renderer: Callable[[Any, str, Dict], T]) -> None:
        """Register a custom renderer for a specific type.

        Args:
            type_cls: The type class to register a renderer for
            renderer: A function that takes (streamlit_app, key, property) and returns the rendered value
        """
        cls._custom_type_renderers[type_cls] = renderer

    def _render_property(self, streamlit_app: Any, key: str, property: Dict) -> Any:
        """Override _render_property to support custom type renderers."""
        # Check if we have a custom renderer for this type
        if property.get("init_value") is not None:
            value_type = type(property["init_value"])
            if value_type in self._custom_type_renderers:
                return self._custom_type_renderers[value_type](streamlit_app, key, property)

        # # Check if this is a list of enums using our extended check
        # if ExtendedSchemaUtils.is_multi_enum_property_extended(property, self._schema_references):
        #     return self._render_multi_enum_input_extended(streamlit_app, key, property)

        # Fallback to parent class implementation
        return super()._render_property(streamlit_app, key, property)

    def _render_multi_enum_input_extended(self, streamlit_app: Any, key: str, property: Dict) -> Any:
        """Render any list of enums input."""
        streamlit_kwargs = self._get_default_streamlit_input_kwargs(key, property)
        overwrite_kwargs = self._get_overwrite_streamlit_kwargs(key, property)

        # Get the enum values either from direct enum or reference
        select_options = []
        if property.get("items", {}).get("enum"):
            select_options = property["items"]["enum"]
        else:
            # Get from reference
            reference_item = schema_utils.resolve_reference(property["items"]["$ref"], self._schema_references)
            select_options = reference_item["enum"]

        # Get current value or default
        current_value = []
        if property.get("init_value"):
            current_value = [opt.value if isinstance(opt, Enum) else opt for opt in property["init_value"]]
        elif property.get("default"):
            try:
                current_value = property.get("default")
            except Exception:
                pass

        # Create multiselect
        selected_values = streamlit_app.multiselect(
            **{**streamlit_kwargs, "options": select_options, "default": current_value, **overwrite_kwargs}
        )

        # If we have a reference, convert back to enum values
        if property.get("items", {}).get("$ref"):
            # Import the enum dynamically based on the reference name
            ref_name = property["items"]["$ref"].split("/")[-1]
            enum_class = None

            # Try to find the enum class in the references
            reference_item = self._schema_references.get(ref_name)
            if reference_item and reference_item.get("enum"):
                # Get module path from the reference if available
                module_path = reference_item.get("module_path")
                if module_path:
                    import importlib

                    try:
                        module = importlib.import_module(module_path)
                        enum_class = getattr(module, ref_name)
                    except (ImportError, AttributeError):
                        pass

            if enum_class and issubclass(enum_class, Enum):
                return [enum_class(val) for val in selected_values]

        return selected_values

    def _render_dict_add_button(self, key: str, streamlit_app: Any, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        next_key = str(len(data_dict) + 1)
        if values := get_dict_key_literal_values(cast(BaseModel, self._input_class), key):
            remaining_values = [val for val in values if val not in data_dict]
            if remaining_values:
                next_key = remaining_values[0]
            else:
                return data_dict

        if streamlit_app.button(
            "Add Item",
            key=self._key + "-" + key + "-add-item",
        ):
            data_dict[next_key] = None

        return data_dict

    def _render_dict_item(
        self,
        streamlit_app: Any,
        parent_key: str,
        in_value: tuple[str, Any],
        index: int,
        property: Dict[str, Any],
    ) -> Any:
        new_key = self._key + "-" + parent_key + "." + str(index)
        item_placeholder = streamlit_app.empty()

        with item_placeholder.container():
            key_col, value_col, button_col = streamlit_app.columns([4, 4, 3])

            dict_key = in_value[0]
            dict_value = in_value[1]

            dict_key_key = new_key + "-key"
            dict_value_key = new_key + "-value"

            button_col.markdown("##")

            if self._remove_button_allowed(index, property):
                remove = False
            else:
                remove = button_col.button("Remove", key=new_key + "-remove")

            if not remove:
                with key_col:
                    if (annotation := self._input_class.model_fields[parent_key].annotation) and get_origin(
                        get_args(annotation)[0]
                    ) == Literal:
                        values = get_args(get_args(annotation)[0])
                        new_key_property = {
                            "title": "Key",
                            "default": values[0],
                            "init_value": dict_key,
                            "enum": values,
                        }
                        updated_key = self._render_property(streamlit_app, dict_key_key, new_key_property)

                    else:
                        updated_key = streamlit_app.text_input(
                            "Key",
                            value=dict_key,
                            key=dict_key_key,
                            disabled=property.get("readOnly", False),
                        )

                with value_col:
                    new_property = {
                        "title": "Value",
                        "init_value": dict_value,
                        "is_item": True,
                        "readOnly": property.get("readOnly"),
                        **property["additionalProperties"],
                    }
                    with value_col:
                        updated_value = self._render_property(streamlit_app, dict_value_key, new_property)

                    return updated_key, updated_value

            else:
                # when the remove button is clicked clear the placeholder and return None
                item_placeholder.empty()
                return None, None

    def _render_single_color_input(self, streamlit_app: Any, key: str, property: Dict) -> Any:
        streamlit_kwargs = self._get_default_streamlit_input_kwargs(key, property)
        overwrite_kwargs = self._get_overwrite_streamlit_kwargs(key, property)

        def ensure_hex_format(color: Color | str) -> str:
            if isinstance(color, Color):
                return color.as_hex(format="long")
            return color

        if property.get("init_value") is not None:
            streamlit_kwargs["value"] = ensure_hex_format(property["init_value"])
        elif property.get("default") is not None:
            streamlit_kwargs["value"] = ensure_hex_format(property["default"])
        elif property.get("example") is not None:
            streamlit_kwargs["value"] = ensure_hex_format(property["example"])

        return streamlit_app.color_picker(**{**streamlit_kwargs, **overwrite_kwargs})


def pydantic_input(
    key: str,
    model: BaseModel | Type[BaseModel],
    group_optional_fields: GroupOptionalFieldsStrategy = GroupOptionalFieldsStrategy.NO,
    lowercase_labels: bool = False,
    ignore_empty_values: bool = False,
    with_form: bool = False,
) -> Dict:
    """Extended version of pydantic_input that uses CustomInputUI."""
    model_instance = {}
    if isinstance(model, BaseModel):
        model_instance = model.model_dump()
        model = model.__class__

    ctx = conditional_context_manager(with_form, st.form(key))
    ui = CustomInputUI(
        key,
        model,
        group_optional_fields=group_optional_fields,
        lowercase_labels=lowercase_labels,
        ignore_empty_values=ignore_empty_values,
    )
    reset_values = st.button("Reset values")
    if model_instance:
        for key, value in model_instance.items():
            if reset_values or ui._get_value(key) is None:
                ui._store_value(key, value)

    with ctx:
        if with_form:
            st.form_submit_button("Submit")

        return ui.render_ui()


# Example of how to register a custom renderer:
# def render_my_custom_type(streamlit_app: Any, key: str, property: Dict) -> Any:
#     # Custom rendering logic here
#     return streamlit_app.text_input(**property)
#
# CustomInputUI.register_type_renderer(MyCustomType, render_my_custom_type)
