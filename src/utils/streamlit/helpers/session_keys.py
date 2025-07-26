from typing import Any, Callable, ClassVar, Generic, Optional, Type, TypeVar, cast, get_args, get_origin

import streamlit as st
from streamlit_pydantic.ui_renderer import GroupOptionalFieldsStrategy, InputUI

TSessionKey = TypeVar("TSessionKey")


class SessionKey(Generic[TSessionKey]):
    """A strongly typed wrapper around streamlit session state values."""

    def __init__(
        self,
        key: str,
        default_value: TSessionKey | None = None,
        allow_none: Optional[bool] = None,
        original_name: str | None = None,
    ):
        self._key = key
        self.default_value = default_value
        self._ever_changed = False
        self._allow_none = default_value is None if allow_none is None else allow_none
        self._original_name = original_name if original_name is not None else key

    def exists(self) -> bool:
        return self.key in st.session_state

    def delete(self):
        if self.exists():
            del st.session_state[self.key]

    def _update(self, value: TSessionKey | None):
        self._ever_changed = True
        if self._allow_none or value is not None:
            st.session_state[self.key] = value
        else:
            self.delete()

    def init(self, value: TSessionKey):
        if not self.exists() or (not self._allow_none and self.value is None):
            st.session_state[self.key] = value

    def init_default(self):
        if not self.exists() or (not self._allow_none and self.value is None):
            self.reset_value()

    @property
    def key(self) -> str:
        return self._key

    @property
    def _key_need_external_update(self) -> "SessionKey[bool]":
        sk = SessionKey(f"{self.key}_need_external_update")
        sk.init(False)
        return sk

    @property
    def _key_next_external_update_value(self) -> "SessionKey[TSessionKey | None]":
        sk = SessionKey(f"{self.key}_next_external_update_value")
        sk.init(None)
        return sk

    @property
    def _key_for_prev_value(self) -> "SessionKey[TSessionKey | None]":
        sk = SessionKey(f"{self.key}_prev_value")
        if self._allow_none:
            sk.init(None)
        return sk

    @property
    def is_changed(self) -> bool:
        """
        You need to check this value *before* the call for the component.
        """
        if self.is_erroneous:
            return self._key_for_prev_value.exists()
        return self._key_for_prev_value.value != self.value

    def restore_prev_value(self):
        self._update(self._key_for_prev_value.value)

    @property
    def prev_value(self) -> TSessionKey | None:
        return self._key_for_prev_value.value

    @property
    def key_for_component(self) -> str:
        """
        This is a workaround to allow external updates to the value.
        And allow is_changed to work.
        Use this as the key for components that need this functionality.
        """
        if self._key_need_external_update.value:
            self._update(self._key_next_external_update_value.value)
            self._key_need_external_update.value = False
            self._key_next_external_update_value.value = None

        self._key_for_prev_value.value = self.value
        return self.key

    def post_external_update(self, value: TSessionKey | None, with_rerun: bool = True):
        """
        This is a workaround to allow external updates to the value after the component has been rendered.
        Use this method only for post render updates, else use update.
        """
        self._key_next_external_update_value.value = value
        self._key_need_external_update.value = True
        if with_rerun:
            st.rerun()

    @property
    def is_erroneous(self) -> bool:
        """
        Consider the key erroneous if it doesn't exist and is not allowed to be none.
        """
        return not self.exists() and not self._allow_none

    @property
    def value(self) -> TSessionKey:
        """Get the current value. Raises KeyError if erroneous."""
        if self.is_erroneous:
            if st.button("Reset Value"):
                self.reset_value()
            raise KeyError(f"Session key '{self.key}' not initialized and has no default value")

        return cast(TSessionKey, st.session_state[self.key] if self.exists() else self.default_value)

    @value.setter
    def value(self, new_value: TSessionKey):
        """Set the current value."""
        self._update(new_value)

    def __str__(self) -> str:
        """Return the current value as string, useful for streamlit widgets."""
        return str(self.value)

    @property
    def ever_changed(self) -> bool:
        """Whether the value has ever been changed from its default."""
        return self._ever_changed

    def equal_if_exists(self, func: Callable[[TSessionKey], bool]) -> bool:
        if self.exists():
            return func(self.value)
        return False

    def exists_and_not_none(self) -> bool:
        return self.equal_if_exists(lambda val: val is not None)

    def update_button(self, value: TSessionKey, label: str):
        st.button(label=label, key=label, on_click=lambda: self._update(value))

    def reset_value(self):
        self._update(self.default_value)

    def post_external_reset_value(self, with_rerun: bool = True):
        self.post_external_update(self.default_value, with_rerun=with_rerun)

    def create_input_widget(
        self,
        label: str | None = None,
        streamlit_container: Any = st,
        group_optional_fields: GroupOptionalFieldsStrategy = GroupOptionalFieldsStrategy.NO,
        lowercase_labels: bool = False,
        ignore_empty_values: bool = False,
    ) -> None:
        """Create an input widget for this session key using streamlit_pydantic's UI renderer.

        Args:
            label: Label for the input widget
            streamlit_container: Streamlit container to render in (default: st)
            group_optional_fields: How to group optional fields (default: NO)
            lowercase_labels: Whether to lowercase labels (default: False)
            ignore_empty_values: Whether to ignore empty values (default: False)
        """
        # Create a minimal Pydantic model for this single value
        if label is None:
            label = self._original_name
        from pydantic import BaseModel, Field, create_model

        # Get the actual type of the value by inspecting the generic parameters
        value_type = type(self.value) if self.value is not None else Any
        if hasattr(value_type, "__origin__"):  # Handle generic types like List, Dict etc
            origin = get_origin(value_type)
            args = get_args(value_type)
            if origin is not None and args:
                value_type = origin[args]

        # Create model dynamically to preserve type information
        SingleValueModel = create_model(
            "SingleValueModel", value=(value_type, Field(title=label, default=self.value)), __base__=BaseModel
        )

        # Use InputUI to render the widget
        input_ui = InputUI(
            key=self.key,
            model=SingleValueModel,
            streamlit_container=streamlit_container,
            group_optional_fields=group_optional_fields,
            lowercase_labels=lowercase_labels,
            ignore_empty_values=ignore_empty_values,
        )

        # Render and update value
        result = input_ui.render_ui()
        if result and "value" in result:
            self.value = result["value"]


is_global_refresh_marker_sk = SessionKey[True]("_global_refresh_marker")


def mark_finished_global_refresh():
    """
    WORKAROUND:
    Mark the current page as being refreshed.
    Need to be called after a successful run in order to allow the dependent features to work correctly.
    Therefore only nice to have features should be dependent on this value
    """
    is_global_refresh_marker_sk.value = True


def is_in_global_refresh() -> bool:
    """Check if the current page is being refreshed."""
    return not is_global_refresh_marker_sk.exists()


class SessionKeyDescriptor(Generic[TSessionKey]):
    """A descriptor that creates SessionKey instances with automatic prefixing."""

    def __init__(self, default_value: TSessionKey | None = None, allow_none: Optional[bool] = None):
        self.default_value = default_value
        self.key: str | None = None
        self._original_name: str | None = None
        self.allow_none = allow_none

    def __set_name__(self, owner: Any, name: str):
        # Add prefix based on class name
        prefix = owner.__name__.lower().strip("_")
        self.key = f"{prefix}_{name}"
        self._original_name = name

    @property
    def instance_key(self) -> str:
        return f"_{self.key}_instance"

    def __get__(self, obj: Any, objtype: Any = None) -> SessionKey[TSessionKey]:
        if obj is None:
            raise ValueError("SessionKeyDescriptor must be used as a class attribute")

        # Create or get SessionKey instance
        if not hasattr(obj, self.instance_key):
            assert self.key is not None, "SessionKeyDescriptor not properly initialized with __set_name__"
            session_key = SessionKey(self.key, self.default_value, self.allow_none, self._original_name)
            if self.allow_none or self.default_value is not None:
                session_key.init(cast(TSessionKey, self.default_value))
            setattr(obj, self.instance_key, session_key)
        else:
            session_key = cast(SessionKey[TSessionKey], getattr(obj, self.instance_key))
            if session_key.is_erroneous:
                if session_key.is_changed:
                    session_key.restore_prev_value()
                else:
                    if not is_in_global_refresh():
                        print(f"resetting value {session_key.key} -> {session_key.default_value}")
                    # if global refresh is in progress, meaning that we need to reset the values,
                    # else, it will raise an error later
                    session_key.reset_value()
        return session_key


_T_SESSION_KEYS_BASE = TypeVar("_T_SESSION_KEYS_BASE", bound="SessionKeysBase[Any]")


class SessionKeysBase(Generic[_T_SESSION_KEYS_BASE]):
    """Base class for session key containers that ensures singleton pattern."""

    _instance: ClassVar[dict[Type[Any], Any]] = {}

    def __new__(cls) -> _T_SESSION_KEYS_BASE:
        if cls not in cls._instance:
            cls._instance[cls] = super().__new__(cls)
        return cast(_T_SESSION_KEYS_BASE, cls._instance[cls])
