import contextvars
import datetime as dt
from typing import Callable, Generic, Optional, ParamSpec, Union

import humanize
import streamlit as st
import streamlit_antd_components as sac

from src.utils.streamlit.helpers.component import OutputType
from src.utils.streamlit.helpers.session_keys import SessionKey

P = ParamSpec("P")


class CachedFunction(Generic[P, OutputType]):
    """A strongly typed wrapper for a cached function with recursive clearing and UI rendering."""

    def global_store(self):
        from src.utils.streamlit.helpers.global_store import _get_global_store

        return _get_global_store()

    def __init__(self, func: Callable[P, OutputType], cached_func: Callable[P, OutputType], is_disabled: bool = False):
        self.func = func
        self.cached_func = cached_func
        self.func_name = func.__name__
        self.execution_time: dt.timedelta | None = None
        self.is_failed = False
        self.is_disabled = is_disabled
        # Register this instance
        self.global_store().add_instance(self.func_name, self)

    @property
    def selection_sk(self) -> SessionKey[Optional[str]]:
        return SessionKey(f"{self.func_name}_selected", default_value=None, allow_none=True)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> OutputType:
        """Call the cached function and track dependencies."""
        caller_instance = _current_function.get()
        _current_function.set(self)  # Mark this function as active

        start_time = dt.datetime.now()
        with st.spinner(f"Running {self.func.__name__}...", show_time=True):
            func = self.func if self.is_disabled else self.cached_func
            result = func(*args, **kwargs)
        end_time = dt.datetime.now()
        execution_time = end_time - start_time

        if self.execution_time is None:
            self.execution_time = execution_time

        _current_function.set(caller_instance)  # Restore the previous caller

        # Register dependency if called within another cached function
        if caller_instance:
            self.global_store().add_dependency(caller_instance.func_name, self.func_name)

        return result

    def clear(self):
        """Clears this function's cache and all upstream dependencies recursively."""
        store = self.global_store()
        # Clear all downstream dependencies first
        for dep_name in store.get_downstream_deps(self.func_name):
            dep_instance = store.get_instance(dep_name)
            if dep_instance:
                dep_instance.clear()

        # Clear this function's cache
        self.cached_func.clear()  # type: ignore
        store.reset_instance_deps(self.func_name)
        self.execution_time = None

    @property
    def execution_time_str(self) -> str:
        """Get the execution time as a human-readable string."""
        if self.is_failed:
            return "Failed"
        if self.is_disabled:
            return "Disabled"
        if self.execution_time is None:
            return "Never run"
        return humanize.precisedelta(
            self.execution_time,
            minimum_unit="milliseconds",
            suppress=["milliseconds"],
            format="%0.2f",
        )

    def render(self):
        """Renders Streamlit buttons for clearing caches in the dependency chain."""
        store = self.global_store()

        # Show upstream dependencies (functions that this one depends on)
        upstream_deps = store.get_upstream_deps(self.func_name)

        def recursively_build_items(deps: dict) -> list[Union[str, dict, sac.TreeItem]]:
            return [
                sac.TreeItem(
                    label=dep_name,
                    children=recursively_build_items(dep_upstream_deps),
                    icon="arrow-clockwise",
                    tag=(instance.execution_time_str if (instance := store.get_instance(dep_name)) else "Never run"),
                )
                for dep_name, dep_upstream_deps in deps.items()
            ]

        cols = st.columns([5, 2])
        with cols[1]:
            if instance := store.get_instance(self.selection_sk.value):
                if st.button(f"Clear Cache for {self.selection_sk.value}"):
                    instance.clear()
                    st.rerun()
        with cols[0]:
            sac.tree(
                items=recursively_build_items({self.func_name: upstream_deps}),
                label="Clear Dependencies",
                size="lg",
                open_all=True,
                checkbox_strict=True,
                on_change=lambda: print("hi"),
                key=self.selection_sk.key_for_component,
            )

    def call_and_render(self, *args: P.args, **kwargs: P.kwargs) -> OutputType:
        """
        This function is used to call the cached function and render the dependencies.
        It also make sure that even if the function raise, the dependencies will be rendered.
        """
        try:
            self.is_failed = False
            return self(*args, **kwargs)
        except Exception as e:
            self.is_failed = True
            raise e
        finally:
            self.render()


class CacheWithDependencies:
    """Class decorator wrapping @st.cache_data with strong typing, dependency tracking, and UI rendering."""

    def __init__(self, *st_args, disable_cache: bool = False, is_resource: bool = False, **st_kwargs):
        self.st_args = st_args
        self.st_kwargs = st_kwargs
        self.disable_cache = disable_cache
        self.is_resource = is_resource

    def __call__(self, func: Callable[P, OutputType]) -> CachedFunction[P, OutputType]:
        if self.is_resource:
            cached_func = st.cache_resource(*self.st_args, **self.st_kwargs)(func)
        else:
            cached_func = st.cache_data(*self.st_args, **self.st_kwargs)(func)
        return CachedFunction(func, cached_func, is_disabled=self.disable_cache)


_current_function = contextvars.ContextVar[Optional[CachedFunction]]("current_function", default=None)
