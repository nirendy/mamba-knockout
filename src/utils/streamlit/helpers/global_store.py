from collections import defaultdict
from typing import Optional

import streamlit as st

from src.utils.streamlit.helpers.cache import CachedFunction


class StreamlitUtilsGlobalStore:
    def __init__(self):
        self._cache_dependencies: dict[str, set[str]] = defaultdict(set)
        self._instances: dict[str, CachedFunction] = {}

    def add_dependency(self, caller_name: str, callee_name: str):
        """Add a dependency where caller depends on callee."""
        if caller_name not in self._instances or callee_name not in self._instances:
            self.rebuild_instances()
        assert caller_name in self._instances
        assert callee_name in self._instances
        self._cache_dependencies[caller_name].add(callee_name)

    def get_upstream_deps(self, func_name: str, visited: set[str] | None = None) -> dict:
        """Get all functions that this function depends on (recursively)."""
        if visited is None:
            visited = set()

        if func_name in visited:
            return {}

        visited.add(func_name)
        return {dep_name: self.get_upstream_deps(dep_name, visited) for dep_name in self._cache_dependencies[func_name]}

    def get_downstream_deps(self, func_name: str, visited: set[str] | None = None) -> set[str]:
        """Get all functions that depend on this function (recursively)."""
        if visited is None:
            visited = set()

        if func_name in visited:
            return set()

        visited.add(func_name)
        deps = set()
        for caller, callees in self._cache_dependencies.items():
            if func_name in callees:
                deps.add(caller)
                deps.update(self.get_downstream_deps(caller, visited))
        return deps

    def add_instance(self, func_name: str, instance: CachedFunction):
        self._instances[func_name] = instance

    def reset_instance_deps(self, func_name: str):
        self._cache_dependencies[func_name] = set()

    def get_instance(self, func_name: Optional[str] = None) -> Optional[CachedFunction]:
        """Get instance by function name, falling back to module search if needed."""
        if func_name is None:
            return None
        return self._instances.get(func_name)

    def rebuild_instances(self):
        import inspect
        import sys

        for module in list(sys.modules.values()):
            if module is None:
                continue
            try:
                for _, obj in inspect.getmembers(module):
                    if isinstance(obj, CachedFunction):
                        # Update the instances map for future use
                        self._instances[obj.func_name] = obj
            except Exception:
                pass

    def clear_all_instances(self):
        """Clears all cached functions in the system."""
        # Clear all instances we know about
        for instance in self._instances.values():
            instance.clear()
        self._instances.clear()


@st.cache_resource(show_spinner=False)
def _get_global_store() -> StreamlitUtilsGlobalStore:
    """Get or create the cache store for dependencies and instances."""
    return StreamlitUtilsGlobalStore()
