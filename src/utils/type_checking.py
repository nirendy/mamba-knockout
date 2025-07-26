import os
from typing import Callable, TypeVar

from beartype import beartype as typechecker
from jaxtyping import jaxtyped

F = TypeVar("F", bound=Callable)


def tensor_type_check(func: F) -> F:
    """
    Wrapper for tensor type checking.
    """
    is_env_var_set = bool(os.getenv("ENABLE_TYPE_CHECKING", False))
    is_pytest_running = "PYTEST_VERSION" in os.environ
    ENABLE_TYPE_CHECKING = is_env_var_set or is_pytest_running

    if not ENABLE_TYPE_CHECKING:
        return func

    return jaxtyped(typechecker=typechecker)(func)  # type:ignore[return-value]
