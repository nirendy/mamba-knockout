"""
Uses a context manager to temporarily allow nested layouts without permanently modifying Streamlit's code.
"""

from contextlib import contextmanager
from functools import wraps
from typing import Callable


@contextmanager
def allow_nested_st_elements():
    """
    Context manager that temporarily allows nesting of Streamlit elements that are
    normally restricted (like columns inside columns or nested expanders).

    Example:
        ```python
        import streamlit as st
        from src.utils.streamlit.helpers.nested import allow_nested_st_elements

        with allow_nested_st_elements():
            with st.expander("Outer expander"):
                with st.expander("Inner expander"):  # Would normally raise an error
                    st.write("This works now!")

                col1, col2 = st.columns(2)
                with col1:
                    subcol1, subcol2 = st.columns(2)  # Would normally be limited to one level
                    with subcol1:
                        st.write("Deeply nested column content")
        ```
    """

    # Define a replacement function that does nothing
    def noop_check_nested_element_violation(*args, **kwargs):
        # This function does nothing, allowing any nesting
        pass

    # Store the reference to original function
    import streamlit.delta_generator

    original_function = streamlit.delta_generator._check_nested_element_violation

    try:
        # Replace the function with our no-op version
        streamlit.delta_generator._check_nested_element_violation = noop_check_nested_element_violation

        # Execute the code within the context manager
        yield
    finally:
        # Restore the original function when leaving the context
        streamlit.delta_generator._check_nested_element_violation = original_function


def decorator_allow_nested_st_elements(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with allow_nested_st_elements():
            return func(*args, **kwargs)

    return wrapper
