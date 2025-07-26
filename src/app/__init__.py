import logging

import streamlit as st

st.set_option("logger.enableRich", True)

# Get the logger
logger = logging.getLogger("streamlit.watcher.local_sources_watcher")


# Create a custom filter to ignore specific messages
class IgnoreSpecificMessageFilter(logging.Filter):
    def __init__(self):
        super().__init__("IgnoreSpecificMessageFilter")

    def filter(self, record):
        if "torch.classes" in record.getMessage():
            assert record.exc_info is not None
            assert record.exc_info[1] is not None
            assert record.exc_info[1].args is not None
            assert len(record.exc_info[1].args) > 0
            if (
                "Tried to instantiate class '__path__._path', but it does not exist!"
                " Ensure that it is registered via torch::class_" in record.exc_info[1].args[0]
            ):
                return False
        return True


# Add the filter to the logger
if logger.filters is not None and not any(
    getattr(f, "name", None) == "IgnoreSpecificMessageFilter" for f in logger.filters
):
    logger.addFilter(IgnoreSpecificMessageFilter())
