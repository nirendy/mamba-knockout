import sys
from contextlib import contextmanager
from io import StringIO


@contextmanager
def st_redirect(src, dst, placeholder, overwrite):
    output_func = getattr(placeholder.empty(), dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            is_newline = b == "\n"
            if is_newline:
                return

            old_write(b)
            buffer.write(b + "\r\n")

            # Without this condition, will cause infinite loop because we can't write to the streamlit from thread
            # TODO: st.script_run_context not found, fix this
            # if getattr(current_thread(), st.script_run_context.SCRIPT_RUN_CONTEXT_ATTR_NAME, None) is None:
            #     if overwrite:
            #         buffer.truncate(0)
            #         buffer.seek(0)
            #     return

            output_func(buffer.getvalue())

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst, placeholder, overwrite):
    "this will show the prints"
    with st_redirect(sys.stdout, dst, placeholder, overwrite):
        yield


@contextmanager
def st_stderr(dst, placeholder, overwrite):
    "This will show the logging"
    with st_redirect(sys.stderr, dst, placeholder, overwrite):
        yield
