import inspect
import re
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Callable

from git import Optional
from mypy import api

MYPY_ERROR_PATTERN = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+): error: (?P<message>[^[]+)\[(?P<error_code>.+)\]$")


@dataclass
class MypyError:
    file_path: str
    line: int
    message: str
    error_code: str


def run_mypy_check_on_function(
    func: Callable,
    strings_to_remove: Optional[list[str]] = None,
) -> list[MypyError]:
    """Extracts the source code of a function and runs mypy on it."""
    if not strings_to_remove:
        strings_to_remove = [
            "type: ignore",
        ]
    source_lines = inspect.getsource(func)
    dedented_source_lines = textwrap.dedent(source_lines).splitlines()
    for string in strings_to_remove:
        dedented_source_lines = [line.replace(string, "") for line in dedented_source_lines]
    code = "\n".join(dedented_source_lines)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".py") as temp_file:
        temp_file.write(code.encode())
        temp_file.flush()
        temp_file_path = temp_file.name

        result, error, _ = api.run(
            [
                # "--strict",
                # "--pretty=",
                temp_file_path,
            ]
        )
        if error:
            raise ValueError(error)

    errors = []
    for line in result.splitlines():
        match = MYPY_ERROR_PATTERN.match(line)
        if match:
            errors.append(
                MypyError(
                    file_path=match.group("file"),
                    line=int(match.group("line")),
                    message=match.group("message").strip(),
                    error_code=match.group("error_code").strip(),
                )
            )
    return errors
