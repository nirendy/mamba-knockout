from pathlib import Path
from typing import Any, Callable, Generic, List, Optional, Sequence, Tuple, TypeVar, Union, assert_never, cast

from src.utils.file_system import fast_relative_to

_ATTRIBUTE_TYPE = TypeVar("_ATTRIBUTE_TYPE")


class OutputKey(Generic[_ATTRIBUTE_TYPE]):
    def __init__(
        self,
        key_name: str,
        convert_to_str: Optional[Callable[[_ATTRIBUTE_TYPE], str]] = None,
        key_display_name: Optional[str] = None,
        skip_condition: Optional[Callable[[_ATTRIBUTE_TYPE], bool]] = None,
        suffix: str = "",
    ):
        """

        Args:
            key_name: key name in the object
            convert_to_str: function to convert the value to a string. Defaults to str.
            key_display_name: display name of the key. Defaults to None.
            skip_condition: condition to skip the key. Defaults to no skipping.
        """
        self.key_name = key_name
        self.convert_to_str = convert_to_str
        self.key_display_name = f"{key_name}=" if key_display_name is None else key_display_name
        self.skip_condition = skip_condition
        self.suffix = suffix

    def should_skip(self, obj: object) -> bool:
        if self.skip_condition is None:
            return False
        return self.skip_condition(self.get_value(obj))

    def get_value(self, obj: object) -> _ATTRIBUTE_TYPE:
        assert hasattr(obj, self.key_name), f"Object {obj} does not have attribute {self.key_name}"
        return cast(_ATTRIBUTE_TYPE, getattr(obj, self.key_name))

    def display(self, obj: object) -> str:
        value = self.get_value(obj)
        converted_value = self.convert_to_str(value) if self.convert_to_str is not None else str(value)
        return f"{self.key_display_name}{converted_value}{self.suffix}"

    def extract_value_from_str(self, value_str: str) -> str:
        """Extract the original value from a path component string.

        Args:
            value_str: The string value from the path component

        Returns:
            The extracted value without the display name prefix

        Raises:
            NotImplementedError: If the key has custom conversion or skip condition
            ValueError: If the value doesn't match the expected format
        """
        if self.convert_to_str is not None or self.skip_condition is not None:
            raise NotImplementedError(
                f"Cannot extract values for key {self.key_name} with custom conversion or skip condition"
            )

        # Remove the display name prefix if it exists
        if value_str.startswith(self.key_display_name) and value_str.endswith(self.suffix):
            value_str = value_str[len(self.key_display_name) :]
            if len(self.suffix) > 0:
                value_str = value_str[: -len(self.suffix)]
            return value_str
        else:
            raise ValueError(f"Value {value_str} does not start with {self.key_display_name}")

    def __repr__(self) -> str:
        return f"{self.key_name}={self.key_display_name}{self.suffix}"


def combine_output_keys(
    obj: object,
    keys: list[OutputKey[Any]],
    sep: str = "/",
) -> str:
    res = []
    for output_key in keys:
        if not output_key.should_skip(obj):
            res.append(output_key.display(obj))
    return sep.join(res)


IPathComponent = Union[str, OutputKey]


def dict_to_obj(d: dict[str, str]) -> object:
    class Config:
        pass

    for key, value in d.items():
        setattr(Config, key, value)
    config = Config()
    return config


def resolve_path_component(component: IPathComponent, obj: object) -> Union[Path, str]:
    if isinstance(component, Path) or isinstance(component, str):
        return component
    else:
        return combine_output_keys(obj, [component])


class OutputPath:
    def __init__(self, base_path: Path, path_components: Sequence[IPathComponent]):
        self.base_path = base_path
        self.path_components = path_components

    def to_path(self, obj: object) -> Path:
        path = self.base_path
        for component in self.path_components:
            path /= resolve_path_component(component, obj)
        return path

    def add(self, component: Sequence[IPathComponent]) -> "OutputPath":
        return OutputPath(self.base_path, list(self.path_components) + list(component))

    def enforce_value(self, key_name: str, value: str) -> "OutputPath":
        new_components: list[IPathComponent] = []
        for component in self.path_components:
            if isinstance(component, OutputKey) and component.key_name == key_name:
                new_components.append(value)
            else:
                new_components.append(component)
        return OutputPath(self.base_path, new_components)

    def get_key_names(self) -> list[str]:
        """Extract all key names from path components."""
        key_names = []
        for component in self.path_components:
            if isinstance(component, OutputKey):
                key_names.append(component.key_name)
        return key_names

    def extract_values_from_path(self, path: Path, allow_extra_parts: bool = True) -> dict[str, str]:
        """Extract values from a path according to the path structure."""
        relative_path = fast_relative_to(path, self.base_path, allow_slow=False)
        path_parts = list(relative_path.parts)

        values = {}
        current_part_idx = 0

        for component in self.path_components:
            if current_part_idx >= len(path_parts):
                raise ValueError(f"Path {path} does not match the expected structure")

            if isinstance(component, OutputKey):
                values[component.key_name] = component.extract_value_from_str(path_parts[current_part_idx])
                current_part_idx += 1
            elif isinstance(component, str):
                if path_parts[current_part_idx] != component:
                    raise ValueError(f"Path {path}, expected {component}, got {path_parts[current_part_idx]}")
                current_part_idx += 1
            else:
                assert_never(component)

        if not allow_extra_parts and current_part_idx < len(path_parts):
            raise ValueError(
                f"Path {path} has more components than expected: extra parts {path_parts[current_part_idx:]}"
            )

        return values

    def process_path(self) -> Tuple[List[Tuple[Path, dict[str, str]]], List[Tuple[Path, str]]]:
        """Process a directory at the given depth in the path structure.

        Args:
            current_path: The current directory path
            depth: Current depth in the path components
            collected_values: Values collected from parent directories

        Returns:
            List of moves to perform (old_path, new_path, values)
        """

        def rec_process_path(
            current_path: Path, depth: int
        ) -> Tuple[List[Tuple[Path, dict[str, str]]], List[Tuple[Path, str]]]:
            sub_path = OutputPath(self.base_path, self.path_components[:depth])

            try:
                values = sub_path.extract_values_from_path(current_path)
            except ValueError as e:
                return [], [(current_path, str(e))]

            if depth == len(self.path_components):
                return [(current_path, values)], []
            else:
                moves: List[Tuple[Path, dict[str, str]]] = []
                errors: List[Tuple[Path, str]] = []
                if current_path.is_dir():
                    for item in current_path.iterdir():
                        res = rec_process_path(item, depth + 1)
                        moves.extend(res[0])
                        errors.extend(res[1])

            return moves, errors

        return rec_process_path(self.base_path, 0)
