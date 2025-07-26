from src.utils.tests_utils import run_mypy_check_on_function


def test_mypy_typing_error():
    def problematic_function():
        def add_numbers(x: int, y: int) -> str:
            return x + y  # type: ignore

        pass  # Avoid running the code

    type_check_result = run_mypy_check_on_function(problematic_function)
    print(f"{type_check_result = }")
    assert len(type_check_result) == 1
    assert "Incompatible return value type" in type_check_result[0].message
