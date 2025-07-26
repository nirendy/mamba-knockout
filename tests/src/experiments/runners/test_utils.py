"""Shared test utilities for path management and data copying."""

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pytest
from datasets import DatasetDict

from src.core.consts import PathsConfig
from src.core.names import COLS
from src.core.types import (
    ALL_SPLITS_LITERAL,
    MODEL_ARCH,
    SPLIT,
    TPromptOriginalIndex,
)
from src.data_ingestion.datasets.download_dataset import DatasetName, load_splitted_counter_fact

# Import shared test constants
from tests.shared.constants import (
    CREATE_RUN_ID_PATH,
    GET_COMMIT_HASH_PATH,
    INFO_FLOW_PRINT_INTERVAL_PATH,
    PATHS_PROJECT_DIR_PATH,
    TEST_TOLERANCE,
)

# Import test parameters from baseline builder
from ..baseline_builder import ORIGINAL_IDS, TEST_BASE_PATH, TEST_MODEL_CONFIGS


class TestPathManager:
    """Manages paths and data copying for test execution."""

    def __init__(self, temp_dir: Path, baseline_dir: Path):
        self.temp_dir = temp_dir
        self.baseline_dir = baseline_dir
        self.temp_paths = PathsConfig(PROJECT_DIR=temp_dir)
        self.baseline_paths = PathsConfig(PROJECT_DIR=baseline_dir)

    def copy_filtered_dataset_to_temp(self) -> None:
        """Copy baseline test data (filtered dataset) from baseline location to temporary directory."""
        # Create the dataset directory in temp location
        temp_dataset_dir = self.temp_paths.dataset_dir(DatasetName.counter_fact) / "splitted"
        temp_dataset_dir.parent.mkdir(parents=True, exist_ok=True)

        # Check if baseline dataset exists
        baseline_dataset_dir = self.baseline_paths.dataset_dir(DatasetName.counter_fact) / "splitted"

        if baseline_dataset_dir.exists():
            # Copy existing baseline dataset
            shutil.copytree(baseline_dataset_dir, temp_dataset_dir, dirs_exist_ok=True)
        else:
            # Generate filtered dataset if baseline doesn't exist
            self._generate_filtered_dataset_in_temp()

    def _generate_filtered_dataset_in_temp(self) -> None:
        """Generate filtered dataset in temporary directory using ORIGINAL_IDS."""
        # Load full dataset and filter by ORIGINAL_IDS for each split
        dataset = {
            split: load_splitted_counter_fact(
                ALL_SPLITS_LITERAL,
                align_to_known=False,
            ).filter(lambda x: x[COLS.ORIGINAL_IDX] in original_ids)
            for split, original_ids in ORIGINAL_IDS.items()
        }

        # Save filtered dataset to temp directory
        temp_dataset_dir = self.temp_paths.dataset_dir(DatasetName.counter_fact) / "splitted"
        DatasetDict(dataset).save_to_disk(temp_dataset_dir)

    def copy_model_dependency_data(self, model_arch: MODEL_ARCH, model_size: str) -> None:
        """Copy relevant dependency data for specific model from baseline to temp directory."""
        # Copy experiment results for the specific model
        baseline_results_dir = self.baseline_paths.RESULTS_DIR
        temp_results_dir = self.temp_paths.RESULTS_DIR

        if baseline_results_dir.exists():
            # Create temp results directory
            temp_results_dir.mkdir(parents=True, exist_ok=True)

            # Copy all results (they may be interdependent)
            for item in baseline_results_dir.iterdir():
                if item.is_dir():
                    dest_path = temp_results_dir / item.name
                    if not dest_path.exists():
                        shutil.copytree(item, dest_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, temp_results_dir / item.name)

    def load_baseline_results(self) -> dict:
        """Load baseline results from original baseline location (unpatched path)."""
        baseline_results_file = self.baseline_dir / "serialized_results.json"
        if baseline_results_file.exists():
            import json

            return json.loads(baseline_results_file.read_text())
        return {}

    def get_baseline_path(self) -> Path:
        """Get the original baseline path for loading baseline results."""
        return self.baseline_dir

    def get_temp_path(self) -> Path:
        """Get the temporary directory path for test execution."""
        return self.temp_dir


@contextmanager
def test_environment_context(
    model_arch: MODEL_ARCH, model_size: str, normalizing_outputs: bool = True
) -> Generator[TestPathManager, None, None]:
    """Context manager for PROJECT_DIR patching during test execution.

    This context manager:
    1. Creates a temporary directory for test execution
    2. Copies baseline test data to the temporary directory
    3. Patches PROJECT_DIR to point to temp directory during test execution
    4. Provides access to both temp directory for execution and baseline for comparison
    5. Cleans up temporary directory after test completion
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        path_manager = TestPathManager(temp_dir, TEST_BASE_PATH)

        # Copy filtered dataset and model dependencies to temp directory
        path_manager.copy_filtered_dataset_to_temp()
        path_manager.copy_model_dependency_data(model_arch, model_size)

        # Set up monkey patches for test execution
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(PATHS_PROJECT_DIR_PATH, temp_dir)
            mp.setattr(INFO_FLOW_PRINT_INTERVAL_PATH, 1)
            if normalizing_outputs:
                mp.setattr(GET_COMMIT_HASH_PATH, lambda *args, **kwargs: "test_commit_hash")
                mp.setattr(CREATE_RUN_ID_PATH, lambda *args, **kwargs: "test_run_id")

            yield path_manager


def get_baseline_model_configs() -> list[tuple[MODEL_ARCH, str]]:
    """Get model configurations from baseline builder."""
    return TEST_MODEL_CONFIGS


def compare_results_with_tolerance(
    actual_results: dict, baseline_results: dict, tolerance: float = TEST_TOLERANCE
) -> None:
    """Compare results with numerical tolerance for cross-system compatibility."""
    import json

    import numpy as np

    def _compare_values(actual, expected, path=""):
        if isinstance(actual, dict) and isinstance(expected, dict):
            # Compare dictionary keys
            actual_keys = set(actual.keys())
            expected_keys = set(expected.keys())

            if actual_keys != expected_keys:
                missing_keys = expected_keys - actual_keys
                extra_keys = actual_keys - expected_keys
                raise AssertionError(f"Key mismatch at {path}: missing keys: {missing_keys}, extra keys: {extra_keys}")

            # Recursively compare values
            for key in actual_keys:
                _compare_values(actual[key], expected[key], f"{path}.{key}" if path else key)

        elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                raise AssertionError(f"Length mismatch at {path}: actual={len(actual)}, expected={len(expected)}")

            for i, (a, e) in enumerate(zip(actual, expected)):
                _compare_values(a, e, f"{path}[{i}]")

        elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            # Handle NaN values specially
            if np.isnan(actual) and np.isnan(expected):
                return  # Both NaN, consider equal
            elif np.isnan(actual) or np.isnan(expected):
                raise AssertionError(f"NaN mismatch at {path}: actual={actual}, expected={expected}")
            elif not np.isclose(actual, expected, atol=tolerance, rtol=tolerance):
                raise AssertionError(
                    f"Numerical mismatch at {path}: actual={actual}, expected={expected}, "
                    f"difference={abs(actual - expected)}, tolerance={tolerance}"
                )

        elif isinstance(actual, str) and isinstance(expected, str):
            # Try to parse as JSON and compare numerically if possible
            try:
                actual_parsed = json.loads(actual)
                expected_parsed = json.loads(expected)
                _compare_values(actual_parsed, expected_parsed, path)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, compare as strings
                if actual != expected:
                    raise AssertionError(f"String mismatch at {path}: actual={actual}, expected={expected}")

        else:
            if actual != expected:
                raise AssertionError(f"Value mismatch at {path}: actual={actual}, expected={expected}")

    _compare_values(actual_results, baseline_results)


def get_original_ids() -> dict[SPLIT, list[TPromptOriginalIndex]]:
    """Get the original IDs used for filtering test data."""
    return ORIGINAL_IDS


def get_test_base_path() -> Path:
    """Get the test base path where baselines are stored."""
    return TEST_BASE_PATH
