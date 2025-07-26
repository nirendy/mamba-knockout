"""Shared test constants and configuration.

This module centralizes test constants that are used across multiple test modules,
following the same pattern as the main project's core constants.
"""

from pathlib import Path
from typing import Final

# Test execution constants
TEST_TIMEOUT: Final[int] = 300  # 5 minutes
TEST_TOLERANCE: Final[float] = 1e-4

# Test data constants
TEST_BATCH_SIZE: Final[int] = 2
TEST_MAX_SAMPLES: Final[int] = 10

# Path constants
TESTS_ROOT: Final[Path] = Path(__file__).parent.parent
TEST_DATA_DIR: Final[Path] = TESTS_ROOT / "data"
TEST_FIXTURES_DIR: Final[Path] = TESTS_ROOT / "fixtures"

# Monkey patch paths (centralized for consistency)
PATHS_PROJECT_DIR_PATH: Final[str] = "src.core.consts.PATHS.PROJECT_DIR"
INFO_FLOW_PRINT_INTERVAL_PATH: Final[str] = "src.experiments.runners.info_flow.PRINT_INTERVAL"
GET_COMMIT_HASH_PATH: Final[str] = "src.experiments.infrastructure.base_runner.get_git_commit_hash"
CREATE_RUN_ID_PATH: Final[str] = "src.experiments.infrastructure.base_runner.create_run_id"
