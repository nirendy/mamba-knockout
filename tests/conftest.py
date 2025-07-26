"""Pytest configuration and shared fixtures for SSM Analysis tests.

This module provides project-wide test configuration and fixtures that can be
used across all test modules without explicit imports.
"""

from pathlib import Path

import pytest

# Make test utilities available project-wide
pytest_plugins = [
    "tests.src.experiments.runners.test_utils",
]


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Get the test data directory."""
    return project_root / "tests" / "data"


# Add any other project-wide fixtures here
