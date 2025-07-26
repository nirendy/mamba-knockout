"""Test runner for Heatmap experiment comparing against baseline."""

import json

import pytest

from src.analysis.experiment_results.helpers import serialize_result_bank
from src.analysis.experiment_results.results_bank import get_experiment_results_bank
from src.core.types import MODEL_ARCH
from src.experiments.runners.heatmap import HeatmapRunner

from ..baseline_builder import get_test_full_pipeline_config_per_model_arch
from .test_utils import (
    TEST_TOLERANCE,
    compare_results_with_tolerance,
    get_baseline_model_configs,
    test_environment_context,
)


class TestHeatmapRunner:
    """Tests for Heatmap runner comparing against baseline."""

    def setup_method(self) -> None:
        """Set up for each test method."""
        # Setup is handled by the test_environment_context manager
        pass

    def teardown_method(self) -> None:
        """Clean up after each test method."""
        # Cleanup is handled by the test_environment_context manager
        pass

    @pytest.mark.parametrize("model_arch,model_size", get_baseline_model_configs())
    def test_heatmap(self, model_arch: MODEL_ARCH, model_size: str) -> None:
        """Test heatmap generation for specific model against baseline."""
        with test_environment_context(model_arch, model_size, normalizing_outputs=True) as path_manager:
            # Get configuration for the specific model
            config = get_test_full_pipeline_config_per_model_arch(
                code_version_name="test_baseline",
                model_arch=model_arch,
                model_size=model_size,
                with_plotting=False,  # Disable plotting for faster tests
            )

            # Extract the Heatmap runner from the full pipeline config
            heatmap_runner = self._extract_heatmap_runner_from_config(config)
            assert heatmap_runner is not None, f"Heatmap runner not found for {model_arch} {model_size}"

            # Run Heatmap with compute_dependencies to ensure all required data is available
            heatmap_runner.compute_dependencies(rec_depth=-1)
            heatmap_runner.run(with_dependencies=True)

            # Serialize current results for comparison
            current_results = serialize_result_bank(get_experiment_results_bank())
            current_results_dict = json.loads(current_results)

            # Load baseline results from original baseline location (unpatched path)
            baseline_results_dict = path_manager.load_baseline_results()

            # Filter results to only compare Heatmap results for this specific model
            current_heatmap_results = self._extract_heatmap_results(current_results_dict, model_arch, model_size)
            baseline_heatmap_results = self._extract_heatmap_results(baseline_results_dict, model_arch, model_size)

            # Compare test results against baseline using tolerance for numerical values
            if baseline_heatmap_results:
                # Extract the actual result data (ignore the keys which may differ)
                current_result_data = list(current_heatmap_results.values())[0] if current_heatmap_results else None
                baseline_result_data = list(baseline_heatmap_results.values())[0] if baseline_heatmap_results else None

                if current_result_data and baseline_result_data:
                    compare_results_with_tolerance(current_result_data, baseline_result_data, tolerance=TEST_TOLERANCE)
                else:
                    assert False, f"Missing result data for {model_arch} {model_size}"
            else:
                # If no baseline exists, this is the first run - just ensure results exist
                assert current_heatmap_results, f"No results generated for {model_arch} {model_size}"

    def _extract_heatmap_results(self, results_data, model_arch: MODEL_ARCH, model_size: str) -> dict:
        """Extract Heatmap results for specific model from serialized results."""
        heatmap_results = {}

        if not results_data:
            return heatmap_results

        # Handle case where results_data is a list of results
        if isinstance(results_data, list):
            # Check if it's a single result (list with experiment data)
            if results_data and isinstance(results_data[0], str):
                results_list = [results_data]
            else:
                # It's a list of results
                results_list = results_data
        # Handle case where results_data is a dict with multiple results
        elif isinstance(results_data, dict):
            results_list = list(results_data.values())
        else:
            return heatmap_results

        # Look for Heatmap experiment results
        for i, result_item in enumerate(results_list):
            if isinstance(result_item, list) and len(result_item) >= 2:
                experiment_name = result_item[0]
                variant_params = result_item[1] if len(result_item) > 1 else {}

                # Check if this is a Heatmap experiment for our model
                if (
                    experiment_name == "heatmap"
                    and variant_params.get("model_arch") == model_arch.value
                    and variant_params.get("model_size") == model_size
                ):
                    heatmap_results[f"result_{i}"] = result_item

        return heatmap_results

    def _extract_heatmap_runner_from_config(self, config) -> HeatmapRunner:
        """Extract Heatmap runner from FullPipeline dependencies."""
        dependencies = config.get_runner_dependencies()

        # Check heatmap dependencies directly
        if "heatmap" in dependencies:
            return dependencies["heatmap"]

        return None  # type: ignore[return-value]

    def test_heatmap_output_structure(self) -> None:
        """Test that Heatmap produces expected output structure."""
        # Test with first model configuration
        model_arch, model_size = get_baseline_model_configs()[0]

        with test_environment_context(model_arch, model_size, normalizing_outputs=True):
            # Get configuration for the specific model
            config = get_test_full_pipeline_config_per_model_arch(
                code_version_name="test_baseline",
                model_arch=model_arch,
                model_size=model_size,
                with_plotting=False,
            )

            # Extract the Heatmap runner
            heatmap_runner = self._extract_heatmap_runner_from_config(config)
            assert heatmap_runner is not None

            # Run the experiment
            heatmap_runner.compute_dependencies(rec_depth=-1)
            heatmap_runner.run(with_dependencies=True)

            # Check that output file exists and has expected structure
            assert heatmap_runner.output_hdf5_path.path.exists()

            # Check that we can load the outputs
            outputs = heatmap_runner.get_outputs()
            assert outputs is not None
            assert len(outputs) > 0  # Should have some results

            # Check that the outputs are in the expected format (dict of prompt_idx -> heatmap)
            for prompt_idx, heatmap in outputs.items():
                assert isinstance(prompt_idx, int)  # Should be TPromptOriginalIndex
                assert heatmap is not None  # Should have heatmap data

            # Check that the runner reports as computed
            assert heatmap_runner.is_computed()
