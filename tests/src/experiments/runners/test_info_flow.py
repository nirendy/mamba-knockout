"""Test runner for InfoFlow experiment comparing against baseline."""

import json

import pytest

from src.analysis.experiment_results.helpers import serialize_result_bank
from src.analysis.experiment_results.results_bank import get_experiment_results_bank
from src.core.types import MODEL_ARCH
from src.experiments.runners.info_flow import InfoFlowRunner

from ..baseline_builder import get_test_full_pipeline_config_per_model_arch
from .test_utils import (
    TEST_TOLERANCE,
    compare_results_with_tolerance,
    get_baseline_model_configs,
    test_environment_context,
)


class TestInfoFlowRunner:
    """Tests for InfoFlow runner comparing against baseline."""

    def setup_method(self) -> None:
        """Set up for each test method."""
        # Setup is handled by the test_environment_context manager
        pass

    def teardown_method(self) -> None:
        """Clean up after each test method."""
        # Cleanup is handled by the test_environment_context manager
        pass

    @pytest.mark.parametrize("model_arch,model_size", get_baseline_model_configs())
    def test_info_flow(self, model_arch: MODEL_ARCH, model_size: str) -> None:
        """Test info flow analysis for specific model against baseline."""
        with test_environment_context(model_arch, model_size, normalizing_outputs=True) as path_manager:
            # Get configuration for the specific model
            config = get_test_full_pipeline_config_per_model_arch(
                code_version_name="test_baseline",
                model_arch=model_arch,
                model_size=model_size,
                with_plotting=False,  # Disable plotting for faster tests
            )

            # Extract InfoFlow runners from the full pipeline config
            info_flow_runners = self._extract_info_flow_runners_from_config(config)
            assert info_flow_runners, f"InfoFlow runners not found for {model_arch} {model_size}"

            # Run InfoFlow experiments with compute_dependencies to ensure all required data is available
            for info_flow_runner in info_flow_runners:
                info_flow_runner.compute_dependencies(rec_depth=-1)
                info_flow_runner.run(with_dependencies=True)

            # Serialize current results for comparison
            current_results = serialize_result_bank(get_experiment_results_bank())
            current_results_dict = json.loads(current_results)

            # Load baseline results from original baseline location (unpatched path)
            baseline_results_dict = path_manager.load_baseline_results()

            # Filter results to only compare InfoFlow results for this specific model
            current_info_flow_results = self._extract_info_flow_results(current_results_dict, model_arch, model_size)
            baseline_info_flow_results = self._extract_info_flow_results(baseline_results_dict, model_arch, model_size)

            # Compare test results against baseline using tolerance for numerical values
            if baseline_info_flow_results:
                # Compare each InfoFlow result
                for key in current_info_flow_results:
                    if key in baseline_info_flow_results:
                        compare_results_with_tolerance(
                            current_info_flow_results[key], baseline_info_flow_results[key], tolerance=TEST_TOLERANCE
                        )
                    else:
                        # New result that wasn't in baseline - just ensure it exists
                        assert current_info_flow_results[key], f"Empty result for {key}"
            else:
                # If no baseline exists, this is the first run - just ensure results exist
                assert current_info_flow_results, f"No results generated for {model_arch} {model_size}"

    def _extract_info_flow_results(self, results_data, model_arch: MODEL_ARCH, model_size: str) -> dict:
        """Extract InfoFlow results for specific model from serialized results."""
        info_flow_results = {}

        if not results_data:
            return info_flow_results

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
            return info_flow_results

        # Look for InfoFlow experiment results
        for i, result_item in enumerate(results_list):
            if isinstance(result_item, list) and len(result_item) >= 2:
                experiment_name = result_item[0]
                variant_params = result_item[1] if len(result_item) > 1 else {}

                # Check if this is an InfoFlow experiment for our model
                if (
                    experiment_name == "info_flow"
                    and variant_params.get("model_arch") == model_arch.value
                    and variant_params.get("model_size") == model_size
                ):
                    # Create a unique key for this InfoFlow configuration
                    source = variant_params.get("source", "unknown")
                    target = variant_params.get("target", "unknown")
                    feature_category = variant_params.get("feature_category", "unknown")
                    key = f"info_flow_{source}_{target}_{feature_category}_{i}"
                    info_flow_results[key] = result_item

        return info_flow_results

    def _extract_info_flow_runners_from_config(self, config) -> list[InfoFlowRunner]:
        """Extract InfoFlow runners from FullPipeline dependencies."""
        info_flow_runners = []
        dependencies = config.get_runner_dependencies()

        # Check info_flow dependencies
        if "info_flow" in dependencies:
            for target_token, source_flows in dependencies["info_flow"].items():
                for source_flow in source_flows.values():
                    info_flow_runners.append(source_flow)

        return info_flow_runners

    def test_info_flow_output_structure(self) -> None:
        """Test that InfoFlow produces expected output structure."""
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

            # Extract InfoFlow runners
            info_flow_runners = self._extract_info_flow_runners_from_config(config)
            assert info_flow_runners, "No InfoFlow runners found"

            # Test the first InfoFlow runner
            info_flow_runner = info_flow_runners[0]

            # Run the experiment
            info_flow_runner.compute_dependencies(rec_depth=-1)
            info_flow_runner.run(with_dependencies=True)

            # Check that output file exists and has expected structure
            assert info_flow_runner.output_file.path.exists()

            # Check that we can load the outputs
            outputs = info_flow_runner.get_outputs()
            assert outputs is not None
            assert len(outputs) > 0  # Should have some results

            # Check that the outputs are in the expected format
            for prompt_idx, metric_data in outputs.items():
                assert isinstance(prompt_idx, int)  # Should be TPromptOriginalIndex
                assert isinstance(metric_data, dict)  # Should be dict of metric -> values

                # Check that we have the expected metrics
                expected_metrics = {"hit", "true_probs", "diffs"}
                assert expected_metrics.issubset(set(metric_data.keys()))

                # Check that each metric has list values
                for metric_name, values in metric_data.items():
                    if metric_name in expected_metrics:
                        assert isinstance(values, list)  # Should be list of values per layer
                        assert len(values) > 0  # Should have some layer values

            # Check that the runner reports as computed
            assert info_flow_runner.is_computed()
