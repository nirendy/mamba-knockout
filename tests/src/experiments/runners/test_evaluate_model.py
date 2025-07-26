"""Test runner for EvaluateModel experiment comparing against baseline."""

import json

import pytest

from src.analysis.experiment_results.helpers import serialize_result_bank
from src.analysis.experiment_results.results_bank import get_experiment_results_bank
from src.core.types import MODEL_ARCH
from src.experiments.runners.evaluate_model import EvaluateModelRunner
from tests.shared.constants import TEST_TOLERANCE

from ..baseline_builder import get_test_full_pipeline_config_per_model_arch
from .test_utils import (
    compare_results_with_tolerance,
    get_baseline_model_configs,
    test_environment_context,
)


class TestEvaluateModelRunner:
    """Tests for EvaluateModel runner comparing against baseline."""

    def setup_method(self) -> None:
        """Set up for each test method."""
        # Setup is handled by the test_environment_context manager
        pass

    def teardown_method(self) -> None:
        """Clean up after each test method."""
        # Cleanup is handled by the test_environment_context manager
        pass

    @pytest.mark.parametrize("model_arch,model_size", get_baseline_model_configs())
    def test_evaluate_model(self, model_arch: MODEL_ARCH, model_size: str) -> None:
        """Test model evaluation for specific model against baseline."""
        with test_environment_context(model_arch, model_size, normalizing_outputs=True) as path_manager:
            # Get configuration for the specific model
            config = get_test_full_pipeline_config_per_model_arch(
                code_version_name="test_baseline",
                model_arch=model_arch,
                model_size=model_size,
                with_plotting=False,  # Disable plotting for faster tests
            )

            # Extract the EvaluateModel runner from the nested dependencies
            evaluate_model_runner = self._extract_evaluate_model_runner_from_config(config)
            assert evaluate_model_runner is not None, f"EvaluateModel runner not found for {model_arch} {model_size}"

            # Run EvaluateModel with compute_dependencies to ensure all required data is available
            evaluate_model_runner.compute_dependencies(rec_depth=-1)
            evaluate_model_runner.run(with_dependencies=True)

            # Serialize current results for comparison
            current_results = serialize_result_bank(get_experiment_results_bank())
            current_results_dict = json.loads(current_results)

            # Load baseline results from original baseline location (unpatched path)
            baseline_results_dict = path_manager.load_baseline_results()

            # Filter results to only compare EvaluateModel results for this specific model
            current_evaluate_results = self._extract_evaluate_model_results(
                current_results_dict, model_arch, model_size
            )
            baseline_evaluate_results = self._extract_evaluate_model_results(
                baseline_results_dict, model_arch, model_size
            )

            # Compare test results against baseline using tolerance for numerical values
            if baseline_evaluate_results:
                # Extract the actual result data (ignore the keys which may differ)
                current_result_data = list(current_evaluate_results.values())[0] if current_evaluate_results else None
                baseline_result_data = (
                    list(baseline_evaluate_results.values())[0] if baseline_evaluate_results else None
                )

                if current_result_data and baseline_result_data:
                    compare_results_with_tolerance(current_result_data, baseline_result_data, tolerance=TEST_TOLERANCE)
                else:
                    assert False, f"Missing result data for {model_arch} {model_size}"
            else:
                # If no baseline exists, this is the first run - just ensure results exist
                assert current_evaluate_results, f"No results generated for {model_arch} {model_size}"

    def _extract_evaluate_model_results(self, results_data, model_arch: MODEL_ARCH, model_size: str) -> dict:
        """Extract EvaluateModel results for specific model from serialized results."""
        evaluate_results = {}

        if not results_data:
            return evaluate_results

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
            return evaluate_results

        # Look for EvaluateModel experiment results
        for i, result_item in enumerate(results_list):
            if isinstance(result_item, list) and len(result_item) >= 2:
                experiment_name = result_item[0]
                variant_params = result_item[1] if len(result_item) > 1 else {}

                # Check if this is an EvaluateModel experiment for our model
                if (
                    experiment_name == "evaluate_model"
                    and variant_params.get("model_arch") == model_arch.value
                    and variant_params.get("model_size") == model_size
                ):
                    evaluate_results[f"result_{i}"] = result_item

        return evaluate_results

    def _extract_evaluate_model_runner_from_config(self, config) -> EvaluateModelRunner:
        """Extract EvaluateModel runner from nested FullPipeline dependencies."""
        dependencies = config.get_runner_dependencies()

        # Check heatmap dependencies
        if "heatmap" in dependencies:
            heatmap_deps = dependencies["heatmap"].get_runner_dependencies()
            if "evaluate_model" in heatmap_deps:
                return heatmap_deps["evaluate_model"]

        # Check info_flow dependencies
        if "info_flow" in dependencies:
            for target_token, source_flows in dependencies["info_flow"].items():
                for source_flow in source_flows.values():
                    info_flow_deps = source_flow.get_runner_dependencies()
                    if "evaluate_model" in info_flow_deps:
                        return info_flow_deps["evaluate_model"]

        return None  # type: ignore[return-value]

    def test_evaluate_model_output_structure(self) -> None:
        """Test that EvaluateModel produces expected output structure."""
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

            # Extract the EvaluateModel runner
            evaluate_model_runner = self._extract_evaluate_model_runner_from_config(config)
            assert evaluate_model_runner is not None

            # Run the experiment
            evaluate_model_runner.compute_dependencies(rec_depth=-1)
            evaluate_model_runner.run(with_dependencies=True)

            # Check that output file exists and has expected structure
            assert evaluate_model_runner.output_result_path.exists()

            # Check that we can load the outputs
            outputs = evaluate_model_runner.get_outputs()
            assert outputs is not None
            assert len(outputs) > 0  # Should have some results

            # Check that prompt data can be loaded
            prompt_data = evaluate_model_runner.get_prompt_data()
            assert prompt_data is not None
