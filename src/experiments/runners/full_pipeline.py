"""
FullPipelineExperiment: Orchestrates all experiments in sequence

This experiment runs:
1. Data Construction - Creates the dataset
2. Model Evaluation - Evaluates model performance
3. Heatmap Analysis - Analyzes layer effects
4. Information Flow Analysis - Analyzes semantic information flow

The experiment ensures proper data flow between experiments and maintains
consistent configuration across all steps.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, TypedDict

import matplotlib.pyplot as plt

from src.analysis.plots.info_flow_confidence import (
    InfoFlowPlotConfig,
    TMetricType,
    create_confidence_plot,
)
from src.core.consts import TOKEN_TYPE_COLORS, TOKEN_TYPE_LINE_STYLES
from src.core.names import ExperimentName, InfoFlowVariantParam
from src.core.types import FeatureCategory, TokenType, TWindowSize
from src.experiments.infrastructure.base_prompt_filteration import BasePromptFilteration
from src.experiments.infrastructure.base_runner import BaseRunner, BaseVariantParams, InputParams
from src.experiments.runners.heatmap import HEATMAP_PLOT_FUNCS, HeatmapParams, HeatmapRunner
from src.experiments.runners.info_flow import InfoFlowParams, InfoFlowRunner
from src.utils.types_utils import first_dict_value


class FullPipelineDependencies(TypedDict):
    heatmap: HeatmapRunner
    info_flow: dict[TokenType, dict[tuple[TokenType, FeatureCategory], InfoFlowRunner]]


@dataclass(frozen=True)
class FullPipelineParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = field(init=False, default=ExperimentName.full_pipeline)
    knockout_map: dict[TokenType, list[tuple[TokenType, FeatureCategory]]]
    info_flow_window_size: TWindowSize

    heatmap_window_size: TWindowSize
    heatmap_prompts: BasePromptFilteration

    with_plotting: bool = False
    enforce_no_missing_outputs: bool = True
    with_generation: bool = True


@dataclass(frozen=True)
class FullPipelineRunner(BaseRunner):
    """Configuration for the full experiment pipeline."""

    variant_params: FullPipelineParams

    @staticmethod
    def _get_variant_params():
        return FullPipelineParams

    @classmethod
    def get_variant_output_keys(cls):
        return super().get_variant_output_keys()

    def target_plot_path(self, target_token: TokenType, plot_name: str) -> Path:
        info_flow_config = first_dict_value(self.get_runner_dependencies()["info_flow"][target_token])
        path = info_flow_config.variation_paths.plots_path
        print(1, path)
        while not path.name.startswith(InfoFlowVariantParam.target):
            path = path.parent
        print(2, path)
        path = path.parent / info_flow_config.variation_paths.plots_path.name / f"target={target_token}{plot_name}.png"
        print(3, path)
        return path

    def get_outputs(self) -> dict:
        """Get outputs from all experiments."""
        return {}

    def _compute_impl(self) -> None:
        main_local(self)

    def get_runner_dependencies(self) -> FullPipelineDependencies:  # type: ignore
        info_flow_deps: dict[TokenType, dict[tuple[TokenType, FeatureCategory], InfoFlowRunner]] = {}
        for target_token, source in self.variant_params.knockout_map.items():
            info_flow_deps[target_token] = {}
            for source_token, feature_category in source:
                config = InfoFlowRunner.init_from_runner(
                    runner=self,
                    variant_params=InfoFlowParams(
                        model_arch=self.variant_params.model_arch,
                        model_size=self.variant_params.model_size,
                        window_size=self.variant_params.info_flow_window_size,
                        source=source_token,
                        feature_category=feature_category,
                        target=target_token,
                    ),
                )
                if not config.variant_params.should_skip_task():
                    info_flow_deps[target_token][(source_token, feature_category)] = config

        return FullPipelineDependencies(
            heatmap=HeatmapRunner.init_from_runner(
                runner=self,
                variant_params=HeatmapParams(
                    model_arch=self.variant_params.model_arch,
                    model_size=self.variant_params.model_size,
                    window_size=self.variant_params.heatmap_window_size,
                ),
                input_params=InputParams(
                    filteration=self.variant_params.heatmap_prompts,
                ),
            ),
            info_flow=info_flow_deps,
        )

    def is_computed(self) -> bool:
        return not self.variant_params.with_plotting


def main_local(args: FullPipelineRunner):
    """Run the full pipeline of experiments."""
    print("Starting Full Pipeline Experiment")
    print(
        " ".join(
            [
                f"{args.variant_params.with_generation=}",
                f"{args.variant_params.with_plotting=}",
                f"{args.variant_params.enforce_no_missing_outputs=}",
            ]
        )
    )
    print(args)

    if args.variant_params.with_plotting:
        print("\nPlotting all heatmaps...")
        try:
            args.get_runner_dependencies()["heatmap"].plot(HEATMAP_PLOT_FUNCS._simple_diff_fixed_0_3)
        except Exception as e:
            print(f"Error plotting heatmaps: {e}")

    if args.variant_params.with_plotting:
        print("\nPlotting all info flow blocks...")
        for target_token, source_info_flows in args.get_runner_dependencies()["info_flow"].items():
            title = (
                " - ".join(
                    [
                        args.variant_params.model_arch,
                        args.variant_params.model_size,
                        f"window_size={args.variant_params.info_flow_window_size}",
                    ]
                )
                + f"\nKnocking out flow to {target_token}"
            )
            lines_metadata = {}
            colors = []
            linestyles = []
            labels = []
            for source_token, feature_category in source_info_flows.keys():
                info_flow_config = source_info_flows[(source_token, feature_category)]
                lines_metadata[f"{source_token} - {feature_category}"] = info_flow_config.get_outputs()
                colors.append(TOKEN_TYPE_COLORS.get(source_token, "#000000"))
                linestyles.append(TOKEN_TYPE_LINE_STYLES.get(feature_category, "-"))
                labels.append(f"{source_token} - {feature_category}")

            for with_fixed_limits in [True, False]:
                plot_name = "_fixed_limits" if with_fixed_limits else ""
                fig = create_confidence_plot(
                    lines=lines_metadata,
                    confidence_level=0.95,
                    config=InfoFlowPlotConfig(
                        custom_colors=dict(zip(labels, colors)),
                        custom_line_styles=dict(zip(labels, linestyles)),
                        title=title,
                        metrics_to_show=[TMetricType.ACC, TMetricType.DIFF],
                    ),
                )
                path = args.target_plot_path(target_token, plot_name)
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path)
                plt.close(fig)

    print("\nFull Pipeline Experiment Complete!")
