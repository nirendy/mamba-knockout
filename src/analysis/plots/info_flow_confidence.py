from collections import defaultdict
from enum import StrEnum
from typing import Dict, Literal, Self, Type, TypedDict, cast

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator
from numpy.typing import NDArray
from pydantic import BaseModel, Field, create_model
from pydantic_extra_types.color import Color
from scipy import stats

from src.core.consts import CONVERT_TO_PLOTLY_LINE_STYLE, TOKEN_TYPE_COLORS
from src.core.names import COLS
from src.core.types import TInfoFlowOutput, TLineStyle
from src.utils.pydantic_utils import create_literal_value
from src.utils.streamlit.components.extended_streamlit_pydantic import annotate_dict_with_literal_values
from src.utils.streamlit.st_pydantic_v2.input import SpecialFieldKeys


class TMetricType(StrEnum):
    ACC = "acc"
    DIFF = "diff"


class PlotMetadata(BaseModel):
    title: str = Field(
        description="Title of the plot",
    )
    ylabel: str = Field(
        description="Label of the y-axis",
    )
    with_fixed_limits: bool = Field(
        default=True,
        description="Use fixed limits for y-axis",
    )
    axhline_value: float = Field(
        description="Value of the horizontal line",
    )
    ylim_min: float = Field(
        description="Minimum value of the y-axis",
        json_schema_extra={SpecialFieldKeys.column_group: "y_axis_limits"},
    )
    ylim_max: float = Field(
        description="Maximum value of the y-axis",
        json_schema_extra={SpecialFieldKeys.column_group: "y_axis_limits"},
    )

    @property
    def ylim(self) -> tuple[float, float]:
        return self.ylim_min, self.ylim_max


class InfoFlowPlotConfig(BaseModel):
    """Configuration for Info Flow confidence plots."""

    class Config:
        json_encoders = {Color: lambda c: c.as_hex(format="long")}

    # Basic Config
    title: str = Field(
        default="",
        description="Custom plot title (leave empty for default)",
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for intervals",
        ge=0.5,
        le=0.99,
        json_schema_extra={SpecialFieldKeys.column_group: "basic_config"},
    )
    alpha: float = Field(
        default=0.2,
        description="Alpha (transparency) for confidence intervals",
        ge=0.0,
        le=1.0,
        json_schema_extra={SpecialFieldKeys.column_group: "basic_config"},
    )
    metrics_to_show: list[TMetricType] = Field(
        default=[TMetricType.DIFF],
        description="Which metrics to show in the plot",
    )

    # Display Options
    x_axis_as_percentage: bool = Field(
        default=True,
        description="Show X-axis (layer positions) as percentages instead of indices",
    )
    x_tick_count: int = Field(
        default=6,
        description="Number of tick marks on the x-axis",
        json_schema_extra={SpecialFieldKeys.column_group: "display_options"},
    )
    y_tick_count: int = Field(
        default=5,
        description="Number of tick marks on the y-axis",
        json_schema_extra={SpecialFieldKeys.column_group: "display_options"},
    )
    show_number_of_points: Literal["min", "per_line", "both", "none", "auto"] = Field(
        default="per_line",
        title="\\# Points format",
        description="The way the points are displayed, per line will show on the legend",
        json_schema_extra={
            SpecialFieldKeys.column_group: "display_options",
        },
    )
    add_number_of_points_in_box: bool = Field(
        default=False,
        description="Add the number of points in the box",
    )
    x_axis_margin: float = Field(
        default=0.0,
        description="Margin on the x-axis",
        ge=0.0,
        json_schema_extra={SpecialFieldKeys.column_group: "display_options"},
    )

    x_tick_shift: dict[int, float] = Field(
        default_factory=dict,
        description="Shift the x-ticks for each line",
    )

    diff_plot_meta_data: PlotMetadata = Field(
        default_factory=lambda: PlotMetadata(
            title="Probability Difference",
            ylabel="Probability Diff. (%)",
            axhline_value=0,
            with_fixed_limits=True,
            ylim_min=-70,
            ylim_max=40,
        ),
        json_schema_extra={SpecialFieldKeys.expander: "diff plot"},
    )
    acc_plot_meta_data: PlotMetadata = Field(
        default_factory=lambda: PlotMetadata(
            title="Accuracy",
            ylabel="Accuracy (%)",
            axhline_value=100,
            with_fixed_limits=True,
            ylim_min=60,
            ylim_max=105,
        ),
        json_schema_extra={SpecialFieldKeys.expander: "acc plot"},
    )

    # Separator
    sep1: None = Field(default=None, json_schema_extra={SpecialFieldKeys.separator: True})

    # Figure Settings
    figure_width: float = Field(
        default=6.0,
        description="Figure width in inches",
        ge=1.0,
        json_schema_extra={SpecialFieldKeys.column_group: "figure_settings"},
    )
    figure_height: float = Field(
        default=6.0,
        description="Figure height in inches",
        ge=1.0,
        json_schema_extra={SpecialFieldKeys.column_group: "figure_settings"},
    )
    tight_layout: bool = Field(
        default=True,
        description="Tight layout for the figure",
        json_schema_extra={SpecialFieldKeys.column_group: "figure_settings"},
    )
    ylabel_x_coord: float = Field(
        default=-0.3,
        description="X-coordinate of the y-axis label",
        json_schema_extra={SpecialFieldKeys.column_group: "figure_settings"},
    )
    ylabel_y_coord: float = Field(
        default=0.5,
        description="Y-coordinate of the y-axis label",
        json_schema_extra={SpecialFieldKeys.column_group: "figure_settings"},
    )

    # Font Settings
    title_fontsize: int = Field(
        default=12,
        title="Title",
        description="Font size for title",
        ge=4,
        json_schema_extra={SpecialFieldKeys.column_group: "font_settings"},
    )
    axis_fontsize: int = Field(
        default=25,
        title="Axis",
        description="Font size for axis labels",
        ge=4,
        json_schema_extra={SpecialFieldKeys.column_group: "font_settings"},
    )
    legend_fontsize: int = Field(
        default=10,
        title="Legend",
        description="Font size for legend",
        ge=4,
        json_schema_extra={SpecialFieldKeys.column_group: "font_settings"},
    )

    tick_pad: float = Field(
        default=5,
        description="Tick length",
        ge=0.0,
        json_schema_extra={SpecialFieldKeys.column_group: "_font_settings"},
    )

    x_label_pad: float = Field(
        default=15,
        description="Label pad",
        ge=0.0,
        json_schema_extra={SpecialFieldKeys.column_group: "_font_settings"},
    )

    grid_linewidth: float = Field(
        default=1,
        description="Grid line width",
        ge=0.0,
        json_schema_extra={SpecialFieldKeys.column_group: "grid_settings"},
    )
    border_width: float = Field(
        default=2,
        description="Border width",
        ge=1,
        json_schema_extra={SpecialFieldKeys.column_group: "grid_settings"},
    )

    # Legend Settings
    legend_loc: Literal["lower center", "upper center", "lower right", "upper right"] = Field(
        default="upper center",
        description="Location of legend",
        json_schema_extra={SpecialFieldKeys.column_group: "legend_settings"},
    )
    tight_layout_rect_y: float = Field(
        default=0.85,
        description="Y-coordinate of legend location",
        json_schema_extra={SpecialFieldKeys.column_group: "legend_settings"},
    )
    legend_loc_x: float = Field(
        default=0.5,
        description="X-coordinate of legend location",
        json_schema_extra={SpecialFieldKeys.column_group: "legend_settings"},
    )
    show_legend: bool = Field(
        default=True,
        description="Show legend",
    )

    # Separator
    sep2: None = Field(default=None, json_schema_extra={SpecialFieldKeys.separator: True})

    # Custom Colors
    custom_colors: Dict[str, Color] = Field(
        default_factory=dict, description="Custom colors mapping token types to hex color codes"
    )

    # Custom Line Styles
    custom_line_styles: Dict[str, TLineStyle] = Field(
        default_factory=dict, description="Custom line styles mapping feature categories to styles"
    )

    # Custom Line Labels
    custom_line_labels: Dict[str, str] = Field(
        default_factory=dict, description="Custom line labels mapping feature categories to labels"
    )

    @classmethod
    def specify_config(cls, lines_options: list[str]) -> Type[Self]:
        literal_lines_options = create_literal_value(lines_options)
        return create_model(
            f"{cls.__name__}Config",
            __base__=cls,
            custom_colors=annotate_dict_with_literal_values(
                lines_options,
                Color,
                default_factory=lambda: {
                    option: Color(value)
                    for option, value in zip(
                        literal_lines_options,
                        TOKEN_TYPE_COLORS.values(),
                    )
                },
            ),
            custom_line_styles=annotate_dict_with_literal_values(
                lines_options,
                TLineStyle,
            ),
            custom_line_labels=annotate_dict_with_literal_values(
                lines_options,
                str,
            ),
        )


class MetricData(TypedDict):
    mean: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]


class Confidence(TypedDict):
    mean: float
    ci_lower: float
    ci_upper: float


type MetricsDict = dict[TMetricType, MetricData]


# region Confidence Calculation
def calculate_ci(data: NDArray[np.float64], confidence_level: float = 0.95) -> Confidence:
    """Calculate confidence intervals for a given data set using standard error."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    n_samples = len(data)

    if std == 0 or n_samples == 1:
        # If no variance or single sample, CI is just the mean
        return {
            "mean": float(mean),
            "ci_lower": float(mean),
            "ci_upper": float(mean),
        }

    # Calculate standard error and confidence interval
    se = std / np.sqrt(n_samples)
    ci = stats.t.interval(confidence_level, df=n_samples - 1, loc=mean, scale=se)

    return {
        "mean": float(mean),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
    }


def calculate_pi(data: NDArray[np.float64], confidence_level: float = 0.95) -> Confidence:
    """Calculate prediction intervals for a given data set."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    n_samples = len(data)

    if std == 0 or n_samples == 1:
        # If no variance or single sample, PI is just the mean
        return {
            "mean": float(mean),
            "ci_lower": float(mean),
            "ci_upper": float(mean),
        }

    # For prediction interval, we need to account for both
    # the uncertainty in the mean and the spread of future observations
    pi_scale = std * np.sqrt(1 + 1 / n_samples)
    pi = stats.t.interval(confidence_level, df=n_samples - 1, loc=mean, scale=pi_scale)

    return {
        "mean": float(mean),
        "ci_lower": float(pi[0]),
        "ci_upper": float(pi[1]),
    }


def calculate_bootstrap(
    data: NDArray[np.float64], confidence_level: float = 0.95, n_bootstrap: int = 10000
) -> Confidence:
    """Calculate bootstrap confidence intervals for a given data set."""
    mean = np.mean(data)
    n_samples = len(data)

    if n_samples == 1:
        return {
            "mean": float(mean),
            "ci_lower": float(mean),
            "ci_upper": float(mean),
        }

    # Generate bootstrap samples
    rng = np.random.default_rng()
    bootstrap_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        bootstrap_sample = rng.choice(data, size=n_samples, replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)

    # Calculate percentile confidence intervals
    alpha = (1 - confidence_level) / 2
    ci_lower, ci_upper = np.percentile(bootstrap_means, [100 * alpha, 100 * (1 - alpha)])

    return {
        "mean": float(mean),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def calculate_se(data: NDArray[np.float64], confidence_level: float = 0.95) -> Confidence:
    """Calculate standard error based intervals for a given data set."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    n_samples = len(data)

    if std == 0 or n_samples == 1:
        return {
            "mean": float(mean),
            "ci_lower": float(mean),
            "ci_upper": float(mean),
        }

    # Calculate standard error
    se = std / np.sqrt(n_samples)

    # Use normal distribution (simpler than t-distribution)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin = z_score * se

    return {
        "mean": float(mean),
        "ci_lower": float(mean - margin),
        "ci_upper": float(mean + margin),
    }


def calculate_confidence(
    confidence_method: Literal["CI", "PI", "bootstrap", "SE"],
    data: NDArray[np.float64],
    confidence_level: float = 0.95,
) -> Confidence:
    """Calculate confidence intervals for a given data set."""
    if confidence_method == "CI":
        return calculate_ci(data, confidence_level)
    elif confidence_method == "PI":
        return calculate_pi(data, confidence_level)
    elif confidence_method == "bootstrap":
        return calculate_bootstrap(data, confidence_level)
    elif confidence_method == "SE":
        return calculate_se(data, confidence_level)
    else:
        raise ValueError(f"Invalid confidence method: {confidence_method}")


# endregion


def calculate_metrics_with_confidence(
    window_outputs: TInfoFlowOutput,
    metric_types: list[TMetricType],
    confidence_level: float = 0.95,
    confidence_method: Literal["CI", "PI", "bootstrap", "SE"] = "CI",
) -> MetricsDict:
    """
    Calculate metrics with confidence intervals from raw window outputs.

    Returns:
        Dictionary with keys 'acc' and 'diff', each containing:
            - 'mean': mean values per window
            - 'ci_lower': lower confidence bound
            - 'ci_upper': upper confidence bound
    """
    metrics: Dict[str, Dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    metric_to_name = {
        "acc": COLS.INFO_FLOW.HIT,
        "diff": COLS.INFO_FLOW.DIFFS,
    }

    for window_idx in window_outputs.keys():
        window_data = window_outputs[window_idx]

        for metric_type in metric_types:
            confidence = calculate_confidence(
                confidence_method=confidence_method,
                data=np.array(window_data[metric_to_name[metric_type].value]),
                confidence_level=confidence_level,
            )
            for key in confidence:
                metrics[metric_type][key].append(float(confidence[key]))  # type: ignore

    return cast(
        MetricsDict,
        {
            metric_type: {key: np.array(value) for key, value in metrics[metric_type].items()}
            for metric_type in metric_types
        },
    )


def plot_with_confidence(
    metrics: MetricsDict,
    metric_type: TMetricType,
    label: str,
    color: str,
    linestyle: str,
    ax: Axes,
    alpha: float = 0.2,
):
    """Plot a single metric with confidence intervals."""
    layers = np.arange(len(metrics[metric_type]["mean"]))

    # Plot mean line
    ax.plot(
        layers,
        metrics[metric_type]["mean"] * (100 if metric_type == TMetricType.ACC else 1),
        label=label,
        color=color,
        linestyle=linestyle,
    )

    # Plot confidence interval
    ax.fill_between(
        layers,
        metrics[metric_type]["ci_lower"] * (100 if metric_type == TMetricType.ACC else 1),
        metrics[metric_type]["ci_upper"] * (100 if metric_type == TMetricType.ACC else 1),
        color=color,
        alpha=alpha,
    )


def create_confidence_plot(
    lines: dict[str, TInfoFlowOutput],
    confidence_level: float,
    config: InfoFlowPlotConfig,
) -> Figure:
    """Create plots with confidence intervals for all metrics.

    Args:
        lines_metadata: List of metadata for each line to plot
        confidence_level: Confidence level for intervals (0-1)
        title: Title for the plot
        plots_meta_data: Dictionary mapping metric types to their plot metadata
        config: Optional plot configuration

    Returns:
        The matplotlib figure containing the plots
    """
    plots_meta_data: dict[TMetricType, PlotMetadata] = {}
    if TMetricType.ACC in config.metrics_to_show:
        plots_meta_data[TMetricType.ACC] = config.acc_plot_meta_data
    if TMetricType.DIFF in config.metrics_to_show:
        plots_meta_data[TMetricType.DIFF] = config.diff_plot_meta_data

    # Create figure with subplots side by side, with possibly different widths
    fig, axes = plt.subplots(
        1,
        len(plots_meta_data),
        figsize=(config.figure_width * len(plots_meta_data), config.figure_height),
    )

    # If only one subplot, wrap in list
    if len(plots_meta_data) == 1:
        axes = [axes]

    # Dictionary to store unique handles by their properties
    unique_handles = {}

    # Get number of points from first window of first block
    points_per_line = {line_id: len(line_data[0][COLS.INFO_FLOW.HIT.value]) for line_id, line_data in lines.items()}
    min_points = min(points_per_line.values())
    diff_points = max(points_per_line.values()) - min_points
    if config.show_number_of_points == "auto":
        config.show_number_of_points = "min" if diff_points / min_points < 1 / 20 else "per_line"
    max_layers = max(len(line) for line in lines.values())
    # Process each metric type (accuracy and diff)
    for i, metric_type in enumerate(config.metrics_to_show):
        plot_metadata = plots_meta_data[metric_type]
        ax: Axes = axes[i]

        if config.x_axis_as_percentage:
            x_axis_label = "Layer Depth (%)"
            x_normalized_steps = np.linspace(0, 100, max_layers)

            percentages = np.linspace(0, 100, config.x_tick_count)
            x_ticks = [int(p / 100 * (max_layers - 1)) for p in percentages]
            x_ticks_labels = [str(int(x)) for x in percentages]

        else:
            x_axis_label = "Layer Depth"
            x_ticks = np.linspace(0, max_layers - 1, config.x_tick_count, dtype=int)
            x_ticks_labels = [str(int(x)) for x in x_ticks]

        # Plot data for each block
        for line_id, line_data in lines.items():
            # Apply custom colors if provided
            color = "#000000"
            if config.custom_colors is not None and line_id in config.custom_colors:
                color = str(config.custom_colors[line_id])

            # Apply custom line styles if provided
            linestyle = "solid"
            if config.custom_line_styles is not None and line_id in config.custom_line_styles:
                linestyle = config.custom_line_styles[line_id].value

            label = f"{line_id}"
            if config.show_number_of_points == "per_line" or config.show_number_of_points == "both":
                label += f" ({points_per_line[line_id]} points)"

            if config.custom_line_labels is not None and line_id in config.custom_line_labels:
                label = config.custom_line_labels[line_id]

            metrics = calculate_metrics_with_confidence(line_data, [metric_type], confidence_level)

            # Get layer indices and convert to percentage if requested

            num_layers = len(metrics[metric_type]["mean"])

            if config.x_axis_as_percentage:
                # Convert layer indices to percentages (0-100%)
                def interpolate_to_percentage(values):
                    # interpolate to get the x_ticks
                    return np.interp(x_normalized_steps, np.linspace(0, 100, num_layers), values)

                metrics[metric_type]["mean"] = interpolate_to_percentage(metrics[metric_type]["mean"])
                metrics[metric_type]["ci_lower"] = interpolate_to_percentage(metrics[metric_type]["ci_lower"])
                metrics[metric_type]["ci_upper"] = interpolate_to_percentage(metrics[metric_type]["ci_upper"])

            # Draw the plot
            plot_with_confidence(
                metrics=metrics,
                metric_type=metric_type,
                label=label,
                color=color,
                linestyle=linestyle,
                ax=ax,
                alpha=config.alpha,
            )

            # Only collect handles and labels from the first subplot
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    assert isinstance(handle, Line2D)
                    # Create a unique key based on the handle's visual properties
                    key = (label, handle.get_color(), handle.get_linestyle())
                    if key not in unique_handles:
                        unique_handles[key] = (handle, label)

        # Customize subplot
        ax.grid(True, which="both", linestyle="-", linewidth=config.grid_linewidth)

        # Set X-axis label and ticks
        ax.set_xlabel(
            x_axis_label,
            fontsize=config.axis_fontsize,
            labelpad=config.x_label_pad,
        )
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            x_ticks_labels,
            fontsize=config.axis_fontsize,
        )
        ax.axhline(plot_metadata.axhline_value, color="gray", linewidth=1)
        ax.set_ylabel(
            plot_metadata.ylabel,
            fontsize=config.axis_fontsize,
        )
        ax.yaxis.set_label_coords(config.ylabel_x_coord, config.ylabel_y_coord)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=config.y_tick_count))  # Try to use 5 ticks
        if plot_metadata.with_fixed_limits:
            ax.set_ylim(plot_metadata.ylim)
        ax.margins(x=config.x_axis_margin)

        # Adjust tick parameters
        ax.tick_params(
            axis="both",
            which="both",
            pad=config.tick_pad,
            labelsize=config.axis_fontsize,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(config.border_width)  # Thicker border

        if config.x_tick_shift:
            # Get current tick positions and labels
            current_positions = ax.get_xticks()
            current_labels = [t.get_text() for t in ax.get_xticklabels()]
            d = (current_positions[-1] - current_positions[0]) / 1e2
            modified_positions = [x for x in current_positions]
            for tick_idx, tick_shift in config.x_tick_shift.items():
                modified_positions[tick_idx] += d * tick_shift
            ax.xaxis.set_major_locator(FixedLocator(modified_positions))
            ax.set_xticklabels(current_labels)

    # Extract unique handles and labels
    # Create a single legend for the entire figure
    if config.show_legend:
        all_handles, all_labels = zip(*unique_handles.values()) if unique_handles else ([], [])
        max_items_per_row = 2 if config.show_number_of_points in ["per_line", "both"] else 4
        ncols = min(max_items_per_row, len(all_handles))

        fig.legend(
            all_handles,
            all_labels,
            # loc=config.legend_loc,
            # bbox_to_anchor=(config.legend_loc_x, config.legend_loc_y),
            ncol=ncols,
            fontsize=config.legend_fontsize,
            frameon=False,
            borderaxespad=0.3,  # space between legend and axes
        )

    points_label = ""
    if config.show_number_of_points == "min" or config.show_number_of_points == "both":
        points_label = f" {min_points} Points"
        if diff_points > 0:
            points_label += " (Min)"

    if config.add_number_of_points_in_box:
        for ax in axes:
            ax.text(
                0.01,
                0.95,
                points_label,
                transform=ax.transAxes,
                fontsize=config.axis_fontsize,
                verticalalignment="top",
                # bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )

    # Set overall title
    custom_title = config.title + points_label
    fig.suptitle(
        f"{custom_title}",
        fontsize=config.title_fontsize,
    )

    if config.tight_layout:
        fig.tight_layout(rect=(0, 0, 1, config.tight_layout_rect_y))
    return fig


def create_plotly_confidence_chart(
    targets_window_outputs: list[TInfoFlowOutput],
    metric_type: TMetricType,
    colors: list[str],
    line_styles: list[TLineStyle],
    legend_labels: list[str],
    confidence_level: float = 0.95,
) -> go.Figure:
    """
    Create a Plotly figure with confidence intervals for the specified metric.

    Args:
        targets_window_outputs: Dictionary mapping sources to their window outputs
        metric_type: Type of metric to plot ('acc' for accuracy or 'diff' for probability difference)
        confidence_level: Confidence level for intervals (0-1)
        title: Title for the plot
        custom_colors: Optional dictionary mapping sources to custom colors
        custom_line_styles: Optional dictionary mapping sources to custom line styles

    Returns:
        Plotly figure with confidence intervals
    """
    fig = go.Figure()

    # Set up y-axis parameters based on metric type
    if metric_type == TMetricType.ACC:
        y_title = "Accuracy (%)"
        y_range = [0, 100]
        axhline_value = 100
        multiplier = 100  # Convert to percentage
    else:  # diff
        y_title = "Probability Difference"
        y_range = None
        axhline_value = 0
        multiplier = 1

    # Add horizontal reference line
    fig.add_hline(
        y=axhline_value,
        line_dash="dash",
        line_color="gray",
        line_width=1,
    )

    max_layers = max(len(info_flow.keys()) for info_flow in targets_window_outputs)

    # Process each source
    for i, window_outputs in enumerate(targets_window_outputs):
        # Calculate metrics with confidence intervals
        metrics = calculate_metrics_with_confidence(window_outputs, [metric_type], confidence_level)

        # Get color and line style
        x_values = list(window_outputs.keys())

        # Create a legend group for this source
        legend_group = f"group_{i}"

        # Add main line
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=metrics[metric_type]["mean"] * multiplier,
                mode="lines",
                line=dict(
                    color=colors[i],
                    dash=CONVERT_TO_PLOTLY_LINE_STYLE[line_styles[i]],
                    width=2,
                ),
                name=legend_labels[i],
                legendgroup=legend_group,
                showlegend=True,
            )
        )

        # Add confidence interval as a filled area
        fig.add_trace(
            go.Scatter(
                x=x_values + x_values[::-1],
                y=list(metrics[metric_type]["ci_upper"] * multiplier)
                + list(metrics[metric_type]["ci_lower"] * multiplier)[::-1],
                fill="toself",
                fillcolor=colors[i],
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                legendgroup=legend_group,
                showlegend=False,
                opacity=0.2,
            )
        )

    # Update layout
    fig.update_layout(
        xaxis_title="Layers",
        yaxis_title=y_title,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(
        tickmode="linear",
        dtick=max(1, max_layers // 10),  # Adjust tick spacing based on max layer
    )

    # Only show legend if there are multiple flows
    if len(targets_window_outputs) <= 1:
        fig.update_layout(showlegend=False)

    # Set y-axis range if specified
    if y_range:
        fig.update_yaxes(range=y_range)

    return fig
