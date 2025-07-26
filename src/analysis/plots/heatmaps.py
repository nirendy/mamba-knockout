from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from pydantic import BaseModel, Field

from src.core.consts import reverse_model_id
from src.utils.streamlit.st_pydantic_v2.input import SpecialFieldKeys


class HeatmapPlotConfig(BaseModel):
    """Configuration for heatmap plots."""

    # Basic Configuration
    title: str = Field(
        default="",
        description="Custom plot title (leave empty for default)",
    )
    minimal_title: bool = Field(
        default=False,
        description="Use minimal title without extra details",
        json_schema_extra={SpecialFieldKeys.column_group: "basic_config"},
    )
    is_base_prob_in_title: bool = Field(
        default=True,
        description="Show the base probability in the title",
        json_schema_extra={SpecialFieldKeys.column_group: "basic_config"},
    )
    show_base_prob_annotation: bool = Field(
        default=True,
        description="Show the base probability as a text annotation on the plot",
        json_schema_extra={SpecialFieldKeys.column_group: "basic_config"},
    )

    # X-axis Options
    x_axis_as_percentage: bool = Field(
        default=True,
        description="Show X-axis as percentages",
        json_schema_extra={SpecialFieldKeys.column_group: "x_axis_options"},
    )
    x_tick_count: int = Field(
        default=6,
        description="Number of tick marks on the x-axis",
        ge=2,
        le=12,
        json_schema_extra={SpecialFieldKeys.column_group: "x_axis_options"},
    )

    # Separator
    sep1: None = Field(default=None, json_schema_extra={SpecialFieldKeys.separator: True})

    # Figure Settings
    figure_width: float = Field(
        default=4.0,
        description="Figure width in inches",
        ge=2.0,
        le=12.0,
        json_schema_extra={SpecialFieldKeys.column_group: "figure_settings"},
    )
    figure_height: float = Field(
        default=3.0,
        description="Figure height in inches",
        ge=2.0,
        le=10.0,
        json_schema_extra={SpecialFieldKeys.column_group: "figure_settings"},
    )
    is_tight_layout: bool = Field(
        default=True,
        description="Use tight layout",
        json_schema_extra={SpecialFieldKeys.column_group: "_figure_settings1"},
    )
    tight_layout_rect_y: float = Field(
        default=0.95,
        description="Rectangle of the tight layout",
        ge=0.0,
        le=1.0,
        json_schema_extra={SpecialFieldKeys.column_group: "_figure_settings1"},
    )
    title_position_x: float = Field(
        default=0.45,
        description="Position of the title (x, y)",
        json_schema_extra={SpecialFieldKeys.column_group: "_figure_settings1"},
    )
    title_position_y: float = Field(
        default=0.95,
        description="Position of the title (x, y)",
        json_schema_extra={SpecialFieldKeys.column_group: "_figure_settings1"},
    )

    # Font Settings
    fontsize: int = Field(
        default=12,
        description="Font size for labels and title",
        ge=8,
        json_schema_extra={SpecialFieldKeys.column_group: "font_settings"},
    )
    title_fontsize: int = Field(
        default=12,
        description="Font size for the title",
        ge=8,
        json_schema_extra={SpecialFieldKeys.column_group: "font_settings"},
    )
    tick_fontsize: int = Field(
        default=10,
        description="Font size for tick labels",
        ge=6,
        json_schema_extra={SpecialFieldKeys.column_group: "font_settings"},
    )
    base_prob_text_fontsize: int = Field(
        default=10,
        description="Font size for the base probability text annotation",
        ge=6,
        json_schema_extra={SpecialFieldKeys.column_group: "font_settings"},
    )

    # Separator
    sep2: None = Field(default=None, json_schema_extra={SpecialFieldKeys.separator: True})

    # Axis & Labels
    x_axis_label: str = Field(
        default="Depth %",
        description="Label for x-axis",
        json_schema_extra={SpecialFieldKeys.column_group: "axis_labels"},
    )
    y_axis_label: str = Field(
        default="",
        description="Label for y-axis",
        json_schema_extra={SpecialFieldKeys.column_group: "axis_labels"},
    )
    colorbar_nbins: int = Field(
        default=5,
        description="Number of bins in the colorbar",
        ge=3,
        le=10,
        json_schema_extra={SpecialFieldKeys.column_group: "axis_labels"},
    )

    # Color Settings
    colormap: str = Field(
        default="RdYlGn",
        description="Colormap name (e.g., 'RdYlGn', 'coolwarm', 'viridis')",
        json_schema_extra={SpecialFieldKeys.column_group: "color_settings"},
    )
    reverse_colormap: bool = Field(
        default=False,
        description="Reverse the colormap direction",
        json_schema_extra={SpecialFieldKeys.column_group: "color_settings"},
    )

    # Separator
    sep3: None = Field(default=None, json_schema_extra={SpecialFieldKeys.separator: True})

    # Annotation Settings
    base_prob_text_x_pos: float = Field(
        default=0.7,  # Adjusted to be to the left of a typical colorbar
        description="X position of the base probability text annotation (figure coordinates)",
        ge=0.0,
        le=1.0,
        json_schema_extra={SpecialFieldKeys.column_group: "annotation_settings"},
    )
    base_prob_text_y_pos: float = Field(
        default=0.5,
        description="Y position of the base probability text annotation (figure coordinates)",
        ge=0.0,
        le=1.0,
        json_schema_extra={SpecialFieldKeys.column_group: "annotation_settings"},
    )

    # Normalization Settings
    with_fixed_diff: bool = Field(
        default=True,
        description="Use fixed difference value for colormap scaling",
        json_schema_extra={SpecialFieldKeys.column_group: "normalization"},
    )
    fixed_diff: float = Field(
        default=0.3,
        description="Fixed difference value for colormap scaling",
        ge=0.01,
        le=1.0,
        json_schema_extra={SpecialFieldKeys.column_group: "normalization"},
    )
    is_diff_probs: bool = Field(
        default=True,
        description="Normalize values by subtracting the base probability",
    )
    two_slopes_normalization: bool = Field(
        default=False,
        description="Use two slopes normalization",
    )
    is_robust_normalization: bool = Field(
        default=False,
        description="Use robust normalization",
    )


def simple_diff_fixed(
    prob_mat,
    model_id,
    window_size,
    last_tok,
    base_prob,
    target_rank,
    true_word,
    toks,
    config: Optional[HeatmapPlotConfig] = None,
):
    """
    Creates a diverging heatmap with a specified baseline value and returns the plot object.

    Parameters:
    - prob_mat: 2D numpy array of probabilities to visualize.
    - model_id: Identifier for the model (for the title).
    - window_size: Window size parameter (for the title).
    - last_tok: Token intervened upon (for the title).
    - base_prob: Base probability (for the title).
    - true_word: The true word label for the colorbar.
    - toks: List of tokens for y-axis labels.
    - fixed_diff: Fixed difference value for colormap scaling.
    - fontsize: Font size for labels and title.
    - minimal_title: Whether to use minimal title.
    - config: Optional HeatmapPlotConfig with additional customization options.

    Returns:
    - fig, ax: The matplotlib figure and axis objects.
    """
    # Use provided config or default parameters
    if config is None:
        config = HeatmapPlotConfig()

    # Create figure with specified dimensions
    fig, ax = plt.subplots(figsize=(config.figure_width, config.figure_height))
    center = base_prob if config.is_diff_probs else 0
    # Normalize values if requested
    plot_data = prob_mat - center

    # Set scaling values for the colormap
    fixed_diff_value = config.fixed_diff

    model_arch_and_size = reverse_model_id(model_id)

    sub_params = {}

    if config.two_slopes_normalization:
        sub_params["norm"] = TwoSlopeNorm(vmin=-fixed_diff_value, vmax=fixed_diff_value, vcenter=center)
    elif config.with_fixed_diff:
        sub_params["vmin"] = -fixed_diff_value
        sub_params["vmax"] = fixed_diff_value

    # Select colormap
    cmap = config.colormap
    if config.reverse_colormap:
        cmap = f"{config.colormap}_r"

    # Plot the heatmap
    sns.heatmap(
        plot_data,
        cbar=True,
        cmap=cmap,
        robust=config.is_robust_normalization,
        ax=ax,
        **sub_params,
    )
    title = "Knockout to last token '" r"$\bf{" f"{last_tok}" r"}$" "'"
    if not config.minimal_title:
        title += f"\n{model_arch_and_size.model_name} - Window Size: {window_size}"

    # Set title with appropriate formatting
    plt.suptitle(
        title,
        position=(config.title_position_x, config.title_position_y),
        fontsize=config.title_fontsize,
    )

    # Customize X-axis ticks
    n_cols = plot_data.shape[1]

    if config.x_axis_as_percentage:
        # Calculate positions for percentages based on the specified number of ticks
        percentages = np.linspace(0, 100, config.x_tick_count)

        # Convert percentages to positions in the matrix
        x_ticks = np.array([int((p / 100) * (n_cols - 1)) for p in percentages])
        x_ticks_labels = [str(int(i)) for i in percentages]
        x_axis_label = config.x_axis_label or "Depth %"
    else:
        # If not using percentages, use indices
        x_ticks = np.linspace(0, n_cols - 1, config.x_tick_count, dtype=int)
        x_ticks_labels = [str(i) for i in x_ticks]
        x_axis_label = config.x_axis_label or "Layer"

    # Set axis labels
    ax.set_xlabel(x_axis_label, fontsize=config.fontsize)
    ax.set_ylabel(config.y_axis_label, fontsize=config.fontsize)

    # Set ticks and labels
    ax.set_xticks(x_ticks + 0.5)
    ax.set_xticklabels(x_ticks_labels, rotation=0, fontsize=config.tick_fontsize)

    # Set Y-axis ticks
    ax.set_yticks(np.arange(plot_data.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=config.tick_fontsize)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="both", length=0, labelsize=config.tick_fontsize)

    # fig.subplots_adjust(top=0.8)

    if config.is_tight_layout:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            bbox = cbar.ax.get_position()
            pad = (config.base_prob_text_fontsize / 72) / config.figure_width
            fig.tight_layout(rect=(0, 0, bbox.x1 + pad, config.tight_layout_rect_y))
        else:
            fig.tight_layout(rect=(0, 0, 1, config.tight_layout_rect_y))

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.locator = MaxNLocator(nbins=config.colorbar_nbins)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=config.tick_fontsize)
        # Shift colorbar right based on base probability text font size
        # Convert font size (points) to figure-coordinate offset: (pts/72in) / figure width in inches
        offset_x = (config.base_prob_text_fontsize / 72) / config.figure_width
        orig = cbar.ax.get_position()
        new_pos = Bbox.from_bounds(orig.x0 + offset_x, orig.y0, orig.width, orig.height)
        cbar.ax.set_position(new_pos)

    # Add base probability text annotation
    if config.show_base_prob_annotation:
        annotation_text = f"Base Prob: {round(base_prob, 2)}"
        if target_rank != 1:
            annotation_text += f"\n(Rank: {target_rank})"
        # Position base probability annotation between plot and colorbar
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            ax_pos = ax.get_position()
            cb_pos = cbar.ax.get_position()
            text_x = (ax_pos.x1 + cb_pos.x0) / 2
            text_y = ax_pos.y0 + ax_pos.height * config.base_prob_text_y_pos
            fig.text(
                text_x,
                text_y,
                annotation_text,
                fontsize=config.base_prob_text_fontsize,
                ha="center",
                va="center",
                rotation=-90,
            )

    return fig, ax
