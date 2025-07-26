from enum import Enum
from typing import Any, List, Union

import streamlit as st

from src.analysis.plots.image_combiner import (
    GridOrganizer,
    ImageGridParams,
    combine_image_grid,
    organize_images_to_grid_with_keys,
)
from src.app.app_consts import GLOBAL_APP_CONSTS, AppSessionKeys
from src.app.components.inputs import choose_heatmap_parms
from src.core.consts import GRAPHS_ORDER
from src.core.types import MODEL_SIZE_CAT, TPromptOriginalIndex
from src.experiments.infrastructure.base_prompt_filteration import SelectivePromptFilteration
from src.experiments.infrastructure.base_runner import InputParams, MetadataParams
from src.experiments.runners.heatmap import HeatmapParams, HeatmapRunner
from src.utils.streamlit.components.extended_streamlit_pydantic import pydantic_input
from src.utils.streamlit.helpers.component import StreamlitComponent


def convert_to_enum_if_needed(value: Any, enum_class: type) -> Any:
    """Convert a string value to an enum if needed."""
    if isinstance(value, str) and issubclass(enum_class, Enum):
        try:
            return enum_class[value]
        except (KeyError, TypeError):
            pass
    return value


class HeatmapPlotGenerationComponent(StreamlitComponent):
    def __init__(self, prompt_idx: TPromptOriginalIndex):
        self.prompt_idx = prompt_idx

    def render(self):
        assert self.prompt_idx is not None

        # Apply model filters to get qualifying prompts
        with st.sidebar:
            heatmap_parms = choose_heatmap_parms()
            image_grid_params = ImageGridParams.model_validate(pydantic_input(key="my_form", model=ImageGridParams))

            # Create the grid organizer with UI-configurable properties
            grid_organizer = GridOrganizer.model_validate(pydantic_input(key="grid_organizer", model=GridOrganizer))

        # Create GridOrganizer with default column order for model sizes
        default_col_order: List[Union[str, int, Enum]] = [
            MODEL_SIZE_CAT.SMALL,
            MODEL_SIZE_CAT.MEDIUM,
            MODEL_SIZE_CAT.LARGE,
            MODEL_SIZE_CAT.HUGE,
        ]

        # Create grid organizer with our preferred ordering
        grid_organizer.col_order = default_col_order

        # Collect plots with their metadata
        plot_items_with_keys = []
        rows_count = len(GLOBAL_APP_CONSTS.MODELS_COMBINATIONS)
        progress_bar = st.progress(0, text="Plotting...")

        for i, model_arch_and_size in enumerate(GLOBAL_APP_CONSTS.MODELS_COMBINATIONS):
            model_arch, model_size = model_arch_and_size
            config = HeatmapRunner(
                variant_params=HeatmapParams(
                    model_arch=model_arch,
                    model_size=model_size,
                    window_size=AppSessionKeys.window_size.value,
                ),
                input_params=InputParams(
                    filteration=SelectivePromptFilteration(
                        prompt_ids=(self.prompt_idx,),
                    ),
                ),
                metadata_params=MetadataParams(
                    code_version=AppSessionKeys.code_version.value,
                ),
            )

            # Check if the prompt exists in the HDF5 file
            prompt_exists = False
            existing_prompts = config.output_hdf5_path.get_existing_prompt_idx()
            if self.prompt_idx in existing_prompts:
                prompt_exists = True

            if not prompt_exists:
                continue

            plots_path = config.get_plot_output_path(self.prompt_idx, heatmap_parms.plot_name)
            if not plots_path.exists():
                config.plot(heatmap_parms.plot_name)

            progress = min((i + 1) / rows_count, 1.0)
            progress_bar.progress(progress, text=f"Plotting {i + 1}/{rows_count}")

            # Extract keys for grid organization
            row_key = model_arch  # Architecture as row key
            col_key = GRAPHS_ORDER[model_arch_and_size]  # Size category as column key

            # Add plot path and pre-extracted keys
            plot_items_with_keys.append((plots_path, row_key, col_key))

        progress_bar.empty()

        # Organize the images into a grid
        if plot_items_with_keys:
            try:
                grid = organize_images_to_grid_with_keys(plot_items_with_keys, grid_organizer)

                # Create the combined image
                if grid and any(any(row) for row in grid):
                    # Remove None values from grid rows for display
                    non_none_grid = [[p for p in row if p is not None] for row in grid if any(row)]
                    if image_grid_params is not None:
                        combined_image = combine_image_grid(
                            non_none_grid, image_grid_params, legend_items=[], row_labels=[], col_labels=[]
                        )
                        if combined_image is not None:
                            st.image(combined_image)
            except Exception as e:
                st.error(f"Error organizing images: {str(e)}")
                import traceback

                st.code(traceback.format_exc())
