"""
Purpose: Manage prompt filteration presets and provide a comprehensive interface for creating complex filterations.

High Level Outline:
- Imports and type definitions
- Helper components for preset details and filteration creation
- Main page component for preset management
- Regions for different sections of functionality

Outline Issues:
- Need to break down the large render method into smaller components
- Should extract filteration creation logic into a separate component
- Should add better type hints for menu items
-

Outline Compatibility Issues:
- Need to organize code into regions
- Need to extract components into proper hierarchy
-
"""

from typing import Dict, List, Optional, Union, cast

import streamlit as st
import streamlit_antd_components as sac
from streamlit.delta_generator import DeltaGenerator

from src.analysis.prompt_filterations import (
    AllPromptFilteration,
    Correctness,
    ModelCorrectPromptFilteration,
)
from src.app.components.prompt_filter import (
    ShowPromptFilterationComponent,
)
from src.core.consts import DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION, MODEL_SIZES_PER_ARCH_TO_MODEL_ID
from src.core.names import DatasetName
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    SPLIT,
    TModelSize,
    TPresetID,
    TPromptOriginalIndex,
    TSplitChoise,
)
from src.data_ingestion.data_defs.data_defs import PromptFilterationsPresets
from src.experiments.infrastructure.base_prompt_filteration import (
    BasePromptFilteration,
    LogicalOperationType,
    LogicalPromptFilteration,
    SamplePromptFilteration,
    SelectivePromptFilteration,
)
from src.utils.streamlit.helpers.component import StreamlitComponent, StreamlitPage
from src.utils.streamlit.helpers.session_keys import SessionKey

# region Helper Components


def render_preset_details(
    preset_name: str, preset: BasePromptFilteration, container: Optional[DeltaGenerator] = None
) -> None:
    """Render detailed information about a preset."""
    target = container or st
    with target.expander(f"Details for '{preset_name}'", expanded=True):
        # Show filteration structure
        ShowPromptFilterationComponent(
            prompt_filteration=preset,
            key=f"preset_details_{preset_name}",
        ).render()

        # Show prompt count
        st.info(f"Number of prompts: {len(preset.get_prompt_ids())}")


class FilterationCreator(StreamlitComponent[BasePromptFilteration]):
    """Component for creating complex filterations."""

    def __init__(self, key: str):
        self.key = key
        self.current_filteration_sk = SessionKey[BasePromptFilteration](
            f"{key}_current_filteration", default_value=AllPromptFilteration()
        )
        self.current_filteration_sk.init_default()

    def render_basic_filteration_creator(self) -> None:
        st.subheader("Basic Filteration")

        filteration_type = st.selectbox(
            "Filteration Type", ["All", "Selective", "Model Correct"], key=f"{self.key}_basic_type"
        )

        if filteration_type == "All":
            dataset = st.selectbox("Dataset", list(DatasetName), key=f"{self.key}_dataset")
            split = st.selectbox("Split", ["all", *[s.value for s in SPLIT]], key=f"{self.key}_split")
            if st.button("Create All Filteration"):
                self.current_filteration_sk.value = AllPromptFilteration(
                    dataset_name=dataset, split=cast(TSplitChoise, split)
                )

        elif filteration_type == "Selective":
            prompt_ids_str = st.text_input("Prompt IDs (comma-separated)", key=f"{self.key}_prompt_ids")
            if st.button("Create Selective Filteration") and prompt_ids_str:
                try:
                    prompt_ids = [TPromptOriginalIndex(int(x.strip())) for x in prompt_ids_str.split(",")]
                    self.current_filteration_sk.value = SelectivePromptFilteration(prompt_ids=tuple(prompt_ids))
                except ValueError:
                    st.error("Invalid prompt IDs format. Please use comma-separated numbers.")

        elif filteration_type == "Model Correct":
            col1, col2 = st.columns(2)
            with col1:
                dataset = st.selectbox("Dataset", list(DatasetName), key=f"{self.key}_mc_dataset")
                model_arch = st.selectbox(
                    "Model Architecture",
                    [arch for arch in MODEL_SIZES_PER_ARCH_TO_MODEL_ID.keys()],
                    key=f"{self.key}_mc_arch",
                )
            with col2:
                model_sizes: List[str] = list(MODEL_SIZES_PER_ARCH_TO_MODEL_ID[model_arch].keys())
                model_size = st.selectbox("Model Size", model_sizes, key=f"{self.key}_mc_size")
                correctness = st.selectbox("Correctness Type", list(Correctness), key=f"{self.key}_mc_correctness")

            if st.button("Create Model Correct Filteration"):
                self.current_filteration_sk.value = ModelCorrectPromptFilteration(
                    dataset_name=dataset,
                    model_arch_and_size=MODEL_ARCH_AND_SIZE(
                        arch=cast(MODEL_ARCH, model_arch),
                        size=cast(TModelSize, model_size),
                    ),
                    correctness=correctness,
                    code_version=DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION,
                )

    def render_composition_creator(self) -> None:
        st.subheader("Composition Operations")

        operation = st.radio("Operation", ["Sample", "Union", "Intersection"], key=f"{self.key}_operation")

        if operation == "Sample":
            sample_size = st.number_input(
                "Sample Size",
                min_value=1,
                max_value=len(self.current_filteration_sk.value.get_prompt_ids()),
                value=min(50, len(self.current_filteration_sk.value.get_prompt_ids())),
                key=f"{self.key}_sample_size",
            )
            seed = st.number_input("Random Seed", value=42, key=f"{self.key}_sample_seed")
            if st.button("Apply Sampling"):
                self.current_filteration_sk.value = SamplePromptFilteration(
                    base_prompt_filteration=self.current_filteration_sk.value,
                    sample_size=sample_size,
                    seed=seed,
                )

        elif operation in ["Union", "Intersection"]:
            presets = PromptFilterationsPresets.load()
            selected_preset = st.selectbox(
                "Select Preset to Compose With", list(presets.keys()), key=f"{self.key}_compose_preset"
            )

            if st.button(f"Apply {operation}"):
                other_filteration = presets[cast(TPresetID, selected_preset)]
                if operation == "Union":
                    if (
                        isinstance(self.current_filteration_sk.value, LogicalPromptFilteration)
                        and self.current_filteration_sk.value.operation_type == LogicalOperationType.OR
                    ):
                        self.current_filteration_sk.value = self.current_filteration_sk.value.or_with(other_filteration)
                    else:
                        self.current_filteration_sk.value = LogicalPromptFilteration.create_or(
                            [self.current_filteration_sk.value, other_filteration]
                        )
                else:  # Intersection
                    self.current_filteration_sk.value = LogicalPromptFilteration.create_and(
                        [self.current_filteration_sk.value, other_filteration]
                    )

    def render(self) -> BasePromptFilteration:
        st.write("Create New Filteration")

        # Show current filteration
        st.subheader("Current Filteration")
        ShowPromptFilterationComponent(
            prompt_filteration=self.current_filteration_sk.value, key=f"{self.key}_current"
        ).render()

        # Create tabs for different creation methods
        tab1, tab2 = st.tabs(["Basic Filteration", "Composition"])

        with tab1:
            self.render_basic_filteration_creator()

        with tab2:
            self.render_composition_creator()

        return self.current_filteration_sk.value


# endregion

# region Main Page Component


class PromptFilterationPresetsPage(StreamlitPage):
    """Page for managing prompt filteration presets."""

    NEW_PRESET_LABEL = "Create New Preset"

    def render(self) -> None:
        """Main function to render the preset management page."""
        st.title("Prompt Filteration Presets Management")

        # Load presets
        presets = PromptFilterationsPresets.load()

        # Layout with two columns
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Available Presets")
            # Create menu items for each preset
            menu_items: List[Union[str, Dict, sac.MenuItem]] = [
                sac.MenuItem(
                    label=self.NEW_PRESET_LABEL,
                    icon="plus",
                ),
                *[
                    sac.MenuItem(
                        label=preset_name,
                        icon="filter",  # Using filter icon for presets
                    )
                    for preset_name in presets.keys()
                ],
            ]

            selected_preset = sac.menu(items=menu_items, format_func="title", open_all=True, key="preset_menu")

        with col2:
            if selected_preset == self.NEW_PRESET_LABEL:
                new_preset_name = st.text_input("New Preset Name", key="new_preset_name")
                new_preset = FilterationCreator(key="new_preset").render()
                if st.button("Create Preset"):
                    if new_preset_name in presets:
                        st.error(f"Preset name '{new_preset_name}' already exists!")
                    else:
                        presets.add_preset(new_preset_name, new_preset)
                        st.success(f"Created new preset '{new_preset_name}'")
                        st.rerun()
            elif selected_preset:
                selected_preset_str = cast(str, selected_preset)
                st.subheader(f"Manage Preset: {selected_preset_str}")

                # Show preset details
                render_preset_details(selected_preset_str, presets[selected_preset_str])

                # Preset actions
                action = st.radio("Choose Action", ["Edit", "Rename", "Delete", "Duplicate"], horizontal=True)

                if action == "Edit":
                    st.write("Edit Preset Configuration")
                    new_preset = FilterationCreator(key=f"edit_preset_{selected_preset_str}").render()

                    if st.button("Save Changes"):
                        presets._items[cast(TPresetID, selected_preset_str)] = new_preset
                        presets.save()
                        st.success(f"Successfully updated preset '{selected_preset_str}'")
                        st.rerun()

                elif action == "Rename":
                    new_name = st.text_input("New Preset Name", value=selected_preset_str)
                    if st.button("Rename Preset") and new_name and new_name != selected_preset_str:
                        if new_name in presets:
                            st.error(f"Preset name '{new_name}' already exists!")
                        else:
                            presets._items[cast(TPresetID, new_name)] = presets[cast(TPresetID, selected_preset_str)]
                            del presets._items[cast(TPresetID, selected_preset_str)]
                            presets.save()
                            st.success(f"Renamed preset from '{selected_preset_str}' to '{new_name}'")
                            st.rerun()

                elif action == "Delete":
                    st.warning(f"Are you sure you want to delete preset '{selected_preset_str}'?")
                    if st.button("Confirm Delete"):
                        del presets._items[cast(TPresetID, selected_preset_str)]
                        presets.save()
                        st.success(f"Deleted preset '{selected_preset_str}'")
                        st.rerun()

                elif action == "Duplicate":
                    new_name = st.text_input("New Preset Name", value=f"{selected_preset_str}_copy")
                    if st.button("Duplicate Preset") and new_name:
                        if new_name in presets:
                            st.error(f"Preset name '{new_name}' already exists!")
                        else:
                            presets._items[cast(TPresetID, new_name)] = presets[cast(TPresetID, selected_preset_str)]
                            presets.save()
                            st.success(f"Duplicated preset '{selected_preset_str}' to '{new_name}'")
                            st.rerun()


# endregion
