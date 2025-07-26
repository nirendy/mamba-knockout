from functools import lru_cache
from typing import Callable, Optional, Union, cast

import pandas as pd
import streamlit as st
import streamlit_antd_components_mod as sac
from annotated_text import annotated_text, annotation
from pandas import DataFrame
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode
from streamlit.delta_generator import DeltaGenerator

from src.analysis.experiment_results.model_prompt_combination import ModelCombination
from src.analysis.experiment_results.prompt_filteration_factory import (
    AllImportantModelsFilterationFactory,
    ContextModelsFilterationFactory,
    CurrentModelFilterationFactory,
    ExistingPromptsFilterationFactory,
    FilterationSource,
    ModelCorrectnessFilterationFactory,
    PresetFilterationFactory,
    PromptFilterationFactoryUnion,
)
from src.analysis.prompt_filterations import (
    Correctness,
    get_shared_models_correctness_prompt_filteration,
)
from src.app.app_consts import GLOBAL_APP_CONSTS, AppSessionKeys
from src.app.app_utils import (
    filter_combinations,
    get_steamlit_dataframe_selected_row,
)
from src.app.components.inputs import select_enum
from src.app.data_store import get_merged_evaluations
from src.app.texts import HEATMAP_TEXTS
from src.core.names import COLS, HeatmapCols
from src.core.types import MODEL_ARCH_AND_SIZE, TPresetID, TPromptData, TPromptDataFlat, TPromptOriginalIndex
from src.data_ingestion.data_defs.data_defs import ModelCombinationsPrompts, PromptFilterationsPresets
from src.data_ingestion.datasets.download_dataset import (
    df_safe_operation,
    indexed_to_flat_prompt_data,
)
from src.data_ingestion.helpers.dataframe import (
    index_to_row_position,
    validate_one_selected_row_dataframe,
)
from src.data_ingestion.helpers.logits_utils import Prompt
from src.experiments.infrastructure.base_prompt_filteration import (
    BasePromptFilteration,
    LogicalOperationType,
    LogicalPromptFilteration,
    SamplePromptFilteration,
)
from src.utils.streamlit.components.aagrid import (
    SelectionMode,
    base_grid_builder,
    set_pre_selected_rows,
)
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.streamlit.helpers.session_keys import SessionKey
from src.utils.types_utils import str_enum_values


def show_prompt(prompt: Prompt):
    annotated_text(
        [
            annotation(val.format(""), col)
            for col in [
                COLS.COUNTER_FACT.RELATION_PREFIX,
                COLS.COUNTER_FACT.SUBJECT,
                COLS.COUNTER_FACT.RELATION_SUFFIX,
                COLS.COUNTER_FACT.TARGET_TRUE,
            ]
            if pd.notna(val := prompt.get_column(col))
        ]
    )


class SamplePrompts(StreamlitComponent[BasePromptFilteration]):
    def __init__(
        self,
        prompts_filteration: BasePromptFilteration,
    ):
        self.prompts_filteration = prompts_filteration

    def render(
        self,
    ):
        sample_results = st.checkbox("Sample results", value=True)

        prompt_ids = self.prompts_filteration.get_prompt_ids()
        if sample_results:
            sample_results_count = st.slider(
                "Sample results count",
                value=min(50, len(prompt_ids)),
                min_value=min(50, len(prompt_ids)),
                max_value=len(prompt_ids),
                step=50,
            )

            seed = st.number_input(
                "Seed",
                value=42,
                min_value=0,
                max_value=1000000,
                step=1,
            )

            return SamplePromptFilteration(
                base_prompt_filteration=self.prompts_filteration,
                sample_size=sample_results_count,
                seed=seed,
            )
        return self.prompts_filteration


class ShowModelCombinations(StreamlitComponent[tuple[pd.DataFrame, Optional[int]]]):
    def __init__(
        self,
        model_combinations_prompts: ModelCombinationsPrompts,
        representative_model_evaluations: pd.DataFrame,
        filters_container: Optional[DeltaGenerator] = None,
    ):
        self.model_combinations_prompts = model_combinations_prompts
        self.representative_model_evaluations = representative_model_evaluations
        self.filters_container = filters_container

    def render(self):
        display_df = self.model_combinations_prompts.to_display_df(GLOBAL_APP_CONSTS.MODELS_COMBINATIONS)
        assert display_df[HeatmapCols.PROMPT_COUNT].sum() == len(self.representative_model_evaluations), (
            "Display df prompt count mismatch, "
            f"{display_df[HeatmapCols.PROMPT_COUNT].sum()} != {len(self.representative_model_evaluations)}"
        )

        filter_container = self.filters_container
        if filter_container is None:
            filter_container = st.sidebar.expander(HEATMAP_TEXTS.MODEL_COMBINATIONS_FILTERING, expanded=True)
        with filter_container:
            filtered_df = filter_combinations(
                display_df,
                [model_name_and_size.model_name for model_name_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS],
            )

        selected_combination_row = get_steamlit_dataframe_selected_row(
            st.dataframe(
                filtered_df,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    HeatmapCols.PROMPT_COUNT: st.column_config.NumberColumn(pinned=True),
                    HeatmapCols.SELECTED_PROMPT: st.column_config.TextColumn(pinned=True),
                    **{
                        model_name_and_size.model_name: st.column_config.TextColumn()
                        for model_name_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS
                    },
                },
            )
        )

        return filtered_df, selected_combination_row


class SelectPromptsComponent(StreamlitComponent[Optional[pd.DataFrame]]):
    def __init__(
        self,
        prompts_df: TPromptDataFlat,
        selection_mode: SelectionMode,
        key: str,
        pre_selected_rows: Optional[list[str]] = None,
    ):
        cols = [COLS.ORIGINAL_IDX, *str_enum_values(COLS.COUNTER_FACT)]
        remaining_cols = [col for col in prompts_df.columns if col not in cols]
        self.prompts_df = prompts_df[cols + remaining_cols]
        self.selection_mode = selection_mode
        self.key = key
        self.pre_selected_rows = pre_selected_rows

    def render_grid(self):
        df, grid_builder = base_grid_builder(self.prompts_df, self.selection_mode, [])
        grid_builder.configure_column(COLS.ORIGINAL_IDX, pinned=True)
        grid_builder.configure_column(COLS.COUNTER_FACT.PROMPT, pinned=True)
        grid_builder.configure_column(COLS.COUNTER_FACT.TARGET_TRUE, pinned=True)

        set_pre_selected_rows(grid_builder, self.pre_selected_rows)
        grid_options = grid_builder.build()
        grid_results = AgGrid(
            df,
            key=self.key,
            gridOptions=grid_options,
            height=300,
            # enable_enterprise_modules=False,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            # update_mode=GridUpdateMode.MANUAL,
            # fit_columns_on_grid_load=True,
            data_return_mode=DataReturnMode.AS_INPUT,
            floatingFilter=True,
            allow_unsafe_jscode=True,
        )
        return grid_results

    def render(self):
        grid_results = self.render_grid()
        return grid_results.selected_data

    def render_validate_single_selection(self):
        grid_results = self.render_grid()
        return validate_one_selected_row_dataframe(grid_results.selected_data)


class PromptSelectionForCombinationComponent(StreamlitComponent[TPromptOriginalIndex]):
    # TODO: remove this component
    def __init__(
        self,
        combination_row: ModelCombination,
        representative_model_evaluations: TPromptData,
        model_combinations_prompts: ModelCombinationsPrompts,
    ):
        self.combination_row = combination_row
        self.representative_model_evaluations = representative_model_evaluations
        self.model_combinations_prompts = model_combinations_prompts

    def render(self):
        possible_prompts = df_safe_operation(
            self.representative_model_evaluations,
            lambda df: df.loc[self.combination_row.prompts],
        )
        chosen_prompt_idx = self.combination_row.chosen_prompt
        selected_prompt_row = SelectPromptsComponent(
            indexed_to_flat_prompt_data(possible_prompts),
            SelectionMode.SINGLE,
            key=f"prompt_selection_component_{chosen_prompt_idx}",
            pre_selected_rows=(
                [] if chosen_prompt_idx is None else [str(index_to_row_position(possible_prompts, chosen_prompt_idx))]
            ),
        ).render_validate_single_selection()
        if selected_prompt_row is not None:
            selected_prompt_idx_new = int(selected_prompt_row[COLS.ORIGINAL_IDX])
            if selected_prompt_idx_new != chosen_prompt_idx:
                # Update the selected prompt index
                chosen_prompt_idx = TPromptOriginalIndex(selected_prompt_idx_new)

        if chosen_prompt_idx is not None:
            show_prompt(Prompt(possible_prompts.loc[chosen_prompt_idx]))
            model_evals: DataFrame = get_merged_evaluations(chosen_prompt_idx, AppSessionKeys.code_version.value)
            st.dataframe(
                (
                    model_evals.pipe(
                        lambda df: df[[col for col in df.columns if col not in str_enum_values(COLS.COUNTER_FACT)]]
                    )
                ),
                hide_index=True,
                column_config={
                    "model_arch": st.column_config.TextColumn(pinned=True),
                    "model_size": st.column_config.TextColumn(pinned=True),
                    COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUTS: st.column_config.ListColumn(),
                },
            )
        return chosen_prompt_idx


class ShowPromptFilterationComponent(StreamlitComponent[Optional[BasePromptFilteration]]):
    def __init__(
        self,
        prompt_filteration: BasePromptFilteration,
        key: str,
    ):
        self.prompt_filteration = prompt_filteration
        self.selected_tree_item_sk = SessionKey[Optional[str]](key)

    def _get_selected_tree_item(self) -> Optional[int]:
        selected_tree_item = self.selected_tree_item_sk.value
        if not selected_tree_item:
            return None
        if isinstance(selected_tree_item, list):
            selected_tree_item = selected_tree_item[0]
        return int(selected_tree_item)

    @staticmethod
    @lru_cache(maxsize=5)
    def render_show_tree(prompt_filteration: BasePromptFilteration):
        key_to_prompt_filteration: list[BasePromptFilteration] = []

        def register_label(label: str, prompt_filteration: BasePromptFilteration):
            key_to_prompt_filteration.append(prompt_filteration)
            return label

        def recursive_build_items(prompt_filteration: BasePromptFilteration) -> Union[str, dict, sac.TreeItem]:
            children = None
            if isinstance(prompt_filteration, LogicalPromptFilteration) and prompt_filteration.operation_type in [
                LogicalOperationType.AND,
                LogicalOperationType.OR,
            ]:
                children = [
                    recursive_build_items(sub_prompt_filteration)
                    for sub_prompt_filteration in prompt_filteration.operands
                ]

            return sac.TreeItem(
                children=cast(list, children),
                label=register_label(str(prompt_filteration.display_name()), prompt_filteration),
                tag=f"Filters {len(prompt_filteration.get_prompt_ids())}",
            )

        items = [recursive_build_items(prompt_filteration)]

        return (
            items,
            tuple(key_to_prompt_filteration),
        )

    def render(self) -> Optional[BasePromptFilteration]:
        self.selected_tree_item_sk.init_default()

        (
            items,
            key_to_prompt_filteration,
        ) = self.render_show_tree(self.prompt_filteration)

        selected_item = self._get_selected_tree_item()
        if selected_item is not None and selected_item < len(items):
            self.selected_tree_item_sk.reset_value()

        sac.tree(
            items=items,
            label="Prompt Filteration",
            size="lg",
            open_all=True,
            checkbox_strict=True,
            return_index=True,
            height=40 * len(key_to_prompt_filteration),
            key=self.selected_tree_item_sk.key_for_component,
        )

        selected_item = self._get_selected_tree_item()
        if selected_item is not None:
            selected_filteration = key_to_prompt_filteration[int(selected_item)]
            st.write(str(selected_filteration))
            return selected_filteration

        return None


def get_default_prompt_filteration() -> dict[TPresetID, Callable[[list[MODEL_ARCH_AND_SIZE]], BasePromptFilteration]]:
    presets = PromptFilterationsPresets.load()

    default_prompt_filteration: dict[str, Callable[[list[MODEL_ARCH_AND_SIZE]], BasePromptFilteration]] = {
        "selected_correct": lambda selected_model_arch_and_sizes: get_shared_models_correctness_prompt_filteration(
            selected_model_arch_and_sizes,
            correctness=Correctness.correct,
        ),
    }
    for preset_name in presets:
        default_prompt_filteration[preset_name] = lambda _, preset=preset_name: presets[preset]

    return default_prompt_filteration


class SelectFilterationFactoryComponent(StreamlitComponent[PromptFilterationFactoryUnion]):
    def __init__(
        self,
        key: str,
        default_filteration_factory: Optional[PromptFilterationFactoryUnion] = None,
    ):
        self.key = key
        if default_filteration_factory is None:
            self.default_filteration_factory = CurrentModelFilterationFactory(
                type=FilterationSource.current_model,
                correctness=Correctness.correct,
                combine_with_existing=False,
            )
        else:
            self.default_filteration_factory = default_filteration_factory

    def render(self):
        selected_source_sk = SessionKey[FilterationSource](
            f"{self.key}_select_filteration_source",
            self.default_filteration_factory.type,
        )
        selected_source_sk.init_default()
        select_correctness_filteration_sk = SessionKey[Correctness](
            f"{self.key}_select_correctness_filteration", allow_none=False
        )
        if isinstance(self.default_filteration_factory, ModelCorrectnessFilterationFactory):
            select_correctness_filteration_sk.init(self.default_filteration_factory.correctness)
        combine_with_existing_sk = SessionKey[bool](f"{self.key}_combine_with_existing")
        combine_with_existing_sk.init(False)
        preset_id_sk = SessionKey[str](f"{self.key}_preset_id")
        preset_id_sk.init("all")

        cols = st.columns([3, 3, 1])
        with cols[0]:
            selected_source = select_enum(
                "Select Filteration Source",
                FilterationSource,
                session_key=selected_source_sk,
            )
        if selected_source == FilterationSource.preset:
            # For preset source, show preset selector
            presets = PromptFilterationsPresets.load()
            preset_options = list(presets.keys())

            selected_preset = cols[1].selectbox(
                "Select Preset",
                preset_options,
                index=preset_options.index(preset_id_sk.value) if preset_id_sk.value in preset_options else 0,
                key=f"{self.key}_preset_selector",
            )
            preset_id_sk.value = selected_preset

            return PresetFilterationFactory(type=FilterationSource.preset, preset_id=selected_preset)

        elif selected_source == FilterationSource.existing_prompts:
            # For existing prompts, no additional options needed
            return ExistingPromptsFilterationFactory(type=FilterationSource.existing_prompts)

        else:
            with cols[1]:
                # For model correctness sources, show correctness selector and combine option
                correctness = select_enum(
                    "Select Correctness",
                    Correctness,
                    session_key=select_correctness_filteration_sk,
                )

            combine_with_existing = cols[2].checkbox(
                "Filter to existing prompts only",
                value=combine_with_existing_sk.value,
                key=f"{self.key}_combine_with_existing",
            )

            # Create the appropriate factory based on selected source
            match selected_source:
                case FilterationSource.current_model:
                    return CurrentModelFilterationFactory(
                        type=FilterationSource.current_model,
                        correctness=correctness,
                        combine_with_existing=combine_with_existing,
                    )
                case FilterationSource.context_models:
                    return ContextModelsFilterationFactory(
                        type=FilterationSource.context_models,
                        correctness=correctness,
                        combine_with_existing=combine_with_existing,
                    )
                case FilterationSource.all_important_models:
                    return AllImportantModelsFilterationFactory(
                        type=FilterationSource.all_important_models,
                        correctness=correctness,
                        combine_with_existing=combine_with_existing,
                    )
                case _:
                    # This shouldn't happen with proper enum handling
                    raise ValueError(f"Unsupported source: {selected_source}")


class FilterPromptsComponent(StreamlitComponent[BasePromptFilteration]):
    """Component for selecting a prompt filteration with optional sampling.

    This simplified component focuses on:
    1. Selecting a filteration preset
    2. Optionally filtering based on available prompts in result bank context
    3. Applying sampling with configurable parameters

    When base_prompt_filteration is provided (typically AnyExistingPromptFilteration),
    the component will ensure the resulting filteration is intersected with
    available prompt IDs from that filteration.
    """

    def __init__(
        self,
        key: str,
        default_preset: TPresetID = "all",
        base_prompt_filteration: Optional[BasePromptFilteration] = None,
    ):
        """Initialize the component.

        Args:
            key: Unique key for the component
            default_preset: Default preset ID to use (used if base_prompt_filteration is None)
            base_prompt_filteration: Optional base prompt filteration to use instead of a preset.
                                     Typically an AnyExistingPromptFilteration to limit to available prompts.
        """
        self.key = key
        self.base_prompt_filteration = base_prompt_filteration
        self.customized_filteration_sk = SessionKey[BasePromptFilteration](
            f"{key}", get_default_prompt_filteration()[default_preset]([])
        )
        self.customized_filteration_sk.init_default()

    def render(self) -> BasePromptFilteration:
        """Render the component and return the selected prompt filteration."""
        # Load presets
        presets = PromptFilterationsPresets.load()

        # Preset selection
        col1, col2 = st.columns([3, 1])
        with col1:
            preset_options = list(presets.keys())
            selected_preset = st.selectbox(
                "Reset from Preset",
                preset_options,
                index=preset_options.index("all") if "all" in preset_options else 0,
                key=f"{self.key}_preset_selector",
            )

        with col2:
            if st.button("Reset to Preset", key="load_btn"):
                # Get the preset and ensure it's intersected with available prompt IDs from context
                self.customized_filteration_sk.value = presets[selected_preset]
                st.success(f"Loaded preset: {selected_preset}")

        # Display current filteration
        st.subheader("Current Filteration")
        total = self.customized_filteration_sk.value
        if self.base_prompt_filteration:
            available_count = len(self.base_prompt_filteration.get_prompt_ids())
            st.info(f"Filtering based on available prompts. {available_count} prompt(s) available in context.")
            total = self.base_prompt_filteration & total

        ShowPromptFilterationComponent(
            prompt_filteration=total,
            key=f"{self.key}_current_filteration",
        ).render()

        total_prompt_count = len(total.get_prompt_ids())
        if total_prompt_count and st.checkbox(
            "Apply sampling", value=True, key=f"{self.customized_filteration_sk.key}_sample_checkbox"
        ):
            col1, col2 = st.columns(2)
            with col1:
                sample_results_count = st.slider(
                    "Sample size",
                    value=min(50, total_prompt_count),
                    min_value=1,
                    max_value=total_prompt_count,
                    step=1,
                    key=f"{self.customized_filteration_sk.key}_sample_size",
                )

            with col2:
                seed = st.number_input(
                    "Random seed",
                    value=42,
                    min_value=0,
                    max_value=1000000,
                    step=1,
                    key=f"{self.customized_filteration_sk.key}_seed",
                )

            return SamplePromptFilteration(
                base_prompt_filteration=total,
                sample_size=sample_results_count,
                seed=seed,
            )

        return total
