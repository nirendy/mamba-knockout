from __future__ import annotations

import functools
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    NewType,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
)

import pandas as pd
import tqdm

from src.core.consts import PATHS
from src.core.names import (
    COLS,
    VARIANT_PARAM_NAME,
    EvaluateModelMetricName,
    ExperimentName,
    HeatmapCols,
    ModelCombinationCols,
    ResultBankParamNames,
    SummarizedDataFulfilledReqsCols,
    ToClassifyNames,
)
from src.core.types import (
    MODEL_ARCH_AND_SIZE,
    TPlotID,
    TPresetID,
    TPromptDataFlat,
    TPromptOriginalIndex,
    TTokenizer,
)
from src.data_ingestion.helpers.logits_utils import Prompt
from src.experiments.infrastructure.base_prompt_filteration import (
    BasePromptFilteration,
    LogicalPromptFilteration,
    SelectivePromptFilteration,
)
from src.experiments.infrastructure.base_runner import (
    BaseRunner,
    BaseVariantParams,
)
from src.experiments.infrastructure.setup_models import get_tokenizer, get_tokenizer_config_from_hub
from src.experiments.runners.evaluate_model import EvaluateModelRunner
from src.experiments.runners.info_flow import InfoFlowRunner, TWindowLayerStartIndex
from src.utils.infra.data_object import (
    DataObject,
    IndexableDataObject,
    IterableDataObject,
)
from src.utils.jsonable import JSONAble
from src.utils.types_utils import (
    compare_dicts,
    get_dict_keys_by_condition,
    get_enum_or_literal_options,
    select_indexes_from_list,
    subset_dict_by_keys,
)


class DataReqiermentCollection:
    def __init__(self):
        self._data_reqs: dict[BaseVariantParams, LogicalPromptFilteration] = defaultdict(
            lambda: LogicalPromptFilteration.create_or([])
        )

    def add_data_req(self, data_req: BaseVariantParams, prompt_filteration: BasePromptFilteration):
        if data_req.should_skip_task():
            return
        self._data_reqs[data_req] = self._data_reqs[data_req].or_with(prompt_filteration)
        for dependency in prompt_filteration.uncomputed_dependencies():
            self.add_data_req(dependency.variant_params, dependency.input_params.filteration)

    def to_dict(self) -> dict[BaseVariantParams, BasePromptFilteration]:
        return dict(self._data_reqs)


class DataReqs(IndexableDataObject[BaseVariantParams, BasePromptFilteration]):
    def to_fulfilled_reqs(self, result_bank: ResultBank[BaseRunner]) -> FulfilledReqs:
        data_reqs_options: dict[BaseVariantParams, list[BaseRunner]] = {
            data_req: [] for data_req, _ in self._items.items()
        }

        for runner in result_bank:
            if runner.variant_params in self._items:
                prompt_filterations = self._items[runner.variant_params]

                if prompt_filterations.dependencies_are_computed():
                    runner = runner.modify(
                        input_params=runner.input_params.modify(filteration=prompt_filterations.contextualize(runner))
                    )
                    if runner.is_computed():
                        data_reqs_options[runner.variant_params].append(runner)

        return FulfilledReqs(
            {data_req: (self._items[data_req], tuple(options)) for data_req, options in data_reqs_options.items()}
        )

    @classmethod
    def from_data_reqs_collection(cls, data_reqs: DataReqiermentCollection) -> DataReqs:
        return cls(data_reqs.to_dict())


class FulfilledReqs(IndexableDataObject[BaseVariantParams, tuple[BasePromptFilteration, tuple[BaseRunner, ...]]]):
    def __init__(
        self,
        fulfilled_reqs: Dict[BaseVariantParams, tuple[BasePromptFilteration, tuple[BaseRunner, ...]]],
    ):
        super().__init__(fulfilled_reqs)

    def summarize(self) -> SummarizedDataFulfilledReqs:
        return SummarizedDataFulfilledReqs(self)

    def choose_latest_fulfilled(self) -> FulfilledReqs:
        def get_latest_results() -> Dict[BaseVariantParams, tuple[BasePromptFilteration, tuple[BaseRunner, ...]]]:
            return {
                data_req: (filteration, tuple([max(options, key=lambda x: x.metadata_params.code_version)]))
                for data_req, (filteration, options) in self._items.items()
            }

        return FulfilledReqs(get_latest_results())

    def get_config(self) -> Dict[BaseVariantParams, BaseRunner]:
        return {req: runners[0] for req, (_, runners) in self._items.items() if runners}


class SummarizedDataFulfilledReqs(IterableDataObject[dict[str, Any]]):
    def __init__(self, fulfilled_reqs: FulfilledReqs):
        self._fulfilled_reqs = fulfilled_reqs

        raw = []

        for req, (filteration, opts) in fulfilled_reqs._items.items():
            row = {
                **{
                    param: getattr(req, param, None)
                    for param in get_enum_or_literal_options(VARIANT_PARAM_NAME)
                    if param
                    not in [
                        ResultBankParamNames.path,
                        ResultBankParamNames.code_version,
                    ]
                },
                SummarizedDataFulfilledReqsCols.AvailableOptions: len(opts),
                SummarizedDataFulfilledReqsCols.Options: opts,
                SummarizedDataFulfilledReqsCols.Key: str(req),
                SummarizedDataFulfilledReqsCols.filters_requested: len(str(filteration)),
            }
            raw.append(row)
        super().__init__(raw)

    def amount_missing(self) -> int:
        return len([sum_req for sum_req in self if sum_req[SummarizedDataFulfilledReqsCols.AvailableOptions] == 0])

    def to_data_reqs(self) -> DataReqs:
        data_reqs = DataReqiermentCollection()
        for req, (filteration, _) in self._fulfilled_reqs._items.items():
            data_reqs.add_data_req(req, filteration)
        return DataReqs(data_reqs.to_dict())

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._items)


class PlotPlans(IndexableDataObject[TPlotID, "PlotPlan"]):
    @staticmethod
    def get_plot_plan_dir() -> Path:
        return PATHS.FINAL_PLOTS_DIR / "combined_plots"

    @staticmethod
    def get_json_path() -> Path:
        return PATHS.FINAL_PLOTS_DIR / "plot_plans.json"

    @classmethod
    def get_cache_dir(cls, plot_id: TPlotID) -> Path:
        return PATHS.FINAL_PLOTS_DIR / "cache" / plot_id

    @classmethod
    def get_cell_cache_path(cls, plot_plan: PlotPlan, cell: Cell) -> Path:
        return cls.get_cache_dir(plot_plan.plot_id) / f"{cell.get_display_name(plot_plan)}.png"

    @classmethod
    def save_plot_plan(cls, plot_plan: PlotPlan) -> None:
        plot_plans = cls.load()
        plot_plans._items[plot_plan.plot_id] = plot_plan
        plot_plans.save()

    def add_plan(self, plan: PlotPlan):
        if plan.plot_id in self._items:
            raise ValueError(f"Plot plan with title {plan.plot_id} already exists")
        new_items = self._items.copy()
        new_items[plan.plot_id] = plan
        return self.__class__(new_items).order_plans()

    def order_plans(self):
        return self.__class__(dict(sorted(self._items.items(), key=lambda x: x[1].order)))

    def remove_plan(self, plot_id: TPlotID):
        if plot_id not in self._items:
            raise ValueError(f"Plot plan with title {plot_id} does not exist")
        new_items = self._items.copy()
        new_items.pop(plot_id)
        return self.__class__(new_items).order_plans()

    def is_plan_exists(self, plot_id: TPlotID) -> bool:
        return plot_id in self._items

    def get_plan(self, plot_id: TPlotID) -> PlotPlan:
        return self._items[plot_id]

    def save(self) -> None:
        """Save plot plans to JSON file using Pydantic serialization."""
        self.order_plans()
        self.get_json_path().parent.mkdir(parents=True, exist_ok=True)

        # Create a dictionary mapping plot_id to serialized PlotPlan
        serialized_data = {
            plot_id: plan.model_dump(mode="json", exclude_defaults=True) for plot_id, plan in self._items.items()
        }
        # Write the JSON to file
        self.get_json_path().write_text(json.dumps(serialized_data, indent=4))

    def is_empty(self) -> bool:
        return not self._items

    @classmethod
    def load(cls) -> PlotPlans:
        """Load plot plans from JSON file using Pydantic deserialization."""
        from src.analysis.experiment_results.plot_plan import PlotPlan

        if not cls.get_json_path().exists():
            PlotPlans({}).save()
            return PlotPlans({})

        json_data = json.loads(cls.get_json_path().read_text())

        return PlotPlans(
            {TPlotID(plot_id_str): PlotPlan.model_validate(plan_data) for plot_id_str, plan_data in json_data.items()}
        )


class PromptFilterationsPresets(IndexableDataObject[TPresetID, BasePromptFilteration], JSONAble):
    def __init__(self, prompt_filterations: dict[TPresetID, BasePromptFilteration]):
        super().__init__(prompt_filterations)

    @classmethod
    def get_json_path(cls) -> Path:
        return PATHS.FINAL_PLOTS_DIR / "prompt_filterations_presets.json"

    @staticmethod
    def get_default_presets() -> PromptFilterationsPresets:
        from src.analysis.prompt_filterations import (
            AllPromptFilteration,
            Correctness,
            get_shared_models_correctness_prompt_filteration,
        )
        from src.core.consts import ALL_IMPORTANT_MODELS, GRAPHS_ORDER
        from src.experiments.infrastructure.base_prompt_filteration import SelectivePromptFilteration

        return PromptFilterationsPresets(
            {
                "all": AllPromptFilteration(),
                "all_correct": get_shared_models_correctness_prompt_filteration(
                    GRAPHS_ORDER.keys(),
                    Correctness.correct,
                ),
                "all_important_correct": get_shared_models_correctness_prompt_filteration(
                    ALL_IMPORTANT_MODELS.keys(),
                    Correctness.correct,
                ),
                "all_important_top_2_to_5_correct": get_shared_models_correctness_prompt_filteration(
                    ALL_IMPORTANT_MODELS.keys(),
                    Correctness.top_2_to_5_correct,
                ),
                "selective": SelectivePromptFilteration(
                    prompt_ids=tuple(
                        [
                            TPromptOriginalIndex(i)
                            for i in [
                                *[290, 4350, 6403, 14577],  # correct
                                *[6274, 9868, 18562, 12930],  # top 2 to 5 correct
                                *[4734, 4311, 13592, 18117],  # not correct
                            ]
                        ]
                    )
                ),
            }
        )

    def save(self) -> None:
        self.get_json_path().parent.mkdir(parents=True, exist_ok=True)
        self.get_json_path().write_text(self.to_jsonable_json(indent=4))

    def add_preset(self, preset_name: TPresetID, preset: BasePromptFilteration):
        new_presets = PromptFilterationsPresets(
            prompt_filterations={
                **self._items,
                preset_name: preset,
            }
        )
        new_presets.save()
        return new_presets

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> PromptFilterationsPresets:
        if not cls.get_json_path().exists():
            presets = cls.get_default_presets()
            presets.save()
            return presets

        presets = cast(
            dict[TPresetID, BasePromptFilteration],
            cls.from_jsonable_json(cls.get_json_path().read_text()),
        )
        return cls(presets)


class ModelCombinationsPrompts(IterableDataObject["ModelCombination"]):
    def __init__(self, model_combinations: list[ModelCombination]):
        super().__init__(model_combinations)

    def cols_enum(self) -> Type[ModelCombinationCols]:
        return ModelCombinationCols

    def sort_by_prompt_count(self) -> ModelCombinationsPrompts:
        return ModelCombinationsPrompts(sorted(self._items, key=lambda x: len(x.prompts), reverse=True))

    def change_chosen_prompt_by_seed(self, seed: int) -> ModelCombinationsPrompts:
        return ModelCombinationsPrompts([combination.choose_prompt_by_seed(seed) for combination in self._items])

    def to_display_df(self, models_combinations: list[MODEL_ARCH_AND_SIZE]) -> pd.DataFrame:
        table_data: List[Dict[str, Union[int, str]]] = []
        for row in self._items:
            # Create row with model correctness
            table_row: Dict[str, Union[int, str]] = {}

            # Add prompt count and selected prompt first
            table_row[HeatmapCols.PROMPT_COUNT] = len(row.prompts)
            if row.chosen_prompt is not None:
                table_row[HeatmapCols.SELECTED_PROMPT] = str(row.chosen_prompt)
            else:
                table_row[HeatmapCols.SELECTED_PROMPT] = ""

            # Add model columns at the end
            for model_name_and_size in models_combinations:
                model_name = model_name_and_size.model_name
                if model_name_and_size in row.correct_models:
                    table_row[model_name] = "✅"
                elif model_name_and_size in row.incorrect_models:
                    table_row[model_name] = "❌"
                else:
                    table_row[model_name] = "-"
            table_data.append(table_row)
        return pd.DataFrame(table_data)


T_RUNNER_TYPE = TypeVar("T_RUNNER_TYPE", bound=BaseRunner)


class ResultBank(IterableDataObject[T_RUNNER_TYPE]):
    _KEY: ClassVar[str] = "key"

    def to_experiment_results_df(self) -> pd.DataFrame:
        results_data = []
        params_to_include = [
            *get_enum_or_literal_options(VARIANT_PARAM_NAME),
            *get_enum_or_literal_options(ResultBankParamNames),
        ]
        for i, result in enumerate(self._items):
            result_dict: dict = {param: getattr(result.variant_params, param, None) for param in params_to_include}
            result_dict[ResultBankParamNames.path] = str(result.variation_relative_path)
            result_dict[ResultBankParamNames.code_version] = result.metadata_params.code_version
            result_dict[ResultBank._KEY] = i
            results_data.append(result_dict)
        return pd.DataFrame(results_data)

    def from_experiment_results_df(self, experiment_results_df: Optional[pd.DataFrame]):
        if experiment_results_df is None:
            return self.__class__([])
        return self.__class__(select_indexes_from_list(self._items, experiment_results_df[ResultBank._KEY].tolist()))

    def to_info_flow_results(self) -> InfoFlowResults:
        results = [result for result in self if result.variant_params.experiment_name == ExperimentName.info_flow]
        return InfoFlowResults(cast(list[InfoFlowRunner], results))

    def to_evaluate_model_results(self) -> EvaluateModelResults:
        results = [result for result in self if result.variant_params.experiment_name == ExperimentName.evaluate_model]
        return EvaluateModelResults(cast(list[EvaluateModelRunner], results))

    def is_empty(self) -> bool:
        return len(self) == 0

    def get_common_and_different_params(
        self,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if self.is_empty():
            return {}, []

        common_params: Dict[str, Any] = {}
        different_params_list: List[Dict[str, Any]] = []

        all_keys: Set[str] = set()
        for result in self:
            all_keys.update(asdict(result.variant_params).keys())
            all_keys.add(ToClassifyNames.code_version)

        for key in all_keys:
            unique_values = {
                result.metadata_params.code_version
                if key == ToClassifyNames.code_version
                else (getattr(result.variant_params, key, None))
                for result in self
            }

            if len(unique_values) == 1:
                common_params[key] = next(iter(unique_values))
            else:
                for i, result in enumerate(self):
                    if i >= len(different_params_list):
                        different_params_list.append({})
                    if key == ToClassifyNames.code_version:
                        different_params_list[i][key] = result.metadata_params.code_version
                    else:
                        variant_params = asdict(result.variant_params)
                        if key in variant_params:
                            different_params_list[i][key] = variant_params[key]

        return common_params, different_params_list

    def set_prompt_filteration(self, prompt_filteration: BasePromptFilteration):
        return self.__class__(
            [result.modify(input_params=result.input_params.modify(filteration=prompt_filteration)) for result in self]
        )

    def subset_prompts(self, prompt_ids: list[TPromptOriginalIndex]):
        return self.set_prompt_filteration(SelectivePromptFilteration(prompt_ids=tuple(prompt_ids)))

    def unique_model_arch_and_sizes(self) -> list[MODEL_ARCH_AND_SIZE]:
        return list(set(result.variant_params.model_arch_and_size for result in self))


class EvaluateModelResults(ResultBank[EvaluateModelRunner]):
    @lru_cache(maxsize=4)
    def get_hit_per_prompt(self, evaluate_model_metric_name: EvaluateModelMetricName) -> pd.DataFrame:
        hit_per_model: list[pd.Series] = []
        for result in self:
            prompt_data = result.get_prompt_data()
            model_arch_and_size = result.variant_params.model_arch_and_size
            ser = prompt_data[evaluate_model_metric_name]
            ser.name = model_arch_and_size
            hit_per_model.append(ser)
        return pd.DataFrame(hit_per_model).T

    @property
    def model_arch_and_sizes(self) -> list[MODEL_ARCH_AND_SIZE]:
        return [result.variant_params.model_arch_and_size for result in self]


class InfoFlowResults(ResultBank[InfoFlowRunner]):
    def get_common_indices(self) -> set[TPromptOriginalIndex]:
        existing_ids_list = [info_flow.output_file.get_computed_prompt_idx() for info_flow in self]
        return functools.reduce(lambda x, y: x.intersection(y), existing_ids_list)

    def max_layer(self) -> int:
        return max(info_flow.output_file.get_statistics().layers_amount for info_flow in self) - 1

    def min_layer(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return len(self)

    def subset_layers(self, layer_idx_subset: TWindowLayerStartIndex) -> InfoFlowResults:
        return InfoFlowResults(
            [
                info_flow.modify(variant_params=info_flow.variant_params.modify(subset_layers=layer_idx_subset))
                for info_flow in self
            ]
        )


TUniqueTokenizerName = NewType("TUniqueTokenizerName", str)


@dataclass(frozen=True)
class UniqueTokenizerInfo:
    tokenizer: TTokenizer
    model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
    raw_config: dict = field(default_factory=dict)
    _hash: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "_hash", hash(json.dumps(self.raw_config)))

    def __hash__(self) -> int:
        return self._hash

    @property
    def display_name(self) -> TUniqueTokenizerName:
        return TUniqueTokenizerName(f"{self.tokenizer.__class__.__name__}_{hash(self)}")


class Tokenizers(IterableDataObject[UniqueTokenizerInfo]):
    def __init__(self, tokenizers: list[UniqueTokenizerInfo]):
        super().__init__(tokenizers)

    @classmethod
    def from_unique_tokenizers(cls, model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]):
        unique_tokenizers: list[UniqueTokenizerInfo] = []
        configs: list[dict] = []

        for model_arch_and_size in tqdm.tqdm(model_arch_and_sizes, desc="Loading tokenizers"):
            model_arch = model_arch_and_size.arch
            model_size = model_arch_and_size.size

            # Determine the actual tokenizer ID that will be used
            tokenizer_config = get_tokenizer_config_from_hub(model_arch_and_size)

            found_match = False
            for i, compare_tokenizer_config in enumerate(configs):
                is_same, _ = compare_dicts(tokenizer_config, compare_tokenizer_config)

                if is_same:
                    unique_tokenizers[i].model_arch_and_sizes.append(model_arch_and_size)
                    found_match = True
                    break
            if not found_match:
                current_tokenizer = get_tokenizer(model_arch, model_size)
                configs.append(tokenizer_config)
                unique_tokenizers.append(
                    UniqueTokenizerInfo(
                        tokenizer=current_tokenizer,
                        model_arch_and_sizes=[model_arch_and_size],
                        raw_config=tokenizer_config,
                    )
                )

        return cls(unique_tokenizers)


class PromptNew(DataObject):
    def __init__(self, prompt: dict):
        self._prompt = prompt

    def as_prompt(self) -> Prompt:
        return Prompt(self._prompt)  # type: ignore


class Prompts(IndexableDataObject[TPromptOriginalIndex, PromptNew]):
    def __init__(self, df: Dict[TPromptOriginalIndex, PromptNew]):
        super().__init__(df)

    def filter_by_prompt_ids(self, prompt_ids: list[TPromptOriginalIndex]):
        return Prompts(subset_dict_by_keys(self._items, prompt_ids))

    def filter_by_condition(self, condition: Callable[[TPromptOriginalIndex, PromptNew], bool]):
        return self.filter_by_prompt_ids(get_dict_keys_by_condition(self._items, condition))

    @lru_cache(maxsize=5)
    def filter_by_prompt_filteration(self, prompt_filteration: BasePromptFilteration):
        return self.filter_by_prompt_ids(prompt_filteration.get_prompt_ids())

    @property
    def empty(self) -> bool:
        return len(self._items) == 0

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def original_idx(self) -> list[TPromptOriginalIndex]:
        return list(self._items.keys())

    def sample(self, sample_size: int, seed: int) -> Prompts:
        random.seed(seed)
        sampled_indices = random.choices(list(self._items.keys()), k=sample_size)
        return self.filter_by_prompt_ids(sampled_indices)

    def to_df(self) -> TPromptDataFlat:
        """Convert prompts to a DataFrame.

        Returns:
            A DataFrame with all prompt data.
        """
        prompt_dicts = []
        for idx, prompt in self._items.items():
            prompt_dict = prompt._prompt.copy()
            prompt_dict[COLS.ORIGINAL_IDX] = idx
            prompt_dicts.append(prompt_dict)
        return TPromptDataFlat(pd.DataFrame(prompt_dicts))

    def get_prompt(self, prompt_idx: TPromptOriginalIndex) -> Prompt:
        """Get a specific prompt by its original index.

        Args:
            prompt_idx: The prompt original index

        Returns:
            The Prompt object
        """
        return self._items[prompt_idx].as_prompt()

    def to_tokenization_summary_df(self, tokenizers: Tokenizers) -> pd.DataFrame:
        df = self.to_df()

        return df


# Forward References

from src.analysis.experiment_results.model_prompt_combination import ModelCombination  # noqa: E402
from src.analysis.experiment_results.plot_plan import Cell, PlotPlan  # noqa: E402
