from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, cast

import pandas as pd

from src.analysis.prompt_filterations import AllPromptFilteration
from src.core.names import BaseVariantParamName, DatasetName, ExperimentName
from src.core.types import MODEL_ARCH_AND_SIZE, TCodeVersionName, TPromptData
from src.data_ingestion.datasets.download_dataset import flat_to_indexed_prompt_data
from src.experiments.infrastructure.base_prompt_filteration import BasePromptFilteration
from src.experiments.infrastructure.base_runner import (
    BaseRunner,
    BaseVariantParams,
    InputParams,
    MetadataParams,
)
from src.experiments.runners.evaluate_model import EvaluateModelParams, EvaluateModelRunner
from src.experiments.runners.heatmap import HeatmapParams, HeatmapRunner
from src.experiments.runners.info_flow import InfoFlowParams, InfoFlowRunner

if TYPE_CHECKING:
    from src.data_ingestion.data_defs.data_defs import ResultBank


def get_model_evaluations(
    code_version: TCodeVersionName, model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
) -> dict[MODEL_ARCH_AND_SIZE, TPromptData]:
    return {
        model_arch_and_size: flat_to_indexed_prompt_data(
            EvaluateModelRunner(
                variant_params=EvaluateModelParams(
                    model_arch=model_arch_and_size[0],
                    model_size=model_arch_and_size[1],
                ),
                input_params=InputParams(
                    filteration=AllPromptFilteration(dataset_name=DatasetName.counter_fact),
                ),
                metadata_params=MetadataParams(
                    code_version=code_version,
                ),
            ).get_outputs()
        )
        for model_arch_and_size in model_arch_and_sizes
    }


def init_variant_params_from_values(dict_values: dict) -> BaseVariantParams:
    experiment_name = cast(ExperimentName, dict_values.pop(BaseVariantParamName.experiment_name))
    match experiment_name:
        case ExperimentName.evaluate_model:
            return EvaluateModelParams(**dict_values)
        case ExperimentName.info_flow:
            return InfoFlowParams(**dict_values)
        case ExperimentName.heatmap:
            return HeatmapParams(**dict_values)
        case _:
            raise ValueError(f"Unsupported experiment name: {experiment_name}")


def init_runner_from_params(
    variant_params: BaseVariantParams,
    input_params: InputParams,
    metadata_params: MetadataParams,
) -> BaseRunner:
    match variant_params:
        case EvaluateModelParams():
            return EvaluateModelRunner(
                variant_params=variant_params,
                input_params=input_params,
                metadata_params=metadata_params,
            )
        case InfoFlowParams():
            return InfoFlowRunner(
                variant_params=variant_params,
                input_params=input_params,
                metadata_params=metadata_params,
            )
        case HeatmapParams():
            return HeatmapRunner(
                variant_params=variant_params,
                input_params=input_params,
                metadata_params=metadata_params,
            )
        case _:
            raise ValueError(f"Unsupported variant params: {variant_params}")


def serialize_result_bank(result_bank: ResultBank) -> str:
    def rec_serialize_dependencies(item):
        if isinstance(item, BaseRunner):
            return [
                item.variant_params.experiment_name,
                rec_serialize_dependencies(asdict(item.variant_params)),
                rec_serialize_dependencies(asdict(item.input_params)),
                rec_serialize_dependencies(item.get_outputs()),
            ]
        if isinstance(item, dict):
            res = {}
            for k, v in item.items():
                if isinstance(k, tuple):
                    k = str(k)
                res[k] = rec_serialize_dependencies(v)
            return res
        elif isinstance(item, list):
            return [rec_serialize_dependencies(v) for v in item]
        elif isinstance(item, BasePromptFilteration):
            return {item.__class__.__name__: rec_serialize_dependencies(asdict(item))}
        elif isinstance(item, pd.DataFrame):
            return item.to_dict()
        else:
            return item

    def sort_key(item):
        assert len(item) == 4
        experiment_name: ExperimentName = item[0]
        variant_params = item[1]
        return tuple([variant_params.get(col) for col in ExperimentName.get_variant_cols(experiment_name)])

    return json.dumps(sorted(rec_serialize_dependencies([item for item in result_bank]), key=sort_key), indent=4)
