from abc import ABC, abstractmethod
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Generic, Optional, Sequence, Type, TypeVar, cast

from src.analysis.prompt_filterations import AnyExistingCompletePromptFilteration
from src.core.consts import (
    PATHS,
)
from src.core.names import (
    BaseVariantParamName,
    ExperimentName,
    ResultBankParamNames,
    ToClassifyNames,
    WindowedVariantParam,
)
from src.core.types import (
    TCodeVersionName,
)
from src.data_ingestion.data_defs.data_defs import ResultBank
from src.experiments.infrastructure.base_runner import BaseRunner, BaseVariantParams, InputParams, MetadataParams
from src.experiments.runners.evaluate_model import EvaluateModelParams, EvaluateModelRunner
from src.experiments.runners.heatmap import HeatmapParams, HeatmapRunner
from src.experiments.runners.info_flow import InfoFlowParams, InfoFlowRunner
from src.utils.infra.output_path import OutputPath


class RESULTS_BASE_PATH(IntEnum):
    CURRENT = auto()
    TEST = auto()

    @property
    def path(self) -> Path:
        if self == RESULTS_BASE_PATH.TEST:
            return PATHS.PROJECT_DIR / "tests/src/experiments/baselines/full_pipeline/output"
        else:
            return PATHS.OUTPUT_DIR


_Runner = TypeVar("_Runner", bound=BaseRunner)


class ValueResolver(ABC, Generic[_Runner]):
    @classmethod
    @abstractmethod
    def get_experiment_runner_cls(cls) -> Type[_Runner]:
        pass

    @classmethod
    @abstractmethod
    def get_experiment_variant_params_cls(cls) -> Type[BaseVariantParams]:
        pass

    @classmethod
    def get_experiment_name(cls) -> ExperimentName:
        return cls.get_experiment_runner_cls()._get_variant_params().experiment_name

    @classmethod
    def get_results_output_path(cls, results_base_path: Path) -> OutputPath:
        return OutputPath(
            base_path=results_base_path,
            path_components=cls.get_experiment_runner_cls().get_variant_output_keys(),
        ).enforce_value(BaseVariantParamName.experiment_name, cls.get_experiment_name())

    @classmethod
    def process_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        return values

    @classmethod
    @abstractmethod
    def init_from_processed_values(
        cls, values: dict[str, str], path: Path, results_base_path: Path
    ) -> Optional[_Runner]:
        values.pop("_", None)
        code_version = TCodeVersionName(values.pop(ResultBankParamNames.code_version))
        values.pop(ToClassifyNames.dataset_name)
        values = cls.process_values(values)
        input_params = InputParams(filteration=AnyExistingCompletePromptFilteration())
        metadata_params = MetadataParams(
            code_version=code_version, override_base_project_dir=str(results_base_path.parent)
        )

        return cls.get_experiment_runner_cls()(
            variant_params=cls.get_experiment_variant_params_cls()(**cast(dict, values)),
            input_params=input_params,
            metadata_params=metadata_params,
        )

    @classmethod
    @abstractmethod
    def is_valid_record(cls, runner: _Runner) -> bool:
        pass


class EvaluateModelValuesResolver(ValueResolver):
    @classmethod
    def get_experiment_runner_cls(cls) -> Type[BaseRunner]:
        return EvaluateModelRunner

    @classmethod
    def get_experiment_variant_params_cls(cls) -> Type[BaseVariantParams]:
        return EvaluateModelParams

    @classmethod
    def is_valid_record(cls, runner: EvaluateModelRunner) -> bool:
        return runner.output_result_path.exists()


class HeatmapValuesResolver(ValueResolver):
    @classmethod
    def get_experiment_runner_cls(cls) -> Type[BaseRunner]:
        return HeatmapRunner

    @classmethod
    def process_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        values[WindowedVariantParam.window_size] = int(values[WindowedVariantParam.window_size])
        return values

    @classmethod
    def get_experiment_variant_params_cls(cls) -> Type[BaseVariantParams]:
        return HeatmapParams

    @classmethod
    def is_valid_record(cls, runner: HeatmapRunner) -> bool:
        return runner.output_hdf5_path.path.exists()


class InfoFlowValuesResolver(ValueResolver):
    @classmethod
    def get_experiment_runner_cls(cls):
        return InfoFlowRunner

    @classmethod
    def get_experiment_variant_params_cls(cls) -> Type[BaseVariantParams]:
        return InfoFlowParams

    @classmethod
    def process_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        values[WindowedVariantParam.window_size] = int(values[WindowedVariantParam.window_size])
        return values

    @classmethod
    def is_valid_record(cls, runner: InfoFlowRunner) -> bool:
        return runner.output_file.path.exists()


def get_experiment_results_bank(
    results_base_paths: Sequence[RESULTS_BASE_PATH] = (RESULTS_BASE_PATH.CURRENT,),
    experiment_records: Sequence[Type[ValueResolver]] = (
        EvaluateModelValuesResolver,
        HeatmapValuesResolver,
        InfoFlowValuesResolver,
    ),
) -> ResultBank:
    results: list[BaseRunner] = []
    for results_base_path in results_base_paths:
        for experiment_record in experiment_records:
            output_path = experiment_record.get_results_output_path(results_base_path.path)
            in_pattern, _ = output_path.process_path()
            for path, values in in_pattern:
                result_record = experiment_record.init_from_processed_values(
                    values=values,
                    results_base_path=results_base_path.path,
                    path=path,
                )
                if result_record and experiment_record.is_valid_record(result_record):
                    results.append(result_record)
    return ResultBank(results)
