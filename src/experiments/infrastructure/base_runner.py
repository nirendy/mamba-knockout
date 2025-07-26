from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    assert_never,
    cast,
    final,
)

from pydantic import Field

from src.core.consts import (
    BASE_OUTPUT_KEYS,
    MODEL_SIZES_PER_ARCH_TO_MODEL_ID,
    PATHS,
    PathsConfig,
    RunnerPaths,
)
from src.core.names import BaseVariantParamName, ExperimentName, RunningHistoryCols, ToClassifyNames
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    TBatchSize,
    TCodeVersionName,
    TModelID,
    TModelSize,
    TTokenizer,
)
from src.data_ingestion.datasets.download_dataset import DatasetName
from src.experiments.infrastructure.base_prompt_filteration import BasePromptFilteration
from src.experiments.infrastructure.model_interface import ModelInterface, get_model_interface
from src.experiments.infrastructure.setup_models import get_tokenizer
from src.utils.infra.experiment_helper import create_run_id
from src.utils.infra.git import get_git_commit_hash
from src.utils.infra.output_path import OutputKey, combine_output_keys
from src.utils.infra.slurm import SLURM_GPU_TYPE, submit_job
from src.utils.infra.slurm_job_folder import ExperimentHistorySlurmJobsFolder
from src.utils.types_utils import BaseParams, json_dumps_dataclass, str_enum_values

TDependencies = Mapping[str, Union["BaseRunner", "TDependencies"]]


@dataclass(frozen=True)
class BaseVariantParams(BaseParams):
    experiment_name: ClassVar[ExperimentName] = Field(init=False)
    model_arch: MODEL_ARCH
    model_size: TModelSize

    @property
    def model_arch_and_size(self) -> MODEL_ARCH_AND_SIZE:
        return MODEL_ARCH_AND_SIZE(self.model_arch, self.model_size)

    def get_model_interface(self) -> ModelInterface:
        return get_model_interface(self.model_arch_and_size)

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.model_arch][self.model_size]

    @property
    def get_tokenizer(self) -> TTokenizer:
        return get_tokenizer(self.model_arch, self.model_size)

    def should_skip_task(self) -> bool:
        return False


@dataclass(frozen=True)
class InputParams(BaseParams):
    filteration: BasePromptFilteration
    dataset_name: DatasetName = DatasetName.counter_fact


@dataclass(frozen=True)
class MetadataParams(BaseParams):
    code_version: TCodeVersionName
    requested_batch_size: TBatchSize = TBatchSize(1)  # Adjust based on GPU memory
    with_slurm: bool = False
    # slurm_gpu_type: SLURM_GPU_TYPE = SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN
    slurm_gpu_type: SLURM_GPU_TYPE = SLURM_GPU_TYPE.L40S
    slurm_gpus_per_node: int = 1
    overwrite_existing_outputs: bool = False
    override_base_project_dir: Optional[str] = None


_TVariantParams = TypeVar("_TVariantParams", bound=BaseVariantParams)


@dataclass(frozen=True)
class BaseRunner(BaseParams, ABC, Generic[_TVariantParams]):
    """Base configuration class with common parameters across all scripts."""

    variant_params: _TVariantParams
    input_params: "InputParams"
    metadata_params: MetadataParams

    @property
    def experiment_name(self) -> ExperimentName:
        return self.variant_params.experiment_name

    @staticmethod
    @abstractmethod
    def _get_variant_params() -> Type[_TVariantParams]:
        pass

    @classmethod
    def init_from_runner(
        cls,
        runner: "BaseRunner",
        variant_params: _TVariantParams,
        input_params: Optional[InputParams] = None,
        metadata_params: Optional[MetadataParams] = None,
    ):
        return cls(
            variant_params=variant_params,
            input_params=input_params or runner.input_params,
            metadata_params=metadata_params or runner.metadata_params,
        )

    @property
    def global_path_config(self) -> PathsConfig:
        return PathsConfig(PROJECT_DIR=Path(self.metadata_params.override_base_project_dir or PATHS.PROJECT_DIR))

    @property
    def effective_batch_size(self) -> TBatchSize:
        assert self.metadata_params.requested_batch_size == 1, "Batch size must be 1, unless we debug the issue"
        return (
            TBatchSize(1)
            if (self.variant_params.model_arch == MODEL_ARCH.MAMBA2)
            else self.metadata_params.requested_batch_size
        )

    @classmethod
    def get_variant_output_keys(cls) -> list[OutputKey]:
        return [
            BASE_OUTPUT_KEYS.EXPERIMENT_NAME,
            BASE_OUTPUT_KEYS.CODE_VERSION,
            BASE_OUTPUT_KEYS.MODEL_ARCH,
            BASE_OUTPUT_KEYS.MODEL_SIZE,
            BASE_OUTPUT_KEYS.DATASET_NAME,
        ]

    @property
    def shared_param_namespace(self):
        class CombineParams:
            @classmethod
            def __getattr__(cls, item: str) -> Any:
                if item == ToClassifyNames.prompt_filteration:
                    return self.input_params.filteration
                elif item == ToClassifyNames.dataset_name:
                    return self.input_params.dataset_name
                elif item == ToClassifyNames.code_version:
                    return self.metadata_params.code_version
                if item in str_enum_values(BaseVariantParamName):
                    item = cast(BaseVariantParamName, item)
                    match item:
                        case BaseVariantParamName.experiment_name:
                            return self.experiment_name
                        case BaseVariantParamName.model_arch:
                            return self.variant_params.model_arch
                        case BaseVariantParamName.model_size:
                            return self.variant_params.model_size
                        case _:
                            assert_never(item)
                else:
                    return getattr(self.variant_params, item)

        return CombineParams()

    @lru_cache(maxsize=None)
    def combine_output_keys(self, sep: str) -> str:
        return combine_output_keys(
            self.shared_param_namespace,
            self.get_variant_output_keys(),
            sep=sep,
        )

    @property
    def variation_relative_path(self) -> Path:
        return Path(".") / self.combine_output_keys(sep="/")

    @final
    @property
    def variation_paths(self) -> RunnerPaths:
        return RunnerPaths(self.global_path_config.OUTPUT_DIR / self.variation_relative_path)

    @property
    def job_name(self) -> str:
        return self.combine_output_keys(sep="_")

    @abstractmethod
    def get_runner_dependencies(self) -> TDependencies:
        pass

    @abstractmethod
    def _compute_impl(self) -> None:
        pass

    @abstractmethod
    def is_computed(self) -> bool:
        pass

    @abstractmethod
    def get_outputs(self) -> Any:
        pass

    def uncomputed_dependencies(self) -> list["BaseRunner"]:
        def rec_uncomputed_dependencies(dependencies: TDependencies) -> list["BaseRunner"]:
            res = []
            for k, v in dependencies.items():
                if isinstance(v, BaseRunner):
                    if not v.is_computed():
                        res.append(v)
                else:
                    res.extend(rec_uncomputed_dependencies(v))
            return res

        return rec_uncomputed_dependencies(self.get_runner_dependencies())

    def dependencies_are_computed(self) -> bool:
        return len(self.uncomputed_dependencies()) == 0

    def create_experiment_dir(self) -> None:
        self.variation_paths.running_history_path.mkdir(parents=True, exist_ok=True)
        self.variation_paths.plots_path.mkdir(parents=True, exist_ok=True)
        self.variation_paths.outputs_path.mkdir(parents=True, exist_ok=True)

        run_id = create_run_id(None)

        params = asdict(self)
        params[RunningHistoryCols.run_id] = run_id
        params[RunningHistoryCols.git_commit_hash] = get_git_commit_hash()

        self.variation_paths.running_history_json_path(run_id).write_text(json_dumps_dataclass(params, indent=4))

    def compute_dependencies(self, rec_depth: int = -1) -> None:
        """
        Compute the dependencies of the current runner.
        if rec_depth is N, only the N-th level dependencies will be computed.
        If rec_depth is 0 nothing will be computed.
        if rec_depth is -1, all the way down dependencies will be computed.
        """
        if rec_depth == 0:
            return

        def rec_compute_with_dependencies(dependencies: TDependencies) -> None:
            for dependency in dependencies.values():
                if isinstance(dependency, BaseRunner):
                    dependency.compute_dependencies(rec_depth - 1)
                    dependency.run(with_dependencies=True)
                else:
                    rec_compute_with_dependencies(dependency)

        rec_compute_with_dependencies(self.get_runner_dependencies())

    def run(self, with_dependencies: bool) -> None:
        if self.variant_params.should_skip_task():
            return
        if self.is_computed():
            return
        if not self.dependencies_are_computed():
            if with_dependencies:
                self.compute_dependencies(1)
            else:
                raise ValueError("Dependencies are not computed")

        if not self.metadata_params.with_slurm:
            self._compute_impl()
            return
        else:
            if self.variant_params.should_skip_task():
                return
            job = submit_job(
                self._compute_impl,
                log_folder=str(self.global_path_config.get_slurm_job_log_folder(self.job_name, "%j")),
                job_name=self.job_name,
                # timeout_min=1200,
                gpu_type=self.metadata_params.slurm_gpu_type,
                slurm_gpus_per_node=self.metadata_params.slurm_gpus_per_node,
            )
            self.variation_paths.slurm_logs_path.mkdir(parents=True, exist_ok=True)
            # create symlink to slurm logs
            (self.variation_paths.slurm_log_folder(job_id=job.job_id)).symlink_to(
                self.global_path_config.get_slurm_job_log_folder(self.job_name, job.job_id)
            )

            self.global_path_config.get_slurm_job_submission_file_path(self.job_name, job.job_id).symlink_to(
                self.variation_paths.variation_base_path
            )

            print(f"{job}: {self.job_name}")

    @property
    def slurm_job_folder(self) -> ExperimentHistorySlurmJobsFolder:
        return ExperimentHistorySlurmJobsFolder(self.variation_paths.slurm_logs_path)
