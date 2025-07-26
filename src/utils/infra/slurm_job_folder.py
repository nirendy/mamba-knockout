from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cachetools import TTLCache, cached
from submitit.slurm.slurm import SlurmJob

from src.core.names import SlurmStatus


@dataclass(frozen=True)
class SlurmJobFolder:
    job_id: str
    path: Path

    @property
    def slurm_job(self) -> SlurmJob:
        return SlurmJob(folder=self.path, job_id=self.job_id)

    @cached(TTLCache(maxsize=1, ttl=60))
    def get_slurm_status(self) -> SlurmStatus | str:
        state = self.slurm_job.state
        try:
            return SlurmStatus[state]
        except KeyError:
            return state

    @property
    def slurm_job_output_path(self) -> Path:
        return self.path / f"{self.job_id}_0_log.out"

    @property
    def slurm_job_error_path(self) -> Path:
        return self.path / f"{self.job_id}_0_log.err"

    def get_slurm_job_output(self) -> str:
        return self.slurm_job_output_path.read_text()

    def get_slurm_job_error(self) -> str:
        return self.slurm_job_error_path.read_text()


@dataclass(frozen=True)
class ExperimentHistorySlurmJobsFolder:
    path: Path

    def get_all_slurm_jobs(self) -> list[SlurmJobFolder]:
        if not self.path.exists():
            return []
        job_paths = list(self.path.glob("*"))
        if not job_paths:
            return []

        return [SlurmJobFolder(job_path.stem, job_path) for job_path in job_paths]

    def get_latest_slurm_job(self) -> Optional[SlurmJobFolder]:
        all_jobs = self.get_all_slurm_jobs()
        if not all_jobs:
            return None
        return max(all_jobs, key=lambda x: int(x.job_id))

    def get_latest_slurm_job_status(self) -> SlurmStatus | str:
        latest_job = self.get_latest_slurm_job()
        if latest_job is None:
            return SlurmStatus.NOT_SUBMITTED
        return latest_job.get_slurm_status()
