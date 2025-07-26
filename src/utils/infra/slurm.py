import os
import re
import subprocess
from collections import defaultdict
from enum import StrEnum
from functools import cache

import submitit

from src.utils.types_utils import ommit_none

"""
| GPU                       | Speed (TFLOPS) | Memory (GB) |
|---------------------------|----------------|-------------|
| NVIDIA H100-80GB HBM3     | 65.0           | 80          |
| NVIDIA A100-SXM-80GB      | 19.5           | 80          |
| L40S                      | 32.0           | 48          |
| A6000                     | 22.3           | 48          |
| Quadro RTX 8000           | 16.3           | 48          |
| NVIDIA GeForce RTX 3090   | 35.6           | 24          |
| A5000                     | 24.0           | 24          |
| Tesla V100-SXM2-32GB      | 15.7           | 32          |
| NVIDIA GeForce RTX 2080 Ti| 13.4           | 11          |
| Nvidia Titan XP           | 12.1           | 12          |
"""


class SLURM_GPU_TYPE(StrEnum):
    H100 = "h100"
    A100 = "a100"
    L40S = "l40s"
    A6000 = "a6000"
    QUADRO_RTX_8000 = "quadro_rtx_8000"
    GEFORCE_RTX_3090 = "geforce_rtx_3090"
    A5000 = "a5000"
    V100 = "v100"
    GEFORCE_RTX_2080 = "geforce_rtx_2080"
    TITAN_XP_STUDENTRUN = "titan_xp-studentrun"
    TESLA_V100_SXM2_32GB = "tesla_v100_sxm2_32gb"
    TITAN_XP_STUDENTRUN_BATCH = "titan_xp-studentrun-batch"
    TITAN_XP_STUDENTRUN_KILLABLE = "titan_xp-studentrun-killable"

    @property
    def gpu_name(self) -> str:
        match self:
            case (
                SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN
                | SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN_BATCH
                | SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN_KILLABLE
            ):
                return "titan_xp"
            case _:
                return self.value


# ------------------------------------------------------------
# Dynamic, cached lookup built from `sacctmgr` + `sinfo`
# ------------------------------------------------------------
GPU_RE = re.compile(r"gpu:([\w\-]+):")


@cache
def _partition_to_accounts() -> dict[str, set[str]]:
    """
    Query sacctmgr once and return {partition -> {account1, …}} for the
    *current* user.  Requires sacctmgr in $PATH and sufficient perms.
    """
    cmd = [
        "sacctmgr",
        "show",
        "assoc",
        "--parsable2",
        "--noheader",
        f"user={os.environ.get('USER', '')}",
        "format=partition,account",
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).splitlines()
    table: dict[str, set[str]] = defaultdict(set)
    for line in out:
        part, acct = line.strip().split("|")
        table[part].add(acct)
    return table


@cache
def _partition_to_gpus() -> dict[str, set[str]]:
    """
    Query sinfo once and map {partition -> {gpu_constraint1, …}}.
    """
    parts = subprocess.check_output(
        ["sinfo", "--noheader", "--format=%P"], text=True, stderr=subprocess.DEVNULL
    ).split()
    mapping: dict[str, set[str]] = defaultdict(set)
    for p in parts:
        # strip trailing * from default partition
        p_clean = p.rstrip("*")
        gres_lines = subprocess.check_output(
            ["sinfo", "-h", "-o", "%G", "-p", p_clean],
            text=True,
            stderr=subprocess.DEVNULL,
        ).split()
        for gres in gres_lines:
            m = GPU_RE.search(gres)
            if m:
                mapping[p_clean].add(m.group(1))
    return mapping


@cache
def get_partition_account(gpu_type: "SLURM_GPU_TYPE") -> dict[str, str]:
    """
    Return a dict with at least {'partition', 'account'} so that the user
    can submit a job requesting `gpu_type`.  The function consults live
    sacctmgr/sinfo output and is cached for subsequent calls.
    """
    part_to_accts = _partition_to_accounts()
    part_to_gpus = _partition_to_gpus()
    wanted_gpu = gpu_type.gpu_name

    # Prefer partitions in this order if multiple match
    preferred_order = [
        "gpu-h100-killable",
        "killable",
        "gpu-wolf",
        "studentrun",
        "studentbatch",
        "studentkillable",
    ]

    # Build list of candidate partitions that both advertise the GPU AND we have an account on
    candidates = [(p, part_to_accts[p]) for p in part_to_gpus if wanted_gpu in part_to_gpus[p] and p in part_to_accts]

    # Sort by preference list, fallback alphabetical
    candidates.sort(key=lambda x: (preferred_order.index(x[0]) if x[0] in preferred_order else 99, x[0]))

    if not candidates:
        raise ValueError(f"No partition advertises GPU '{wanted_gpu}' that you have access to.")

    chosen_partition, accounts = candidates[0]
    # Pick first account (deterministic order via sorted)
    chosen_account = sorted(accounts)[0]

    return {"partition": chosen_partition, "account": chosen_account}


def submit_job(
    func,
    *args,
    gpu_type: SLURM_GPU_TYPE,
    job_name="test",
    log_folder="log_test/%j",  # %j is replaced by the job id at runtime
    timeout_min=1200,
    memory_required=None,
    slurm_nodes=1,
    tasks_per_node=1,
    slurm_cpus_per_task=1,
    slurm_gpus_per_node=1,
    slurm_nodelist=None,
):
    if gpu_type == SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN:
        timeout_min = 150

    # Determine the appropriate partition and account based on `gpu_type`
    partition_account = get_partition_account(gpu_type)
    slurm_partition = partition_account["partition"]
    slurm_account = partition_account["account"]
    slurm_nodelist = slurm_nodelist or partition_account.get("nodelist", slurm_nodelist)

    # Setup the executor
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name=job_name,
        timeout_min=timeout_min,
        slurm_partition=slurm_partition,
        slurm_account=slurm_account,
        slurm_nodes=slurm_nodes,
        tasks_per_node=tasks_per_node,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_gpus_per_node=slurm_gpus_per_node,
        slurm_mem=memory_required,
        slurm_constraint=gpu_type.gpu_name,
        **ommit_none(
            dict(
                slurm_nodelist=slurm_nodelist,
            )
        ),
    )

    # Submit the job
    job = executor.submit(func, *args)
    return job


def submit_cpu_job(
    func,
    *args,
    job_name="test",
    log_folder="log_test/%j",  # %j is replaced by the job id at runtime
    timeout_min=1200,
    memory_required=None,
    slurm_nodes=1,
    tasks_per_node=1,
    slurm_cpus_per_task=1,
):
    # Map GPU type and account type to partition and account options based on `sinfo` data
    partition_account = "studentbatch"
    account = "gpu-students"

    # Setup the executor
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name=job_name,
        timeout_min=timeout_min,
        slurm_partition=partition_account,
        slurm_account=account,
        slurm_nodes=slurm_nodes,
        tasks_per_node=tasks_per_node,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_mem=memory_required,
    )

    # Submit the job
    job = executor.submit(func, *args)
    return job
