import logging
from pathlib import PosixPath
from typing import TypeVar

import pytest
import rich.logging
import submitit

import remote_slurm_executor
from remote_slurm_executor.slurm_remote import get_slurm_account

logging.basicConfig(
    format="%(message)s", level=logging.INFO, handlers=[rich.logging.RichHandler()]
)
logging.getLogger("submitit").setLevel(logging.DEBUG)
T = TypeVar("T", int, float)


def add(a: T, b: T) -> T:
    return a + b


@pytest.fixture(params=["mila"])
def cluster(request: pytest.FixtureRequest) -> str:
    return getattr(request, "param", "mila")


def test_autoexecutor(cluster: str):
    folder = f"logs/{cluster}/%j"
    dont_care_about_reproducibility = True
    executor = submitit.AutoExecutor(
        folder=folder,  # todo: perhaps we can rename this folder?
        cluster="remoteslurm",
        remoteslurm_cluster=cluster,
        remoteslurm_I_dont_care_about_reproducibility=dont_care_about_reproducibility,
    )
    assert isinstance(executor._executor, remote_slurm_executor.RemoteSlurmExecutor)
    assert executor._executor.folder == PosixPath(folder)
    assert executor._executor.cluster_hostname == cluster
    assert (
        executor._executor.I_dont_care_about_reproducibility
        == dont_care_about_reproducibility
    )


@pytest.fixture()
def executor(cluster: str):
    executor = remote_slurm_executor.RemoteSlurmExecutor(
        folder=f"logs/{cluster}/%j",  # todo: perhaps we can rename this folder?
        cluster_hostname=cluster,
        I_dont_care_about_reproducibility=True,
    )

    if cluster != "mila":
        executor.update_parameters(account=get_slurm_account(cluster))

    try:
        yield executor
    finally:
        pass
        assert executor.remote_dir_mount
        # Comment / uncomment to keep the mount after the test to inspect results.
        # executor.remote_dir_mount.unmount()


def test_submit(executor: remote_slurm_executor.RemoteSlurmExecutor):
    # assert False, list(pkg_resources.iter_entry_points("submitit"))
    # the AutoExecutor class is your interface for submitting function to a cluster or run them locally.
    # The specified folder is used to dump job information, logs and result when finished
    # %j is replaced by the job id at runtime

    # The AutoExecutor provides a simple abstraction over SLURM to simplify switching between local and slurm jobs (or other clusters if plugins are available).
    # specify sbatch parameters (here it will timeout after 4min, and run on dev)
    # This is where you would specify `gpus_per_node=1` for instance
    # Cluster specific options must be appended by the cluster name:
    # Eg.: slurm partition can be specified using `slurm_partition` argument. It
    # will be ignored on other clusters:
    executor.update_parameters(partition="long")
    # The submission interface is identical to concurrent.futures.Executor
    job = executor.submit(add, 5, 7)  # will compute add(5, 7)
    print(job.job_id)  # ID of your job

    output = (
        job.result()
    )  # waits for the submitted function to complete and returns its output
    # if ever the job failed, job.result() will raise an error with the corresponding trace
    assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster
    print(output)


def test_map_array(executor: remote_slurm_executor.RemoteSlurmExecutor):
    # You can also map a function on a list of arguments
    a = [1, 2, 3, 4]
    b = [10, 20, 30, 40]
    # the following line tells the scheduler to only run\
    # at most 2 jobs at once. By default, this is several hundreds
    executor.update_parameters(array_parallelism=2)
    jobs = executor.map_array(add, a, b)  # just a list of jobs

    results = [job.result() for job in jobs]
    assert results == [a_i + b_i for a_i, b_i in zip(a, b)]


def test_batch(executor: remote_slurm_executor.RemoteSlurmExecutor):
    A = [1, 2, 3, 4]
    B = [10, 20, 30, 40]
    jobs = []
    with executor.batch():
        for a, b in zip(A, B):
            job = executor.submit(add, a, b)
            jobs.append(job)

    results = [job.result() for job in jobs]
    assert results == [a + b for a, b in zip(A, B)]
