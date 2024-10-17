import logging
from pathlib import PosixPath, PurePosixPath

import pytest
import rich.logging
import submitit

import remote_slurm_executor

logging.basicConfig(
    format="%(message)s", level=logging.INFO, handlers=[rich.logging.RichHandler()]
)
logging.getLogger("submitit").setLevel(logging.DEBUG)


def add(a, b):
    return a + b


@pytest.fixture(params=["mila", "cedar", "narval"])
def cluster(request: pytest.FixtureRequest) -> str:
    return getattr(request, "param", "mila")


def test_autoexecutor(cluster: str):
    folder = f"logs/{cluster}/%j"
    repo_dir_on_cluster = "repos/remote-submitit-launcher"
    dont_care_about_reproducibility = True
    executor = submitit.AutoExecutor(
        folder=folder,  # todo: perhaps we can rename this folder?
        cluster="remoteslurm",
        remoteslurm_cluster=cluster,
        remoteslurm_repo_dir_on_cluster=repo_dir_on_cluster,
        remoteslurm_I_dont_care_about_reproducibility=dont_care_about_reproducibility,
    )
    assert isinstance(executor._executor, remote_slurm_executor.RemoteSlurmExecutor)
    assert executor._executor.folder == PosixPath(folder)
    assert executor._executor.cluster == cluster
    assert executor._executor.repo_dir == PurePosixPath(repo_dir_on_cluster)
    assert (
        executor._executor.I_dont_care_about_reproducibility
        == dont_care_about_reproducibility
    )


@pytest.fixture()
def executor(cluster: str):

    executor = remote_slurm_executor.RemoteSlurmExecutor(
        folder="logs/%j",  # todo: perhaps we can rename this folder?
        cluster=cluster,
        repo_dir_on_cluster="repos/remote-submitit-launcher",
        I_dont_care_about_reproducibility=True,
    )
    try:
        yield executor
    finally:
        assert executor.remote_dir_mount
        executor.remote_dir_mount.unmount()


def test_add(executor: remote_slurm_executor.RemoteSlurmExecutor):
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
