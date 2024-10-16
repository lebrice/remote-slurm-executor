import logging

import rich.logging

# import submitit
# import submitit.slurm_remote
from remote_slurm_launcher import RemoteSlurmExecutor

logging.basicConfig(
    format="%(message)s", level=logging.INFO, handlers=[rich.logging.RichHandler()]
)
logging.getLogger("submitit").setLevel(logging.DEBUG)


def add(a, b):
    return a + b


def test_add(tmp_path):
    # the AutoExecutor class is your interface for submitting function to a cluster or run them locally.
    # The specified folder is used to dump job information, logs and result when finished
    # %j is replaced by the job id at runtime
    executor = RemoteSlurmExecutor(
        cluster="mila",
        repo_dir_on_cluster="repos/submitit",
        folder="logs",  # todo: perhaps we can rename this folder?
        I_dont_care_about_reproducibility=True,
    )
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
    executor.remote_dir_mount.unmount()
