# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import itertools
import logging
import shlex
import subprocess
import sys
import typing as tp
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import ClassVar

from milatools.utils.local_v2 import LocalV2
from milatools.utils.remote_v2 import RemoteV2
from submitit.core import core, utils
from submitit.slurm import slurm
from submitit.slurm.slurm import SlurmInfoWatcher, SlurmJobEnvironment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteDir:
    login_node: RemoteV2
    remote_dir: PurePosixPath | str
    local_dir: Path

    def mount(self):
        self.login_node.run(f"mkdir -p {self.remote_dir}")
        self.local_dir.mkdir(exist_ok=True, parents=True)
        LocalV2.run(
            (
                "sshfs",
                f"{self.login_node.hostname}:{self.remote_dir}",
                str(self.local_dir),
            ),
            display=True,
        )

        logger.info(
            f"Remote dir {self.login_node.hostname}:{self.remote_dir} is now mounted at {self.local_dir}"
        )

    def unmount(self):
        LocalV2.run(("fusermount", "--unmount", str(self.local_dir)), display=True)

    @contextmanager
    def context(self):
        if not self.is_mounted():
            self.mount()
        yield
        self.unmount()

    def is_mounted(self) -> bool:
        # Check mounted filesystems using the 'mount' command
        output = LocalV2.get_output("mount")
        # Search for the specific mount point in the output
        if any(
            f"{self.login_node.hostname}:{self.remote_dir} on {self.local_dir.absolute()} type fuse.sshfs"
            in line
            for line in output.splitlines()
        ):
            return True
        return False


class RemoteSlurmInfoWatcher(SlurmInfoWatcher):
    def __init__(self, cluster: str, delay_s: int = 60) -> None:
        super().__init__(delay_s)
        self.cluster = cluster

    def _make_command(self) -> tp.Optional[tp.List[str]]:
        # asking for array id will return all status
        # on the other end, asking for each and every one of them individually takes a huge amount of time
        cmd = super()._make_command()
        if not cmd:
            return None
        return ["ssh", self.cluster] + cmd


def get_first_id_independent_folder(folder: tp.Union[PurePath, str]) -> PurePosixPath:
    """Returns the closest folder which is id independent"""
    parts = PurePath(folder).parts
    tags = ["%j", "%t", "%A", "%a"]
    indep_parts = itertools.takewhile(
        lambda x: not any(tag in x for tag in tags), parts
    )
    return PurePosixPath(*indep_parts)


class RemoteSlurmJob(core.Job[core.R]):
    _cancel_command = "scancel"
    watchers: ClassVar[dict[str, RemoteSlurmInfoWatcher]] = {}
    watcher: RemoteSlurmInfoWatcher

    def __init__(
        self,
        cluster: str,
        folder: tp.Union[str, Path],
        job_id: str,
        tasks: tp.Sequence[int] = (0,),
    ) -> None:
        self.cluster = cluster
        # watcher*s*, since this could be using different clusters.
        # Also: `watcher` is now an instance variable, not a class variable.
        self.watcher = type(self).watchers.setdefault(
            self.cluster, RemoteSlurmInfoWatcher(cluster=cluster, delay_s=600)
        )
        super().__init__(folder=folder, job_id=job_id, tasks=tasks)

    def _interrupt(self, timeout: bool = False) -> None:
        """Sends preemption or timeout signal to the job (for testing purpose)

        Parameter
        ---------
        timeout: bool
            Whether to trigger a job time-out (if False, it triggers preemption)
        """
        cmd = ["ssh", self.cluster, "scancel", self.job_id, "--signal"]
        # in case of preemption, SIGTERM is sent first
        if not timeout:
            subprocess.check_call(cmd + ["SIGTERM"])
        subprocess.check_call(cmd + [SlurmJobEnvironment.USR_SIG])

    def cancel(self, check: bool = True) -> None:
        (subprocess.check_call if check else subprocess.call)(
            ["ssh", self.cluster, self._cancel_command, f"{self.job_id}"], shell=False
        )


class RemoteSlurmExecutor(slurm.SlurmExecutor):
    """Executor for a remote SLURM cluster.

    - Installs `uv` on the remote cluster.
    - Syncs dependencies with `uv sync --all-extras` on the login node.
    """

    job_class: ClassVar[type[RemoteSlurmJob]] = RemoteSlurmJob

    def __init__(
        self,
        folder: PurePath | str,
        *,
        cluster: str,
        repo_dir_on_cluster: str | PurePosixPath,
        internet_access_on_compute_nodes: bool = True,
        max_num_timeout: int = 3,
        python: str | None = None,
        I_dont_care_about_reproducibility: bool = False,
    ) -> None:
        self._original_folder = folder  # save this argument that we'll modify.

        # Example: `folder="logs_test/%j"`

        # Locally:
        # ./logs_test/mila/%j
        # Remote:
        # $SCRATCH/.submitit/logs_test/%j

        self.cluster = cluster
        self.repo_dir = PurePosixPath(repo_dir_on_cluster)
        self.internet_access_on_compute_nodes = internet_access_on_compute_nodes
        self.I_dont_care_about_reproducibility = I_dont_care_about_reproducibility

        folder = Path(folder)
        assert not folder.is_absolute()
        self.login_node = RemoteV2(self.cluster)

        # "base" folder := dir without any %j %t, %A, etc.
        base_folder = get_first_id_independent_folder(folder)
        rest_of_folder = folder.relative_to(base_folder)

        self.local_base_folder = Path(base_folder)
        self.local_folder = self.local_base_folder / rest_of_folder

        # todo: include our hostname / something unique so we don't overwrite anything on the
        # remote?
        self.remote_base_folder = (
            PurePosixPath(self.login_node.get_output("echo $SCRATCH"))
            / ".submitit"
            / base_folder
        )
        self.remote_folder = self.remote_base_folder / rest_of_folder

        self.remote_dir_mount: RemoteDir | None = RemoteDir(
            self.login_node,
            remote_dir=self.remote_base_folder,
            local_dir=self.local_base_folder,
        )

        assert python is None, "TODO: Can't use something else than uv for now."
        self._uv_path: str = self.setup_uv()
        _python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        python = f"{self._uv_path} run --python={_python_version} python"
        if not self.remote_dir_mount.is_mounted():
            self.remote_dir_mount.mount()

        super().__init__(
            folder=self.local_folder, max_num_timeout=max_num_timeout, python=python
        )

        # No need to make it absolute. Revert it back to a relative path?
        assert not self.local_folder.is_absolute()
        assert self.folder == self.local_folder.absolute(), (
            self.folder,
            self.local_folder.absolute(),
        )
        self.folder = self.local_folder

        self.sync_source_code()
        self.sync_dependencies()

        # chdir to the repo so that `uv run` uses the dependencies, etc.
        self.update_parameters(
            srun_args=[f"--chdir={self.repo_dir}"], stderr_to_stdout=True
        )

    def sync_dependencies(self):
        # if not self.internet_access_on_compute_nodes:
        #     logger.info("Syncing the dependencies on the login node.")
        self.login_node.run(f"cd {self.repo_dir} && {self._uv_path} sync --all-extras")
        # IDEA: IF there is internet access on the compute nodes, then perhaps we could sync the
        # dependencies on a compute node?

    def sync_source_code(self):
        # IDEA: Could also mount a folder with sshfs and then use a
        # `git clone . /path/to/mount/source` to sync the source code.
        #  + the job can't break because of a change in the source code.
        #  - Not as good for reproducibility: not forcing the user to commit and push the code..

        if not self.I_dont_care_about_reproducibility:
            if LocalV2.get_output("git status --porcelain"):
                print(
                    "You have uncommitted changes, please commit and push them before trying again.",
                    file=sys.stderr,
                )
                exit()
            LocalV2.run("git push")

        current_branch = LocalV2.get_output("git rev-parse --abbrev-ref HEAD")
        current_commit = LocalV2.get_output("git rev-parse HEAD")
        repo_url = LocalV2.get_output("git config --get remote.origin.url")

        # If the repo doesn't exist on the remote, clone it:
        if self.login_node.run(
            f"test -d {self.repo_dir}", warn=True, hide=True
        ).returncode:
            self.login_node.run(
                f"git clone {repo_url} -b {current_branch} {self.repo_dir}"
            )
        self.login_node.run(
            f"cd {self.repo_dir} && git fetch && git checkout {current_branch} && git pull"
        )
        if not self.I_dont_care_about_reproducibility:
            self.login_node.run(f"cd {self.repo_dir} && git checkout {current_commit}")

    def setup_uv(self) -> str:
        if not (uv_path := self._get_uv_path()):
            logger.info(
                f"Setting up [uv](https://docs.astral.sh/uv/) on {self.cluster}"
            )
            self.login_node.run(
                "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.cargo/env"
            )
            uv_path = self._get_uv_path()
            if uv_path is None:
                raise RuntimeError(
                    f"Unable to setup `uv` on the {self.cluster} cluster!"
                )
        return uv_path

    def _get_uv_path(self) -> str | None:
        return (
            LocalV2.get_output(
                ("ssh", self.cluster, "which", "uv"),
                warn=True,
            )
            or LocalV2.get_output(
                ("ssh", self.cluster, "bash", "-l", "which", "uv"),
                warn=True,
            )
            or None
        )

    @property
    def _submitit_command_str(self) -> str:
        # Changed from the base class: Points to the remote folder instead of the local folder.
        # Also: `self.python` is `uv run --python=X.Y python`
        # return " ".join([self.python, "-u -m submitit.core._submit", shlex.quote(str(self.folder))])
        return f"{self.python} -u -m submitit.core._submit {shlex.quote(str(self.remote_folder))}"

    def _submit_command(self, command: str) -> core.Job:
        # Copied and adapted from PicklingExecutor.
        tmp_uuid = uuid.uuid4().hex
        local_submission_file_path = Path(
            self.local_base_folder / f".submission_file_{tmp_uuid}.sh"
        )
        remote_submission_file_path = (
            self.remote_base_folder / f".submission_file_{tmp_uuid}.sh"
        )

        with local_submission_file_path.open("w") as f:
            f.write(self._make_submission_file_text(command, tmp_uuid))

        # remote_content = self.login_node.get_output(
        #     f"cat {remote_submission_file_path}"
        # )
        # local_content = local_submission_file_path.read_text()

        command_list = self._make_submission_command(remote_submission_file_path)

        output = utils.CommandFunction(command_list, verbose=False)()  # explicit errors
        job_id = self._get_job_id_from_submission_command(output)
        tasks_ids = list(range(self._num_tasks()))

        job = self.job_class(
            cluster=self.cluster,
            folder=self.local_folder,
            job_id=job_id,
            tasks=tasks_ids,
        )
        # Equivalent of `_move_temporarity_file` call (expanded to be more explicit):
        # job.paths.move_temporary_file(
        #     local_submission_file_path, "submission_file", keep_as_symlink=True
        # )
        # Local submission file.
        job.paths.submission_file.parent.mkdir(parents=True, exist_ok=True)
        local_submission_file_path.rename(job.paths.submission_file)
        # Might not work!
        local_submission_file_path.symlink_to(job.paths.submission_file)
        # TODO: The rest here isn't used?
        self._write_job_id(job.job_id, tmp_uuid)
        self._set_job_permissions(job.paths.folder)
        return job

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[utils.DelayedSubmission]
    ) -> tp.List[core.Job[tp.Any]]:
        logger.info(f"Processing {len(delayed_submissions)} submissions")
        logger.debug(delayed_submissions[0])
        if len(delayed_submissions) == 1:
            # TODO: Why is this case here?
            return super()._internal_process_submissions(delayed_submissions)

        # array
        folder = utils.JobPaths.get_first_id_independent_folder(self.folder)
        folder.mkdir(parents=True, exist_ok=True)
        timeout_min = self.parameters.get("time", 5)
        pickle_paths = []
        for d in delayed_submissions:
            pickle_path = folder / f"{uuid.uuid4().hex}.pkl"
            d.set_timeout(timeout_min, self.max_num_timeout)
            d.dump(pickle_path)
            pickle_paths.append(pickle_path)
        n = len(delayed_submissions)
        # Make a copy of the executor, since we don't want other jobs to be
        # scheduled as arrays.
        array_ex = type(self)(
            cluster=self.cluster,
            repo_dir=self.repo_dir,
            folder=self._original_folder,
            max_num_timeout=self.max_num_timeout,
        )
        array_ex.update_parameters(**self.parameters)
        array_ex.parameters["map_count"] = n
        self._throttle()

        first_job: core.Job[tp.Any] = array_ex._submit_command(
            self._submitit_command_str
        )
        tasks_ids = list(range(first_job.num_tasks))
        jobs: tp.List[core.Job[tp.Any]] = [
            RemoteSlurmJob(
                cluster=self.cluster,
                folder=self.folder,
                job_id=f"{first_job.job_id}_{a}",
                tasks=tasks_ids,
            )
            for a in range(n)
        ]
        for job, pickle_path in zip(jobs, pickle_paths):
            job.paths.move_temporary_file(pickle_path, "submitted_pickle")
        return jobs

    def _make_submission_file_text(self, command: str, uid: str) -> str:
        # todo: there might still be issues with absolute paths with this folder here!
        return _make_sbatch_string(
            command=command, folder=self.remote_folder, **self.parameters
        )
        # content_with_remote_paths = content_with_local_paths.replace(
        #     str(self.local_base_folder.absolute()), str(self.remote_base_folder)
        # )
        # return content_with_remote_paths

    def _num_tasks(self) -> int:
        nodes: int = self.parameters.get("nodes", 1)
        tasks_per_node: int = max(1, self.parameters.get("ntasks_per_node", 1))
        return nodes * tasks_per_node

    def _make_submission_command(self, submission_file_path: PurePath) -> tp.List[str]:
        return ["ssh", self.cluster, "sbatch", str(submission_file_path)]

    @classmethod
    def affinity(cls) -> int:
        return 2
        # return -1 if shutil.which("srun") is None else 2


# pylint: disable=too-many-arguments,unused-argument, too-many-locals
def _make_sbatch_string(
    command: str,
    folder: tp.Union[str, PurePath],
    job_name: str = "submitit",
    partition: tp.Optional[str] = None,
    time: int = 5,
    nodes: int = 1,
    ntasks_per_node: tp.Optional[int] = None,
    cpus_per_task: tp.Optional[int] = None,
    cpus_per_gpu: tp.Optional[int] = None,
    num_gpus: tp.Optional[int] = None,  # legacy
    gpus_per_node: tp.Optional[int] = None,
    gpus_per_task: tp.Optional[int] = None,
    qos: tp.Optional[str] = None,  # quality of service
    setup: tp.Optional[tp.List[str]] = None,
    mem: tp.Optional[str] = None,
    mem_per_gpu: tp.Optional[str] = None,
    mem_per_cpu: tp.Optional[str] = None,
    signal_delay_s: int = 90,
    comment: tp.Optional[str] = None,
    constraint: tp.Optional[str] = None,
    exclude: tp.Optional[str] = None,
    account: tp.Optional[str] = None,
    gres: tp.Optional[str] = None,
    mail_type: tp.Optional[str] = None,
    mail_user: tp.Optional[str] = None,
    nodelist: tp.Optional[str] = None,
    dependency: tp.Optional[str] = None,
    exclusive: tp.Optional[tp.Union[bool, str]] = None,
    array_parallelism: int = 256,
    wckey: str = "submitit",
    stderr_to_stdout: bool = False,
    map_count: tp.Optional[int] = None,  # used internally
    additional_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None,
    srun_args: tp.Optional[tp.Iterable[str]] = None,
    use_srun: bool = True,
) -> str:
    """Creates the content of an sbatch file with provided parameters

    Parameters
    ----------
    See slurm sbatch documentation for most parameters:
    https://slurm.schedmd.com/sbatch.html

    Below are the parameters that differ from slurm documentation:

    folder: str/Path
        folder where print logs and error logs will be written
    signal_delay_s: int
        delay between the kill signal and the actual kill of the slurm job.
    setup: list
        a list of command to run in sbatch before running srun
    map_size: int
        number of simultaneous map/array jobs allowed
    additional_parameters: dict
        Forces any parameter to a given value in sbatch. This can be useful
        to add parameters which are not currently available in submitit.
        Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    srun_args: List[str]
        Add each argument in the list to the srun call

    Raises
    ------
    ValueError
        In case an erroneous keyword argument is added, a list of all eligible parameters
        is printed, with their default values
    """
    nonslurm = [
        "nonslurm",
        "folder",
        "command",
        "map_count",
        "array_parallelism",
        "additional_parameters",
        "setup",
        "signal_delay_s",
        "stderr_to_stdout",
        "srun_args",
        "use_srun",  # if False, un python directly in sbatch instead of through srun
    ]
    parameters = {
        k: v for k, v in locals().items() if v is not None and k not in nonslurm
    }
    # rename and reformat parameters
    parameters["signal"] = f"{SlurmJobEnvironment.USR_SIG}@{signal_delay_s}"
    if num_gpus is not None:
        warnings.warn(
            '"num_gpus" is deprecated, please use "gpus_per_node" instead (overwritting with num_gpus)'
        )
        parameters["gpus_per_node"] = parameters.pop("num_gpus", 0)
    if "cpus_per_gpu" in parameters and "gpus_per_task" not in parameters:
        warnings.warn(
            '"cpus_per_gpu" requires to set "gpus_per_task" to work (and not "gpus_per_node")'
        )
    # add necessary parameters

    # Local paths to read from?

    # Paths to put in the sbatch file
    # paths = utils.JobPaths(folder=folder)  # changed!
    stdout = str(PurePosixPath(folder) / "%j_%t_log.out")  # changed!
    stderr = str(PurePosixPath(folder) / "%j_%t_log.err")  # changed!
    # Job arrays will write files in the form  <ARRAY_ID>_<ARRAY_TASK_ID>_<TASK_ID>
    if map_count is not None:
        assert isinstance(map_count, int) and map_count
        parameters["array"] = f"0-{map_count - 1}%{min(map_count, array_parallelism)}"
        stdout = stdout.replace("%j", "%A_%a")
        stderr = stderr.replace("%j", "%A_%a")
    parameters["output"] = stdout.replace("%t", "0")
    if not stderr_to_stdout:
        parameters["error"] = stderr.replace("%t", "0")
    parameters["open-mode"] = "append"
    if additional_parameters is not None:
        parameters.update(additional_parameters)
    # now create
    lines = ["#!/bin/bash", "", "# Parameters"]
    for k in sorted(parameters):
        lines.append(_as_sbatch_flag(k, parameters[k]))
    # environment setup:
    if setup is not None:
        lines += ["", "# setup"] + setup
    # commandline (this will run the function and args specified in the file provided as argument)
    # We pass --output and --error here, because the SBATCH command doesn't work as expected with a filename pattern

    if use_srun:
        # using srun has been the only option historically,
        # but it's not clear anymore if it is necessary, and using it prevents
        # jobs from scheduling other jobs
        stderr_flags = [] if stderr_to_stdout else ["--error", stderr]
        if srun_args is None:
            srun_args = []
        srun_cmd = _shlex_join(
            ["srun", "--unbuffered", "--output", stdout, *stderr_flags, *srun_args]
        )
        command = " ".join((srun_cmd, command))

    lines += [
        "",
        "# command",
        "export SUBMITIT_EXECUTOR=slurm",
        # The input "command" is supposed to be a valid shell command
        command,
        "",
    ]
    return "\n".join(lines)


def _convert_mem(mem_gb: float) -> str:
    if mem_gb == int(mem_gb):
        return f"{int(mem_gb)}GB"
    return f"{int(mem_gb * 1024)}MB"


def _as_sbatch_flag(key: str, value: tp.Any) -> str:
    key = key.replace("_", "-")
    if value is True:
        return f"#SBATCH --{key}"

    value = shlex.quote(str(value))
    return f"#SBATCH --{key}={value}"


def _shlex_join(split_command: tp.List[str]) -> str:
    """Same as shlex.join, but that was only added in Python 3.8"""
    return " ".join(shlex.quote(arg) for arg in split_command)
