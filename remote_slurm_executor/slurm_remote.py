# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import asyncio
import contextlib
import functools
import itertools
import logging
import shlex
import subprocess
import sys
import time
import typing as tp
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Mapping,
    ParamSpec,
    TypeVar,
    overload,
)
import warnings

import rich
from milatools.cli import console
from milatools.cli.utils import SSH_CONFIG_FILE
from milatools.utils.local_v2 import LocalV2
from milatools.utils.remote_v2 import RemoteV2
from submitit.core import core, utils
from submitit.slurm import slurm
from submitit.slurm.slurm import SlurmInfoWatcher, SlurmJobEnvironment
from typing_extensions import override

OutT = TypeVar("OutT", covariant=True)
P = ParamSpec("P")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteDirSync:
    login_node: RemoteV2
    remote_dir: PurePosixPath
    local_dir: Path

    def sync_to_remote(self):
        # TODO: Need to double-check, not sure what rsync does if the target dir already exists, seems
        # to create a new folder in it, e.g. logs/mila --> logs/mila/mila
        self.login_node.run(f"mkdir -p {self.remote_dir}")
        self.local_dir.mkdir(exist_ok=True, parents=True)
        subprocess.check_call(
            shlex.split(
                f"rsync --recursive --links --safe-links --update {self.local_dir} {self.login_node.hostname}:{self.remote_dir.parent}"
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(
            f"Local dir {self.local_dir} was copied to {self.remote_dir} on the "
            f"{self.login_node.hostname} cluster."
        )

    def sync_from_remote(self):
        self.local_dir.mkdir(exist_ok=True, parents=True)
        self.login_node.local_runner.run(
            f"rsync --recursive --links --safe-links --update {self.login_node.hostname}:{self.remote_dir} {self.local_dir}"
        )
        logger.info(
            f"Local dir {self.local_dir} was updated with contents from {self.remote_dir} on the "
            f"{self.login_node.hostname} cluster."
        )

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


@dataclass(init=False)
class DelayedSubmission(utils.DelayedSubmission, Generic[P, OutT]):
    function: Callable[P, OutT]
    args: tuple
    kwargs: Mapping

    def __init__(
        self, function: Callable[P, OutT], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        super().__init__(function, *args, **kwargs)

    def result(self) -> OutT:
        return super().result()


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
        cluster_hostname: str,
        internet_access_on_compute_nodes: bool = True,
        max_num_timeout: int = 3,
        python: str | None = None,
        I_dont_care_about_reproducibility: bool = False,
    ) -> None:
        self._original_folder = folder  # save this argument that we'll modify.

        folder = PurePosixPath(folder)
        assert not folder.is_absolute()

        self.cluster_hostname = cluster_hostname
        self.login_node = LoginNode(self.cluster_hostname)
        self.internet_access_on_compute_nodes = internet_access_on_compute_nodes
        self.I_dont_care_about_reproducibility = I_dont_care_about_reproducibility

        self.remote_home = PurePosixPath(self.login_node.get_output("echo $HOME"))
        self.remote_scratch = PurePosixPath(self.login_node.get_output("echo $SCRATCH"))
        # NOTE: Could allow passing this is, but it could cause trouble, since we'd have to check
        # that it's in $HOME (or, more precisely, on the same filesystem as the worktrees will be
        # created, which is currently in $HOME/worktrees

        # "base" folder := dir without any %j %t, %A, etc.
        base_folder = get_first_id_independent_folder(folder)
        rest_of_folder = folder.relative_to(base_folder)

        # Local folder is a real "local" folder that we'll just sync as needed with the remote.
        self.local_base_folder = Path(base_folder).absolute()
        self.local_folder = self.local_base_folder / rest_of_folder

        # This is the folder where we store the pickle files on the remote.
        self.remote_base_folder = self.remote_scratch / ".submitit" / base_folder
        self.remote_folder = self.remote_base_folder / rest_of_folder

        self.remote_dir_sync = RemoteDirSync(
            self.login_node,
            local_dir=self.local_base_folder,
            remote_dir=self.remote_base_folder,
        )

        assert python is None, "TODO: Can't use something else than uv for now."
        self._uv_path: str = self.setup_uv()
        _python_version = ".".join(map(str, sys.version_info[:3]))
        python = f"{self._uv_path} run --python={_python_version} python"

        # try:
        # Try without mounting.
        # if not self.remote_dir_sync.is_mounted():
        #     self.remote_dir_sync.mount()

        super().__init__(
            folder=self.local_folder, max_num_timeout=max_num_timeout, python=python
        )
        # No need to make it absolute. Revert it back to a relative path?
        assert self.folder == self.local_folder.absolute()

        # We create a git worktree on the remote, at that particular commit.
        repo_dir_on_cluster = self.remote_home / "repos" / current_repo_name()
        self.worktree_path = self.sync_source_code(repo_dir_on_cluster)

        if not self.internet_access_on_compute_nodes:
            self.predownload_dependencies()

        # chdir to the repo so that `uv run` uses the dependencies, etc at that commit.
        srun_args: list[str] = self.parameters.setdefault("srun_args", [])
        srun_args.append(f"--chdir={self.worktree_path}")
        self.parameters.setdefault("stderr_to_stdout", True)

    def submit(
        self, fn: Callable[P, OutT], *args: P.args, **kwargs: P.kwargs
    ) -> core.Job[OutT]:
        ds = DelayedSubmission(fn, *args, **kwargs)

        if self._delayed_batch is not None:
            job: core.Job[OutT] = core.DelayedJob(self)
            self._delayed_batch.append((job, ds))
            assert type(job) is core.Job  # pylint: disable=unidiomatic-typecheck
            return job
        return self.process_submission(ds)

    def process_submission(
        self, ds: DelayedSubmission[..., OutT]
    ) -> RemoteSlurmJob[OutT]:
        # NOTE: Expanded (copied) from the base class, just to understand whats going on.
        eq_dict = self._equivalence_dict()
        timeout_min = self.parameters.get(
            eq_dict["timeout_min"] if eq_dict else "timeout_min", 5
        )
        tmp_uuid = uuid.uuid4().hex
        pickle_path = (
            utils.JobPaths.get_first_id_independent_folder(self.folder)
            / f"{tmp_uuid}.pkl"
        )
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        ds.set_timeout(timeout_min, self.max_num_timeout)
        ds.dump(pickle_path)

        self.remote_dir_sync.sync_to_remote()

        self._throttle()
        self._last_job_submitted = time.time()
        job = self._submit_command(self._submitit_command_str)

        # TODO: Need to sync here?

        # job.paths.move_temporary_file(pickle_path, "submitted_pickle")
        tmp_path = pickle_path
        job.paths.folder.mkdir(parents=True, exist_ok=True)
        Path(tmp_path).rename(job.paths.submitted_pickle)

        self.remote_dir_sync.sync_to_remote()
        return job

    @overload
    def map_array(
        self,
        fn: Callable[[A], OutT],
        _a: Iterable[A],
        /,
    ) -> list[core.Job[OutT]]: ...

    @overload
    def map_array(
        self,
        fn: Callable[[A, B], OutT],
        _a: Iterable[A],
        _b: Iterable[B],
        /,
    ) -> list[core.Job[OutT]]: ...

    @overload
    def map_array(
        self,
        fn: Callable[[A, B, C], OutT],
        _a: Iterable[A],
        _b: Iterable[B],
        _c: Iterable[C],
        /,
    ) -> list[core.Job[OutT]]: ...

    def map_array(
        self, fn: Callable[..., OutT], *iterable: Iterable[Any]
    ) -> list[RemoteSlurmJob[OutT]]:
        submissions = [utils.DelayedSubmission(fn, *args) for args in zip(*iterable)]
        if len(submissions) == 0:
            warnings.warn("Received an empty job array")
            return []
        return self._internal_process_submissions(submissions)

    def predownload_dependencies(self):
        logger.info(
            "Syncing the dependencies on the login node once, so that they are in the cache "
            "and available for the job later."
        )
        with self.login_node.chdir(self.worktree_path):
            self.login_node.run(f"{self._uv_path} sync --all-extras --frozen")
            # Remove the venv since we just want the dependencies to be downloaded to the cache)
            self.login_node.run("rm -r .venv")

        # IDEA: IF there is internet access on the compute nodes, then perhaps we could sync the
        # dependencies on a compute node instead of on the login nodes?

    def sync_source_code(self, repo_dir_on_cluster: PurePosixPath) -> PurePosixPath:
        """Sync the local source code with the remote cluster."""

        if not self.I_dont_care_about_reproducibility:
            if LocalV2.get_output("git status --porcelain"):
                rich.print(
                    "You have uncommitted changes, please commit and push them before re-running the command.\n"
                    "(This is necessary in order to sync local code with the remote cluster, and is also a good "
                    "practice for reproducibility.)"
                )
                exit(1)
            # Local git repo is clean, push HEAD to the remote.
            LocalV2.run("git push")

        current_branch_name = LocalV2.get_output("git rev-parse --abbrev-ref HEAD")
        current_commit = LocalV2.get_output("git rev-parse HEAD")

        ref = (
            current_branch_name
            if self.I_dont_care_about_reproducibility
            else current_commit
        )

        repo_url = LocalV2.get_output("git config --get remote.origin.url")
        repo_name = repo_url.split("/")[-1].removesuffix(".git")

        # If the repo doesn't exist on the remote, clone it:
        if not self.login_node.dir_exists(repo_dir_on_cluster):
            self.login_node.run(f"mkdir -p {repo_dir_on_cluster.parent}")
            self.login_node.run(f"git clone {repo_url} {repo_dir_on_cluster}")
        else:
            # Else, fetch the latest changes:
            self.login_node.run(f"cd {repo_dir_on_cluster} && git fetch")

        remote_worktree_path = self.remote_home / "worktrees" / f"{repo_name}-{ref}"

        if not self.login_node.dir_exists(remote_worktree_path):
            self.login_node.run(f"mkdir -p {remote_worktree_path.parent}")
            self.login_node.run(
                f"cd {repo_dir_on_cluster} && git worktree add {remote_worktree_path} {ref}"
                # IDK what this "--lock" thing does, I'd like it to prevent users from modifying the code.
                # Could also pass a reason for locking (if locking actually does what we want)
                # '''--lock --reason "Please don't modify the code here. This is locked for reproducibility."'''
            )
        return remote_worktree_path

    def setup_uv(self) -> str:
        if not (uv_path := self._get_uv_path()):
            logger.info(
                f"Setting up [uv](https://docs.astral.sh/uv/) on {self.cluster_hostname}"
            )
            self.login_node.run(
                "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.cargo/env"
            )
            uv_path = self._get_uv_path()
            if uv_path is None:
                raise RuntimeError(
                    f"Unable to setup `uv` on the {self.cluster_hostname} cluster!"
                )
        return uv_path

    def _get_uv_path(self) -> str | None:
        return (
            LocalV2.get_output(
                ("ssh", self.cluster_hostname, "which", "uv"),
                warn=True,
            )
            or LocalV2.get_output(
                ("ssh", self.cluster_hostname, "bash", "-l", "which", "uv"),
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

    def _submit_command(self, command: str) -> RemoteSlurmJob:
        # Copied and adapted from PicklingExecutor.
        tmp_uuid = uuid.uuid4().hex

        submission_file_path = (
            utils.JobPaths.get_first_id_independent_folder(self.folder)
            / f"submission_file_{tmp_uuid}.sh"
        )
        with submission_file_path.open("w") as f:
            f.write(self._make_submission_file_text(command, tmp_uuid))

        assert submission_file_path.is_relative_to(self.local_base_folder.absolute())
        self.remote_dir_sync.sync_to_remote()

        relative_submission_file_path = submission_file_path.relative_to(
            self.local_base_folder.absolute()
        )
        path_to_submission_file_on_remote = (
            self.remote_base_folder / relative_submission_file_path
        )

        command_list = self._make_submission_command(path_to_submission_file_on_remote)
        # run
        output = utils.CommandFunction(command_list, verbose=False)()  # explicit errors
        job_id = self._get_job_id_from_submission_command(output)
        tasks_ids = list(range(self._num_tasks()))

        job = self.job_class(
            cluster=self.cluster_hostname,
            folder=self.local_folder,
            job_id=job_id,
            tasks=tasks_ids,
        )
        # This will probably not work!
        job.paths.move_temporary_file(
            submission_file_path, "submission_file", keep_as_symlink=True
        )
        # TODO: The rest here isn't used? Seems to be meant for another executor
        # (maybe a conda-based executor (chronos?) internal to FAIR?) (hinted at
        # in doctrings and such).
        self._write_job_id(job.job_id, tmp_uuid)
        self._set_job_permissions(job.paths.folder)
        return job

    def _internal_process_submissions(
        self, delayed_submissions: list[DelayedSubmission[P, OutT]]
    ) -> list[RemoteSlurmJob[OutT]]:
        logger.info(f"Processing {len(delayed_submissions)} submissions")
        logger.debug(delayed_submissions[0])
        if len(delayed_submissions) == 1:
            return [self.process_submission(delayed_submissions[0])]

        # Job Array

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

        # NOTE: I don't yet understand this part here. Seems like poor design to me.

        # Make a copy of the executor, since we don't want other jobs to be
        # scheduled as arrays.
        array_ex = type(self)(
            folder=self._original_folder,
            cluster_hostname=self.cluster_hostname,
            internet_access_on_compute_nodes=self.internet_access_on_compute_nodes,
            max_num_timeout=self.max_num_timeout,
            python=None,
            I_dont_care_about_reproducibility=self.I_dont_care_about_reproducibility,
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
                cluster=self.cluster_hostname,
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
        # return _make_sbatch_string(
        #     command=command, folder=self.remote_folder, **self.parameters
        # )
        content_with_local_paths = slurm._make_sbatch_string(
            command=command, folder=self.local_folder, **self.parameters
        )
        content_with_remote_paths = content_with_local_paths.replace(
            str(self.local_base_folder.absolute()), str(self.remote_base_folder)
        )

        # Note: annoying, but seems like `srun_args` is fed through shlex.quote or
        # something, which causes issues with the evaluation of variables.
        chdir_to_worktree = "--chdir=$WORKTREE_LOCATION"
        return content_with_remote_paths.replace(
            f"'{chdir_to_worktree}'", chdir_to_worktree
        )

    def _num_tasks(self) -> int:
        nodes: int = self.parameters.get("nodes", 1)
        tasks_per_node: int = max(1, self.parameters.get("ntasks_per_node", 1))
        return nodes * tasks_per_node

    def _make_submission_command(self, submission_file_path: PurePath) -> tp.List[str]:
        return [
            "ssh",
            self.cluster_hostname,
            "cd",
            "$SCRATCH",
            "&&",
            "sbatch",
            str(submission_file_path),
        ]

    @classmethod
    def affinity(cls) -> int:
        return 2
        # return -1 if shutil.which("srun") is None else 2


from milatools.utils.remote_v2 import SSH_CONFIG_FILE, Hide, RemoteV2


class LoginNode(RemoteV2):
    # Tiny improvements / changes to the RemoteV2 class from milatools.
    def __init__(
        self,
        hostname: str,
        *,
        control_path: Path | None = None,
        ssh_config_path: Path = SSH_CONFIG_FILE,
        command_prefix: str = "",
        _start_control_socket: bool = True,
    ):
        super().__init__(
            hostname,
            control_path=control_path,
            ssh_config_path=ssh_config_path,
            _start_control_socket=_start_control_socket,
        )
        self._start_control_socket = _start_control_socket
        self.command_prefix = command_prefix

    def dir_exists(self, remote_dir: PurePosixPath | str) -> bool:
        return (
            self.run(
                f"test -d {remote_dir}",
                warn=True,
                hide=True,
                display=False,
            ).returncode
            == 0
        )

    def file_exists(self, remote_dir: PurePosixPath | str) -> bool:
        return (
            self.run(
                f"test -f {remote_dir}",
                warn=True,
                hide=True,
                display=False,
            ).returncode
            == 0
        )

    @contextlib.contextmanager
    def chdir(self, remote_dir: PurePosixPath | str):
        cd_command = f"cd {remote_dir}"
        if self.command_prefix:
            added = f"&& {cd_command}"
        else:
            added = cd_command

        self.command_prefix += added

        yield

        self.command_prefix = self.command_prefix.removesuffix(added)

    def cd(self, remote_dir: PurePosixPath | str):
        cd_command = f"cd {remote_dir}"
        if self.command_prefix:
            new_prefix = f"{self.command_prefix} && {cd_command}"
        else:
            new_prefix = cd_command
        return type(self)(
            hostname=self.hostname,
            control_path=self.control_path,
            ssh_config_path=self.ssh_config_path,
            _start_control_socket=self._start_control_socket,
            command_prefix=new_prefix,
        )

    def display(self, command: str, input: str | None = None, _stack_offset: int = 3):
        message = f"({self.hostname}) $ {command}"
        if input:
            message += f"\n{input}"

        console.log(message, style="green", _stack_offset=_stack_offset)

    @override
    def run(
        self,
        command: str,
        *,
        input: str | None = None,
        display: bool = True,
        warn: bool = False,
        hide: Hide = False,
    ):
        if display:
            self.display(command, input=input, _stack_offset=3)
        return super().run(
            self.command_prefix + command,
            input=input,
            display=display,
            warn=warn,
            hide=hide,
        )

    @override
    async def run_async(
        self,
        command: str,
        *,
        input: str | None = None,
        display: bool = True,
        warn: bool = False,
        hide: Hide = False,
    ) -> subprocess.CompletedProcess[str]:
        if display:
            self.display(command, input=input, _stack_offset=3)
        return await super().run_async(
            self.command_prefix + command,
            input=input,
            display=False,
            warn=warn,
            hide=hide,
        )


def current_repo_name() -> str:
    repo_url = LocalV2.get_output("git config --get remote.origin.url")
    return repo_url.split("/")[-1]


@functools.lru_cache
def get_slurm_account(cluster: str) -> str:
    """Gets the SLURM account of the user using sacctmgr on the slurm cluster.

    When there are multiple accounts, this selects the first account, alphabetically.

    On DRAC cluster, this uses the `def` allocations instead of `rrg`, and when
    the rest of the accounts are the same up to a '_cpu' or '_gpu' suffix, it uses
    '_cpu'.

    For example:

    ```text
    def-someprofessor_cpu  <-- this one is used.
    def-someprofessor_gpu
    rrg-someprofessor_cpu
    rrg-someprofessor_gpu
    ```
    """
    logger.info(
        f"Fetching the list of SLURM accounts available on the {cluster} cluster."
    )
    result = RemoteV2(cluster).run(
        "sacctmgr --noheader show associations where user=$USER format=Account%50"
    )
    accounts = [line.strip() for line in result.stdout.splitlines()]
    assert accounts
    logger.info(f"Accounts on the slurm cluster {cluster}: {accounts}")
    account = sorted(accounts)[0]
    logger.info(f"Using account {account} to launch jobs.")
    return account
