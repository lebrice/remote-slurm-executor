# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
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
import warnings
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

from milatools.cli import console
from milatools.cli.utils import SSH_CONFIG_FILE
from milatools.utils.local_v2 import LocalV2
from milatools.utils.remote_v2 import Hide, RemoteV2
from submitit.core import core, utils
from submitit.slurm import slurm
from submitit.slurm.slurm import SlurmInfoWatcher, SlurmJobEnvironment
from typing_extensions import override

OutT = TypeVar("OutT", covariant=True)
P = ParamSpec("P")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteDirSync:
    login_node: "LoginNode"
    remote_dir: PurePosixPath
    local_dir: Path

    def copy_to_remote(self, local_path: Path) -> PurePosixPath:
        remote_path = self._get_remote_path(local_path)
        self._copy_to_remote(local_path, remote_path)
        return remote_path

    def get_from_remote(
        self, remote_path: PurePosixPath | None = None, local_path: Path | None = None
    ) -> Path:
        assert bool(remote_path) ^ bool(
            local_path
        ), "Exactly one of remote_path or local_path should be passed."
        if remote_path:
            local_path = self._get_local_path(remote_path)
        else:
            assert local_path
            remote_path = self._get_remote_path(local_path)
        self._get_from_remote(remote_path=remote_path, local_path=local_path)
        return local_path

    def _get_remote_path(self, local_path: Path) -> PurePosixPath:
        return self.remote_dir / (
            local_path.absolute().relative_to(self.local_dir.absolute())
        )

    def _get_local_path(self, remote_path: PurePosixPath) -> Path:
        return self.local_dir / (remote_path.relative_to(self.remote_dir))

    def _copy_to_remote(self, local_path: Path, remote_path: PurePosixPath):
        assert local_path.is_file()
        self.login_node.run(f"mkdir -p {remote_path.parent}")
        # if self.login_node.file_exists(remote_path):
        self.login_node.local_runner.run(
            f"scp {local_path} {self.login_node.hostname}:{remote_path}", display=False
        )
        # else:
        #     assert self.login_node.dir_exists(remote_path)
        #     self.login_node.local_runner.run(
        #         f"scp -r {local_path} {self.login_node.hostname}:{remote_path}", display=False
        #     )
        # Could also perhaps use rsync?
        # self.login_node.local_runner.run(
        #     f"rsync --recursive --links --safe-links --update "
        #     f"{self.local_dir} {self.login_node.hostname}:{self.remote_dir.parent}"
        # )

    def _get_from_remote(self, remote_path: PurePosixPath, local_path: Path):
        # todo: switch between rsync and scp based on in the remote path is a file or a dir?
        local_path.parent.mkdir(exist_ok=True, parents=True)
        if self.login_node.file_exists(remote_path):
            self.login_node.local_runner.run(
                f"scp {self.login_node.hostname}:{remote_path} {local_path}",
                display=False,
            )
        else:
            assert self.login_node.dir_exists(remote_path)
            self.login_node.local_runner.run(
                f"scp -r {self.login_node.hostname}:{remote_path} {local_path}",
                display=False,
            )
        # Could also perhaps use rsync?
        # self.login_node.local_runner.run(
        #     f"rsync --recursive --links --safe-links --update "
        #     f"{self.login_node.hostname}:{self.remote_dir} {self.local_dir.parent}"
        # )


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


class RemoteSlurmJob(core.Job[OutT]):
    _cancel_command = "scancel"
    watchers: ClassVar[dict[str, RemoteSlurmInfoWatcher]] = {}
    watcher: RemoteSlurmInfoWatcher

    def __init__(
        self,
        cluster: str,
        folder: tp.Union[str, Path],
        job_id: str,
        remote_dir_sync: RemoteDirSync,
        tasks: tp.Sequence[int] = (0,),
    ) -> None:
        self.cluster = cluster
        # watcher*s*, since this could be using different clusters.
        # Also: `watcher` is now an instance variable, not a class variable.
        self.watcher = type(self).watchers.setdefault(
            self.cluster, RemoteSlurmInfoWatcher(cluster=cluster, delay_s=600)
        )
        self.remote_dir_sync = remote_dir_sync
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

    def wait(self) -> None:
        super().wait()
        logger.info(f"Copying folder {self.paths.folder} from the remote.")
        self.remote_dir_sync.get_from_remote(local_path=self.paths.folder)
        self.remote_dir_sync.get_from_remote(local_path=self.paths.stdout)


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

    ## TODOs:
    - [ ] Unable to launch jobs from the `master` branch of a repo, because we try to create a
          worktree and the branch is already checked out in the cloned repo!
    - [ ] Having issues on narval where the venv can't be created?
    """

    job_class: ClassVar[type[RemoteSlurmJob]] = RemoteSlurmJob

    def __init__(
        self,
        folder: PurePath | str,
        *,
        cluster_hostname: str,
        repo_dir_on_cluster: PurePosixPath | str | None = None,
        internet_access_on_compute_nodes: bool = True,
        max_num_timeout: int = 3,
        python: str | None = None,
        I_dont_care_about_reproducibility: bool = False,
    ) -> None:
        """ Create a new remote slurm executor.

        Args:
            folder: The output folder (same idea as the base class)
            cluster_hostname: Hostname of the cluster to connect to.
            repo_dir_on_cluster: The directory on the cluster where the repo is cloned. If not \
                passed, the repo is cloned in `$HOME/repos/<repo_name>`.
            internet_access_on_compute_nodes: Whether compute nodes on that cluster have access to \
                the internet.
            max_num_timeout: Maximum number of job timeouts before giving up (from the base \
                class constructor).
            python: Python command. Defaults to `uv run --python=X.Y python`. Cannot be \
                customized for now.
            I_dont_care_about_reproducibility: Whether you should be forced to commit and push \
                your changes before submitting jobs. Things might break if unset, use at your peril.
        """
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

        # Where we clone the repo on the cluster.
        self.repo_dir_on_cluster = PurePosixPath(
            repo_dir_on_cluster or (self.remote_home / "repos" / current_repo_name())
        )

        # "base" folder := dir without any %j %t, %A, etc.
        base_folder = get_first_id_independent_folder(folder)
        rest_of_folder = folder.relative_to(base_folder)

        self.local_base_folder = Path(base_folder).absolute()
        self.local_folder = Path(folder).absolute()

        # This is the folder where we store the pickle files on the remote.
        self.remote_base_folder = self.remote_scratch / base_folder
        self.remote_folder = self.remote_base_folder / rest_of_folder

        self.remote_dir_sync = RemoteDirSync(
            self.login_node,
            local_dir=self.local_base_folder,
            remote_dir=self.remote_base_folder,
        )

        assert python is None, "TODO: Can't use something else than uv for now."

        # note: seems like we really need to specify the path to uv since `srun --pty uv` doesn't
        # work.
        self._uv_path: str = self.setup_uv()
        _python_version = ".".join(map(str, sys.version_info[:3]))
        offline = "--offline " if not self.internet_access_on_compute_nodes else ""
        python = f"{self._uv_path} run {offline} --python={_python_version} python"

        super().__init__(
            folder=Path(folder), max_num_timeout=max_num_timeout, python=python
        )
        # No need to make it absolute. Revert it back to a relative path?
        assert self.folder == self.local_folder.absolute()

        # We create a git worktree on the remote, at that particular commit.
        self.worktree_path = self.sync_source_code(self.repo_dir_on_cluster)

        if not self.internet_access_on_compute_nodes:
            self.predownload_dependencies()

        # chdir to the repo so that `uv run` uses the dependencies, etc at that commit.
        # Note: Here we avoid mutating the passed in lists or dicts.
        self.parameters["srun_args"] = self.parameters.get("srun_args", []) + [
            f"--chdir={self.worktree_path}"
        ]
        self.parameters.setdefault("stderr_to_stdout", True)
        self.parameters["setup"] = self.parameters.get("setup", []) + [
            f"# {cluster_hostname=}",
        ]

    def submit(
        self, fn: Callable[P, OutT], *args: P.args, **kwargs: P.kwargs
    ) -> core.DelayedJob[OutT] | RemoteSlurmJob[OutT]:
        ds = DelayedSubmission(fn, *args, **kwargs)
        super().submit
        if self._delayed_batch is not None:
            job: core.Job[OutT] = core.DelayedJob(self)
            self._delayed_batch.append((job, ds))
        else:
            job = self.process_submission(ds)
        # IDK why this is in the base class?
        if type(job) is core.Job:  # pylint: disable=unidiomatic-typecheck
            raise RuntimeError(
                "Executors should never return a base Job class (implementation issue)"
            )
        return job

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

        remote_pickle_path = self.remote_dir_sync.copy_to_remote(pickle_path)
        # self.remote_dir_sync.sync_to_remote()

        self._throttle()
        self._last_job_submitted = time.time()
        job = self._submit_command(self._submitit_command_str)

        # job.paths.move_temporary_file(pickle_path, "submitted_pickle")
        # job.paths.folder.mkdir(parents=True, exist_ok=True)
        # Path(pickle_path).rename(job.paths.submitted_pickle)
        _get_remote_path = self.remote_dir_sync._get_remote_path
        new_pickle_path = _get_remote_path(job.paths.submitted_pickle)
        self.login_node.run(f"mkdir -p {new_pickle_path.parent}")
        self.login_node.run(f"mv {remote_pickle_path} {new_pickle_path}")
        # Also reflect this change locally?
        job.paths.submitted_pickle.parent.mkdir(exist_ok=True, parents=True)
        pickle_path.rename(job.paths.submitted_pickle)

        # self.remote_dir_sync.sync_to_remote()
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

    @overload
    def map_array(
        self,
        fn: Callable[[A, B, C, D], OutT],
        _a: Iterable[A],
        _b: Iterable[B],
        _c: Iterable[C],
        _d: Iterable[D],
        /,
    ) -> list[core.Job[OutT]]: ...

    @overload
    def map_array(
        self,
        fn: Callable[[A, B, C, D, E], OutT],
        _a: Iterable[A],
        _b: Iterable[B],
        _c: Iterable[C],
        _d: Iterable[D],
        _e: Iterable[E],
        /,
    ) -> list[core.Job[OutT]]: ...

    def map_array(
        self, fn: Callable[..., OutT], *iterable: Iterable[Any]
    ) -> list[RemoteSlurmJob[OutT]]:
        submissions = [DelayedSubmission(fn, *args) for args in zip(*iterable)]
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
            self.login_node.run(f"{self._uv_path} sync --offline --all-extras --frozen")
            # Remove the venv since we just want the dependencies to be downloaded to the cache)
            # self.login_node.run("rm -r .venv")

        # IDEA: IF there is internet access on the compute nodes, then perhaps we could sync the
        # dependencies on a compute node instead of on the login nodes?

    def sync_source_code(self, repo_dir_on_cluster: PurePosixPath) -> PurePosixPath:
        """Sync the local source code with the remote cluster."""

        if not self.I_dont_care_about_reproducibility:
            if LocalV2.get_output("git status --porcelain"):
                console.print(
                    "You have uncommitted changes, please commit and push them before re-running the command.\n"
                    "(This is necessary in order to sync local code with the remote cluster, and is also a good "
                    "practice for reproducibility.)",
                    style="orange3",  # Why the hell isn't 'orange' a colour?!
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

        submission_file_on_remote = self.remote_dir_sync.copy_to_remote(
            submission_file_path
        )

        command_list = self._make_submission_command(submission_file_on_remote)
        # run the sbatch command.
        output = utils.CommandFunction(command_list, verbose=False)()  # explicit errors
        job_id = self._get_job_id_from_submission_command(output)
        tasks_ids = list(range(self._num_tasks()))

        job = RemoteSlurmJob(
            cluster=self.cluster_hostname,
            folder=self.local_folder,
            job_id=job_id,
            tasks=tasks_ids,
            remote_dir_sync=self.remote_dir_sync,
        )

        # TODO: Need to do those on the remote? Or can we get away with doing it
        # locally and rsync-ing the dir? (which will reflect this with symlink
        # changes and such?)
        # env = self.login_node.get_output(f"srun --pty --overlap --jobid {job_id} env")

        ## NOTE: This function call in the base class:

        # job.paths.move_temporary_file(
        #     submission_file_path, "submission_file", keep_as_symlink=True
        # )

        ## Gets expanded to this, with `tmp_path=submission_file_path`:

        # job.paths.folder.mkdir(parents=True, exist_ok=True)
        # Path(submission_file_path).rename(job.paths.submission_file)
        # Path(submission_file_path).symlink_to(job.paths.submission_file)

        ## And executing the equivalent over SSH looks like this:
        _get_remote_path = self.remote_dir_sync._get_remote_path
        self.login_node.run(f"mkdir -p {_get_remote_path(job.paths.folder)}")
        self.login_node.run(
            f"mv {submission_file_on_remote} {_get_remote_path(job.paths.submission_file)}"
        )
        self.login_node.run(
            f"ln -s {_get_remote_path(job.paths.submission_file)} {submission_file_on_remote}"
        )
        # TODO: Also reflect this locally?
        job.paths.submission_file.parent.mkdir(parents=True, exist_ok=True)
        submission_file_path.rename(job.paths.submission_file)
        submission_file_path.symlink_to(job.paths.submission_file)

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
        local_pickle_paths: list[Path] = []
        for d in delayed_submissions:
            _pickle_path = folder / f"{uuid.uuid4().hex}.pkl"
            d.set_timeout(timeout_min, self.max_num_timeout)
            d.dump(_pickle_path)
            local_pickle_paths.append(_pickle_path)

        n = len(delayed_submissions)

        # self.remote_dir_sync.sync_to_remote()

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
        jobs = [
            RemoteSlurmJob[OutT](
                cluster=self.cluster_hostname,
                folder=self.folder,
                job_id=f"{first_job.job_id}_{a}",
                tasks=tasks_ids,
                remote_dir_sync=self.remote_dir_sync,
            )
            for a in range(n)
        ]
        # TODO: Handle these more explicitly.
        for job, local_pickle_path in zip(jobs, local_pickle_paths):
            # job.paths.move_temporary_file(pickle_path, "submitted_pickle")
            # self._copy_to_remote(job.paths.submitted_pickle)
            _get_remote_path = self.remote_dir_sync._get_remote_path

            remote_pickle_path = self.remote_dir_sync.copy_to_remote(local_pickle_path)

            self.login_node.run(
                f"mkdir -p {_get_remote_path(job.paths.folder)}", display=False
            )
            self.login_node.run(
                f"mv {remote_pickle_path} {_get_remote_path(job.paths.submitted_pickle)}",
                display=False,
            )
            # job.paths.folder.mkdir(parents=True, exist_ok=True)
            # Path(pickle_path).rename(job.paths.submitted_pickle)

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
        display: bool = False,  # changed to default of False
        warn: bool = False,
        hide: Hide = False,
    ):
        if display:
            self.display(command, input=input, _stack_offset=3)
        return super().run(
            self.command_prefix + command,
            input=input,
            display=False,
            warn=warn,
            hide=hide,
        )

    @override
    async def run_async(
        self,
        command: str,
        *,
        input: str | None = None,
        display: bool = False,  # changed to default of False
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

    On DRAC cluster, this uses the `rrg` allocations instead of `def`, and when
    the rest of the accounts are the same up to a '_cpu' or '_gpu' suffix, it uses
    '_gpu'.

    For example:

    ```text
    def-someprofessor_cpu
    def-someprofessor_gpu
    rrg-someprofessor_cpu
    rrg-someprofessor_gpu  <-- this one is used.
    ```
    """
    logger.info(
        f"Fetching the list of SLURM accounts available on the {cluster} cluster."
    )
    result = LoginNode(cluster).run(
        "sacctmgr --noheader show associations where user=$USER format=Account%50",
        display=True,
        hide=False,
    )
    accounts = [line.strip() for line in result.stdout.splitlines()]
    assert accounts
    logger.info(f"Accounts on the slurm cluster {cluster}: {accounts}")
    account = sorted(accounts)[-1]
    logger.info(f"Using account {account} to launch jobs.")
    return account
