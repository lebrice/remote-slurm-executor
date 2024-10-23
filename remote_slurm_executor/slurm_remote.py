# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import dataclasses
import functools
import itertools
import logging
import shlex
import subprocess
import sys
import textwrap
import typing as tp
import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import (
    Any,
    ClassVar,
    Generic,
    ParamSpec,
    TypeVar,
    overload,
)

from milatools.cli import console
from milatools.utils.local_v2 import LocalV2
from submitit.core import core, utils
from submitit.slurm import slurm
from submitit.slurm.slurm import SlurmInfoWatcher, SlurmJobEnvironment

from remote_slurm_executor.utils import LoginNode, RemoteDirSync

OutT = TypeVar("OutT", covariant=True)
P = ParamSpec("P")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")

logger = logging.getLogger(__name__)
CURRENT_PYTHON = ".".join(map(str, sys.version_info[:3]))


# Could also perhaps use rsync?
# self.login_node.local_runner.run(
#     f"rsync --recursive --links --safe-links --update "
#     f"{self.login_node.hostname}:{self.remote_dir} {self.local_dir.parent}"
# )


class RemoteSlurmInfoWatcher(SlurmInfoWatcher):
    def __init__(self, cluster: str, delay_s: int = 60) -> None:
        super().__init__(delay_s)
        self.cluster = cluster

    def _make_command(self) -> list[str] | None:
        # asking for array id will return all status
        # on the other end, asking for each and every one of them individually takes a huge amount of time
        cmd = super()._make_command()
        if not cmd:
            return None
        return ["ssh", self.cluster] + cmd


class RemoteSlurmJob(core.Job[OutT]):
    _cancel_command = "scancel"
    watchers: ClassVar[dict[str, RemoteSlurmInfoWatcher]] = {}
    watcher: RemoteSlurmInfoWatcher

    def __init__(
        self,
        cluster: str,
        folder: str | Path,
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
        # self.remote_dir_sync.get_from_remote(local_path=self.paths.folder)
        self.remote_dir_sync.login_node.local_runner.run(
            f"rsync --recursive --links --safe-links --update --verbose "
            f"{self.remote_dir_sync.login_node.hostname}:{self.remote_dir_sync._get_remote_path(self.paths.folder)} {self.paths.folder.parent}"
        )
        # self.remote_dir_sync.get_from_remote(local_path=self.paths.stdout)
        # self.remote_dir_sync.get_from_remote(local_path=self.paths.result_pickle)


@dataclass(init=False)
class DelayedSubmission(utils.DelayedSubmission, Generic[P, OutT]):
    function: Callable[P, OutT]
    args: tuple
    kwargs: Mapping

    def __init__(self, function: Callable[P, OutT], *args: P.args, **kwargs: P.kwargs) -> None:
        super().__init__(function, *args, **kwargs)

    def result(self) -> OutT:
        return super().result()


@dataclasses.dataclass(init=False)
class RemoteSlurmExecutor(slurm.SlurmExecutor):
    """Executor for a remote SLURM cluster.

     - Installs `uv` on the remote cluster.
     - Syncs dependencies with `uv sync --all-extras` on the login node.

     ## TODOs / ideas:
    - [ ] Add a tag like {cluster_name}-{job_id} to the commit once we know the job id?
    - [ ] Prevent the warning being logged twice (and syncing of source code as well) when launching
          an array job. Avoid re-creating an executor when submitting array jobs if possible.
    """

    folder: Path
    """The output folder, for example, "logs/%j".

    NOTE: if `remote_folder` is unset, this must be a relative path.
    """

    cluster_hostname: str
    """Hostname of the cluster to connect to."""

    remote_folder: str = ""
    """Where `folder` will be mirrored on the remote cluster.

    Set to "$SCRATCH/`folder`" by default.
    """

    repo_dir_on_cluster: str = ""
    """The directory on the cluster where the repo is cloned.

    If not passed, the repo is cloned in `$HOME/repos/<repo_name>`.
    """

    internet_access_on_compute_nodes: bool = True
    """Whether compute nodes on that cluster have access to the internet."""

    max_num_timeout: int = 3
    """Maximum number of job timeouts before giving up (from the base class)."""

    python: str | None = None
    """Python command.

    Defaults to `uv run --python=X.Y python`. Cannot be customized for now.
    """

    job_class: ClassVar[type[RemoteSlurmJob]] = RemoteSlurmJob

    def __init__(
        self,
        folder: PurePath | str,
        *,
        cluster_hostname: str,
        remote_folder: PurePosixPath | str | None = None,
        repo_dir_on_cluster: PurePosixPath | str | None = None,
        internet_access_on_compute_nodes: bool = True,
        reproducibility_mode: bool = False,
        max_num_timeout: int = 3,
        python: str | None = f"uv run --python={CURRENT_PYTHON} python",
    ) -> None:
        """Create a new remote slurm executor."""
        self._original_folder = folder  # save this argument that we'll modify.
        folder = Path(folder)

        self.cluster_hostname = cluster_hostname
        self.login_node = LoginNode(self.cluster_hostname)
        self.internet_access_on_compute_nodes = internet_access_on_compute_nodes
        self.reproducibility_mode = reproducibility_mode  # TODO: Add tags for jobs (locally only).

        # Where we clone the repo on the cluster.
        if not repo_dir_on_cluster:
            repo_dir_on_cluster = (
                PurePosixPath(self.login_node.get_output("echo $HOME"))
                / "repos"
                / _current_repo_name()
            )
        # repo_dir_on_cluster = PurePosixPath(repo_dir_on_cluster)
        self.repo_dir_on_cluster = str(repo_dir_on_cluster)

        # "base folder" := last part of `folder` that isn't dependent on the job id or task id (no
        # %j %t, %A, etc).
        # For example: "logs/%j" -> "logs"
        base_folder = get_first_id_independent_folder(folder)
        self.local_base_folder = Path(base_folder).absolute()

        # This is the folder where we store the pickle files on the remote.
        if not remote_folder:
            assert not folder.is_absolute()
            _remote_scratch = PurePosixPath(self.login_node.get_output("echo $SCRATCH"))
            remote_folder = _remote_scratch / folder.relative_to(base_folder)
        self.remote_folder = str(remote_folder)
        self.remote_base_folder = get_first_id_independent_folder(
            PurePosixPath(self.remote_folder)
        )

        self.remote_dir_sync = RemoteDirSync(
            self.login_node,
            local_dir=self.local_base_folder,
            remote_dir=self.remote_base_folder,
        )

        super().__init__(folder=folder, max_num_timeout=max_num_timeout, python=python)

        # Note: It seems like we really need to specify the full path to uv since `srun --pty uv`
        # doesn't work?
        _uv_path: str = self.setup_uv()
        # todo: Is this --offline flag actually useful?
        # offline = "--offline " if not self.internet_access_on_compute_nodes else ""
        # self.python = f"{uv_path} run --python={CURRENT_PYTHON} python"
        sync_dependencies_command = f"uv sync --python={CURRENT_PYTHON} --all-extras --frozen"

        repo_url = _current_repo_url()
        repo_name = _current_repo_name()
        commit = _current_commit()
        commit_short = _current_commit_short()
        login_node = self.login_node

        sync_source_code(
            login_node=login_node,
            repo_dir_on_cluster=PurePosixPath(self.repo_dir_on_cluster),
            repo_url=repo_url,
            commit=commit,
        )

        if not internet_access_on_compute_nodes:
            logger.info(
                "Syncing the dependencies on the login node once, so that they are in the cache "
                "and available for the job later."
            )
            with login_node.chdir(repo_dir_on_cluster):
                # Assume that we've already synced the code and the dependencies.
                # IDEA: IF there is internet access on the compute nodes, then perhaps we could sync
                # the dependencies on a compute node instead of on the login nodes?
                login_node.run(sync_dependencies_command, display=True)
                # NOTE: We could also remove the venv since we mainly want the dependencies to be downloaded
                # to the cache.
                # self.login_node.run("rm -r .venv")

        # Note: Here we avoid mutating the passed in lists or dicts.
        # self.parameters["srun_args"] = (
        #     self.parameters.get("srun_args", [])
        #     # TODO: How do we --chdir to $SLURM_TMPDIR, since that might (in principle) be different
        #     # within each srun context when working with multiple nodes?
        #     # + [f"--chdir={self.worktree_path}"]
        # )

        # worktrees = [
        #     (_path := PurePosixPath((_parts := line.split())[0]), _ref := _parts[1])
        #     for line in login_node.cd(repo_dir_on_cluster)
        #     .get_output("git worktree list")
        #     .splitlines()
        # ]
        # worktree_doesnt_exist = commit_short not in [p[1] for p in worktrees]

        _worktree_path = f"$SLURM_TMPDIR/{repo_name}-{commit_short}"

        added_setup_block = [
            f"### Added by the {type(self).__name__}",
            f"# {cluster_hostname=}",
            # TODO: Would love to set this, but the --link-mode raises an error below.
            "set -e  # Exit immediately if a command exits with a non-zero status.",
            "export UV_PYTHON_PREFERENCE='managed'  # Prefer using the python managed by uv over the system's python.",
            "export UV_LINK_MODE='copy'  # Don't quit the job if we can't use hardlinks from the cache.",
            "source $HOME/.cargo/env  # Needed so we can run `uv` in a non-interactive job, apparently.",
            (
                f"git clone {repo_dir_on_cluster} {_worktree_path}"
                # f"git worktree add {_worktree_path} {commit} --force --force --detach --lock "
                # '--reason="Locked for reproducibility: This code was used in job $SLURM_JOB_ID"'
                # if worktree_doesnt_exist
                # MEGA HACK: https://stackoverflow.com/questions/42822869/how-can-i-recover-a-staged-changes-from-a-deleted-git-worktree
                # mkdir -p {worktree_path}
                # echo "gitdir: {repo_dir_on_cluster}/.git/worktrees/{worktree_name}" > {worktree_path}/.git
                # gitdir: /home/mila/n/normandf/repos/remote-slurm-executor/.git/worktrees/remote-slurm-executor-1057174
                # else f"git worktree repair {_worktree_path}"
            ),
            f"cd {_worktree_path}",
            f"git reset --hard {commit}",
            "###",
            # Trying out the idea of creating the venv in $SLURM_TMPDIR instead of in the worktree in $HOME.
        ]
        self.parameters["setup"] = self.parameters.get("setup", []) + added_setup_block
        self.parameters.setdefault("stderr_to_stdout", True)
        logger.debug(f"Setup: {self.parameters['setup']}")

    def submit(
        self, fn: Callable[P, OutT], *args: P.args, **kwargs: P.kwargs
    ) -> core.DelayedJob[OutT] | RemoteSlurmJob[OutT]:
        ds = DelayedSubmission(fn, *args, **kwargs)
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

    def process_submission(self, ds: DelayedSubmission[..., OutT]) -> RemoteSlurmJob[OutT]:
        # NOTE: Expanded (copied) from the base class, just to see what's going on.
        time = self.parameters.get("time", 5)
        if isinstance(time, int):
            timeout_min = time
        else:
            from datetime import datetime, timedelta

            t = datetime.strptime(time, "%D-%H:%M:%S")
            # ...and use datetime's hour, min and sec properties to build a timedelta
            delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
            timeout_min = int(delta.total_seconds() // 60)

        tmp_uuid = uuid.uuid4().hex
        local_pickle_path = get_first_id_independent_folder(self.folder) / f"{tmp_uuid}.pkl"
        local_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        ds.set_timeout(timeout_min, self.max_num_timeout)
        ds.dump(local_pickle_path)

        remote_pickle_path = self.remote_dir_sync.copy_to_remote(local_pickle_path)
        # self.remote_dir_sync.sync_to_remote()

        self._throttle()
        self._last_job_submitted = time.time()
        # NOTE: Choosing to remove the submission file, instead of making it a symlink like in the
        # base class.
        job = self._submit_command(self._submitit_command_str, _keep_sbatch_file_as_symlink=False)

        # job.paths.move_temporary_file(pickle_path, "submitted_pickle")
        # job.paths.folder.mkdir(parents=True, exist_ok=True)
        # Path(pickle_path).rename(job.paths.submitted_pickle)
        _get_remote_path = self.remote_dir_sync._get_remote_path
        new_pickle_path = _get_remote_path(job.paths.submitted_pickle)
        self.login_node.run(f"mkdir -p {new_pickle_path.parent}")
        self.login_node.run(f"mv {remote_pickle_path} {new_pickle_path}")
        # Also reflect this change locally?
        job.paths.submitted_pickle.parent.mkdir(exist_ok=True, parents=True)
        local_pickle_path.rename(job.paths.submitted_pickle)

        # self.remote_dir_sync.sync_to_remote()
        return job

    @overload
    def map_array(
        self,
        fn: Callable[[A], OutT],
        _a: Iterable[A],
        /,
    ) -> list[core.Job[OutT]]:
        ...

    @overload
    def map_array(
        self,
        fn: Callable[[A, B], OutT],
        _a: Iterable[A],
        _b: Iterable[B],
        /,
    ) -> list[core.Job[OutT]]:
        ...

    @overload
    def map_array(
        self,
        fn: Callable[[A, B, C], OutT],
        _a: Iterable[A],
        _b: Iterable[B],
        _c: Iterable[C],
        /,
    ) -> list[core.Job[OutT]]:
        ...

    @overload
    def map_array(
        self,
        fn: Callable[[A, B, C, D], OutT],
        _a: Iterable[A],
        _b: Iterable[B],
        _c: Iterable[C],
        _d: Iterable[D],
        /,
    ) -> list[core.Job[OutT]]:
        ...

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
    ) -> list[core.Job[OutT]]:
        ...

    def map_array(
        self, fn: Callable[..., OutT], *iterable: Iterable[Any]
    ) -> list[RemoteSlurmJob[OutT]]:
        submissions = [DelayedSubmission(fn, *args) for args in zip(*iterable)]
        if len(submissions) == 0:
            warnings.warn("Received an empty job array")
            return []
        return self._internal_process_submissions(submissions)

    # TODO: Move this out?

    def setup_uv(self) -> str:
        if not (uv_path := self._get_uv_path()):
            logger.info(f"Setting up [uv](https://docs.astral.sh/uv/) on {self.cluster_hostname}")
            self.login_node.run(
                "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.cargo/env"
            )
            uv_path = self._get_uv_path()
            if uv_path is None:
                raise RuntimeError(f"Unable to setup `uv` on the {self.cluster_hostname} cluster!")
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

    def _submit_command(self, command: str, _keep_sbatch_file_as_symlink=True) -> RemoteSlurmJob:
        # Copied and adapted from PicklingExecutor.
        # NOTE: Weird that this is a 'private' method in the base class, which is always called
        # with the same argument. Is it this indented to be passed a different argument for testing?
        # assert command == self._submitit_command_str

        tmp_uuid = uuid.uuid4().hex

        submission_file_path = (
            utils.JobPaths.get_first_id_independent_folder(self.folder)
            / f"submission_file_{tmp_uuid}.sh"
        )
        with submission_file_path.open("w") as f:
            f.write(self._make_submission_file_text(command, tmp_uuid))

        submission_file_on_remote = self.remote_dir_sync.copy_to_remote(submission_file_path)

        command_list = self._make_submission_command(submission_file_on_remote)
        # run the sbatch command.
        output = utils.CommandFunction(command_list, verbose=False)()  # explicit errors
        job_id = self._get_job_id_from_submission_command(output)
        tasks_ids = list(range(self._num_tasks()))

        job = RemoteSlurmJob(
            cluster=self.cluster_hostname,
            folder=self.folder,
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
        if _keep_sbatch_file_as_symlink:
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

        # base_folder = utils.JobPaths.get_first_id_independent_folder(self.folder)
        local_base_folder = self.local_base_folder
        local_base_folder.mkdir(parents=True, exist_ok=True)
        timeout_min = self.parameters.get("time", 5)
        local_pickle_paths: list[Path] = []
        for d in delayed_submissions:
            _pickle_path = local_base_folder / f"{uuid.uuid4().hex}.pkl"
            d.set_timeout(timeout_min, self.max_num_timeout)
            d.dump(_pickle_path)
            local_pickle_paths.append(_pickle_path)

        n = len(delayed_submissions)

        # self.remote_dir_sync.sync_to_remote()

        # NOTE: I don't yet understand this part here. Why do they create a cloned executor?
        # Seems like poor design to me.

        # "Make a copy of the executor, since we don't want other jobs to be
        # scheduled as arrays." (TODO: What does this mean?)
        # array_ex = RemoteSlurmExecutor(
        #     folder=self.folder,
        #     cluster_hostname=self.cluster_hostname,
        #     remote_folder=self.remote_folder,
        #     repo_dir_on_cluster=self.repo_dir_on_cluster,
        #     internet_access_on_compute_nodes=self.internet_access_on_compute_nodes,
        #     max_num_timeout=self.max_num_timeout,
        #     python=self.python,
        #     reproducibility_mode=self.reproducibility_mode,
        # )
        # array_ex.update_parameters(**self.parameters)
        # array_ex.parameters["map_count"] = n

        array_ex = copy.deepcopy(self)
        array_ex._delayed_batch = None  # I imagine that this is what they wanted to not propagate?
        array_ex.update_parameters(map_count=n)

        # slurm._make_sbatch_string  # Uncomment to look at the code with ctrl+click.
        # THis is where "map_count" is used in _make_sbatch_string:
        # if map_count is not None:
        #     assert isinstance(map_count, int) and map_count
        #     parameters["array"] = f"0-{map_count - 1}%{min(map_count, array_parallelism)}"
        #     stdout = stdout.replace("%j", "%A_%a")
        #     stderr = stderr.replace("%j", "%A_%a")

        self._throttle()
        # Maybe that's an example of a case where we want to keep the submission file as a symlink?
        # WHY not use `array_ex.submit` here?
        first_job: core.Job[tp.Any] = array_ex._submit_command(
            self._submitit_command_str, _keep_sbatch_file_as_symlink=True
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
            _get_remote_path = self.remote_dir_sync._get_remote_path
            remote_pickle_path = self.remote_dir_sync.copy_to_remote(local_pickle_path)

            self.login_node.run(f"mkdir -p {_get_remote_path(job.paths.folder)}", display=False)
            self.login_node.run(
                f"mv {remote_pickle_path} {_get_remote_path(job.paths.submitted_pickle)}",
                display=False,
            )
            # Reflect this locally as well?
            job.paths.submitted_pickle.parent.mkdir(exist_ok=True, parents=True)
            local_pickle_path.rename(job.paths.submitted_pickle)

        return jobs

    def _make_submission_file_text(self, command: str, uid: str) -> str:
        # return _make_sbatch_string(
        #     command=command, folder=self.remote_folder, **self.parameters
        # )
        content_with_local_paths = slurm._make_sbatch_string(
            command=command, folder=self.folder, **self.parameters
        )
        content_with_remote_paths = content_with_local_paths.replace(
            str(self.local_base_folder.absolute()), str(self.remote_base_folder)
        )

        # Note: annoying, but seems like `srun_args` is fed through shlex.quote or
        # something, which causes issues with the evaluation of variables.
        chdir_to_worktree = "--chdir=$WORKTREE_LOCATION"
        return content_with_remote_paths.replace(f"'{chdir_to_worktree}'", chdir_to_worktree)

    def _num_tasks(self) -> int:
        nodes: int = self.parameters.get("nodes", 1)
        tasks_per_node: int = max(1, self.parameters.get("ntasks_per_node", 1))
        return nodes * tasks_per_node

    def _make_submission_command(self, submission_file_path: PurePath) -> list[str]:
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
        # return -1 if shutil.which("srun") is None else 2
        return 2


_Path = TypeVar("_Path", bound=PurePath)


def get_first_id_independent_folder(folder: _Path) -> _Path:
    """Returns the closest folder which is id independent."""
    # This is different than in the `core.JobPaths.get_first_id_independent_folder` method:
    # we don't try to resolve the path to be absolute since it might be a pure (remote) path.
    tags = ["%j", "%t", "%A", "%a"]
    indep_parts = itertools.takewhile(
        lambda path_part: not any(tag in path_part for tag in tags), folder.parts
    )
    return type(folder)(*indep_parts)


def _current_repo_url():
    return LocalV2.get_output("git config --get remote.origin.url", display=False)


def _current_repo_name() -> str:
    return _current_repo_url().split("/")[-1].removesuffix(".git")


def _current_branch_name():
    return LocalV2.get_output("git rev-parse --abbrev-ref HEAD", display=False)


def _current_commit():
    return LocalV2.get_output("git rev-parse HEAD", display=False)


def _current_commit_short():
    return LocalV2.get_output("git rev-parse --short HEAD", display=False)


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
    logger.info(f"Fetching the list of SLURM accounts available on the {cluster} cluster.")
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


def sync_source_code(
    login_node: LoginNode,
    repo_dir_on_cluster: PurePosixPath,
    repo_url: str,
    commit: str,
):
    """Sync the local source code with the remote cluster."""

    if uncommitted_changes := LocalV2.get_output("git status --porcelain"):
        console.log(
            UserWarning(
                "You have uncommitted changes! Please consider adding and committing them before re-running the command.\n"
                "(This the best way to guarantee that the same code will be used on the remote cluster "
                "and helps make your experiments easier to reproduce in the future.)\n"
                "Uncommitted changes:\n\n" + textwrap.indent(uncommitted_changes, "\t")
            ),
            style="orange3",  # Why the hell isn't 'orange' a colour?!
        )
    LocalV2.run("git push", display=True)

    # If the repo doesn't exist on the remote, clone it:
    if not login_node.dir_exists(repo_dir_on_cluster):
        login_node.run(f"mkdir -p {repo_dir_on_cluster.parent}")
        login_node.run(
            f"git clone {repo_url} {repo_dir_on_cluster}",
            display=True,
        )

    # In any case, fetch the latest changes on the remote and checkout that commit.
    with login_node.chdir(repo_dir_on_cluster):
        login_node.run("git fetch", display=True)
        login_node.run(f"git checkout {commit}", display=True)
