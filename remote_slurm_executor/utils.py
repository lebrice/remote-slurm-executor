import contextlib
import dataclasses
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import milatools.utils.local_v2
from milatools.cli import console
from milatools.cli.utils import SSH_CONFIG_FILE
from milatools.utils.local_v2 import Hide
from milatools.utils.remote_v2 import RemoteV2
from typing_extensions import override


@dataclasses.dataclass(init=False)
class LoginNode(RemoteV2):
    # Tiny improvements / changes to the RemoteV2 class from milatools.

    command_prefix: str = ""
    displayed_prefix: str = ""

    def __init__(
        self,
        hostname: str,
        *,
        control_path: Path | None = None,
        ssh_config_path: Path = SSH_CONFIG_FILE,
        _start_control_socket: bool = True,
        command_prefix: str = "",
        displayed_prefix: str | None = None,
    ):
        super().__init__(
            hostname,
            control_path=control_path,
            ssh_config_path=ssh_config_path,
            _start_control_socket=_start_control_socket,
        )
        self._start_control_socket = _start_control_socket
        self.command_prefix = command_prefix
        self.displayed_prefix = displayed_prefix or command_prefix

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
        before = self.command_prefix
        before_text = self.displayed_prefix
        if self.command_prefix:
            self.command_prefix = f"{self.command_prefix} && cd {remote_dir} && "
        else:
            self.command_prefix = f"cd {remote_dir} && "
        # Change the display of commands.
        if self.displayed_prefix:
            self.displayed_prefix = f"{self.displayed_prefix} ({remote_dir}) "
        else:
            self.displayed_prefix = f"({remote_dir}) "

        yield

        self.command_prefix = before
        self.displayed_prefix = before_text

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
            self.display(self.displayed_prefix + command, input=input, _stack_offset=3)
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
            self.display(self.displayed_prefix + command, input=input, _stack_offset=3)
        return await super().run_async(
            self.command_prefix + command,
            input=input,
            display=False,
            warn=warn,
            hide=hide,
        )


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
        return self.remote_dir / (local_path.absolute().relative_to(self.local_dir.absolute()))

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


def run(
    program_and_args: tuple[str, ...],
    input: str | None = None,
    warn: bool = False,
    hide: Hide = False,
) -> subprocess.CompletedProcess[str]:
    """Runs the command *synchronously* in a subprocess and returns the result.

    Parameters
    ----------
    program_and_args: The program and arguments to pass to it. This is a tuple of \
        strings, same as in `subprocess.Popen`.
    input: The optional 'input' argument to `subprocess.Popen.communicate()`.
    warn: When `True` and an exception occurs, warn instead of raising the exception.
    hide: Controls the printing of the subprocess' stdout and stderr.

    Returns
    -------
    The `subprocess.CompletedProcess` object with the result of the subprocess.

    Raises
    ------
    subprocess.CalledProcessError
        If an error occurs when running the command and `warn` is `False`.
    """
    logger = milatools.utils.local_v2.logger
    displayed_command = shlex.join(program_and_args)
    if not input:
        logger.debug(f"Calling `subprocess.run` with {program_and_args=}")
    else:
        logger.debug(f"Calling `subprocess.run` with {program_and_args=} and {input=}")
    try:
        result = subprocess.run(
            program_and_args,
            shell=False,
            capture_output=True,
            text=True,
            check=not warn,
            input=input,
        )
    ### Changed:
    except subprocess.CalledProcessError as e:
        if hide not in [True, "err", "stderr"]:
            print(e.stderr, file=sys.stderr)
        raise e
    ### End of change.
    assert result.returncode is not None
    if warn and result.returncode != 0:
        message = (
            f"Command {displayed_command!r}"
            + (f" with {input=!r}" if input else "")
            + f" exited with {result.returncode}: {result.stderr=}"
        )
        logger.debug(message)
        if hide is not True:  # don't warn if hide is True.
            logger.warning(RuntimeWarning(message), stacklevel=2)

    if result.stdout:
        if hide not in [True, "out", "stdout"]:
            print(result.stdout)
        logger.debug(f"{result.stdout=}")
    if result.stderr:
        if hide not in [True, "err", "stderr"]:
            print(result.stderr, file=sys.stderr)
        logger.debug(f"{result.stderr=}")
    return result


# temporary patch until it's added to milatools.
milatools.utils.local_v2.run = run
