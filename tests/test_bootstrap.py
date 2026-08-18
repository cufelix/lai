"""The one-line installer.

`curl … | sh` is the path most people will actually take, and it is the one
place where a mistake happens before any Python exists to catch it. These tests
drive the real script against a throwaway git repository.

They deliberately stop at `--no-setup`: building a virtualenv and installing
packages is the installer's job, not the bootstrap's, and is covered by running
it for real rather than in the suite.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

BOOTSTRAP = Path(__file__).resolve().parents[1] / "packaging" / "bootstrap.sh"


def git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=str(cwd), check=True, capture_output=True,
    )


@pytest.fixture
def origin(tmp_path: Path) -> Path:
    """A minimal repository the bootstrap can clone."""
    repo = tmp_path / "origin"
    (repo / "packaging").mkdir(parents=True)
    # A stand-in installer: records that it ran, with its arguments.
    (repo / "packaging" / "install.sh").write_text(
        '#!/usr/bin/env bash\necho "INSTALLER RAN args=$*"\n', encoding="utf-8"
    )
    (repo / "marker.txt").write_text("v1\n", encoding="utf-8")
    git("init", "-q", "-b", "main", ".", cwd=repo)
    git("add", "-A", cwd=repo)
    git("commit", "-qm", "first", cwd=repo)
    return repo


def run(origin: Path | str, target: Path, *args: str, ref: str = "main", piped: bool = True):
    """Run the bootstrap the way a user would — piped into sh."""
    env = dict(
        os.environ,
        LAI_REPO=f"file://{origin}" if not str(origin).startswith("file://") else str(origin),
        LAI_REF=ref,
        LAI_DIR=str(target),
    )
    command = ["sh", "-s", "--", *args] if piped else ["sh", str(BOOTSTRAP), *args]
    return subprocess.run(
        command,
        input=BOOTSTRAP.read_text(encoding="utf-8") if piped else None,
        capture_output=True, text=True, env=env, timeout=120,
    )


def test_the_script_is_valid_posix_sh():
    """It runs before anything is known about the machine, so it cannot need bash."""
    result = subprocess.run(["sh", "-n", str(BOOTSTRAP)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_it_clones_and_hands_over_to_the_installer(origin, tmp_path):
    target = tmp_path / "install"
    result = run(origin, target)
    assert result.returncode == 0, result.stderr
    assert "INSTALLER RAN" in result.stdout
    assert (target / ".git").is_dir()
    assert (target / "marker.txt").read_text().strip() == "v1"


def test_flags_are_passed_through_to_the_installer(origin, tmp_path):
    result = run(origin, tmp_path / "install", "--no-setup")
    assert "INSTALLER RAN args=--no-setup" in result.stdout


def test_re_running_updates_instead_of_failing(origin, tmp_path):
    target = tmp_path / "install"
    assert run(origin, target).returncode == 0

    (origin / "marker.txt").write_text("v2\n", encoding="utf-8")
    git("add", "-A", cwd=origin)
    git("commit", "-qm", "second", cwd=origin)

    result = run(origin, target)
    assert result.returncode == 0, result.stderr
    assert "already cloned" in result.stdout
    assert (target / "marker.txt").read_text().strip() == "v2", "an update must land"


def test_local_modifications_do_not_block_an_update(origin, tmp_path):
    """A user poking at the checkout must not wedge their next install."""
    target = tmp_path / "install"
    run(origin, target)
    (target / "marker.txt").write_text("locally edited\n", encoding="utf-8")

    result = run(origin, target)
    assert result.returncode == 0, result.stderr
    assert (target / "marker.txt").read_text().strip() == "v1"


def test_a_non_repository_target_is_refused(origin, tmp_path):
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "important.txt").write_text("do not delete me", encoding="utf-8")

    result = run(origin, target)
    assert result.returncode != 0
    assert "not a git checkout" in result.stderr
    assert (target / "important.txt").exists(), "it must not touch what it did not create"


def test_an_unknown_ref_fails_clearly(origin, tmp_path):
    result = run(origin, tmp_path / "install", ref="no-such-branch")
    assert result.returncode != 0
    assert "could not clone" in result.stderr


def test_an_unreachable_repository_fails_clearly(tmp_path):
    result = run(tmp_path / "nothing-here", tmp_path / "install")
    assert result.returncode != 0
    assert "could not clone" in result.stderr


def test_the_baked_in_url_is_a_real_repository():
    """The shipped default must be the real repo, never the OWNER placeholder."""
    body = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'LAI_REPO:-https://github.com/cufelix/lai.git' in body


def test_an_explicit_placeholder_url_refuses_to_run(tmp_path):
    """A fork that re-placeholders the URL must fail, not clone something wrong."""
    env = dict(os.environ, LAI_DIR=str(tmp_path / "install"), LAI_REPO="https://github.com/OWNER/lai.git")
    result = subprocess.run(
        ["sh", "-s"], input=BOOTSTRAP.read_text(encoding="utf-8"),
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert result.returncode != 0
    assert "no repository URL" in result.stderr
    assert not (tmp_path / "install").exists()


def test_it_warns_when_there_is_no_terminal(origin, tmp_path):
    """Piped into sh with no tty, the wizard silently takes defaults — say so."""
    result = run(origin, tmp_path / "install")
    assert "no terminal available" in result.stdout
    assert "packaging/install.sh" in result.stdout


def test_a_missing_installer_is_reported(tmp_path):
    """A checkout without the second stage must fail loudly, not half-install."""
    repo = tmp_path / "origin"
    repo.mkdir()
    (repo / "readme.md").write_text("no installer here\n", encoding="utf-8")
    git("init", "-q", "-b", "main", ".", cwd=repo)
    git("add", "-A", cwd=repo)
    git("commit", "-qm", "first", cwd=repo)

    result = run(repo, tmp_path / "install")
    assert result.returncode != 0
    assert "no packaging/install.sh" in result.stderr


@pytest.mark.skipif(shutil.which("dash") is None, reason="dash is not installed")
def test_it_runs_under_dash(origin, tmp_path):
    """`sh` is dash on Debian and Ubuntu, so bashisms would break the one-liner."""
    env = dict(os.environ, LAI_REPO=f"file://{origin}", LAI_REF="main",
               LAI_DIR=str(tmp_path / "install"))
    result = subprocess.run(
        ["dash", "-s", "--", "--no-setup"], input=BOOTSTRAP.read_text(encoding="utf-8"),
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "INSTALLER RAN" in result.stdout


def test_it_reattaches_the_terminal_when_piped(origin, tmp_path):
    """The whole point of `curl … | sh`: the wizard must still be able to ask.

    Piped into sh, stdin *is* the script, so a naive installer reads EOF instead
    of the human and silently takes every default. This models the real thing —
    stdin is a pipe, the terminal is a separate pty — and proves the handover
    reaches the person at the keyboard.
    """
    import pty
    import select
    import time

    (origin / "packaging" / "install.sh").write_text(
        '#!/usr/bin/env bash\nprintf "ask> "\nread -r answer\necho "READ:[$answer]"\n',
        encoding="utf-8",
    )
    git("add", "-A", cwd=origin)
    git("commit", "-qm", "asking installer", cwd=origin)

    env = dict(os.environ, LAI_REPO=f"file://{origin}", LAI_REF="main",
               LAI_DIR=str(tmp_path / "install"))
    script = BOOTSTRAP.read_bytes()

    read_fd, write_fd = os.pipe()
    pid, pty_fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        os.close(write_fd)
        os.dup2(read_fd, 0)
        os.close(read_fd)
        os.execve("/bin/sh", ["/bin/sh", "-s"], env)

    os.close(read_fd)
    os.write(write_fd, script)
    os.close(write_fd)

    output, typed, deadline = b"", False, time.time() + 60
    while time.time() < deadline:
        ready, _, _ = select.select([pty_fd], [], [], 1.0)
        if not ready:
            continue
        try:
            chunk = os.read(pty_fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        output += chunk
        if not typed and b"ask>" in output:
            os.write(pty_fd, b"typed-by-a-human\n")
            typed = True
    os.close(pty_fd)
    os.waitpid(pid, 0)

    text = output.decode("utf-8", "replace")
    assert "READ:[typed-by-a-human]" in text, "the terminal was not reattached"
    assert "no terminal available" not in text
