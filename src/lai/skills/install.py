"""Install skills from the internet.

Sources: a git repository, a zip/tarball URL, or a local directory. Everything
lands in ``~/.lai/skills/<name>``, which is the first search path, so an
installed skill is available on the next run with no restart.

Downloads are treated as untrusted input: archive members are path-checked
before extraction, sizes are capped, and nothing is executed at install time.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..errors import SkillError
from .registry import SKILL_FILENAMES, _find_skill_files, _parse_skill

MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
MAX_MEMBER_BYTES = 20 * 1024 * 1024
CLONE_TIMEOUT = 120
GIT_URL = re.compile(r"^(https?://|git@|ssh://|git://).*", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class InstallResult:
    installed: tuple[str, ...]
    destination: Path
    source: str
    skipped: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "installed": list(self.installed),
            "skipped": list(self.skipped),
            "destination": str(self.destination),
            "source": self.source,
        }


def install(source: str, target_dir: Path, *, name: str | None = None, overwrite: bool = False) -> InstallResult:
    """Install skill(s) from ``source`` into ``target_dir``."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    local = Path(source).expanduser()
    if local.exists():
        return _install_from_dir(local, target_dir, name=name, overwrite=overwrite, source=str(local))

    if source.endswith((".zip", ".tar.gz", ".tgz", ".tar")) and source.startswith(("http://", "https://")):
        return _install_from_archive(source, target_dir, name=name, overwrite=overwrite)

    if GIT_URL.match(source) or source.endswith(".git"):
        return _install_from_git(source, target_dir, name=name, overwrite=overwrite)

    # Bare "owner/repo" is a GitHub shorthand.
    if re.fullmatch(r"[\w.-]+/[\w.-]+", source):
        return _install_from_git(f"https://github.com/{source}.git", target_dir, name=name, overwrite=overwrite)

    raise SkillError(
        f"cannot work out how to install {source!r}",
        detail="give a git URL, a zip/tarball URL, 'owner/repo', or a local directory path",
    )


def _install_from_git(url: str, target_dir: Path, *, name: str | None, overwrite: bool) -> InstallResult:
    if not shutil.which("git"):
        raise SkillError("git is not installed", detail="sudo apt install git")
    with tempfile.TemporaryDirectory(prefix="lai-skill-") as tmp:
        checkout = Path(tmp) / "repo"
        try:
            proc = subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", url, str(checkout)],
                capture_output=True,
                text=True,
                timeout=CLONE_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SkillError(f"git clone timed out after {CLONE_TIMEOUT}s", detail=url) from exc
        if proc.returncode != 0:
            raise SkillError(f"git clone failed for {url}", detail=(proc.stderr or "").strip()[:400])
        return _install_from_dir(checkout, target_dir, name=name, overwrite=overwrite, source=url)


def _install_from_archive(url: str, target_dir: Path, *, name: str | None, overwrite: bool) -> InstallResult:
    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SkillError(f"download failed: {url}", detail=str(exc)) from exc

    payload = response.content
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise SkillError(f"archive is too large ({len(payload)} bytes; limit {MAX_ARCHIVE_BYTES})")

    with tempfile.TemporaryDirectory(prefix="lai-skill-") as tmp:
        extracted = Path(tmp) / "extracted"
        extracted.mkdir()
        if url.endswith(".zip"):
            _extract_zip(payload, extracted)
        else:
            _extract_tar(payload, extracted)
        return _install_from_dir(extracted, target_dir, name=name, overwrite=overwrite, source=url)


def _extract_zip(payload: bytes, destination: Path) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            # Only the members that passed the checks are extracted. Filtering
            # in a loop and then calling extractall() on the whole archive —
            # which is what this used to do — made the size limit advisory:
            # the member was skipped in the loop and written anyway.
            members = []
            for member in archive.infolist():
                if member.file_size > MAX_MEMBER_BYTES:
                    continue
                _safe_extract_path(destination, member.filename)
                members.append(member)
            archive.extractall(destination, members=members)  # noqa: S202 - validated above
    except zipfile.BadZipFile as exc:
        raise SkillError("downloaded file is not a valid zip archive") from exc


def _extract_tar(payload: bytes, destination: Path) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            members = []
            for member in archive.getmembers():
                if member.size > MAX_MEMBER_BYTES or member.issym() or member.islnk():
                    continue
                _safe_extract_path(destination, member.name)
                members.append(member)
            archive.extractall(destination, members=members)  # noqa: S202 - validated above
    except tarfile.TarError as exc:
        raise SkillError("downloaded file is not a valid tar archive") from exc


def _safe_extract_path(root: Path, member_name: str) -> None:
    """Reject absolute paths and ../ traversal before extracting anything.

    Uses path-component containment, not a string prefix: with a destination of
    ``/tmp/skills``, the member ``../skills-evil/x`` resolves to
    ``/tmp/skills-evil/x``, which *does* start with the destination string
    while landing entirely outside it.
    """
    resolved = (root / member_name).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise SkillError(f"archive contains an unsafe path: {member_name!r}")


def _install_from_dir(
    source_dir: Path, target_dir: Path, *, name: str | None, overwrite: bool, source: str
) -> InstallResult:
    source_dir = Path(source_dir)
    if source_dir.is_file() and source_dir.name in SKILL_FILENAMES:
        source_dir = source_dir.parent
    if not source_dir.is_dir():
        raise SkillError(f"not a directory: {source_dir}")

    skill_files = _find_skill_files(source_dir, max_depth=4)
    if not skill_files:
        raise SkillError(
            f"no SKILL.md found in {source}",
            detail="a skill directory must contain SKILL.md with name and description frontmatter",
        )

    installed: list[str] = []
    skipped: list[str] = []
    for skill_file in skill_files:
        skill = _parse_skill(skill_file)
        if skill is None:
            skipped.append(str(skill_file))
            continue
        folder_name = name if (name and len(skill_files) == 1) else skill.name
        destination = target_dir / _safe_name(folder_name)
        if destination.exists():
            if not overwrite:
                skipped.append(f"{skill.name} (already installed)")
                continue
            shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(skill_file.parent, destination, dirs_exist_ok=True)
        installed.append(skill.name)

    if not installed and skipped:
        raise SkillError(
            "nothing installed",
            detail="; ".join(skipped[:6]) + " — pass overwrite=true to replace existing skills",
        )
    return InstallResult(tuple(installed), target_dir, source, tuple(skipped))


def uninstall(name: str, target_dir: Path) -> bool:
    destination = Path(target_dir) / _safe_name(name)
    if not destination.is_dir():
        return False
    shutil.rmtree(destination, ignore_errors=True)
    return True


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    if not cleaned or cleaned in (".", ".."):
        raise SkillError(f"unusable skill name: {name!r}")
    return cleaned[:80]
