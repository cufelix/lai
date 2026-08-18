"""Skill discovery and progressive disclosure.

A skill is a directory containing ``SKILL.md`` with YAML-ish frontmatter:

    ---
    name: my-skill
    description: Use when doing X
    ---
    <instructions>

The format is deliberately Claude Code's, so the thousands of skills already
written for it work here unchanged — the agent gains new procedures without any
code being written.

Progressive disclosure is the point: only names and descriptions go into the
system prompt; a skill's body is loaded on demand via the ``skill_load`` tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import SkillError

SKILL_FILENAMES = ("SKILL.md", "skill.md")
MAX_SKILL_BYTES = 200_000
FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Skill:
    """One reusable procedure."""

    name: str
    description: str
    path: Path
    body: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def scripts(self) -> list[Path]:
        """Executable helpers bundled with the skill."""
        found: list[Path] = []
        for sub in ("scripts", "bin"):
            directory = self.directory / sub
            if directory.is_dir():
                found.extend(sorted(p for p in directory.iterdir() if p.is_file()))
        return found

    @property
    def references(self) -> list[Path]:
        directory = self.directory / "references"
        return sorted(p for p in directory.iterdir() if p.is_file()) if directory.is_dir() else []

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "source": self.source,
            "scripts": [str(p) for p in self.scripts],
            "references": [str(p) for p in self.references],
        }

    def render(self, *, include_files: bool = True) -> str:
        """Full text handed to the model when the skill is loaded."""
        parts = [f"# Skill: {self.name}", "", self.description, "", "---", "", self.body.strip()]
        if include_files:
            scripts = self.scripts
            references = self.references
            if scripts:
                parts += [
                    "",
                    "## Bundled scripts",
                    "Run these with shell_exec (they live next to the skill):",
                    *[f"- {p}" for p in scripts],
                ]
            if references:
                parts += [
                    "",
                    "## Reference files",
                    "Read these with file_read when the skill points you at them:",
                    *[f"- {p}" for p in references],
                ]
        return "\n".join(parts)


class SkillRegistry:
    """Finds skills across every configured directory."""

    def __init__(self, paths: list[Path] | None = None, *, max_depth: int = 3) -> None:
        self.paths = [Path(p) for p in (paths or [])]
        self.max_depth = max_depth
        self._skills: dict[str, Skill] | None = None

    def refresh(self) -> SkillRegistry:
        self._skills = None
        return self

    def _load(self) -> dict[str, Skill]:
        if self._skills is not None:
            return self._skills
        found: dict[str, Skill] = {}
        for root in self.paths:
            if not root.is_dir():
                continue
            for skill_file in _find_skill_files(root, self.max_depth):
                skill = _parse_skill(skill_file, source=str(root))
                # First path wins: earlier entries are higher priority.
                if skill and skill.name not in found:
                    found[skill.name] = skill
        self._skills = found
        return found

    def list(self) -> list[Skill]:
        return sorted(self._load().values(), key=lambda s: s.name)

    def __len__(self) -> int:
        return len(self._load())

    def get(self, name: str) -> Skill:
        skills = self._load()
        skill = skills.get(name)
        if skill is not None:
            return skill

        needle = name.lower().strip()
        exact = [s for s in skills.values() if s.name.lower() == needle]
        if exact:
            return exact[0]
        partial = [s for s in skills.values() if needle in s.name.lower()]
        if len(partial) == 1:
            return partial[0]
        if partial:
            raise SkillError(
                f"skill {name!r} is ambiguous",
                detail=f"matches: {', '.join(sorted(s.name for s in partial)[:10])}",
            )
        raise SkillError(
            f"no skill named {name!r}",
            detail=f"available: {', '.join(sorted(skills)[:25])}" if skills else "no skills installed",
        )

    def search(self, query: str, *, limit: int = 10) -> list[Skill]:
        needle = query.lower().strip()
        if not needle:
            return self.list()[:limit]
        scored: list[tuple[int, Skill]] = []
        for skill in self._load().values():
            score = 0
            if needle == skill.name.lower():
                score = 100
            elif needle in skill.name.lower():
                score = 60
            elif needle in skill.description.lower():
                score = 30
            elif any(word in skill.description.lower() for word in needle.split()):
                score = 10
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [skill for _, skill in scored[:limit]]


def _find_skill_files(root: Path, max_depth: int) -> list[Path]:
    """Locate SKILL.md files without walking an entire home directory."""
    found: list[Path] = []

    def walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir())
        except (OSError, PermissionError):
            return
        for entry in entries:
            if entry.is_file() and entry.name in SKILL_FILENAMES:
                found.append(entry)
            elif entry.is_dir() and not entry.name.startswith((".", "node_modules", "__pycache__")):
                walk(entry, depth + 1)

    walk(root, 0)
    return found


def _parse_skill(path: Path, *, source: str = "") -> Skill | None:
    try:
        if path.stat().st_size > MAX_SKILL_BYTES:
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    match = FRONTMATTER.match(text)
    metadata: dict = {}
    body = text
    if match:
        metadata = _parse_frontmatter(match.group(1))
        body = text[match.end():]

    name = str(metadata.get("name") or path.parent.name).strip()
    description = str(metadata.get("description") or "").strip()
    if not name:
        return None
    if not description:
        # Fall back to the first prose line so the skill is still discoverable.
        for line in body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                description = stripped[:300]
                break
    return Skill(
        name=name,
        description=description or "(no description)",
        path=path,
        body=body,
        source=source,
        metadata=metadata,
    )


def _parse_frontmatter(raw: str) -> dict:
    """Parse the small YAML subset skills actually use.

    Avoids a hard PyYAML dependency: keys, scalar values, quoted strings,
    simple inline lists, and multi-line folded values.
    """
    data: dict = {}
    key: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal key, buffer
        if key is not None:
            joined = " ".join(part.strip() for part in buffer).strip()
            if joined:
                data[key] = _coerce(joined)
        key, buffer = None, []

    for line in raw.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if re.match(r"^\s+", line) and key is not None:
            buffer.append(line)
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*:\s*(.*)$", line)
        if match:
            flush()
            key = match.group(1)
            value = match.group(2).strip()
            if value:
                buffer.append(value)
        elif key is not None:
            buffer.append(line)
    flush()
    return data


def _coerce(value: str):
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [part.strip().strip("\"'") for part in inner.split(",") if part.strip()] if inner else []
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    return text
