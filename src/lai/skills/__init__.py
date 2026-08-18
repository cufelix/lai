"""Skills: reusable procedures, discovered locally or installed from the internet."""

# NB: the function is exported as ``install_skill`` so it does not shadow the
# ``lai.skills.install`` submodule of the same name.
from .install import InstallResult, uninstall
from .install import install as install_skill
from .registry import Skill, SkillRegistry
from .tools import register as register_skill_tools

__all__ = [
    "InstallResult",
    "Skill",
    "SkillRegistry",
    "install_skill",
    "register_skill_tools",
    "uninstall",
]
