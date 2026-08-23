"""LAI — a native OS-level autonomous agent for Linux.

The desktop is the page: LAI perceives it through the accessibility tree, the
window tree and the screen, acts on it through real input events and semantic
accessibility actions, and runs an autonomous observe-act-verify loop over it.
"""

__version__ = "1.7.0"

from .config import Config, load_config
from .errors import LaiError
from .runtime import Runtime, build_runtime

__all__ = ["Config", "LaiError", "Runtime", "__version__", "build_runtime", "load_config"]
