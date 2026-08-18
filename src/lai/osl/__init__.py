"""OS layer: perception and actuation primitives for the Linux desktop."""

from .a11y import A11yTree, Element, Snapshot
from .apps import AppEntry, AppLauncher, LaunchResult
from .clipboard import Clipboard
from .desktop import Desktop, Observation
from .geometry import Monitor, Point, Rect
from .idle import IdleMonitor, IdleState
from .inputs import InputController, InputResult, normalize_key
from .notifications import Notification, NotificationMonitor, send_notification
from .ocr import OCREngine, OCRResult, OCRWord
from .recorder import RecordingInfo, ScreenRecorder, to_gif
from .screen import ScreenCapture, Screenshot, annotate
from .windows import WindowInfo, WindowManager

__all__ = [
    "A11yTree",
    "AppEntry",
    "AppLauncher",
    "Clipboard",
    "Desktop",
    "Element",
    "IdleMonitor",
    "IdleState",
    "InputController",
    "InputResult",
    "LaunchResult",
    "Monitor",
    "Notification",
    "NotificationMonitor",
    "OCREngine",
    "OCRResult",
    "OCRWord",
    "Observation",
    "Point",
    "RecordingInfo",
    "Rect",
    "ScreenCapture",
    "ScreenRecorder",
    "Screenshot",
    "Snapshot",
    "WindowInfo",
    "WindowManager",
    "annotate",
    "normalize_key",
    "send_notification",
    "to_gif",
]
