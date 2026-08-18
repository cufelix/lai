"""Screen recording via ffmpeg's ``x11grab``, for showing what the agent did.

Two design decisions here are not obvious and matter a lot in practice:

1. **Stopping must be graceful.** Killing ``x11grab`` with SIGKILL — or just
   letting the ``Popen`` object be garbage collected while it's still running
   — leaves the MP4's moov atom unwritten, so the container is truncated and
   most players (and re-encoders, like :func:`to_gif`) refuse to open it.
   ffmpeg finalizes the file cleanly if it receives ``'q'`` on stdin (its
   documented interactive-quit key) or SIGINT. We try those, in order, and
   only escalate to SIGKILL if ffmpeg is completely wedged and ignores both —
   at which point the file may still be broken, but at least we didn't leave
   an orphan process capturing the screen forever.
2. **The recorder cleans up after itself.** It works as a context manager,
   and also makes a best-effort kill from ``__del__`` as a last line of
   defence, so a caller that forgets to call ``stop()`` (or crashes) can
   never leave ffmpeg running indefinitely in the background.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from ..errors import BackendUnavailable, LaiError
from .geometry import Rect

STOP_GRACE_SECONDS = 5.0
FFMPEG_TIMEOUT = 120.0
GIF_FPS_DEFAULT = 8
GIF_WIDTH_DEFAULT = 800


@dataclass(frozen=True, slots=True)
class RecordingInfo:
    """State of one recording session."""

    path: Path
    region: Rect | None
    fps: int
    started_at: float
    stopped_at: float | None = None

    @property
    def duration(self) -> float | None:
        return None if self.stopped_at is None else self.stopped_at - self.started_at

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "region": self.region.to_dict() if self.region else None,
            "fps": self.fps,
            "duration": round(self.duration, 2) if self.duration is not None else None,
        }


def _ffmpeg_args(output_path: Path, *, display: str, region: Rect | None, fps: int) -> list[str]:
    """Build the ``ffmpeg -f x11grab ...`` argument list.

    Factored out (and taking no ``self``) so tests can check exactly what
    would be launched without spawning ffmpeg.
    """
    args = ["ffmpeg", "-y", "-f", "x11grab", "-framerate", str(fps)]
    if region is not None:
        args += ["-video_size", f"{region.width}x{region.height}", "-i", f"{display}+{region.x},{region.y}"]
    else:
        args += ["-i", display]
    args += ["-pix_fmt", "yuv420p", str(output_path)]
    return args


class ScreenRecorder:
    """One ffmpeg ``x11grab`` session at a time."""

    def __init__(self, *, display: str | None = None) -> None:
        self._display = display or os.environ.get("DISPLAY", ":0")
        self._ffmpeg = shutil.which("ffmpeg")
        self._proc: subprocess.Popen | None = None
        self._info: RecordingInfo | None = None
        self.last_info: RecordingInfo | None = None
        """The most recently completed recording's info, set by :meth:`stop`."""

    @property
    def available(self) -> bool:
        return self._ffmpeg is not None

    @property
    def recording(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- lifecycle -----------------------------------------------------------

    def start(self, output_path: str | Path, *, region: Rect | None = None, fps: int = 10) -> RecordingInfo:
        if not self.available:
            raise BackendUnavailable("ffmpeg is not installed", detail="install it with: sudo apt install ffmpeg")
        if self.recording:
            raise LaiError("a recording is already in progress; call stop() first")
        if not 1 <= fps <= 60:
            raise LaiError(f"fps must be 1..60, got {fps}")

        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        args = _ffmpeg_args(path, display=self._display, region=region, fps=fps)
        self._proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._info = RecordingInfo(path=path, region=region, fps=fps, started_at=time.time())
        return self._info

    def stop(self, *, grace: float = STOP_GRACE_SECONDS) -> Path:
        if self._proc is None or self._info is None:
            raise LaiError("no recording in progress")
        proc, info = self._proc, self._info

        if proc.poll() is None:
            _terminate_gracefully(proc, grace=grace)

        self._proc = None
        self._info = None
        self.last_info = replace(info, stopped_at=time.time())
        return info.path

    def record(self, seconds: float, output_path: str | Path, *, region: Rect | None = None, fps: int = 10) -> Path:
        """Convenience: record for a fixed duration and return the file path."""
        self.start(output_path, region=region, fps=fps)
        time.sleep(max(0.0, seconds))
        return self.stop()

    def __enter__(self) -> ScreenRecorder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.recording:
            try:
                self.stop()
            except Exception:
                pass

    def __del__(self) -> None:
        # Last line of defence: never let a forgotten recorder leave ffmpeg
        # capturing the screen forever. Best-effort only — __del__ runs
        # during (possibly partial) interpreter teardown, so every failure
        # mode here is swallowed rather than raised.
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.kill()
        except Exception:
            pass


def _terminate_gracefully(proc: subprocess.Popen, *, grace: float) -> None:
    """'q' on stdin (ffmpeg's documented interactive quit) finalizes the
    container cleanly. SIGINT is the signal-based equivalent, for when stdin
    isn't being read yet. SIGKILL is the last resort and *will* corrupt the
    output — it should only ever be reached if ffmpeg is completely wedged.
    """
    try:
        if proc.stdin is not None:
            proc.stdin.write(b"q")
            proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    try:
        proc.kill()
        proc.wait(timeout=2.0)
    except Exception:
        pass


def to_gif(
    video_path: str | Path, gif_path: str | Path, *,
    fps: int = GIF_FPS_DEFAULT, width: int = GIF_WIDTH_DEFAULT,
    timeout: float = FFMPEG_TIMEOUT,
) -> Path:
    """Convert a recording to a GIF using the two-pass palettegen/paletteuse recipe.

    A single-pass GIF encode reuses ffmpeg's fixed, generic 256-colour
    palette and looks banded and dithered on anything but flat UI chrome.
    Two passes — first generate a palette optimised for *this specific clip*
    (``palettegen``), then re-encode against it (``paletteuse``) — is the
    standard ffmpeg recipe for a GIF that doesn't look like it's from 1998,
    at the cost of decoding the input twice.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise BackendUnavailable("ffmpeg is not installed", detail="install it with: sudo apt install ffmpeg")

    video_path, gif_path = Path(video_path), Path(gif_path)
    if not video_path.is_file():
        raise LaiError(f"video file does not exist: {video_path}")
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    filters = f"fps={fps},scale={width}:-1:flags=lanczos"
    with tempfile.TemporaryDirectory() as tmp:
        palette = Path(tmp) / "palette.png"
        _run_ffmpeg(
            [ffmpeg, "-y", "-i", str(video_path), "-vf", f"{filters},palettegen", str(palette)], timeout=timeout
        )
        _run_ffmpeg(
            [
                ffmpeg, "-y", "-i", str(video_path), "-i", str(palette),
                "-lavfi", f"{filters}[x];[x][1:v]paletteuse", str(gif_path),
            ],
            timeout=timeout,
        )
    return gif_path


def _run_ffmpeg(args: list[str], *, timeout: float) -> None:
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BackendUnavailable(f"ffmpeg timed out after {timeout}s", detail=" ".join(args)) from exc
    if proc.returncode != 0:
        raise LaiError(
            "ffmpeg failed", detail=(proc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
        )
