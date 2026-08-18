"""Tests for lai.osl.recorder and the record_* tools in lai.tools.perception.

Argument construction and the graceful-stop escalation ladder ('q' on stdin
-> SIGINT -> SIGKILL) are tested against a fake ``subprocess.Popen`` so no
real ffmpeg process is ever spawned for those cases. The @pytest.mark.x11
tests record a very short real clip via ffmpeg's x11grab against the live
display and convert it to a GIF, since both ffmpeg and a display are known
to be available in this environment.
"""

from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path

import pytest

from lai.config import Config
from lai.errors import BackendUnavailable, LaiError
from lai.osl.geometry import Rect
from lai.osl.recorder import RecordingInfo, ScreenRecorder, _ffmpeg_args, to_gif
from lai.tools.base import ToolContext, ToolRegistry
from lai.tools.perception import register as register_perception


def context(**kwargs) -> ToolContext:
    kwargs.setdefault("config", Config())
    kwargs.setdefault("extra", {})
    return ToolContext(**kwargs)


# -- _ffmpeg_args: pure argument construction ------------------------------------------------------


def test_ffmpeg_args_whole_screen_has_no_video_size():
    args = _ffmpeg_args(Path("/tmp/out.mp4"), display=":0", region=None, fps=15)
    assert "-video_size" not in args
    assert args[args.index("-i") + 1] == ":0"
    assert args[args.index("-framerate") + 1] == "15"
    assert args[-1] == "/tmp/out.mp4"


def test_ffmpeg_args_region_sets_video_size_and_offset():
    region = Rect(100, 200, 640, 480)
    args = _ffmpeg_args(Path("/tmp/out.mp4"), display=":0", region=region, fps=10)
    assert args[args.index("-video_size") + 1] == "640x480"
    assert args[args.index("-i") + 1] == ":0+100,200"


def test_ffmpeg_args_uses_x11grab_and_overwrite_flag():
    args = _ffmpeg_args(Path("/tmp/out.mp4"), display=":1", region=None, fps=10)
    assert "-y" in args
    assert args[args.index("-f") + 1] == "x11grab"


# -- RecordingInfo ------------------------------------------------------


def test_recording_info_duration_is_none_before_stop():
    info = RecordingInfo(path=Path("/tmp/x.mp4"), region=None, fps=10, started_at=100.0)
    assert info.duration is None
    assert info.to_dict()["duration"] is None


def test_recording_info_duration_after_stop():
    info = RecordingInfo(path=Path("/tmp/x.mp4"), region=None, fps=10, started_at=100.0, stopped_at=102.5)
    assert info.duration == pytest.approx(2.5)


# -- ScreenRecorder.available / missing-ffmpeg failure path ------------------------------------------------------


def test_available_is_false_without_ffmpeg(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    recorder = ScreenRecorder()
    assert recorder.available is False


def test_start_raises_backend_unavailable_with_apt_hint(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    recorder = ScreenRecorder()
    with pytest.raises(BackendUnavailable, match="ffmpeg"):
        recorder.start("/tmp/out.mp4")


def test_start_rejects_fps_out_of_range(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    recorder = ScreenRecorder()
    with pytest.raises(LaiError, match="fps"):
        recorder.start("/tmp/out.mp4", fps=0)
    with pytest.raises(LaiError, match="fps"):
        recorder.start("/tmp/out.mp4", fps=61)


def test_stop_without_a_recording_raises():
    recorder = ScreenRecorder()
    with pytest.raises(LaiError, match="no recording"):
        recorder.stop()


# -- start()/stop() against a fake ffmpeg process ------------------------------------------------------


class _FakeStdin:
    def __init__(self) -> None:
        self.written = b""

    def write(self, data: bytes) -> None:
        self.written += data

    def flush(self) -> None:
        pass


class _FakeProc:
    """Stands in for subprocess.Popen without spawning a real process."""

    def __init__(self, *, quits_on: str | None = "stdin") -> None:
        self.stdin = _FakeStdin()
        self._alive = True
        self._quits_on = quits_on
        self.signals: list[int] = []
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        if self._quits_on == "stdin" and self.stdin.written == b"q":
            self._alive = False
        if self._alive:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout or 0)
        return 0

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        if self._quits_on == "sigint" and sig == signal.SIGINT:
            self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False


def _started_recorder(monkeypatch, tmp_path, fake_proc: _FakeProc) -> ScreenRecorder:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: fake_proc)
    recorder = ScreenRecorder()
    recorder.start(tmp_path / "out.mp4", fps=10)
    return recorder


def test_stop_quits_cleanly_via_stdin_q(monkeypatch, tmp_path):
    fake_proc = _FakeProc(quits_on="stdin")
    recorder = _started_recorder(monkeypatch, tmp_path, fake_proc)
    path = recorder.stop()
    assert path == tmp_path / "out.mp4"
    assert fake_proc.stdin.written == b"q"
    assert fake_proc.signals == []  # never had to escalate
    assert fake_proc.killed is False


def test_stop_escalates_to_sigint_when_stdin_is_ignored(monkeypatch, tmp_path):
    fake_proc = _FakeProc(quits_on="sigint")
    recorder = _started_recorder(monkeypatch, tmp_path, fake_proc)
    recorder.stop(grace=0.01)
    assert fake_proc.signals == [signal.SIGINT]
    assert fake_proc.killed is False


def test_stop_escalates_to_sigkill_when_totally_wedged(monkeypatch, tmp_path):
    fake_proc = _FakeProc(quits_on="never")
    recorder = _started_recorder(monkeypatch, tmp_path, fake_proc)
    recorder.stop(grace=0.01)
    assert fake_proc.signals == [signal.SIGINT]
    assert fake_proc.killed is True


def test_recording_property_reflects_process_state(monkeypatch, tmp_path):
    fake_proc = _FakeProc()
    recorder = _started_recorder(monkeypatch, tmp_path, fake_proc)
    assert recorder.recording is True
    recorder.stop()
    assert recorder.recording is False


def test_start_rejects_a_second_concurrent_recording(monkeypatch, tmp_path):
    fake_proc = _FakeProc()
    recorder = _started_recorder(monkeypatch, tmp_path, fake_proc)
    with pytest.raises(LaiError, match="already in progress"):
        recorder.start(tmp_path / "second.mp4")


def test_context_manager_stops_an_active_recording(monkeypatch, tmp_path):
    fake_proc = _FakeProc()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: fake_proc)
    with ScreenRecorder() as recorder:
        recorder.start(tmp_path / "out.mp4")
    assert fake_proc.stdin.written == b"q"


def test_del_kills_a_still_running_process(monkeypatch, tmp_path):
    fake_proc = _FakeProc()
    recorder = _started_recorder(monkeypatch, tmp_path, fake_proc)
    recorder.__del__()
    assert fake_proc.killed is True


def test_record_convenience_starts_sleeps_and_stops(monkeypatch, tmp_path):
    fake_proc = _FakeProc()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: fake_proc)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    recorder = ScreenRecorder()
    path = recorder.record(2.0, tmp_path / "clip.mp4")
    assert path == tmp_path / "clip.mp4"
    assert recorder.recording is False


# -- to_gif: argument construction (two-pass) ------------------------------------------------------


def test_to_gif_raises_without_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(BackendUnavailable, match="ffmpeg"):
        to_gif(tmp_path / "in.mp4", tmp_path / "out.gif")


def test_to_gif_raises_when_source_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    with pytest.raises(LaiError, match="does not exist"):
        to_gif(tmp_path / "ghost.mp4", tmp_path / "out.gif")


def test_to_gif_runs_a_palettegen_then_paletteuse_pass(monkeypatch, tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"not a real video, just needs to exist")
    calls: list[list[str]] = []

    def fake_run_ffmpeg(args, *, timeout):
        calls.append(args)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("lai.osl.recorder._run_ffmpeg", fake_run_ffmpeg)
    out = to_gif(video, tmp_path / "out.gif", fps=6, width=400)

    assert out == tmp_path / "out.gif"
    assert len(calls) == 2
    assert "palettegen" in " ".join(calls[0])
    assert "paletteuse" in " ".join(calls[1])
    assert "fps=6" in " ".join(calls[0])
    assert "scale=400" in " ".join(calls[0])


def test_run_ffmpeg_raises_on_nonzero_exit(monkeypatch):
    from lai.osl.recorder import _run_ffmpeg

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"boom")

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(LaiError, match="ffmpeg failed"):
        _run_ffmpeg(["ffmpeg"], timeout=1.0)


def test_run_ffmpeg_raises_backend_unavailable_on_timeout(monkeypatch):
    from lai.osl.recorder import _run_ffmpeg

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1.0)

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(BackendUnavailable, match="timed out"):
        _run_ffmpeg(["ffmpeg"], timeout=1.0)


# -- record_start / record_stop tools ------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_perception(reg)
    return reg


def test_record_start_tool_failure_path_without_ffmpeg(monkeypatch, registry):
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = registry.call("record_start", {"path": "/tmp/out.mp4"}, context())
    assert result.ok is False
    assert "ffmpeg" in result.content.lower()


def test_record_start_tool_rejects_a_malformed_region(monkeypatch, registry):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    result = registry.call(
        "record_start", {"path": "/tmp/out.mp4", "region": {"x": 1}}, context()
    )
    assert result.ok is False and "region" in result.content


def test_record_start_then_stop_round_trip(monkeypatch, registry, tmp_path):
    fake_proc = _FakeProc()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: fake_proc)
    ctx = context()

    started = registry.call("record_start", {"path": str(tmp_path / "s.mp4"), "fps": 12}, ctx)
    assert started.ok is True and started.data["fps"] == 12

    stopped = registry.call("record_stop", {}, ctx)
    assert stopped.ok is True
    assert stopped.data["path"] == str(tmp_path / "s.mp4")


def test_record_stop_tool_failure_path_when_nothing_is_recording(registry):
    result = registry.call("record_stop", {}, context())
    assert result.ok is False
    assert "no recording" in result.content.lower()


# -- x11: a real, very short recording ------------------------------------------------------


@pytest.mark.x11
def test_real_recording_produces_a_non_empty_file(tmp_path):
    recorder = ScreenRecorder()
    if not recorder.available:
        pytest.skip("ffmpeg is not installed")
    output = recorder.record(1.0, tmp_path / "clip.mp4", fps=5)
    assert output.exists()
    assert output.stat().st_size > 0


@pytest.mark.x11
def test_real_recording_converts_to_gif(tmp_path):
    recorder = ScreenRecorder()
    if not recorder.available:
        pytest.skip("ffmpeg is not installed")
    video = recorder.record(1.0, tmp_path / "clip.mp4", fps=5)
    gif = to_gif(video, tmp_path / "clip.gif", fps=4, width=160)
    assert gif.exists()
    assert gif.stat().st_size > 0
