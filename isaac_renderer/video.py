"""Deterministic viewport capture and H.264 MP4 encoding.

Import this module only after ``SimulationApp`` has started.  Isaac Sim owns
the ``omni.*`` modules used here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import carb
import omni.renderer_capture
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport


class FrameRecorder:
    """Capture numbered viewport PNGs and turn them into a portable MP4."""

    def __init__(
        self,
        simulation_app,
        output_dir: Path,
        *,
        camera_path: str,
        fps: int,
        width: int,
        height: int,
        stem: str,
    ) -> None:
        self._app = simulation_app
        self.output_dir = Path(output_dir).resolve()
        self.frames_dir = self.output_dir / "frames"
        self.video_path = self.output_dir / f"{stem}.mp4"
        self.fps = int(fps)
        self.frame_count = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._clean_own_frames()

        self._viewport = get_active_viewport()
        if self._viewport is None:
            raise RuntimeError("Isaac Sim did not create an active viewport.")
        # Captures should contain the scene, not editor grids, light icons, or
        # selection overlays. This setting is per portable Isaac profile.
        carb.settings.get_settings().set_int("/persistent/app/viewport/displayOptions", 0)
        self._viewport.camera_path = camera_path
        self._viewport.set_texture_resolution((int(width), int(height)))
        for _ in range(6):
            self._app.update()

    def _clean_own_frames(self) -> None:
        """Remove only numbered PNG files owned by this recorder."""

        for path in self.frames_dir.glob("frame_[0-9][0-9][0-9][0-9][0-9].png"):
            if path.is_file() and not path.is_symlink():
                path.unlink()

    def capture(self, *, timeout_seconds: float = 30.0) -> Path:
        output_path = self.frames_dir / f"frame_{self.frame_count:05d}.png"
        if output_path.exists():
            output_path.unlink()

        capture_viewport_to_file(self._viewport, file_path=str(output_path))
        capture_interface = omni.renderer_capture.acquire_renderer_capture_interface()
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            capture_interface.wait_async_capture()
            self._app.update()
            if output_path.exists() and output_path.stat().st_size > 0:
                self.frame_count += 1
                return output_path
            time.sleep(0.01)
        raise TimeoutError(f"Timed out while capturing {output_path}")

    def encode(self, *, crf: int = 18) -> Path:
        if self.frame_count < 1:
            raise RuntimeError("No frames were captured.")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to encode MP4. Install it or use the provided container.")
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-framerate",
            str(self.fps),
            "-i",
            str(self.frames_dir / "frame_%05d.png"),
            "-frames:v",
            str(self.frame_count),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.video_path),
        ]
        subprocess.run(command, check=True)
        return self.video_path


def write_metadata(output_dir: Path, payload: dict[str, Any]) -> Path:
    """Write JSON metadata next to a rendered video."""

    path = Path(output_dir) / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
