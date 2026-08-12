"""Shared command-line options for standalone Isaac Sim examples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RenderOptions:
    headless: bool
    frames: int
    fps: int
    width: int
    height: int
    output_dir: Path
    no_video: bool


def add_render_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output: Path,
    default_frames: int = 240,
) -> None:
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run without the Isaac Sim GUI (default: true).",
    )
    parser.add_argument("--frames", type=int, default=default_frames, help="Number of video frames to capture.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
    parser.add_argument("--width", type=int, default=960, help="Capture width in pixels.")
    parser.add_argument("--height", type=int, default=540, help="Capture height in pixels.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory for frames, MP4, and metadata.")
    parser.add_argument("--no-video", action="store_true", help="Simulate without capturing PNG frames or MP4.")


def validate_render_options(args: argparse.Namespace) -> RenderOptions:
    if args.frames < 1:
        raise ValueError("--frames must be at least 1")
    if args.fps < 1:
        raise ValueError("--fps must be at least 1")
    if args.width < 64 or args.height < 64:
        raise ValueError("--width and --height must both be at least 64")
    if args.width % 2 or args.height % 2:
        raise ValueError("--width and --height must both be even for H.264/yuv420p output")
    return RenderOptions(
        headless=bool(args.headless),
        frames=int(args.frames),
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        no_video=bool(args.no_video),
    )
