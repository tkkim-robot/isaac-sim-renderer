"""Shared terminal-state and collision visualization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class RunResult:
    example: str
    status: str
    frames: int
    simulated_seconds: float
    final_position: list[float]
    goal: list[float]
    collision: dict[str, Any] | None = None
    video: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
