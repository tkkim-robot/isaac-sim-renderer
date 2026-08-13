"""Simulator-independent animation helpers used by rendered robot proxies."""

from __future__ import annotations

import math


def rotor_spin_degrees(frame_index: int, rotor_index: int, spin_scale: float = 1.0) -> float:
    """Return the original SEAMLIS proxy's deterministic rotor angle.

    The slight per-rotor speed difference keeps all four crosses from aliasing
    into the same pose in a 30 FPS recording. A zero scale intentionally snaps
    the blades to rest, matching the former renderer's crash behavior.
    """

    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ValueError("frame_index must be a non-negative integer")
    if not isinstance(rotor_index, int) or isinstance(rotor_index, bool) or rotor_index < 0:
        raise ValueError("rotor_index must be a non-negative integer")
    scale = float(spin_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("spin_scale must be finite and non-negative")
    if scale == 0.0:
        return 0.0
    return float((frame_index * 115.0 * scale * (1.0 + 0.06 * rotor_index)) % 360.0)
