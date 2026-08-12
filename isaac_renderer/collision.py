"""Simulator-independent 2-D circle collision helpers and monitoring.

Isaac Sim examples can call :class:`CollisionMonitor` once per physics step.
The monitor latches the first collision so a later frame cannot overwrite the
information that explains why a run terminated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

VectorLike: TypeAlias = Sequence[float] | NDArray[np.floating]


def _finite_vector(value: VectorLike, size: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def closest_point_on_segment(
    point: VectorLike,
    start: VectorLike,
    end: VectorLike,
) -> tuple[NDArray[np.float64], float]:
    """Return the closest point and its normalized segment parameter.

    A zero-length segment is handled safely and returns ``(start, 0.0)``.
    """

    target = _finite_vector(point, 2, "point")
    segment_start = _finite_vector(start, 2, "start")
    segment_end = _finite_vector(end, 2, "end")
    delta = segment_end - segment_start
    squared_length = float(np.dot(delta, delta))
    if squared_length <= np.finfo(np.float64).eps:
        return segment_start, 0.0
    fraction = float(
        np.clip(np.dot(target - segment_start, delta) / squared_length, 0.0, 1.0)
    )
    return segment_start + fraction * delta, fraction


def point_to_segment_distance(
    point: VectorLike,
    start: VectorLike,
    end: VectorLike,
) -> float:
    """Return the Euclidean distance from a point to a closed segment."""

    target = _finite_vector(point, 2, "point")
    closest, _ = closest_point_on_segment(target, start, end)
    return float(np.linalg.norm(target - closest))


def circle_signed_clearance(
    center_a: VectorLike,
    radius_a: float,
    center_b: VectorLike,
    radius_b: float,
    margin: float = 0.0,
) -> float:
    """Return circle boundary clearance; zero/tangent and negative collide."""

    first = _finite_vector(center_a, 2, "center_a")
    second = _finite_vector(center_b, 2, "center_b")
    threshold = (
        _non_negative(radius_a, "radius_a")
        + _non_negative(radius_b, "radius_b")
        + _non_negative(margin, "margin")
    )
    return float(np.linalg.norm(first - second) - threshold)


def circles_collide(
    center_a: VectorLike,
    radius_a: float,
    center_b: VectorLike,
    radius_b: float,
    margin: float = 0.0,
) -> bool:
    """Return whether two closed circles overlap or are tangent."""

    return circle_signed_clearance(center_a, radius_a, center_b, radius_b, margin) <= 0.0


def segment_circle_first_intersection(
    start: VectorLike,
    end: VectorLike,
    circle_center: VectorLike,
    circle_radius: float,
    margin: float = 0.0,
) -> float | None:
    """Return the first intersection fraction in ``[0, 1]``, or ``None``.

    ``margin`` expands the circle.  Starting inside returns ``0.0`` and a
    zero-length segment is handled as a point query.
    """

    segment_start = _finite_vector(start, 2, "start")
    segment_end = _finite_vector(end, 2, "end")
    center = _finite_vector(circle_center, 2, "circle_center")
    radius = _non_negative(circle_radius, "circle_radius") + _non_negative(
        margin, "margin"
    )
    direction = segment_end - segment_start
    offset = segment_start - center
    c = float(np.dot(offset, offset) - radius * radius)
    if c <= 0.0:
        return 0.0

    a = float(np.dot(direction, direction))
    if a <= np.finfo(np.float64).eps:
        return None
    b = 2.0 * float(np.dot(offset, direction))
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None

    root = float(np.sqrt(max(0.0, discriminant)))
    first = (-b - root) / (2.0 * a)
    second = (-b + root) / (2.0 * a)
    candidates = [value for value in (first, second) if 0.0 <= value <= 1.0]
    return None if not candidates else float(np.clip(min(candidates), 0.0, 1.0))


def segment_circle_intersection(
    start: VectorLike,
    end: VectorLike,
    circle_center: VectorLike,
    circle_radius: float,
    margin: float = 0.0,
) -> bool:
    """Return whether a closed line segment intersects an expanded circle."""

    return (
        segment_circle_first_intersection(
            start, end, circle_center, circle_radius, margin
        )
        is not None
    )


@dataclass(frozen=True, slots=True)
class Circle:
    """A named circular collider in the world XY plane."""

    center: VectorLike
    radius: float
    name: str = "obstacle"
    kind: str = "static"
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        center = _finite_vector(self.center, 2, "center")
        radius = _non_negative(self.radius, "radius")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("kind must be a non-empty string")
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        metadata = {} if self.metadata is None else dict(self.metadata)

        object.__setattr__(self, "center", (float(center[0]), float(center[1])))
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def center_array(self) -> NDArray[np.float64]:
        """Return a new NumPy representation of :attr:`center`."""

        return np.asarray(self.center, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CollisionEvent:
    """Details captured when a monitor observes its first collision."""

    robot_name: str
    obstacle_name: str
    obstacle_kind: str
    obstacle_index: int
    robot_center: tuple[float, float]
    obstacle_center: tuple[float, float]
    robot_radius: float
    obstacle_radius: float
    safety_margin: float
    center_distance: float
    signed_clearance: float
    contact_point: tuple[float, float]
    step: int
    simulation_time: float
    detection_mode: str
    trajectory_fraction: float
    obstacle_metadata: Mapping[str, object]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "obstacle_metadata", MappingProxyType(dict(self.obstacle_metadata))
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def penetration_depth(self) -> float:
        """Non-negative penetration depth (zero for exact tangency)."""

        return max(0.0, -self.signed_clearance)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the event."""

        return {
            "robot_name": self.robot_name,
            "obstacle_name": self.obstacle_name,
            "obstacle_kind": self.obstacle_kind,
            "obstacle_index": self.obstacle_index,
            "robot_center": list(self.robot_center),
            "obstacle_center": list(self.obstacle_center),
            "robot_radius": self.robot_radius,
            "obstacle_radius": self.obstacle_radius,
            "safety_margin": self.safety_margin,
            "center_distance": self.center_distance,
            "signed_clearance": self.signed_clearance,
            "penetration_depth": self.penetration_depth,
            "contact_point": list(self.contact_point),
            "step": self.step,
            "simulation_time": self.simulation_time,
            "detection_mode": self.detection_mode,
            "trajectory_fraction": self.trajectory_fraction,
            "obstacle_metadata": dict(self.obstacle_metadata),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CollisionStatus:
    """The monitor's latched status after a collision check."""

    collided: bool
    event: CollisionEvent | None
    checks: int
    message: str

    def __post_init__(self) -> None:
        if self.collided != (self.event is not None):
            raise ValueError("collided must agree with whether event is present")
        if self.checks < 0:
            raise ValueError("checks must be non-negative")

    @property
    def clean(self) -> bool:
        """Whether no collision has been observed since the last reset."""

        return not self.collided

    @property
    def is_clear(self) -> bool:
        """Readable alias for :attr:`clean`."""

        return self.clean

    @property
    def state(self) -> str:
        """Return ``"clear"`` or ``"collision"`` for logs and reports."""

        return "collision" if self.collided else "clear"

    def __bool__(self) -> bool:
        return self.collided

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly status dictionary."""

        return {
            "state": self.state,
            "collided": self.collided,
            "clean": self.clean,
            "checks": self.checks,
            "message": self.message,
            "event": None if self.event is None else self.event.as_dict(),
        }


class CollisionMonitor:
    """Latch and report the first robot-versus-circle collision."""

    def __init__(
        self,
        robot_radius: float,
        obstacles: Iterable[Circle] = (),
        *,
        robot_name: str = "robot",
        safety_margin: float = 0.0,
    ) -> None:
        self.robot_radius = _non_negative(robot_radius, "robot_radius")
        self.safety_margin = _non_negative(safety_margin, "safety_margin")
        if not isinstance(robot_name, str) or not robot_name:
            raise ValueError("robot_name must be a non-empty string")
        self.robot_name = robot_name
        self._obstacles: tuple[Circle, ...] = ()
        self._first_collision: CollisionEvent | None = None
        self._checks = 0
        self.set_obstacles(obstacles)

    @property
    def obstacles(self) -> tuple[Circle, ...]:
        """The immutable snapshot of currently monitored obstacles."""

        return self._obstacles

    @property
    def first_collision(self) -> CollisionEvent | None:
        """The first event since reset, if one has occurred."""

        return self._first_collision

    @property
    def status(self) -> CollisionStatus:
        """Return the current clean/collision status without another check."""

        if self._first_collision is None:
            return CollisionStatus(
                collided=False,
                event=None,
                checks=self._checks,
                message="No collision detected.",
            )
        event = self._first_collision
        return CollisionStatus(
            collided=True,
            event=event,
            checks=self._checks,
            message=(
                f"Collision detected between {event.robot_name!r} and "
                f"{event.obstacle_name!r} at step {event.step}."
            ),
        )

    def reset(self) -> None:
        """Clear the latched event and check count, retaining obstacles."""

        self._first_collision = None
        self._checks = 0

    def set_obstacles(self, obstacles: Iterable[Circle]) -> None:
        """Replace the collider snapshot."""

        snapshot = tuple(obstacles)
        if any(not isinstance(obstacle, Circle) for obstacle in snapshot):
            raise TypeError("obstacles must contain only Circle instances")
        self._obstacles = snapshot

    def update_obstacle(
        self,
        obstacle: int | str,
        *,
        center: VectorLike | None = None,
        radius: float | None = None,
        kind: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Circle:
        """Update one collider by index or name and return its new snapshot."""

        if isinstance(obstacle, str):
            matches = [i for i, item in enumerate(self._obstacles) if item.name == obstacle]
            if not matches:
                raise KeyError(f"unknown obstacle name: {obstacle!r}")
            index = matches[0]
        elif isinstance(obstacle, int) and not isinstance(obstacle, bool):
            index = obstacle
            if index < 0 or index >= len(self._obstacles):
                raise IndexError("obstacle index out of range")
        else:
            raise TypeError("obstacle must be an integer index or string name")

        current = self._obstacles[index]
        updated = replace(
            current,
            center=current.center if center is None else center,
            radius=current.radius if radius is None else radius,
            kind=current.kind if kind is None else kind,
            metadata=current.metadata if metadata is None else metadata,
        )
        mutable = list(self._obstacles)
        mutable[index] = updated
        self._obstacles = tuple(mutable)
        return updated

    def check(
        self,
        robot_center: VectorLike,
        *,
        step: int | None = None,
        simulation_time: float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> CollisionStatus:
        """Check the robot at one position and latch the first overlap."""

        position = _finite_vector(robot_center, 2, "robot_center")
        return self._check_path(
            position,
            position,
            step=step,
            simulation_time=simulation_time,
            metadata=metadata,
            detection_mode="point",
        )

    def check_segment(
        self,
        start: VectorLike,
        end: VectorLike,
        *,
        step: int | None = None,
        simulation_time: float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> CollisionStatus:
        """Swept-circle check that prevents fast steps tunneling through objects."""

        segment_start = _finite_vector(start, 2, "start")
        segment_end = _finite_vector(end, 2, "end")
        return self._check_path(
            segment_start,
            segment_end,
            step=step,
            simulation_time=simulation_time,
            metadata=metadata,
            detection_mode="swept",
        )

    def _check_path(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        *,
        step: int | None,
        simulation_time: float | None,
        metadata: Mapping[str, object] | None,
        detection_mode: str,
    ) -> CollisionStatus:
        resolved_step = self._checks if step is None else step
        if not isinstance(resolved_step, int) or isinstance(resolved_step, bool):
            raise TypeError("step must be an integer or None")
        if resolved_step < 0:
            raise ValueError("step must be non-negative")
        resolved_time = 0.0 if simulation_time is None else _finite_scalar(
            simulation_time, "simulation_time"
        )
        if resolved_time < 0.0:
            raise ValueError("simulation_time must be non-negative")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")

        self._checks += 1
        if self._first_collision is not None:
            return self.status

        effective_robot_radius = self.robot_radius + self.safety_margin
        for index, obstacle in enumerate(self._obstacles):
            fraction = segment_circle_first_intersection(
                start,
                end,
                obstacle.center,
                obstacle.radius + effective_robot_radius,
            )
            if fraction is None:
                continue
            impact_center = start + fraction * (end - start)
            self._first_collision = self._make_event(
                impact_center,
                obstacle,
                obstacle_index=index,
                step=resolved_step,
                simulation_time=resolved_time,
                detection_mode=detection_mode,
                trajectory_fraction=fraction,
                metadata={} if metadata is None else metadata,
            )
            break
        return self.status

    def _make_event(
        self,
        robot_center: NDArray[np.float64],
        obstacle: Circle,
        *,
        obstacle_index: int,
        step: int,
        simulation_time: float,
        detection_mode: str,
        trajectory_fraction: float,
        metadata: Mapping[str, object],
    ) -> CollisionEvent:
        obstacle_center = obstacle.center_array
        separation = robot_center - obstacle_center
        center_distance = float(np.linalg.norm(separation))
        if center_distance <= np.finfo(np.float64).eps:
            normal = np.asarray((1.0, 0.0), dtype=np.float64)
        else:
            normal = separation / center_distance
        contact = obstacle_center + normal * obstacle.radius
        signed_clearance = (
            center_distance
            - obstacle.radius
            - self.robot_radius
            - self.safety_margin
        )
        return CollisionEvent(
            robot_name=self.robot_name,
            obstacle_name=obstacle.name,
            obstacle_kind=obstacle.kind,
            obstacle_index=obstacle_index,
            robot_center=(float(robot_center[0]), float(robot_center[1])),
            obstacle_center=(float(obstacle_center[0]), float(obstacle_center[1])),
            robot_radius=self.robot_radius,
            obstacle_radius=obstacle.radius,
            safety_margin=self.safety_margin,
            center_distance=center_distance,
            signed_clearance=float(signed_clearance),
            contact_point=(float(contact[0]), float(contact[1])),
            step=step,
            simulation_time=simulation_time,
            detection_mode=detection_mode,
            trajectory_fraction=float(trajectory_fraction),
            obstacle_metadata=obstacle.metadata,
            metadata=metadata,
        )


__all__ = [
    "Circle",
    "CollisionEvent",
    "CollisionMonitor",
    "CollisionStatus",
    "circle_signed_clearance",
    "circles_collide",
    "closest_point_on_segment",
    "point_to_segment_distance",
    "segment_circle_first_intersection",
    "segment_circle_intersection",
]
