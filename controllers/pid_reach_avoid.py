"""A deterministic PID reach-avoid controller for planar tutorial robots.

The module deliberately has no Isaac Sim imports.  It can therefore be unit
tested on the host and reused by examples that drive either a kinematic robot
or a dynamically simulated differential-drive robot.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

VectorLike: TypeAlias = Sequence[float] | NDArray[np.floating]


def _finite_vector(value: VectorLike, size: int, name: str) -> NDArray[np.float64]:
    """Return *value* as a finite, copied vector with an exact shape."""

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


def wrap_angle(angle: float | NDArray[np.floating]) -> float | NDArray[np.float64]:
    """Wrap an angle (or NumPy array of angles) to ``[-pi, pi)``."""

    array = np.asarray(angle, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("angle must contain only finite values")
    wrapped = (array + np.pi) % (2.0 * np.pi) - np.pi
    if array.ndim == 0:
        return float(wrapped)
    return wrapped


@dataclass(frozen=True, slots=True)
class CircleObstacle:
    """A circular obstacle in the world XY plane.

    ``velocity`` is optional.  When it is nonzero the controller can account
    for a short prediction horizon, and :meth:`PIDReachAvoidController.advance_obstacles`
    can move the stored obstacle deterministically between simulation steps.
    """

    center: VectorLike
    radius: float
    velocity: VectorLike = (0.0, 0.0)
    name: str = "obstacle"

    def __post_init__(self) -> None:
        center = _finite_vector(self.center, 2, "center")
        velocity = _finite_vector(self.velocity, 2, "velocity")
        radius = _finite_scalar(self.radius, "radius")
        if radius < 0.0:
            raise ValueError("radius must be non-negative")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

        # Tuples make the frozen dataclass genuinely immutable instead of
        # retaining a caller-owned, mutable NumPy array.
        object.__setattr__(self, "center", (float(center[0]), float(center[1])))
        object.__setattr__(self, "velocity", (float(velocity[0]), float(velocity[1])))
        object.__setattr__(self, "radius", radius)

    @property
    def center_array(self) -> NDArray[np.float64]:
        """Return a new NumPy representation of :attr:`center`."""

        return np.asarray(self.center, dtype=np.float64)

    @property
    def velocity_array(self) -> NDArray[np.float64]:
        """Return a new NumPy representation of :attr:`velocity`."""

        return np.asarray(self.velocity, dtype=np.float64)

    def advanced(self, dt: float) -> CircleObstacle:
        """Return a copy moved with constant velocity for ``dt`` seconds."""

        step = _finite_scalar(dt, "dt")
        if step < 0.0:
            raise ValueError("dt must be non-negative")
        center = self.center_array + self.velocity_array * step
        return replace(self, center=center)


@dataclass(frozen=True, slots=True)
class PIDConfig:
    """Tuning and safety values for :class:`PIDReachAvoidController`."""

    kp: float = 1.2
    ki: float = 0.05
    kd: float = 0.15
    max_speed: float = 1.0
    integral_limit: float = 2.0
    goal_tolerance: float = 0.08
    robot_radius: float = 0.20
    safety_margin: float = 0.10
    influence_distance: float = 1.25
    repulsion_gain: float = 0.35
    max_repulsion: float = 2.0
    prediction_horizon: float = 0.0
    heading_gain: float = 2.5
    max_angular_speed: float = 2.5
    allow_reverse: bool = False

    def __post_init__(self) -> None:
        non_negative = (
            "kp",
            "ki",
            "kd",
            "integral_limit",
            "goal_tolerance",
            "robot_radius",
            "safety_margin",
            "influence_distance",
            "repulsion_gain",
            "max_repulsion",
            "prediction_horizon",
            "heading_gain",
        )
        positive = ("max_speed", "max_angular_speed")
        for name in non_negative:
            value = _finite_scalar(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in positive:
            value = _finite_scalar(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if not isinstance(self.allow_reverse, bool):
            raise TypeError("allow_reverse must be a bool")


@dataclass(frozen=True, slots=True)
class ReachAvoidCommand:
    """A unicycle command together with useful controller diagnostics."""

    linear_velocity: float
    angular_velocity: float
    desired_heading: float
    heading_error: float
    planar_velocity: tuple[float, float]
    goal_distance: float
    reached_goal: bool

    @property
    def v(self) -> float:
        """Short alias commonly used for unicycle forward speed."""

        return self.linear_velocity

    @property
    def omega(self) -> float:
        """Short alias commonly used for unicycle angular speed."""

        return self.angular_velocity

    def as_array(self) -> NDArray[np.float64]:
        """Return ``[linear_velocity, angular_velocity]`` as a new array."""

        return np.asarray((self.linear_velocity, self.angular_velocity), dtype=np.float64)


class PIDReachAvoidController:
    """Planar PID goal tracking plus circular-obstacle potential fields.

    The PID part produces a desired world-frame XY velocity.  Repulsive fields
    are then added for obstacles inside ``influence_distance``.  Output is
    speed limited and integral growth that would push an already-saturated
    command farther into saturation is rejected (conditional anti-windup).
    """

    def __init__(
        self,
        config: PIDConfig | None = None,
        obstacles: Iterable[CircleObstacle] = (),
    ) -> None:
        self.config = config if config is not None else PIDConfig()
        if not isinstance(self.config, PIDConfig):
            raise TypeError("config must be a PIDConfig")
        self._obstacles: tuple[CircleObstacle, ...] = ()
        self._integral = np.zeros(2, dtype=np.float64)
        self._previous_error: NDArray[np.float64] | None = None
        self.set_obstacles(obstacles)

    @property
    def obstacles(self) -> tuple[CircleObstacle, ...]:
        """The immutable snapshot of currently stored obstacles."""

        return self._obstacles

    @property
    def integral_error(self) -> NDArray[np.float64]:
        """A copy of the controller's accumulated error."""

        return self._integral.copy()

    @property
    def previous_error(self) -> NDArray[np.float64] | None:
        """A copy of the previous goal error, if a step has run."""

        return None if self._previous_error is None else self._previous_error.copy()

    def reset(self) -> None:
        """Clear derivative and integral state without changing obstacles."""

        self._integral.fill(0.0)
        self._previous_error = None

    def set_obstacles(self, obstacles: Iterable[CircleObstacle]) -> None:
        """Replace the stored static/dynamic obstacle snapshot."""

        snapshot = tuple(obstacles)
        if any(not isinstance(obstacle, CircleObstacle) for obstacle in snapshot):
            raise TypeError("obstacles must contain only CircleObstacle instances")
        self._obstacles = snapshot

    def update_obstacle(
        self,
        obstacle: int | str,
        *,
        center: VectorLike | None = None,
        radius: float | None = None,
        velocity: VectorLike | None = None,
    ) -> CircleObstacle:
        """Update one stored obstacle by index or name and return the new value."""

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
            velocity=current.velocity if velocity is None else velocity,
        )
        mutable = list(self._obstacles)
        mutable[index] = updated
        self._obstacles = tuple(mutable)
        return updated

    def advance_obstacles(self, dt: float) -> tuple[CircleObstacle, ...]:
        """Advance every stored obstacle using its constant XY velocity."""

        self._obstacles = tuple(obstacle.advanced(dt) for obstacle in self._obstacles)
        return self._obstacles

    def avoidance_velocity(
        self,
        position: VectorLike,
        obstacles: Iterable[CircleObstacle] | None = None,
    ) -> NDArray[np.float64]:
        """Compute only the bounded geometric repulsion at ``position``."""

        robot_position = _finite_vector(position, 2, "position")
        active = self._obstacles if obstacles is None else tuple(obstacles)
        if any(not isinstance(obstacle, CircleObstacle) for obstacle in active):
            raise TypeError("obstacles must contain only CircleObstacle instances")

        config = self.config
        if (
            config.repulsion_gain == 0.0
            or config.max_repulsion == 0.0
            or config.influence_distance == 0.0
        ):
            return np.zeros(2, dtype=np.float64)

        repulsion = np.zeros(2, dtype=np.float64)
        epsilon = 1.0e-6
        for item in active:
            predicted_center = (
                item.center_array + config.prediction_horizon * item.velocity_array
            )
            away = robot_position - predicted_center
            center_distance = float(np.linalg.norm(away))
            direction = (
                np.asarray((1.0, 0.0), dtype=np.float64)
                if center_distance <= epsilon
                else away / center_distance
            )
            occupied_radius = item.radius + config.robot_radius + config.safety_margin
            clearance = center_distance - occupied_radius
            if clearance >= config.influence_distance:
                continue

            if clearance <= epsilon:
                magnitude = config.max_repulsion
            else:
                magnitude = config.repulsion_gain * (
                    (1.0 / clearance) - (1.0 / config.influence_distance)
                ) / (clearance * clearance)
                magnitude = float(np.clip(magnitude, 0.0, config.max_repulsion))
            repulsion += magnitude * direction

        norm = float(np.linalg.norm(repulsion))
        if norm > config.max_repulsion:
            repulsion *= config.max_repulsion / norm
        return repulsion

    def compute_velocity(
        self,
        position: VectorLike,
        goal: VectorLike,
        dt: float,
        obstacles: Iterable[CircleObstacle] | None = None,
    ) -> NDArray[np.float64]:
        """Return a speed-limited desired world-frame velocity ``[vx, vy]``."""

        robot_position = _finite_vector(position, 2, "position")
        target = _finite_vector(goal, 2, "goal")
        step = _finite_scalar(dt, "dt")
        if step <= 0.0:
            raise ValueError("dt must be positive")

        error = target - robot_position
        if float(np.linalg.norm(error)) <= self.config.goal_tolerance:
            self.reset()
            return np.zeros(2, dtype=np.float64)

        derivative = (
            np.zeros(2, dtype=np.float64)
            if self._previous_error is None
            else (error - self._previous_error) / step
        )
        candidate_integral = np.clip(
            self._integral + error * step,
            -self.config.integral_limit,
            self.config.integral_limit,
        )
        repulsion = self.avoidance_velocity(robot_position, obstacles)
        base = self.config.kp * error + self.config.kd * derivative + repulsion
        candidate_command = base + self.config.ki * candidate_integral

        # Conditional integration: reject a candidate integral increment when
        # it pushes an already speed-saturated vector farther outward.
        if (
            float(np.linalg.norm(candidate_command)) > self.config.max_speed
            and float(
                np.dot(
                    self.config.ki * (candidate_integral - self._integral),
                    candidate_command,
                )
            )
            > 0.0
        ):
            command = base + self.config.ki * self._integral
        else:
            self._integral = candidate_integral
            command = candidate_command

        self._previous_error = error.copy()
        speed = float(np.linalg.norm(command))
        if speed > self.config.max_speed:
            command = command * (self.config.max_speed / speed)
        return command.astype(np.float64, copy=False)

    def compute_unicycle(
        self,
        pose: VectorLike,
        goal: VectorLike,
        dt: float,
        obstacles: Iterable[CircleObstacle] | None = None,
    ) -> ReachAvoidCommand:
        """Return forward/angular speeds for a pose ``[x, y, yaw]``.

        With the default ``allow_reverse=False``, forward speed smoothly drops
        to zero while the desired velocity lies behind the robot.  This makes
        the command suitable for common differential-drive bases.
        """

        robot_pose = _finite_vector(pose, 3, "pose")
        target = _finite_vector(goal, 2, "goal")
        planar = self.compute_velocity(robot_pose[:2], target, dt, obstacles)
        goal_distance = float(np.linalg.norm(target - robot_pose[:2]))
        reached = goal_distance <= self.config.goal_tolerance
        planar_speed = float(np.linalg.norm(planar))

        if reached or planar_speed <= np.finfo(np.float64).eps:
            desired_heading = float(wrap_angle(robot_pose[2]))
            heading_error = 0.0
            linear = 0.0
            angular = 0.0
        else:
            desired_heading = float(np.arctan2(planar[1], planar[0]))
            heading_error = float(wrap_angle(desired_heading - robot_pose[2]))
            if self.config.allow_reverse:
                linear = planar_speed * float(np.cos(heading_error))
            else:
                linear = planar_speed * max(0.0, float(np.cos(heading_error)))
            angular = float(
                np.clip(
                    self.config.heading_gain * heading_error,
                    -self.config.max_angular_speed,
                    self.config.max_angular_speed,
                )
            )

        return ReachAvoidCommand(
            linear_velocity=linear,
            angular_velocity=angular,
            desired_heading=desired_heading,
            heading_error=heading_error,
            planar_velocity=(float(planar[0]), float(planar[1])),
            goal_distance=goal_distance,
            reached_goal=reached,
        )

    def compute_command(
        self,
        pose: VectorLike,
        goal: VectorLike,
        dt: float,
        obstacles: Iterable[CircleObstacle] | None = None,
    ) -> ReachAvoidCommand:
        """Alias for :meth:`compute_unicycle` for simulation-loop readability."""

        return self.compute_unicycle(pose, goal, dt, obstacles)


__all__ = [
    "CircleObstacle",
    "PIDConfig",
    "PIDReachAvoidController",
    "ReachAvoidCommand",
    "wrap_angle",
]
