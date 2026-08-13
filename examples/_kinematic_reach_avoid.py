"""Shared runner for the two intentionally kinematic reach-avoid examples.

This module is imported after ``SimulationApp`` starts.  It keeps a useful
controller/renderer separation: control creates state, while Isaac Sim owns
the USD scene, cameras, and pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from controllers import CircleObstacle, PIDConfig, PIDReachAvoidController
from isaac_renderer.cli import RenderOptions
from isaac_renderer.collision import Circle, CollisionMonitor
from isaac_renderer.paths import ASSET_ROOT
from isaac_renderer.quadrotor import (
    AnimatedQuadrotor,
    add_quadrotor_accent_lights,
    spawn_procedural_quadrotor,
)
from isaac_renderer.scene import (
    add_camera,
    add_cube,
    add_cylinder,
    add_ground,
    add_lights,
    add_sphere,
    add_trail,
    create_material,
    import_urdf,
    new_stage,
    set_camera,
    set_pose,
    set_transform_srt,
    set_visible,
    stage,
    update_trail,
)
from isaac_renderer.status import RunResult
from isaac_renderer.video import FrameRecorder, write_metadata


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    urdf_name: str
    motion: Literal["holonomic", "unicycle"]
    start: tuple[float, float]
    goal: tuple[float, float]
    altitude: float
    visual_scale: float
    robot_radius: float
    visual_style: Literal["urdf", "procedural_quadrotor"]
    camera_mode: Literal["fixed", "follow"]
    camera_eye: tuple[float, float, float]
    camera_target: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class RobotHandle:
    """Visible robot plus an optional hidden imported-URDF backing model."""

    visible_path: str
    visual_scale: float
    backing_path: str | None = None
    quadrotor: AnimatedQuadrotor | None = None


def _dynamic_center(time_s: float, *, ground_robot: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return center and velocity of a deterministic moving obstacle."""

    phase = 0.55 * float(time_s)
    x = 0.2 if ground_robot else 0.8
    amplitude = 1.35 if ground_robot else 1.4
    return (
        np.asarray((x, amplitude * math.sin(phase)), dtype=float),
        np.asarray((0.0, amplitude * 0.55 * math.cos(phase)), dtype=float),
    )


def _obstacles(time_s: float, *, ground_robot: bool) -> list[CircleObstacle]:
    dynamic_center, dynamic_velocity = _dynamic_center(time_s, ground_robot=ground_robot)
    if ground_robot:
        static = [
            CircleObstacle((-1.0, 0.25), 0.55, name="static_crate"),
            CircleObstacle((1.0, -0.65), 0.48, name="static_pillar"),
        ]
    else:
        static = [
            CircleObstacle((-1.0, -0.65), 0.52, name="static_crate"),
            CircleObstacle((1.25, 0.75), 0.48, name="static_pillar"),
        ]
    return static + [
        CircleObstacle(dynamic_center, 0.38, dynamic_velocity, name="dynamic_obstacle"),
    ]


def _collision_circles(obstacles: list[CircleObstacle]) -> list[Circle]:
    circles = []
    for obstacle in obstacles:
        circles.append(
            Circle(
                obstacle.center,
                obstacle.radius,
                name=obstacle.name,
                kind="dynamic" if obstacle.name.startswith("dynamic") else "static",
                metadata={"velocity": list(obstacle.velocity)},
            )
        )
    return circles


def _set_robot_state(
    handle: RobotHandle,
    position: tuple[float, float, float],
    rpy: tuple[float, float, float],
    *,
    frame_index: int,
    rotor_spin_scale: float,
) -> None:
    if handle.quadrotor is None:
        set_transform_srt(
            handle.visible_path,
            position,
            rpy,
            scale=(handle.visual_scale,) * 3,
        )
        return

    handle.quadrotor.set_world_pose(position, rpy)
    handle.quadrotor.animate_rotors(frame_index, spin_scale=rotor_spin_scale)
    if handle.backing_path is not None:
        # Keep the invisible imported URDF synchronized as the structural
        # backing for users who want to replace the proxy with real dynamics.
        set_transform_srt(
            handle.backing_path,
            position,
            rpy,
            scale=(handle.visual_scale,) * 3,
        )


def _update_follow_camera(
    camera_path: str,
    position: np.ndarray,
    yaw: float,
    altitude: float,
    state: dict[str, np.ndarray | None],
) -> None:
    """Keep a smooth, close view behind and above the robot."""

    heading = np.asarray((math.cos(yaw), math.sin(yaw)), dtype=float)
    left = np.asarray((-heading[1], heading[0]), dtype=float)
    desired_eye = np.asarray(
        (
            position[0] - 2.2 * heading[0] - 0.65 * left[0],
            position[1] - 2.2 * heading[1] - 0.65 * left[1],
            3.35,
        ),
        dtype=float,
    )
    desired_target = np.asarray(
        (
            position[0] + 0.30 * heading[0],
            position[1] + 0.30 * heading[1],
            altitude,
        ),
        dtype=float,
    )
    if state["eye"] is None:
        state["eye"] = desired_eye
        state["target"] = desired_target
    else:
        state["eye"] = 0.86 * state["eye"] + 0.14 * desired_eye
        state["target"] = 0.80 * state["target"] + 0.20 * desired_target
    set_camera(camera_path, state["eye"], state["target"])


def _build_scene(simulation_app, scenario: Scenario) -> tuple[RobotHandle, str, str, str, str]:
    new_stage(simulation_app)
    add_ground(size=18.0)
    add_lights()
    if scenario.visual_style == "procedural_quadrotor":
        add_quadrotor_accent_lights()

    obstacle_material = create_material("/World/Looks/StaticObstacle", (0.80, 0.24, 0.12), roughness=0.42)
    dynamic_material = create_material("/World/Looks/DynamicObstacle", (0.95, 0.67, 0.08), roughness=0.32)
    goal_material = create_material("/World/Looks/Goal", (0.10, 0.88, 0.44), roughness=0.20)
    collision_material = create_material("/World/Looks/Collision", (1.0, 0.03, 0.03), roughness=0.18)

    initial_obstacles = _obstacles(0.0, ground_robot=scenario.motion == "unicycle")
    first, second, moving = initial_obstacles
    add_cube(
        "/World/Obstacles/StaticCrate",
        center=(*first.center, 0.50),
        size=(0.78, 0.78, 1.0),
        material=obstacle_material,
        collision=True,
    )
    add_cylinder(
        "/World/Obstacles/StaticPillar",
        center=(*second.center, 0.60),
        radius=second.radius,
        height=1.2,
        material=obstacle_material,
        collision=True,
    )
    moving_path = add_cylinder(
        "/World/Obstacles/Dynamic",
        center=(*moving.center, 0.42),
        radius=moving.radius,
        height=0.84,
        material=dynamic_material,
        collision=True,
    )
    add_sphere(
        "/World/Goal",
        center=(*scenario.goal, 0.20),
        radius=0.20,
        material=goal_material,
    )
    collision_marker = add_sphere(
        "/World/CollisionMarker",
        center=(0.0, 0.0, 0.08),
        radius=0.15,
        material=collision_material,
    )
    set_visible(collision_marker, False)
    trail_path = add_trail("/World/RobotTrail", (0.16, 0.72, 1.0), width=0.055)
    camera_path = add_camera(eye=scenario.camera_eye, target=scenario.camera_target)

    articulation_path = import_urdf(
        simulation_app,
        ASSET_ROOT / "robots" / scenario.urdf_name,
        # Physics never plays in these renderer-only examples, so a free base
        # avoids authoring a world-fixed joint that would fight root poses.
        fix_base=False,
    )
    # The importer may return `/robot/root_joint`, including for this free-base
    # import. Kinematic examples must move its parent model prim so visuals,
    # collision geometry, and every link move together.
    imported_model_path = articulation_path.rsplit("/", 1)[0]
    if not imported_model_path or not stage().GetPrimAtPath(imported_model_path).IsValid():
        raise RuntimeError(f"Could not resolve model root from {articulation_path!r}")

    if scenario.visual_style == "procedural_quadrotor":
        # Keep the URDF as an invisible, synchronized structural backing so the
        # tutorial demonstrates both a self-contained import and a customizable
        # procedural rendering layer.
        set_visible(imported_model_path, False)
        quadrotor = spawn_procedural_quadrotor()
        robot_handle = RobotHandle(
            visible_path=quadrotor.root_path,
            visual_scale=scenario.visual_scale,
            backing_path=imported_model_path,
            quadrotor=quadrotor,
        )
    else:
        robot_handle = RobotHandle(
            visible_path=imported_model_path,
            visual_scale=scenario.visual_scale,
        )
    _set_robot_state(
        robot_handle,
        (*scenario.start, scenario.altitude),
        (0.0, 0.0, 0.0),
        frame_index=0,
        rotor_spin_scale=1.0,
    )
    robot_prim = stage().GetPrimAtPath(imported_model_path)
    print(
        f"[{scenario.name}] imported URDF at {imported_model_path}; visible robot={robot_handle.visible_path} "
        f"children={[child.GetName() for child in robot_prim.GetChildren()]}",
        flush=True,
    )
    return robot_handle, moving_path, collision_marker, trail_path, camera_path


def run_kinematic_reach_avoid(simulation_app, options: RenderOptions, scenario: Scenario) -> RunResult:
    """Run a deterministic PID reach-avoid rollout and optionally encode MP4."""

    robot_handle, moving_path, collision_marker, trail_path, camera_path = _build_scene(simulation_app, scenario)
    ground_robot = scenario.motion == "unicycle"
    controller = PIDReachAvoidController(
        PIDConfig(
            kp=0.75,
            ki=0.01,
            kd=0.08,
            max_speed=1.05 if not ground_robot else 1.20,
            integral_limit=1.0,
            goal_tolerance=0.10,
            robot_radius=scenario.robot_radius,
            safety_margin=0.12,
            influence_distance=1.20,
            repulsion_gain=0.35,
            max_repulsion=4.0,
            prediction_horizon=0.25,
            heading_gain=4.0 if ground_robot else 3.0,
            max_angular_speed=3.0,
        )
    )
    monitor = CollisionMonitor(scenario.robot_radius, robot_name=scenario.name)
    recorder = None
    if not options.no_video:
        recorder = FrameRecorder(
            simulation_app,
            options.output_dir,
            camera_path=camera_path,
            fps=options.fps,
            width=options.width,
            height=options.height,
            stem=scenario.name,
        )

    dt = 1.0 / float(options.fps)
    position = np.asarray(scenario.start, dtype=float)
    goal = np.asarray(scenario.goal, dtype=float)
    yaw = 0.0
    trail: list[list[float]] = []
    collision_time: float | None = None
    collision_payload = None
    reached_goal = False
    follow_camera_state: dict[str, np.ndarray | None] = {"eye": None, "target": None}

    for frame in range(options.frames):
        time_s = frame * dt
        active_obstacles = _obstacles(time_s, ground_robot=ground_robot)
        controller.set_obstacles(active_obstacles)
        monitor.set_obstacles(_collision_circles(active_obstacles))

        moving = active_obstacles[-1]
        set_pose(moving_path, (*moving.center, 0.42), 0.0)

        previous = position.copy()
        if collision_time is None and not reached_goal:
            if scenario.motion == "holonomic":
                velocity = controller.compute_velocity(position, goal, dt)
                position = position + velocity * dt
                if float(np.linalg.norm(velocity)) > 1.0e-8:
                    yaw = float(math.atan2(velocity[1], velocity[0]))
            else:
                command = controller.compute_unicycle((position[0], position[1], yaw), goal, dt)
                yaw += command.angular_velocity * dt
                position = position + command.linear_velocity * np.asarray((math.cos(yaw), math.sin(yaw))) * dt
                reached_goal = command.reached_goal

            status = monitor.check_segment(previous, position, step=frame, simulation_time=time_s)
            if status.collided:
                collision_time = time_s
                collision_payload = status.event.as_dict() if status.event else None
                if status.event:
                    position = np.asarray(status.event.robot_center, dtype=float)
                    set_pose(collision_marker, (*status.event.contact_point, 0.12), 0.0)
                    set_visible(collision_marker, True)

        reached_goal = reached_goal or float(np.linalg.norm(position - goal)) <= controller.config.goal_tolerance
        if collision_time is None:
            _set_robot_state(
                robot_handle,
                (*position, scenario.altitude),
                (0.0, 0.0, yaw),
                frame_index=frame,
                rotor_spin_scale=1.0,
            )
        else:
            # Stop the failed vehicle, mark impact, and make an aerial robot
            # fall and tumble so collision behavior is visually unambiguous.
            crash_age = max(0.0, time_s - collision_time)
            if ground_robot:
                _set_robot_state(
                    robot_handle,
                    (*position, scenario.altitude),
                    (0.0, min(0.65, crash_age), yaw),
                    frame_index=frame,
                    rotor_spin_scale=0.0,
                )
            else:
                z = max(0.08, scenario.altitude - 0.5 * 9.81 * crash_age * crash_age)
                _set_robot_state(
                    robot_handle,
                    (*position, z),
                    (2.1 * crash_age, 1.4 * crash_age, yaw),
                    frame_index=frame,
                    rotor_spin_scale=0.0,
                )

        if scenario.camera_mode == "follow":
            _update_follow_camera(
                camera_path,
                position,
                yaw,
                scenario.altitude,
                follow_camera_state,
            )

        trail.append([float(position[0]), float(position[1]), 0.055])
        update_trail(trail_path, trail)
        simulation_app.update()
        if recorder is not None:
            recorder.capture()

    video_path = recorder.encode() if recorder is not None else None
    if collision_payload is not None:
        status_name = "collision"
    elif reached_goal:
        status_name = "reached_goal"
    else:
        status_name = "time_limit"
    result = RunResult(
        example=scenario.name,
        status=status_name,
        frames=options.frames,
        simulated_seconds=options.frames * dt,
        final_position=position.tolist(),
        goal=goal.tolist(),
        collision=collision_payload,
        video=str(video_path) if video_path else None,
    )
    write_metadata(options.output_dir, result.to_dict())
    print(f"[{scenario.name}] status={result.status} final={result.final_position} video={result.video}", flush=True)
    return result
