"""Drive a differential robot through Isaac PhysX and push a dynamic box.

Unlike examples 01 and 02, this script only sets the robot's initial pose and
never teleports it during rollout. It sends wheel-speed targets to an imported
articulation; contacts, inertia, friction, and the final trajectory are all
determined by Isaac Sim physics.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from _bootstrap import PROJECT_ROOT  # noqa: F401  (also updates sys.path)

from isaac_renderer.cli import add_render_arguments, validate_render_options
from isaac_renderer.paths import ASSET_ROOT, output_directory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_render_arguments(
        parser,
        default_output=output_directory("03_differential_drive_dynamics"),
        default_frames=240,
    )
    return parser.parse_args()


ARGS = parse_args()
OPTIONS = validate_render_options(ARGS)

from isaacsim import SimulationApp

APP = SimulationApp(
    {
        "headless": OPTIONS.headless,
        "renderer": "RaytracedLighting",
        "width": OPTIONS.width,
        "height": OPTIONS.height,
    }
)

from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from pxr import UsdPhysics

from isaac_renderer.scene import (
    add_camera,
    add_cube,
    add_lights,
    create_material,
    import_urdf,
    new_stage,
    set_camera,
    stage,
)
from isaac_renderer.status import RunResult
from isaac_renderer.video import FrameRecorder, write_metadata


def yaw_from_quaternion_wxyz(quaternion) -> float:
    w, x, y, z = map(float, quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def configure_wheel_velocity_drives() -> None:
    """Make imported wheel joints true velocity drives.

    URDF importers commonly create position drives by default. A zero
    stiffness plus non-zero damping makes the ArticulationAction velocity
    targets produce physical wheel torque instead of fighting a pose target.
    """

    wanted = {"left_wheel_joint", "right_wheel_joint"}
    found = set()
    for prim in stage().Traverse():
        if prim.GetName() not in wanted:
            continue
        drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive_api:
            drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive_api.CreateStiffnessAttr(0.0)
        drive_api.CreateDampingAttr(35.0)
        drive_api.CreateMaxForceAttr(16.0)
        found.add(prim.GetName())
    missing = wanted - found
    if missing:
        raise RuntimeError(f"Imported URDF is missing wheel joints: {sorted(missing)}")


def main() -> RunResult:
    new_stage(APP)
    # Two fixed physics substeps per captured frame keep metadata and motion
    # duration correct for every requested output FPS (60 Hz at the default
    # 30 FPS), while leaving capture at exactly the user's video frame rate.
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / (2.0 * OPTIONS.fps),
        rendering_dt=1.0 / OPTIONS.fps,
    )
    world.scene.add_default_ground_plane()
    add_lights()

    block_material = create_material("/World/Looks/PushBlock", (0.95, 0.42, 0.08), roughness=0.48)
    block_path = add_cube(
        "/World/DynamicPushBlock",
        center=(0.30, 0.0, 0.16),
        size=(0.30, 0.30, 0.32),
        material=block_material,
        collision=True,
    )
    block_prim = stage().GetPrimAtPath(block_path)
    UsdPhysics.RigidBodyAPI.Apply(block_prim)
    mass = UsdPhysics.MassAPI.Apply(block_prim)
    mass.CreateMassAttr(0.45)

    robot_path = import_urdf(
        APP,
        ASSET_ROOT / "robots" / "tutorial_diff_drive.urdf",
        fix_base=False,
        merge_fixed_joints=True,
    )
    configure_wheel_velocity_drives()
    robot = world.scene.add(
        WheeledRobot(
            prim_path=robot_path,
            name="physics_diff_drive",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=False,
            position=np.asarray((-2.0, 0.0, 0.0)),
        )
    )
    drive = DifferentialController(
        name="unicycle_to_wheels",
        wheel_radius=0.065,
        wheel_base=0.260,
        max_linear_speed=0.70,
        max_angular_speed=1.5,
        max_wheel_speed=18.0,
    )
    camera_path = add_camera(eye=(5.7, -6.6, 5.0), target=(0.2, 0.0, 0.15))
    world.reset()

    recorder = None
    if not OPTIONS.no_video:
        recorder = FrameRecorder(
            APP,
            OPTIONS.output_dir,
            camera_path=camera_path,
            fps=OPTIONS.fps,
            width=OPTIONS.width,
            height=OPTIONS.height,
            stem="differential_drive_dynamics",
        )

    last_speed = 0.0
    for frame in range(OPTIONS.frames):
        # Drive into the movable block, arc around it, then finish straight.
        fraction = frame / max(1, OPTIONS.frames - 1)
        if fraction < 0.62:
            command = np.asarray((0.55, 0.0))
        elif fraction < 0.82:
            command = np.asarray((0.38, 1.05))
        else:
            command = np.asarray((0.48, 0.0))
        robot.apply_wheel_actions(drive.forward(command))
        for _ in range(2):
            world.step(render=False)
        world.render()

        position, orientation = robot.get_world_pose()
        yaw = yaw_from_quaternion_wxyz(orientation)
        last_speed = float(np.linalg.norm(robot.get_linear_velocity()))
        eye = (float(position[0] - 3.6), float(position[1] - 4.5), 3.3)
        target = (float(position[0] + 0.6 * math.cos(yaw)), float(position[1] + 0.6 * math.sin(yaw)), 0.12)
        set_camera(camera_path, eye, target)
        if recorder is not None:
            # Viewport capture needs Kit updates. Pause first so those async
            # updates cannot add an unpredictable number of physics substeps.
            world.pause()
            recorder.capture()
            world.play()

    robot.apply_wheel_actions(drive.forward(np.zeros(2)))
    final_position, _ = robot.get_world_pose()
    block_translation = stage().GetPrimAtPath(block_path).GetAttribute("xformOp:translate").Get()
    video_path = recorder.encode() if recorder is not None else None
    result = RunResult(
        example="differential_drive_dynamics",
        status="completed",
        frames=OPTIONS.frames,
        simulated_seconds=OPTIONS.frames / float(OPTIONS.fps),
        final_position=list(map(float, final_position)),
        goal=[],
        collision={
            "note": "PhysX contacts are enabled; the movable block displacement demonstrates collision response.",
            "block_final_position": list(map(float, block_translation)),
            "final_robot_speed_mps": last_speed,
        },
        video=str(video_path) if video_path else None,
    )
    write_metadata(OPTIONS.output_dir, result.to_dict())
    print(
        f"[differential_drive_dynamics] final={result.final_position} "
        f"block={result.collision['block_final_position']} video={result.video}",
        flush=True,
    )
    return result


try:
    main()
finally:
    APP.close()
