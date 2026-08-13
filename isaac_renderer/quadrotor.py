"""Faithful procedural port of the quadrotor visual used by SEAMLIS.

The former renderer did not load a Crazyflie mesh or URDF for its visible
robot. It assembled this proxy from ordinary USD primitives and rotated four
child Xforms every captured frame. Keeping the implementation procedural makes
the tutorial self-contained and stable across Isaac Sim releases.

Import this module only after ``SimulationApp`` has started.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pxr import Gf, UsdGeom, UsdLux

from isaac_renderer.animation import rotor_spin_degrees
from isaac_renderer.scene import (
    add_cube,
    add_cylinder,
    create_material,
    set_pose,
    set_pose_rpy,
    stage,
)

SEAMLIS_QUADROTOR_SCALE = 0.65


@dataclass(frozen=True, slots=True)
class AnimatedQuadrotor:
    """Paths and local offsets needed to pose and animate one proxy."""

    root_path: str
    rotor_roots: tuple[str, ...]
    rotor_offsets: tuple[tuple[float, float, float], ...]

    def set_world_pose(self, position: Sequence[float], rpy: Sequence[float]) -> None:
        set_pose_rpy(self.root_path, position, rpy)

    def animate_rotors(self, frame_index: int, *, spin_scale: float = 1.0) -> None:
        for rotor_index, (rotor_root, offset) in enumerate(zip(self.rotor_roots, self.rotor_offsets, strict=True)):
            angle_degrees = rotor_spin_degrees(frame_index, rotor_index, spin_scale)
            set_pose(rotor_root, offset, math.radians(angle_degrees))


def _define_xform(path: str) -> str:
    UsdGeom.Xform.Define(stage(), path)
    return path


def _add_cube_with_yaw(
    path: str,
    *,
    center: Sequence[float],
    size: Sequence[float],
    material: str,
    yaw: float = 0.0,
) -> str:
    add_cube(path, center=center, size=size, material=material)
    if yaw:
        set_pose(path, center, yaw)
    return path


def _create_materials(prefix: str) -> dict[str, str]:
    accent = np.asarray((0.15, 0.77, 0.98), dtype=float)
    return {
        "accent": create_material(
            f"{prefix}/Accent",
            accent,
            roughness=0.26,
            metallic=0.08,
            emissive=np.clip(accent * 0.12, 0.0, 1.0),
        ),
        "rotor": create_material(
            f"{prefix}/RotorDark",
            (0.12, 0.13, 0.14),
            roughness=0.45,
            metallic=0.10,
        ),
        "body": create_material(
            f"{prefix}/BodyDark",
            (0.16, 0.17, 0.20),
            roughness=0.42,
            metallic=0.18,
        ),
        "glass": create_material(
            f"{prefix}/GlassDark",
            (0.14, 0.18, 0.21),
            roughness=0.10,
            opacity=0.45,
        ),
    }


def add_seamlis_accent_lights() -> None:
    """Add the cool fill and warm rim lights used by the old visual style."""

    fill = UsdLux.SphereLight.Define(stage(), "/World/Lights/QuadrotorFill")
    fill.CreateIntensityAttr(950.0)
    fill.CreateRadiusAttr(2.2)
    fill.CreateColorAttr(Gf.Vec3f(0.48, 0.60, 0.76))
    UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-4.5, -3.5, 5.8))

    rim = UsdLux.SphereLight.Define(stage(), "/World/Lights/QuadrotorRim")
    rim.CreateIntensityAttr(720.0)
    rim.CreateRadiusAttr(2.4)
    rim.CreateColorAttr(Gf.Vec3f(0.95, 0.86, 0.78))
    UsdGeom.Xformable(rim.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(5.0, 4.5, 5.6))


def spawn_seamlis_quadrotor(
    *,
    root_path: str = "/World/Robots/CrazyflieVisual",
    scale: float = SEAMLIS_QUADROTOR_SCALE,
) -> AnimatedQuadrotor:
    """Build the exact layered primitive proxy used by the SEAMLIS renderer."""

    resolved_scale = float(scale)
    if not math.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError("scale must be finite and positive")

    visual_root = f"{root_path}/Visual"
    _define_xform(root_path)
    _define_xform(visual_root)
    materials = _create_materials("/World/Looks/SEAMLisQuadrotor")

    def scaled(values: Sequence[float]) -> tuple[float, float, float]:
        return tuple(float(resolved_scale * value) for value in values)

    _add_cube_with_yaw(
        f"{visual_root}/Body",
        center=(0.0, 0.0, 0.0),
        size=scaled((0.30, 0.16, 0.07)),
        material=materials["accent"],
    )
    _add_cube_with_yaw(
        f"{visual_root}/Top",
        center=scaled((0.0, 0.0, 0.052)),
        size=scaled((0.20, 0.12, 0.05)),
        material=materials["body"],
    )
    _add_cube_with_yaw(
        f"{visual_root}/Nose",
        center=scaled((0.16, 0.0, 0.008)),
        size=scaled((0.09, 0.06, 0.04)),
        material=materials["glass"],
    )
    _add_cube_with_yaw(
        f"{visual_root}/ArmA",
        center=(0.0, 0.0, 0.0),
        size=scaled((0.74, 0.04, 0.028)),
        material=materials["body"],
        yaw=math.pi / 4.0,
    )
    _add_cube_with_yaw(
        f"{visual_root}/ArmB",
        center=(0.0, 0.0, 0.0),
        size=scaled((0.74, 0.04, 0.028)),
        material=materials["body"],
        yaw=-math.pi / 4.0,
    )

    for name, y in (("SkidLeft", 0.10), ("SkidRight", -0.10)):
        _add_cube_with_yaw(
            f"{visual_root}/{name}",
            center=scaled((0.0, y, -0.10)),
            size=scaled((0.28, 0.015, 0.018)),
            material=materials["body"],
        )
    for name, x, y in (
        ("LegFL", 0.14, 0.10),
        ("LegFR", 0.14, -0.10),
        ("LegRL", -0.14, 0.10),
        ("LegRR", -0.14, -0.10),
    ):
        _add_cube_with_yaw(
            f"{visual_root}/{name}",
            center=scaled((x, y, -0.04)),
            size=scaled((0.018, 0.018, 0.11)),
            material=materials["body"],
        )

    rotor_offsets = tuple(
        scaled(offset)
        for offset in (
            (0.25, 0.25, 0.025),
            (0.25, -0.25, 0.025),
            (-0.25, 0.25, 0.025),
            (-0.25, -0.25, 0.025),
        )
    )
    rotor_roots: list[str] = []
    for rotor_index, offset in enumerate(rotor_offsets):
        rotor_root = _define_xform(f"{visual_root}/Rotor_{rotor_index:02d}")
        set_pose(rotor_root, offset, 0.0)
        add_cylinder(
            f"{rotor_root}/Hub",
            center=(0.0, 0.0, 0.0),
            radius=resolved_scale * 0.028,
            height=resolved_scale * 0.018,
            material=materials["rotor"],
        )
        _add_cube_with_yaw(
            f"{rotor_root}/BladeA",
            center=scaled((0.0, 0.0, 0.010)),
            size=scaled((0.18, 0.014, 0.005)),
            material=materials["accent"],
        )
        _add_cube_with_yaw(
            f"{rotor_root}/BladeB",
            center=scaled((0.0, 0.0, 0.010)),
            size=scaled((0.014, 0.18, 0.005)),
            material=materials["accent"],
        )
        rotor_roots.append(rotor_root)

    return AnimatedQuadrotor(
        root_path=root_path,
        rotor_roots=tuple(rotor_roots),
        rotor_offsets=rotor_offsets,
    )
