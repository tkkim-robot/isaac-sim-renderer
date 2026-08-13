"""Small USD scene helpers shared by the examples.

The functions intentionally expose ordinary USD prim paths so tutorial users
can replace any layer without adopting a framework.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import omni.kit.commands
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt


def stage():
    current = omni.usd.get_context().get_stage()
    if current is None:
        raise RuntimeError("No USD stage is open.")
    return current


def new_stage(simulation_app) -> None:
    omni.usd.get_context().new_stage()
    for _ in range(3):
        simulation_app.update()
    UsdGeom.SetStageMetersPerUnit(stage(), 1.0)
    UsdGeom.SetStageUpAxis(stage(), UsdGeom.Tokens.z)


def _xformable(prim_path: str) -> UsdGeom.Xformable:
    prim = stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim does not exist: {prim_path}")
    return UsdGeom.Xformable(prim)


def _op(xformable: UsdGeom.Xformable, op_type, precision):
    for operation in xformable.GetOrderedXformOps():
        if operation.GetOpType() == op_type:
            return operation
    if op_type == UsdGeom.XformOp.TypeTranslate:
        return xformable.AddTranslateOp(precision=precision)
    if op_type == UsdGeom.XformOp.TypeOrient:
        return xformable.AddOrientOp(precision=precision)
    if op_type == UsdGeom.XformOp.TypeScale:
        return xformable.AddScaleOp(precision=precision)
    raise ValueError(f"Unsupported transform operation: {op_type}")


def quaternion_from_yaw(yaw: float) -> Gf.Quatf:
    half = 0.5 * float(yaw)
    return Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half)))


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Gf.Quatf:
    """Return a USD quaternion for extrinsic XYZ roll, pitch, yaw."""

    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def set_pose(prim_path: str, position: Sequence[float], yaw: float = 0.0) -> None:
    xformable = _xformable(prim_path)
    translate = _op(xformable, UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)
    orient = _op(xformable, UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionFloat)
    xyz = np.asarray(position, dtype=float).reshape(3)
    translate.Set(Gf.Vec3d(*map(float, xyz)))
    orient.Set(quaternion_from_yaw(float(yaw)))
    ordered = [translate, orient]
    for operation in xformable.GetOrderedXformOps():
        if operation not in ordered:
            ordered.append(operation)
    xformable.SetXformOpOrder(ordered)


def set_pose_rpy(prim_path: str, position: Sequence[float], rpy: Sequence[float]) -> None:
    xformable = _xformable(prim_path)
    translate = _op(xformable, UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)
    orient = _op(xformable, UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionFloat)
    xyz = np.asarray(position, dtype=float).reshape(3)
    angles = np.asarray(rpy, dtype=float).reshape(3)
    translate.Set(Gf.Vec3d(*map(float, xyz)))
    orient.Set(quaternion_from_rpy(*map(float, angles)))
    ordered = [translate, orient]
    for operation in xformable.GetOrderedXformOps():
        if operation not in ordered:
            ordered.append(operation)
    xformable.SetXformOpOrder(ordered)


def set_transform_srt(
    prim_path: str,
    position: Sequence[float],
    rpy: Sequence[float],
    *,
    scale: Sequence[float] | None = None,
) -> None:
    """Set an imported model's local SRT with Kit's stack-aware command."""

    xyz = np.asarray(position, dtype=float).reshape(3)
    angles_deg = np.degrees(np.asarray(rpy, dtype=float).reshape(3))
    kwargs = {
        "path": prim_path,
        "new_translation": Gf.Vec3d(*map(float, xyz)),
        "new_rotation_euler": Gf.Vec3d(*map(float, angles_deg)),
        "new_rotation_order": Gf.Vec3i(0, 1, 2),
    }
    if scale is not None:
        values = np.asarray(scale, dtype=float).reshape(3)
        kwargs["new_scale"] = Gf.Vec3d(*map(float, values))
    status, _ = omni.kit.commands.execute("TransformPrimSRTCommand", **kwargs)
    if not status:
        raise RuntimeError(f"Failed to transform imported model: {prim_path}")


def set_scale(prim_path: str, scale_xyz: Sequence[float]) -> None:
    xformable = _xformable(prim_path)
    scale = _op(xformable, UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionFloat)
    values = np.asarray(scale_xyz, dtype=float).reshape(3)
    scale.Set(Gf.Vec3f(*map(float, values)))
    ordered = list(xformable.GetOrderedXformOps())
    if scale not in ordered:
        ordered.append(scale)
    xformable.SetXformOpOrder(ordered)


def create_material(
    path: str,
    color: Sequence[float],
    *,
    roughness: float = 0.55,
    metallic: float = 0.0,
    opacity: float = 1.0,
    emissive: Sequence[float] | None = None,
) -> str:
    material = UsdShade.Material.Define(stage(), path)
    shader = UsdShade.Shader.Define(stage(), f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, emissive)))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return path


def bind_material(prim_path: str, material_path: str) -> None:
    UsdShade.MaterialBindingAPI(stage().GetPrimAtPath(prim_path)).Bind(UsdShade.Material.Get(stage(), material_path))


def add_cube(
    prim_path: str,
    *,
    center: Sequence[float],
    size: Sequence[float],
    material: str | None = None,
    collision: bool = False,
) -> str:
    cube = UsdGeom.Cube.Define(stage(), prim_path)
    cube.GetSizeAttr().Set(1.0)
    xformable = UsdGeom.Xformable(cube.GetPrim())
    translate = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    scale = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat)
    translate.Set(Gf.Vec3d(*map(float, center)))
    scale.Set(Gf.Vec3f(*map(float, size)))
    if material:
        bind_material(prim_path, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return prim_path


def add_cylinder(
    prim_path: str,
    *,
    center: Sequence[float],
    radius: float,
    height: float,
    material: str | None = None,
    collision: bool = False,
) -> str:
    cylinder = UsdGeom.Cylinder.Define(stage(), prim_path)
    cylinder.GetRadiusAttr().Set(float(radius))
    cylinder.GetHeightAttr().Set(float(height))
    xformable = UsdGeom.Xformable(cylinder.GetPrim())
    xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, center)))
    if material:
        bind_material(prim_path, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    return prim_path


def add_sphere(
    prim_path: str,
    *,
    center: Sequence[float],
    radius: float,
    material: str | None = None,
) -> str:
    sphere = UsdGeom.Sphere.Define(stage(), prim_path)
    sphere.GetRadiusAttr().Set(float(radius))
    UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*map(float, center))
    )
    if material:
        bind_material(prim_path, material)
    return prim_path


def add_trail(prim_path: str, color: Sequence[float], *, width: float = 0.045) -> str:
    curve = UsdGeom.BasisCurves.Define(stage(), prim_path)
    curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr().Set([2])
    curve.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0), Gf.Vec3f(0.0)]))
    curve.CreateWidthsAttr().Set(Vt.FloatArray([float(width)]))
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr().Set([Gf.Vec3f(*map(float, color))])
    return prim_path


def update_trail(prim_path: str, points: Sequence[Sequence[float]]) -> None:
    values = [tuple(map(float, point)) for point in points]
    if not values:
        values = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    elif len(values) == 1:
        values = [values[0], values[0]]
    curve = UsdGeom.BasisCurves(stage().GetPrimAtPath(prim_path))
    curve.GetCurveVertexCountsAttr().Set([len(values)])
    curve.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*value) for value in values]))


def set_visible(prim_path: str, visible: bool) -> None:
    imageable = UsdGeom.Imageable(stage().GetPrimAtPath(prim_path))
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def add_ground(*, size: float = 20.0) -> str:
    material = create_material("/World/Looks/Ground", (0.16, 0.18, 0.21), roughness=0.8)
    return add_cube(
        "/World/Ground",
        center=(0.0, 0.0, -0.05),
        size=(float(size), float(size), 0.1),
        material=material,
        collision=True,
    )


def add_lights() -> None:
    dome = UsdLux.DomeLight.Define(stage(), "/World/Lights/Dome")
    dome.CreateIntensityAttr(550.0)
    dome.CreateColorAttr(Gf.Vec3f(0.78, 0.84, 1.0))
    key = UsdLux.DistantLight.Define(stage(), "/World/Lights/Key")
    key.CreateIntensityAttr(2200.0)
    key.CreateAngleAttr(0.8)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-48.0, 22.0, 25.0))


def add_camera(
    *,
    path: str = "/World/Camera",
    eye: Sequence[float] = (7.5, -8.0, 7.0),
    target: Sequence[float] = (0.5, 0.0, 0.0),
) -> str:
    camera = UsdGeom.Camera.Define(stage(), path)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
    camera.GetFocalLengthAttr().Set(24.0)
    camera.GetHorizontalApertureAttr().Set(36.0)
    set_camera_view(eye=list(map(float, eye)), target=list(map(float, target)), camera_prim_path=path)
    return path


def set_camera(path: str, eye: Sequence[float], target: Sequence[float]) -> None:
    set_camera_view(eye=list(map(float, eye)), target=list(map(float, target)), camera_prim_path=path)


def import_urdf(
    simulation_app,
    urdf_path: Path,
    *,
    fix_base: bool,
    merge_fixed_joints: bool = False,
    get_articulation_root: bool = True,
) -> str:
    """Import a URDF into the open stage and return its root prim path."""

    urdf_path = Path(urdf_path).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    enable_extension("isaacsim.asset.importer.urdf")
    for _ in range(3):
        simulation_app.update()
    status, config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Isaac Sim could not create a URDF import configuration.")
    config.merge_fixed_joints = bool(merge_fixed_joints)
    config.convex_decomp = False
    config.import_inertia_tensor = True
    config.fix_base = bool(fix_base)
    config.distance_scale = 1.0
    status, root_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=config,
        get_articulation_root=bool(get_articulation_root),
    )
    if not status or not root_path:
        raise RuntimeError(f"Failed to import URDF: {urdf_path}")
    for _ in range(3):
        simulation_app.update()
    return str(root_path)


def iter_valid_prim_paths(paths: Iterable[str]) -> list[str]:
    return [path for path in paths if stage().GetPrimAtPath(path).IsValid()]
