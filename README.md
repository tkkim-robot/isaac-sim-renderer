# Isaac Sim Renderer Tutorial

A small, self-contained starting point for using NVIDIA Isaac Sim as a robot
renderer and physics simulator. The repository demonstrates the complete path
from a controller command to a USD scene, collision reporting, PNG frames, and
an H.264 MP4—without bringing a project-specific optimization stack into a new
application.

The examples intentionally stay simple. Copy one, replace its robot or
controller, and keep the reusable scene/capture utilities.

## What is included

| Example | Robot motion | Obstacles | What it teaches |
| --- | --- | --- | --- |
| [`01_crazyflie_reach_avoid.py`](examples/01_crazyflie_reach_avoid.py) | Kinematic URDF backing + animated USD proxy | Static box/pillar plus a moving cylinder | Procedural quadrotor visual, spinning rotors, PID reach-avoid, collision/failure visualization, follow camera, MP4 |
| [`02_mobile_robot_reach_avoid.py`](examples/02_mobile_robot_reach_avoid.py) | Kinematic unicycle model | Static and dynamic | Ground-robot controller template and the same renderer contract |
| [`03_differential_drive_dynamics.py`](examples/03_differential_drive_dynamics.py) | Isaac PhysX articulation | A movable rigid box | Real wheel-joint velocity drives, mass, friction, contact response, a follow camera, MP4 |

The first two examples deliberately separate control from rendering, making
them useful for replaying externally generated trajectories. The third is the
important contrast: after setting the initial pose, it never teleports the
robot during rollout. Isaac Sim integrates the wheel forces and contacts.

Included reusable pieces:

- a small NumPy PID reach-avoid controller in [`controllers/`](controllers);
- swept-circle collision checks with latched metadata in
  [`isaac_renderer/collision.py`](isaac_renderer/collision.py);
- USD scene, URDF import, camera, material, and trail helpers;
- reusable procedural quadrotor geometry and rotor animation;
- deterministic viewport PNG capture and FFmpeg H.264/yuv420p encoding;
- self-contained primitive-only Crazyflie and differential-drive URDFs;
- Docker Compose lifecycle scripts and host-side unit tests.

## 1. Prerequisites

The container workflow is Linux-only and requires:

1. An NVIDIA RTX GPU with ray-tracing support.
2. A host NVIDIA driver compatible with Isaac Sim 6.0.1.
3. Docker Engine, the Compose plugin, and NVIDIA Container Toolkit 1.17 or
   newer.
4. At least 32 GB RAM, 16 GB GPU memory, and roughly 50 GB free disk for the
   image and caches.

Use NVIDIA's current [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
and [container installation guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
as the authority for driver and platform compatibility.

Configure the NVIDIA runtime once on the host:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --runtime=nvidia --gpus all \
  nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

The `nvcr.io/nvidia/isaac-sim:6.0.1` image already contains Isaac Sim and its
matching CUDA user-space runtime. Do not install a display driver or a second
CUDA stack in the Dockerfile; the host driver is passed through by NVIDIA
Container Toolkit.

## 2. Start and stop the container

From the repository root:

```bash
cp .env.example .env
./scripts/prepare.sh
./scripts/up.sh
```

`prepare.sh` validates Compose, reports the visible GPU/driver, and creates the
only writable bind mount, `outputs/`. The project itself is mounted read-only
at `/workspace`; Isaac's cache, logs, configuration, data, packages, and Hub
cache use named Docker volumes.

Starting the service with `ACCEPT_EULA=Y` indicates acceptance of the
[NVIDIA Isaac Sim EULA](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/).
Privacy telemetry is not enabled by this project.

Stop the service without deleting its warm caches:

```bash
./scripts/down.sh
```

To also delete the named caches later, explicitly run
`docker compose down --volumes`. That is intentionally not the default.

## 3. Run an example

Against the reusable service started above:

```bash
./scripts/exec-example.sh examples/01_crazyflie_reach_avoid.py
```

Or run an ephemeral one-shot container (it builds the image when needed):

```bash
./scripts/run-example.sh examples/02_mobile_robot_reach_avoid.py
./scripts/run-example.sh examples/03_differential_drive_dynamics.py
```

Render every example sequentially:

```bash
./scripts/render-all.sh
```

Common options are identical across scripts:

```text
--frames 360       number of captured frames
--fps 30           simulation/capture rate for tutorial rollouts
--width 960        output width
--height 540       output height
--output-dir PATH  override the example's output directory
--no-video         run the simulation without PNG/MP4 capture
--no-headless      show a native GUI (useful with a local install)
```

Width and height must be even because the portable H.264 `yuv420p` output
format stores chroma in 2×2 pixel blocks.

Container runs are headless by default. For interactive GUI work, a local Isaac
Sim install is usually simpler because X11/Wayland is intentionally not exposed
by this minimal Compose file.

If Isaac Sim is installed locally, always use its bundled Python launcher, not
the system interpreter:

```bash
/path/to/isaac-sim/python.sh \
  examples/01_crazyflie_reach_avoid.py --no-headless
```

## 4. Find the results

Each example creates an isolated directory:

```text
outputs/
├── 01_crazyflie_reach_avoid/
│   ├── crazyflie_reach_avoid.mp4
│   ├── metadata.json
│   └── frames/frame_00000.png ...
├── 02_mobile_robot_reach_avoid/
└── 03_differential_drive_dynamics/
```

Videos use H.264, `yuv420p`, and `+faststart`, so they play in browsers and
ordinary media players. `metadata.json` records the terminal status, duration,
final pose, collision details, and output path. Frame folders are retained for
debugging or re-encoding; delete an example's own output directory when you no
longer need it.

## 5. Understand the reach-avoid loop

The controller has no Isaac dependency and can be tested with regular Python:

```python
from controllers import CircleObstacle, PIDReachAvoidController

controller = PIDReachAvoidController(
    obstacles=[CircleObstacle(center=(1.0, 0.0), radius=0.4)]
)
velocity_xy = controller.compute_velocity(
    position=(0.0, 0.0), goal=(3.0, 1.0), dt=1 / 30
)
command = controller.compute_unicycle(
    pose=(0.0, 0.0, 0.0), goal=(3.0, 1.0), dt=1 / 30
)
print(command.linear_velocity, command.angular_velocity)
```

`kp`, `ki`, and `kd` attract the robot to its goal. Bounded potential-field
repulsion pushes it away from the inflated obstacle boundary. Integral state
is clamped and conditionally disabled during saturation. The controller is a
readable template—not a formal safety guarantee. Use CBF, MPC, a planner, or a
certified supervisor when guarantees matter.

Dynamic obstacles are ordinary `CircleObstacle` objects with a velocity. The
examples update their center every frame and use a short prediction horizon.

For a different controller, preserve only this small boundary:

```text
current robot state + obstacle snapshot -> command or next pose
next pose -> collision monitor -> Isaac scene transform / wheel action
Isaac viewport -> numbered PNGs -> FFmpeg -> MP4
```

## 6. Collision behavior

`CollisionMonitor.check_segment()` sweeps the robot circle from its previous
position to its next position, preventing fast discrete steps from tunneling
through a circular obstacle approximation. It latches the first event with:

- robot and obstacle names, kinds, centers, and radii;
- contact point, signed clearance, and penetration depth;
- simulation step/time and swept-trajectory fraction;
- caller and obstacle metadata.

If a reach-avoid example collides, it stops advancing the controller, places a
red impact marker, writes the event to `metadata.json`, and holds a visible
failure animation. The aerial robot falls and tumbles; the ground robot tips.
The visible marker and metadata are both driven by the same latched
first-contact event.

The PhysX example demonstrates real rigid-body collision response by pushing a
box. For production contact sensing, add Isaac Sim's current experimental
physics contact sensor API rather than treating the geometric monitor as a
force sensor.

## 7. Use another robot or scene

### Replace a URDF

Example 01 deliberately has two robot layers. It imports
`crazyflie_minimal.urdf` as an invisible, synchronized structural backing, then
renders the polished proxy from [`isaac_renderer/quadrotor.py`](isaac_renderer/quadrotor.py).
The visible robot is built from USD cubes and cylinders rather than loaded from
a mesh or URDF. Its four rotor child Xforms rotate independently every captured
frame and stop on crash.

Set `visual_style="urdf"` in example 01 if you specifically want to display the
raw imported URDF instead of the procedural rendering layer.

1. Put a self-contained URDF and any meshes under `assets/robots/`.
2. Copy example 01 or 02.
3. Change `urdf_name`, `visual_scale`, `robot_radius`, start, and goal.
4. If it is a dynamic articulation, set `fix_base=False` and use example 03's
   `WheeledRobot`/joint-action path.
5. Verify returned importer paths. A fixed-base import may return a
   `/model/root_joint` articulation prim; kinematic transforms belong on the
   parent model prim.

The included URDFs use only primitive geometry, SI units, +x forward, +y left,
and +z up. They need no asset server or network downloads.

### Add an obstacle

Add matching entries in both layers:

1. a USD visual/collider using `add_cube()` or `add_cylinder()`; and
2. a `CircleObstacle` for control plus a `Circle` for collision reporting.

This explicit duplication makes the tutorial approximation visible. For exact
mesh collision distance, query PhysX or a signed-distance representation.

### Change the camera or video

- Fixed hero view: edit `camera_eye` and `camera_target` in example 02.
- Follow view: adapt `_update_follow_camera()` for example 01 or the per-frame
  `set_camera()` call in example 03.
- Encoding: edit `FrameRecorder.encode()` in
  [`isaac_renderer/video.py`](isaac_renderer/video.py).
- Higher resolution: pass `--width 1920 --height 1080`.

Capture is deliberately PNG plus explicit FFmpeg rather than an opaque video
writer. A failed run still leaves inspectable frames, and encoding settings are
easy to reproduce.

## 8. Test without launching Isaac Sim

Install only the small host dependency set and run:

```bash
python -m pip install -e '.[dev]'
./scripts/test.sh
```

The tests cover PID behavior, saturation/anti-windup, dynamic obstacle updates,
angle wrapping, circle geometry, swept collision detection, event latching,
and validation failures. CI runs the same pure-Python suite; it does not try to
install the multi-gigabyte Isaac image on a GitHub runner.

For an inexpensive simulator smoke test, skip capture:

```bash
./scripts/run-example.sh examples/01_crazyflie_reach_avoid.py \
  --frames 5 --no-video --output-dir /outputs/smoke
```

## Repository layout

```text
.
├── assets/robots/              self-contained tutorial URDFs
├── controllers/                simulator-independent PID reach-avoid
├── examples/                   executable Isaac Sim tutorials
├── isaac_renderer/             scene, animated quadrotor, collision, status, video
├── scripts/                    Compose lifecycle, rendering, and tests
├── tests/                      fast unit tests
├── Dockerfile                  Isaac Sim 6.0.1 + FFmpeg
└── docker-compose.yaml         GPU, rootless user, cache and output mounts
```

## Troubleshooting

- **Driver or Vulkan initialization failure:** compare `nvidia-smi` with the
  current Isaac Sim 6.0 requirements. Upgrade the host driver; changing CUDA
  packages inside this image will not repair an incompatible driver.
- **Permission denied under `outputs/`:** run `./scripts/prepare.sh`. The Isaac
  image runs rootless as UID/GID 1234.
- **No GPU in Docker:** rerun `nvidia-ctk runtime configure`, restart Docker,
  and verify the CUDA `nvidia-smi` command from the prerequisites.
- **Black or empty frame:** keep the app headless, verify the camera path and
  clipping range, and first try `--frames 2 --width 480 --height 270`.
- **URDF imports but does not move:** for a kinematic renderer, transform the
  model prim; for dynamics, reset the `World`, use the returned articulation
  root, and configure wheel joints as velocity drives with zero stiffness.
- **First run is slow:** shader compilation and asset caches are cold. Named
  Compose volumes make later launches faster.
- **Need network assets:** these examples do not. If you switch to JetBot,
  Kaya, or another Isaac asset, allow outbound HTTPS to NVIDIA's asset host and
  keep the cache volumes.

## Version and license

The container is pinned to Isaac Sim 6.0.1 for repeatability. NVIDIA marks some
of the recommended 6.0 replacements experimental and the older
`isaacsim.core.api` and `isaacsim.robot.wheeled_robots` namespaces deprecated;
their use is isolated to example 03 so the dynamics tutorial stays compact and
the imports can be replaced in one place as the APIs settle.

Repository code and original primitive URDFs are MIT licensed; see
[`LICENSE`](LICENSE). NVIDIA Isaac Sim, its container, and third-party
dependencies retain their respective licenses.
