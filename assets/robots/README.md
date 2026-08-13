# Tutorial robot assets

These URDFs are intentionally simplified, mesh-free stand-ins for learning the
Isaac Sim import, rendering, articulation, and controller workflows. They use
only URDF primitive geometry, so the repository is self-contained and there
are no mesh search paths or downloaded model packages to configure.

All dimensions, masses, inertias, joint limits, and joint velocities use SI
units: meters, kilograms, kilogram-square-meters, radians, and seconds. Both
robots follow the usual mobile-robot convention: **+x is forward, +y is left,
and +z is up**.

## `crazyflie_minimal.urdf`

This is a visually recognizable Crazyflie-style quadrotor for kinematic scene
and rendering examples. Example 01 imports it as a hidden structural backing
and renders a polished procedural quadrotor on top; set that scenario's
`visual_style` to `"urdf"` to display this raw model instead. It is
not a flight-dynamics model and does not include a motor, propeller-thrust,
battery, or aerodynamic plugin. The root is `base_link`; four continuous
joints rotate the four propeller links about +z.
The rotor centers are at x/y = +/-0.046 m. Red marks the front (+x), while blue
marks the rear. The inertial and collision elements make the file importable as
a complete URDF, but tutorial code should move its root pose kinematically
unless it provides a real flight controller and force model.

## `tutorial_diff_drive.urdf`

This is a compact differential-drive ground robot intended for actual rigid
body and articulation examples. Its `base_link` frame sits on the ground plane,
so an initial root translation of z = 0 m places the wheel and caster collision
geometry on the floor. The drive-wheel radius is 0.065 m, and the wheel-center
separation is 0.260 m. The actuated continuous joints are named exactly:

- `left_wheel_joint`
- `right_wheel_joint`

Both wheel axes point along +y. A fixed spherical rear skid supplies the third
contact point; it is deliberately simple so no caster steering joint or
transmission plugin is required. Isaac Sim can command the wheel articulation
DOFs directly, so the URDF intentionally omits ROS control transmissions.

## License and provenance

These two models were created from scratch for this repository. They are not
copies of Bitcraze, Clearpath, NVIDIA, or other vendor assets, and vendor names
are used only to describe the tutorial scenario. The URDFs and this
documentation are released under the same MIT License as the project.
