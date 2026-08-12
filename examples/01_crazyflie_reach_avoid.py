"""Render a Crazyflie-style URDF doing PID reach-avoid.

The motion is intentionally kinematic, like a trajectory-replay renderer.  It
is the smallest example of the controller -> state -> Isaac renderer pattern.
Use example 03 when you need Isaac PhysX to determine robot motion.
"""

from __future__ import annotations

import argparse

from _bootstrap import PROJECT_ROOT  # noqa: F401  (also updates sys.path)

from isaac_renderer.cli import add_render_arguments, validate_render_options
from isaac_renderer.paths import output_directory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_render_arguments(
        parser,
        default_output=output_directory("01_crazyflie_reach_avoid"),
        default_frames=360,
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

from _kinematic_reach_avoid import Scenario, run_kinematic_reach_avoid

try:
    run_kinematic_reach_avoid(
        APP,
        OPTIONS,
        Scenario(
            name="crazyflie_reach_avoid",
            urdf_name="crazyflie_minimal.urdf",
            motion="holonomic",
            start=(-3.0, -2.2),
            goal=(3.2, 2.1),
            altitude=0.85,
            visual_scale=6.0,
            robot_radius=0.30,
            camera_eye=(0.0, 0.0, 13.0),
            camera_target=(0.0, 0.0, 0.0),
        ),
    )
finally:
    APP.close()
