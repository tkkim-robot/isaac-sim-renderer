"""Render a ground robot using the reusable PID reach-avoid controller.

This example uses a unicycle motion model and moves the URDF root directly.
The next example replaces that motion model with Isaac PhysX wheel dynamics.
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
        default_output=output_directory("02_mobile_robot_reach_avoid"),
        default_frames=420,
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
            name="mobile_robot_reach_avoid",
            urdf_name="tutorial_diff_drive.urdf",
            motion="unicycle",
            start=(-3.0, -1.5),
            goal=(3.1, 1.6),
            altitude=0.0,
            visual_scale=1.0,
            robot_radius=0.24,
            camera_eye=(0.0, 0.0, 13.0),
            camera_target=(0.0, 0.0, 0.0),
        ),
    )
finally:
    APP.close()
