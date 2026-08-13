"""Render a polished procedural quadrotor doing PID reach-avoid.

The self-contained Crazyflie URDF is imported as a structural backing, while
the visible layered USD proxy provides animated rotors and richer materials.
Motion remains kinematic, like a trajectory-replay renderer.
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
            altitude=0.96,
            visual_scale=6.0,
            robot_radius=0.30,
            visual_style="procedural_quadrotor",
            camera_mode="follow",
            camera_eye=(-5.2, -3.8, 3.35),
            camera_target=(-2.6, -2.0, 0.96),
        ),
    )
finally:
    APP.close()
