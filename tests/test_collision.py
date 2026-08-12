import numpy as np
import pytest

from isaac_renderer.collision import (
    Circle,
    CollisionMonitor,
    circle_signed_clearance,
    circles_collide,
    closest_point_on_segment,
    point_to_segment_distance,
    segment_circle_first_intersection,
    segment_circle_intersection,
)


def test_closest_point_and_distance_handle_zero_length_segment() -> None:
    closest, fraction = closest_point_on_segment((3.0, 4.0), (1.0, 1.0), (1.0, 1.0))

    np.testing.assert_allclose(closest, (1.0, 1.0))
    assert fraction == 0.0
    assert point_to_segment_distance((3.0, 4.0), (1.0, 1.0), (1.0, 1.0)) == pytest.approx(
        np.sqrt(13.0)
    )


def test_circle_tangency_counts_as_collision() -> None:
    assert circle_signed_clearance((0.0, 0.0), 0.5, (1.0, 0.0), 0.5) == pytest.approx(
        0.0
    )
    assert circles_collide((0.0, 0.0), 0.5, (1.0, 0.0), 0.5)


def test_segment_circle_intersection_returns_first_fraction_and_tangent() -> None:
    fraction = segment_circle_first_intersection(
        (-2.0, 1.0), (2.0, 1.0), (0.0, 0.0), 1.0
    )

    assert fraction == pytest.approx(0.5)
    assert segment_circle_intersection((-2.0, 1.0), (2.0, 1.0), (0.0, 0.0), 1.0)
    assert not segment_circle_intersection(
        (-2.0, 1.01), (2.0, 1.01), (0.0, 0.0), 1.0
    )


def test_monitor_returns_clean_status_then_rich_first_collision() -> None:
    monitor = CollisionMonitor(
        robot_radius=0.25,
        robot_name="turtlebot",
        safety_margin=0.05,
        obstacles=[
            Circle(
                (2.0, 0.0),
                0.25,
                name="moving_box_proxy",
                kind="dynamic",
                metadata={"prim_path": "/World/MovingBox"},
            )
        ],
    )

    clear = monitor.check((0.0, 0.0), step=3, simulation_time=0.3)
    assert clear.clean
    assert clear.state == "clear"
    assert not clear
    assert clear.event is None

    monitor.update_obstacle("moving_box_proxy", center=(0.5, 0.0))
    hit = monitor.check(
        (0.0, 0.0),
        step=4,
        simulation_time=0.4,
        metadata={"episode": 7},
    )

    assert hit.collided
    assert bool(hit)
    assert hit.state == "collision"
    assert hit.event is monitor.first_collision
    assert hit.event is not None
    assert hit.event.robot_name == "turtlebot"
    assert hit.event.obstacle_name == "moving_box_proxy"
    assert hit.event.obstacle_kind == "dynamic"
    assert hit.event.step == 4
    assert hit.event.simulation_time == pytest.approx(0.4)
    assert hit.event.signed_clearance == pytest.approx(-0.05)
    assert hit.event.penetration_depth == pytest.approx(0.05)
    assert hit.event.obstacle_metadata["prim_path"] == "/World/MovingBox"
    assert hit.event.metadata["episode"] == 7
    assert hit.as_dict()["event"]["contact_point"] == pytest.approx([0.25, 0.0])


def test_monitor_latches_only_first_event_until_reset() -> None:
    monitor = CollisionMonitor(
        0.5,
        [
            Circle((0.0, 0.0), 0.5, name="first"),
            Circle((0.0, 0.0), 0.5, name="second"),
        ],
    )

    first_status = monitor.check((0.0, 0.0), step=9)
    later_status = monitor.check((100.0, 100.0), step=10)

    assert first_status.event is not None
    assert first_status.event.obstacle_name == "first"
    assert later_status.event is first_status.event
    assert later_status.checks == 2

    monitor.reset()
    reset_status = monitor.check((100.0, 100.0), step=0)
    assert reset_status.clean
    assert reset_status.checks == 1


def test_swept_monitor_prevents_tunneling() -> None:
    monitor = CollisionMonitor(
        robot_radius=0.25,
        obstacles=[Circle((0.0, 0.0), 0.25, name="post")],
    )

    status = monitor.check_segment(
        (-2.0, 0.0),
        (2.0, 0.0),
        step=1,
        simulation_time=0.1,
    )

    assert status.collided
    assert status.event is not None
    assert status.event.detection_mode == "swept"
    assert status.event.trajectory_fraction == pytest.approx(0.375)
    assert status.event.robot_center == pytest.approx((-0.5, 0.0))
    assert status.event.signed_clearance == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (((0.0, 0.0), -0.1, (1.0, 0.0), 0.1), "radius_a"),
        (((0.0, np.nan), 0.1, (1.0, 0.0), 0.1), "center_a"),
    ],
)
def test_geometry_rejects_invalid_inputs(args: tuple[object, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        circle_signed_clearance(*args)
