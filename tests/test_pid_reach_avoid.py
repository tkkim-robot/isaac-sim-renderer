import numpy as np
import pytest

from controllers import (
    CircleObstacle,
    PIDConfig,
    PIDReachAvoidController,
    wrap_angle,
)


def test_wrap_angle_handles_boundary_and_vector_inputs() -> None:
    assert wrap_angle(np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(-np.pi) == pytest.approx(-np.pi)

    values = np.asarray((-3.0 * np.pi, 0.0, 3.0 * np.pi))
    np.testing.assert_allclose(wrap_angle(values), (-np.pi, 0.0, -np.pi))


def test_obstacle_copies_input_and_advances_with_velocity() -> None:
    caller_owned = np.asarray((1.0, 2.0))
    obstacle = CircleObstacle(
        center=caller_owned,
        radius=0.3,
        velocity=(0.5, -1.0),
        name="moving",
    )
    caller_owned[:] = 100.0

    assert obstacle.center == (1.0, 2.0)
    assert obstacle.advanced(2.0).center == pytest.approx((2.0, 0.0))


def test_pid_velocity_reaches_toward_goal_and_respects_speed_limit() -> None:
    controller = PIDReachAvoidController(
        PIDConfig(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            max_speed=0.75,
            goal_tolerance=0.0,
            repulsion_gain=0.0,
        )
    )

    velocity = controller.compute_velocity((0.0, 0.0), (2.0, 0.0), dt=0.1)

    np.testing.assert_allclose(velocity, (0.75, 0.0))


def test_repulsion_and_dynamic_obstacle_update_change_avoidance_side() -> None:
    controller = PIDReachAvoidController(
        PIDConfig(
            kp=0.2,
            ki=0.0,
            kd=0.0,
            max_speed=2.0,
            robot_radius=0.1,
            safety_margin=0.0,
            influence_distance=1.0,
            repulsion_gain=0.25,
            max_repulsion=1.0,
        ),
        [CircleObstacle((0.0, 0.6), 0.1, name="crossing")],
    )

    above = controller.compute_velocity((0.0, 0.0), (2.0, 0.0), dt=0.1)
    controller.reset()
    updated = controller.update_obstacle("crossing", center=(0.0, -0.6))
    below = controller.compute_velocity((0.0, 0.0), (2.0, 0.0), dt=0.1)

    assert updated.center == (0.0, -0.6)
    assert above[1] < 0.0
    assert below[1] > 0.0


def test_obstacle_prediction_and_explicit_advance_are_deterministic() -> None:
    obstacle = CircleObstacle(
        center=(2.0, 0.0), radius=0.1, velocity=(-1.0, 0.0), name="dynamic"
    )
    controller = PIDReachAvoidController(
        PIDConfig(
            kp=0.0,
            ki=0.0,
            kd=0.0,
            influence_distance=1.5,
            prediction_horizon=1.0,
            robot_radius=0.1,
            safety_margin=0.0,
        ),
        [obstacle],
    )

    predicted = controller.avoidance_velocity((0.0, 0.0))
    snapshots = controller.advance_obstacles(0.5)

    assert predicted[0] < 0.0
    assert snapshots[0].center == pytest.approx((1.5, 0.0))


def test_conditional_anti_windup_and_reset() -> None:
    controller = PIDReachAvoidController(
        PIDConfig(
            kp=1.0,
            ki=1.0,
            kd=0.0,
            max_speed=0.2,
            integral_limit=100.0,
            goal_tolerance=0.0,
            repulsion_gain=0.0,
        )
    )

    for _ in range(20):
        controller.compute_velocity((0.0, 0.0), (10.0, 0.0), dt=0.1)

    np.testing.assert_allclose(controller.integral_error, (0.0, 0.0))
    assert controller.previous_error is not None

    controller.reset()
    np.testing.assert_allclose(controller.integral_error, (0.0, 0.0))
    assert controller.previous_error is None


def test_integral_is_componentwise_bounded_when_output_is_not_saturated() -> None:
    controller = PIDReachAvoidController(
        PIDConfig(
            kp=0.0,
            ki=0.1,
            kd=0.0,
            max_speed=100.0,
            integral_limit=0.25,
            goal_tolerance=0.0,
            repulsion_gain=0.0,
        )
    )
    for _ in range(10):
        controller.compute_velocity((0.0, 0.0), (1.0, -1.0), dt=1.0)

    np.testing.assert_allclose(controller.integral_error, (0.25, -0.25))


def test_unicycle_uses_short_wrapped_turn_and_reports_goal() -> None:
    controller = PIDReachAvoidController(
        PIDConfig(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            max_speed=1.0,
            goal_tolerance=0.05,
            repulsion_gain=0.0,
            heading_gain=2.0,
        )
    )
    command = controller.compute_unicycle(
        pose=(0.0, 0.0, np.pi - 0.01),
        goal=(-1.0, -0.02),
        dt=0.05,
    )

    assert 0.0 < command.heading_error < 0.1
    assert command.angular_velocity > 0.0
    assert command.linear_velocity > 0.9
    np.testing.assert_allclose(command.as_array(), (command.v, command.omega))

    reached = controller.compute_command((0.0, 0.0, 0.7), (0.01, 0.0), dt=0.05)
    assert reached.reached_goal
    assert reached.linear_velocity == 0.0
    assert reached.angular_velocity == 0.0


@pytest.mark.parametrize("bad_dt", [0.0, -0.1, np.inf, np.nan])
def test_controller_rejects_invalid_dt(bad_dt: float) -> None:
    controller = PIDReachAvoidController()
    with pytest.raises(ValueError):
        controller.compute_velocity((0.0, 0.0), (1.0, 0.0), bad_dt)
