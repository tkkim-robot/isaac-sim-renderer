import pytest

from isaac_renderer.animation import rotor_spin_degrees


def test_rotor_spin_uses_deterministic_per_rotor_rates() -> None:
    assert rotor_spin_degrees(0, 0) == 0.0
    assert rotor_spin_degrees(1, 0) == 115.0
    assert rotor_spin_degrees(3, 2) == pytest.approx((3 * 115.0 * 1.12) % 360.0)


def test_rotor_spin_stops_on_failure() -> None:
    assert rotor_spin_degrees(200, 3, spin_scale=0.0) == 0.0


@pytest.mark.parametrize(
    ("frame_index", "rotor_index", "spin_scale"),
    [(-1, 0, 1.0), (0, -1, 1.0), (0, 0, -0.1), (0, 0, float("nan"))],
)
def test_rotor_spin_rejects_invalid_inputs(frame_index: int, rotor_index: int, spin_scale: float) -> None:
    with pytest.raises(ValueError):
        rotor_spin_degrees(frame_index, rotor_index, spin_scale)
