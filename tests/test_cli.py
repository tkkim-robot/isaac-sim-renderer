from argparse import Namespace
from pathlib import Path

import pytest

from isaac_renderer.cli import validate_render_options


def _arguments(**overrides) -> Namespace:
    values = {
        "headless": True,
        "frames": 10,
        "fps": 30,
        "width": 960,
        "height": 540,
        "output_dir": Path("outputs/test"),
        "no_video": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_render_options_accept_even_dimensions() -> None:
    options = validate_render_options(_arguments())

    assert options.width == 960
    assert options.height == 540


@pytest.mark.parametrize("width,height", [(961, 540), (960, 541)])
def test_render_options_reject_odd_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="must both be even"):
        validate_render_options(_arguments(width=width, height=height))
