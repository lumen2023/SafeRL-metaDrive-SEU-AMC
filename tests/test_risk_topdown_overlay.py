import numpy as np
from PIL import Image

from debug_risk_field_topdown_overlay_gif import (
    _blend_risk_map_values,
    _draw_ego_footprint,
    _make_ego_to_pixel_transform,
    build_risk_overlay_args,
)


class DummyEgo:
    position = (0.0, 0.0)
    heading = (1.0, 0.0)
    top_down_length = 4.0
    top_down_width = 2.0


def test_build_risk_overlay_args_defaults_show_ego_and_road_alpha_floor():
    args = build_risk_overlay_args({})

    assert args.show_ego_shape is True
    assert args.road_min_alpha == 28


def test_draw_ego_footprint_marks_center_of_blank_risk_frame():
    ego = DummyEgo()
    base = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
    transform = _make_ego_to_pixel_transform(ego, width=64, height=64, scaling=4.0)

    output = _draw_ego_footprint(base, ego, transform, scaling=4.0)
    pixels = np.asarray(output)

    assert pixels[32, 32, :3].sum() > 0
    assert pixels[0, 0, :3].tolist() == [0, 0, 0]


def test_blend_risk_map_min_alpha_only_raises_visible_opacity():
    base = Image.new("RGBA", (3, 3), (0, 0, 0, 0))
    risk_map = np.zeros((3, 3), dtype=np.float32)
    risk_map[1, 1] = 0.02

    without_floor = _blend_risk_map_values(
        base,
        risk_map,
        min_visible=0.0,
        vmax=1.0,
        max_alpha=100,
        min_alpha=0,
    )
    with_floor = _blend_risk_map_values(
        base,
        risk_map,
        min_visible=0.0,
        vmax=1.0,
        max_alpha=100,
        min_alpha=28,
    )

    without_pixel = np.asarray(without_floor)[1, 1]
    with_pixel = np.asarray(with_floor)[1, 1]

    assert 0 < without_pixel[3] < 28
    assert with_pixel[3] == 28
    assert with_pixel[:3].tolist() == without_pixel[:3].tolist()
