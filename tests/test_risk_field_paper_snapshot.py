import json
from pathlib import Path

import numpy as np

from debug_risk_field_paper_snapshot import (
    align_topdown_image_for_world_plot,
    build_paper_snapshot_figure,
    collect_complex_scene_records,
    extract_route_plot_segments,
    get_paper_style_rcparams,
    resolve_output_paths,
    resolve_selected_scene,
    save_figure_outputs,
)
from risk_field_bev import (
    build_risk_surface_data,
    resolve_effective_resolution,
    sample_risk_field_world_bbox,
)


class DummyRoadNetwork:
    def __init__(self, bbox):
        self._bbox = tuple(bbox)

    def get_bounding_box(self):
        return self._bbox


class DummyMap:
    def __init__(self, bbox):
        self.road_network = DummyRoadNetwork(bbox)


class DummyEnv:
    def __init__(self, bbox):
        self.current_map = DummyMap(bbox)


class DummyVehicle:
    position = (0.0, 0.0)


class DummyCalculator:
    def calculate_at_position(self, env, vehicle, world_position):
        x_value = float(world_position[0])
        y_value = float(world_position[1])
        risk = x_value + y_value
        return risk, {"risk_field_cost": risk}


class DummyLane:
    def __init__(self, points, index):
        self._points = np.asarray(points, dtype=float)
        self.index = index

    def get_polyline(self, interval=2.0):
        return self._points


class DummyAgent:
    def __init__(self, lane_index, checkpoints):
        self.lane_index = lane_index
        self.navigation = type("DummyNavigation", (), {"checkpoints": checkpoints})()


class DummyRouteEnv:
    def __init__(self):
        checkpoints = ["A", "B", "C"]
        graph = {
            "A": {
                "B": [
                    DummyLane([(0.0, 3.0), (10.0, 3.0)], ("A", "B", 0)),
                    DummyLane([(0.0, 0.0), (10.0, 0.0)], ("A", "B", 1)),
                ]
            },
            "B": {
                "C": [
                    DummyLane([(10.0, 6.0), (20.0, 6.0)], ("B", "C", 0)),
                    DummyLane([(10.0, 0.4), (20.0, 0.4)], ("B", "C", 1)),
                ]
            },
        }
        self.agent = DummyAgent(("A", "B", 1), checkpoints)
        self.agents = {"agent0": self.agent}
        road_network = type("DummyRouteRoadNetwork", (), {"graph": graph})()
        self.current_map = type("DummyRouteMap", (), {"road_network": road_network})()


def test_collect_complex_scene_records_deduplicates_by_seed(tmp_path: Path):
    batch_a = tmp_path / "train_batch_a" / "manifests"
    batch_b = tmp_path / "train_batch_b" / "manifests"
    batch_a.mkdir(parents=True)
    batch_b.mkdir(parents=True)

    (batch_a / "manifest.json").write_text(
        json.dumps(
            {
                "split": "train",
                "records": [
                    {"seed": 100, "complexity_score": 21.5, "route_length_m": 310.0},
                    {"seed": 101, "complexity_score": 18.2, "route_length_m": 290.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    (batch_b / "manifest.json").write_text(
        json.dumps(
            {
                "split": "train",
                "records": [
                    {"seed": 100, "complexity_score": 24.0, "route_length_m": 325.0},
                    {"seed": 102, "complexity_score": 19.0, "route_length_m": 305.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    records = collect_complex_scene_records(tmp_path, "train")
    assert [record["seed"] for record in records] == [100, 102, 101]
    assert records[0]["complexity_score"] == 24.0


def test_resolve_selected_scene_uses_auto_rank_or_manual_seed(tmp_path: Path):
    batch = tmp_path / "train_batch_a" / "manifests"
    batch.mkdir(parents=True)
    (batch / "manifest.json").write_text(
        json.dumps(
            {
                "split": "train",
                "records": [
                    {"seed": 123, "complexity_score": 30.0, "route_length_m": 400.0},
                    {"seed": 124, "complexity_score": 28.5, "route_length_m": 350.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    auto_selected = resolve_selected_scene("train", None, tmp_path, 2)
    manual_selected = resolve_selected_scene("train", 555, tmp_path / "missing", 1)

    assert auto_selected["seed"] == 124
    assert auto_selected["auto_selected"] is True
    assert manual_selected["seed"] == 555
    assert manual_selected["auto_selected"] is False
    assert manual_selected["complexity_score"] is None


def test_resolve_effective_resolution_and_world_bbox_sampling_respect_limits_and_mask():
    effective_resolution, grid_shape = resolve_effective_resolution((0.0, 6.0, 0.0, 6.0), 1.0, max_grid_points=16)
    assert effective_resolution > 1.0
    assert grid_shape[0] * grid_shape[1] <= 16

    env = DummyEnv((0.0, 2.0, 0.0, 1.0))
    vehicle = DummyVehicle()
    calculator = DummyCalculator()
    road_grid_mask = np.array([[True, False, True], [False, True, False]], dtype=bool)
    sample = sample_risk_field_world_bbox(
        env,
        vehicle,
        calculator=calculator,
        bbox=(0.0, 2.0, 0.0, 1.0),
        resolution=1.0,
        max_grid_points=16,
        road_grid_mask=road_grid_mask,
    )

    assert sample["grid_shape"] == [2, 3]
    assert np.isnan(sample["risk"][0, 1])
    assert np.isnan(sample["risk"][1, 0])
    assert sample["risk"][1, 1] == 2.0


def test_route_plot_segments_choose_continuous_lanes_without_bridging_gaps():
    route_plot = extract_route_plot_segments(DummyRouteEnv(), sample_interval=2.0, gap_threshold_m=2.0)
    strict_route_plot = extract_route_plot_segments(DummyRouteEnv(), sample_interval=2.0, gap_threshold_m=0.1)

    assert len(route_plot["route_segments_world"]) == 2
    assert route_plot["selected_route_lane_ids"] == ["A/B/1", "B/C/1"]
    assert route_plot["route_gap_count"] == 0
    assert route_plot["max_route_gap_m"] < 1.0
    assert strict_route_plot["route_gap_count"] == 1


def test_align_topdown_image_for_world_plot_flips_vertical_axis():
    base_image = np.array(
        [
            [[10, 0, 0], [20, 0, 0]],
            [[30, 0, 0], [40, 0, 0]],
            [[50, 0, 0], [60, 0, 0]],
        ],
        dtype=np.uint8,
    )

    aligned = align_topdown_image_for_world_plot(base_image)

    assert aligned[:, :, 0].tolist() == [[50, 60], [30, 40], [10, 20]]
    assert not np.shares_memory(aligned, base_image)


def test_paper_snapshot_figure_exports_png_and_pdf(tmp_path: Path):
    base_image = np.full((96, 96, 3), 255, dtype=np.uint8)
    vehicle_sample = {
        "risk": np.array(
            [
                [0.0, 0.05, 0.20],
                [0.10, 0.80, 0.30],
                [0.0, 0.12, 0.0],
            ],
            dtype=float,
        ),
        "x_values": np.array([120.0, 126.0, 132.0], dtype=float),
        "y_values": np.array([0.0, 6.0, 12.0], dtype=float),
        "extent": [120.0, 132.0, 0.0, 12.0],
        "plot_bounds": (120.0, 132.0, 0.0, 12.0),
        "grid_shape": [3, 3],
        "effective_resolution": 6.0,
        "requested_resolution": 6.0,
        "component_key": "risk_field_vehicle_cost",
    }
    lane_sample = dict(vehicle_sample)
    lane_sample["risk"] = np.array(
        [
            [0.0, 0.15, 0.0],
            [0.25, 0.90, 0.22],
            [0.0, 0.18, 0.0],
        ],
        dtype=float,
    )
    lane_sample["component_key"] = "risk_field_lane_cost"
    vehicle_surface = build_risk_surface_data(vehicle_sample, risk_min_visible=0.08)
    lane_surface = build_risk_surface_data(lane_sample, risk_min_visible=0.08)

    fig, meta = build_paper_snapshot_figure(
        base_image,
        surface_extent=(120.0, 132.0, 0.0, 12.0),
        vehicle_surface_data=vehicle_surface,
        lane_surface_data=lane_surface,
        route_segments_world=[np.array([[121.0, 1.0], [126.0, 5.5], [131.0, 10.5]], dtype=float)],
        ego_position=np.array([121.0, 1.0], dtype=float),
        cmap_name="turbo",
    )
    output_paths = resolve_output_paths(tmp_path / "paper_case.png", "both")
    saved = save_figure_outputs(fig, output_paths, dpi=120)

    assert saved["png"] is not None
    assert saved["pdf"] is not None
    assert Path(saved["png"]).exists()
    assert Path(saved["pdf"]).exists()
    assert meta["surface_has_visible_values"] is True
    assert meta["surface_floor_z"] < 0.0
    assert meta["vehicle_panel_component"] == "risk_field_vehicle_cost"
    assert meta["surface_panel_component"] == "risk_field_lane_cost"
    assert meta["raw_panel_component"] == ""
    assert meta["raw_panel_lane_risk_overlay"] is False
    assert meta["base_image_y_aligned"] is True
    assert meta["figure_layout"] == "ab_top_c_bottom"
    assert meta["surface_vertical_projection"] is True
    assert 0 < meta["surface_projection_line_count"] <= 90
    assert meta["vehicle_requested_resolution"] == 6.0
    assert meta["vehicle_effective_resolution"] == 6.0
    assert meta["colorbar_label"] == "Lane-Line Risk Value"
    assert meta["floor_texture"] is True

    raw_axis = fig.axes[0]
    overlay_axis = fig.axes[1]
    surface_axis = fig.axes[2]
    assert raw_axis.get_title() == "Raw Global BEV"
    assert len(raw_axis.images) == 1
    assert len(overlay_axis.images) == 2

    raw_position = raw_axis.get_position()
    overlay_position = overlay_axis.get_position()
    surface_position = surface_axis.get_position()
    assert raw_position.y0 > surface_position.y1
    assert overlay_position.y0 > surface_position.y1
    assert raw_position.x0 < overlay_position.x0
    assert surface_position.width > max(raw_position.width, overlay_position.width) * 1.25

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_paper_style_prefers_serif_ieee_like_fonts():
    rcparams = get_paper_style_rcparams()

    assert rcparams["font.family"] == "serif"
    assert "Times New Roman" in rcparams["font.serif"]
    assert rcparams["figure.facecolor"] == "white"
    assert rcparams["pdf.fonttype"] == 42
