"""Export a global complex-road-network risk-field paper figure.

Default output panels:
    (a) Raw global semantic BEV
    (b) Vehicle-only 2D risk map overlay
    (c) Lane-line-only 3D risk surface
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from risk_field_bev import (
    build_surface_road_mask,
    resolve_map_bbox,
    resolve_risk_cmap,
    sample_risk_surface_world_bbox,
)
from visualize_metadrive_route import build_env, extract_route_geometry, get_primary_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a global 3-panel risk-field paper figure.")
    parser.add_argument("--output", default="debug/risk_field_paper_snapshot.png", help="Output path base or PNG path.")
    parser.add_argument("--output-format", choices=("png", "pdf", "both"), default="both", help="Export format.")
    parser.add_argument("--dpi", type=int, default=300, help="Saved figure DPI.")
    parser.add_argument("--split", choices=("train", "val"), default="train", help="Seed pool used for auto selection.")
    parser.add_argument("--seed", type=int, default=None, help="Explicit scene seed. Omit to auto-select a complex one.")
    parser.add_argument(
        "--complex-search-root",
        default="debug/metadrive_route_batches",
        help="Root directory containing route batch manifests used for auto complex-scene selection.",
    )
    parser.add_argument(
        "--complex-rank",
        type=int,
        default=1,
        help="Use the N-th most complex unique seed when --seed is omitted.",
    )
    parser.add_argument("--warmup-steps", type=int, default=20, help="Environment warmup steps before capture.")
    parser.add_argument("--traffic-density", type=float, default=0.05, help="Traffic density.")
    parser.add_argument("--accident-prob", type=float, default=0.8, help="Static obstacle accident probability.")
    parser.add_argument("--film-size", type=int, default=2600, help="Global topdown canvas size in pixels.")
    parser.add_argument("--scaling", type=float, default=None, help="Optional global topdown scaling in pixels per meter.")
    parser.add_argument("--map-padding-m", type=float, default=8.0, help="Padding around the road-network bbox.")
    parser.add_argument("--resolution", type=float, default=1.5, help="Requested world-grid resolution in meters.")
    parser.add_argument(
        "--vehicle-resolution",
        type=float,
        default=0.75,
        help="Requested world-grid resolution in meters for the 2D vehicle-risk panel.",
    )
    parser.add_argument(
        "--max-grid-points",
        type=int,
        default=180000,
        help="Maximum number of global risk-grid samples before resolution is auto-increased.",
    )
    parser.add_argument(
        "--road-display-margin-px",
        type=int,
        default=3,
        help="Pixel dilation radius for the road-only display mask.",
    )
    parser.add_argument("--risk-min-visible", type=float, default=1e-4, help="Visibility threshold for the risk surface.")
    parser.add_argument("--surface-elev", type=float, default=30.0, help="3D view elevation.")
    parser.add_argument("--surface-azim", type=float, default=-62.0, help="3D view azimuth.")
    parser.add_argument("--surface-zmax", type=float, default=0.0, help="3D z-axis max; <=0 uses auto scaling.")
    parser.add_argument(
        "--paper-cmap",
        choices=("turbo", "jet", "risk_legacy"),
        default="turbo",
        help="Colormap for the global 2D/3D risk field.",
    )
    parser.add_argument("--route-sample-interval", type=float, default=2.0, help="Route centerline sample interval.")
    parser.add_argument(
        "--route-gap-threshold-m",
        type=float,
        default=6.0,
        help="Route segment gaps above this distance are reported and never bridged in the paper plot.",
    )
    parser.add_argument("--turn-threshold-deg", type=float, default=15.0, help="Turn classification threshold.")
    parser.add_argument("--curvature-threshold-deg", type=float, default=10.0, help="Curve classification threshold.")
    parser.add_argument(
        "--z-aspect-ratio",
        type=float,
        default=0.35,
        help="Visual z-axis height as a fraction of the larger X/Y map span for the 3D panel.",
    )
    parser.add_argument("--title", default="", help="Optional in-figure title.")
    parser.add_argument(
        "--show-title",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to draw the figure title.",
    )
    return parser.parse_args()


def get_paper_style_rcparams() -> Dict[str, Any]:
    """Return IEEE-like serif plotting defaults aligned with paper_board styling."""
    return {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Liberation Serif",
            "Nimbus Roman",
            "Nimbus Roman No9 L",
            "Noto Serif CJK SC",
            "Noto Serif CJK JP",
            "Noto Serif CJK TC",
        ],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "black",
        "axes.titlesize": 11.2,
        "axes.labelsize": 10.3,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "grid.color": "#d1d5db",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.30,
        "text.color": "black",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def resolve_output_paths(output: str | Path, output_format: str) -> Dict[str, Optional[Path]]:
    """Resolve PNG/PDF/JSON paths from a user-provided base output path."""
    output_path = Path(output).expanduser().resolve()
    if output_path.suffix.lower() in {".png", ".pdf", ".json"}:
        base_path = output_path.with_suffix("")
    else:
        base_path = output_path
    return {
        "base": base_path,
        "png": base_path.with_suffix(".png") if output_format in ("png", "both") else None,
        "pdf": base_path.with_suffix(".pdf") if output_format in ("pdf", "both") else None,
        "json": base_path.with_suffix(".json"),
    }


def save_figure_outputs(fig: Any, output_paths: Dict[str, Optional[Path]], dpi: int) -> Dict[str, Optional[str]]:
    """Save the rendered paper figure to the requested formats."""
    saved = {"png": None, "pdf": None}
    for key in ("png", "pdf"):
        path = output_paths.get(key)
        if path is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
        saved[key] = str(path)
    return saved


def _warmup_env(env: Any, warmup_steps: int) -> bool:
    terminated_early = False
    for step_idx in range(max(int(warmup_steps), 0)):
        _, _, terminated, truncated, _ = env.step([0, 0])
        if terminated or truncated:
            terminated_early = True
            print(f"[paper-snapshot] warmup terminated early at step {step_idx + 1}")
            break
    return terminated_early


def _manifest_split(payload: Dict[str, Any], path: Path) -> str:
    split = payload.get("split")
    if split:
        return str(split)
    batch_name = path.parent.parent.name
    return "train" if batch_name.startswith("train_") else "val" if batch_name.startswith("val_") else ""


def collect_complex_scene_records(search_root: str | Path, split: str) -> List[Dict[str, Any]]:
    """Collect unique per-seed complexity records from route batch manifests."""
    root = Path(search_root).expanduser().resolve()
    manifests = sorted(root.glob("*/manifests/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No route batch manifests found under {root}")

    best_by_seed: Dict[int, Dict[str, Any]] = {}
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_split(payload, manifest_path) != split:
            continue
        manifest_mtime = float(manifest_path.stat().st_mtime)
        for record in payload.get("records", []):
            seed = int(record["seed"])
            candidate = {
                "seed": seed,
                "complexity_score": float(record.get("complexity_score", 0.0)),
                "route_length_m": float(record.get("route_length_m", 0.0)),
                "curved_segment_count": int(record.get("curved_segment_count", 0)),
                "map_num_blocks": int(record.get("map_num_blocks", 0)),
                "route_block_id_sequence": str(record.get("route_block_id_sequence", "")),
                "manifest_path": str(manifest_path),
                "manifest_mtime": manifest_mtime,
            }
            previous = best_by_seed.get(seed)
            if previous is None or candidate["complexity_score"] > previous["complexity_score"] or (
                math.isclose(candidate["complexity_score"], previous["complexity_score"])
                and candidate["manifest_mtime"] > previous["manifest_mtime"]
            ):
                best_by_seed[seed] = candidate

    records = list(best_by_seed.values())
    records.sort(key=lambda item: (-item["complexity_score"], -item["route_length_m"], item["seed"]))
    return records


def resolve_selected_scene(
    split: str,
    explicit_seed: Optional[int],
    complex_search_root: str | Path,
    complex_rank: int,
) -> Dict[str, Any]:
    """Return the explicit seed or the N-th most complex auto-selected scene."""
    if explicit_seed is not None:
        try:
            records = collect_complex_scene_records(complex_search_root, split)
        except FileNotFoundError:
            records = []
        matched = next((record for record in records if int(record["seed"]) == int(explicit_seed)), None)
        return {
            "seed": int(explicit_seed),
            "auto_selected": False,
            "complexity_score": None if matched is None else float(matched["complexity_score"]),
            "manifest_path": None if matched is None else str(matched["manifest_path"]),
        }

    rank = max(int(complex_rank), 1)
    records = collect_complex_scene_records(complex_search_root, split)
    if not records:
        raise FileNotFoundError(
            "No complex-scene records found for split={!r} under {}. "
            "Run batch_visualize_metadrive_routes.py first or pass --seed explicitly.".format(split, complex_search_root)
        )
    if rank > len(records):
        raise IndexError(f"complex-rank={rank} exceeds available unique seeds ({len(records)}) for split={split}.")
    selected = records[rank - 1]
    return {
        "seed": int(selected["seed"]),
        "auto_selected": True,
        "complexity_score": float(selected["complexity_score"]),
        "manifest_path": str(selected["manifest_path"]),
    }


def resolve_surface_scaling(
    bounds: Tuple[float, float, float, float],
    film_size: int,
    requested_scaling: Optional[float],
) -> float:
    """Mirror MetaDrive's topdown scaling rule, but over the padded global bbox."""
    x_min, x_max, y_min, y_max = [float(v) for v in bounds]
    max_len = max(x_max - x_min, y_max - y_min, 1e-6)
    max_allowed = film_size / max_len - 0.1
    if requested_scaling is None:
        return float(max_allowed)
    return float(min(float(requested_scaling), max_allowed))


def build_global_map_surface(
    env: Any,
    *,
    film_size: int,
    scaling: float,
    semantic_map: bool = True,
    draw_center_line: bool = False,
) -> tuple[Any, np.ndarray, Tuple[float, float, float, float]]:
    """Render a full-map semantic surface and return its world extent."""
    from metadrive.engine.top_down_renderer import draw_top_down_map_native
    from metadrive.obs.top_down_obs_impl import WorldSurface

    surface = draw_top_down_map_native(
        env.current_map,
        semantic_map=semantic_map,
        draw_center_line=draw_center_line,
        return_surface=True,
        film_size=(int(film_size), int(film_size)),
        scaling=float(scaling),
    )
    base_image = WorldSurface.to_cv2_image(surface).copy()
    width = int(surface.get_width())
    height = int(surface.get_height())
    extent = (
        float(surface.origin[0]),
        float(surface.origin[0] + width / surface.scaling),
        float(surface.origin[1]),
        float(surface.origin[1] + height / surface.scaling),
    )
    return surface, base_image, extent


def _lane_polyline(lane: Any, sample_interval: float) -> np.ndarray:
    """Return a finite Nx2 lane polyline for plotting and continuity checks."""
    interval = max(float(sample_interval), 1e-3)
    polyline = np.asarray(lane.get_polyline(interval), dtype=float)
    if polyline.ndim != 2 or polyline.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    polyline = polyline[:, :2]
    finite_rows = np.isfinite(polyline).all(axis=1)
    return polyline[finite_rows]


def _route_lane_id(lane: Any, start_node: Any, end_node: Any, lane_index: int) -> str:
    lane_id = getattr(lane, "index", None)
    if lane_id is None:
        lane_id = (start_node, end_node, lane_index)
    if isinstance(lane_id, (list, tuple)):
        return "/".join(str(part) for part in lane_id)
    return str(lane_id)


def _first_segment_ego_lane_index(agent: Any, start_node: Any, end_node: Any, lane_count: int) -> Optional[int]:
    lane_index = getattr(agent, "lane_index", None)
    if not isinstance(lane_index, (list, tuple)) or len(lane_index) < 3:
        return None
    if lane_index[0] != start_node or lane_index[1] != end_node:
        return None
    try:
        candidate = int(lane_index[2])
    except (TypeError, ValueError):
        return None
    if 0 <= candidate < int(lane_count):
        return candidate
    return None


def _select_route_lane_index(
    lanes: Sequence[Any],
    *,
    start_node: Any,
    end_node: Any,
    previous_end: Optional[np.ndarray],
    agent: Any,
    segment_index: int,
    sample_interval: float,
) -> int:
    """Choose a route lane that avoids visually bridging unrelated lane centerlines."""
    if not lanes:
        raise ValueError(f"Route segment {start_node!r}->{end_node!r} has no lanes.")

    if segment_index == 0:
        ego_lane_index = _first_segment_ego_lane_index(agent, start_node, end_node, len(lanes))
        if ego_lane_index is not None:
            return int(ego_lane_index)

    if previous_end is not None:
        best_index = 0
        best_distance = math.inf
        for lane_index, lane in enumerate(lanes):
            polyline = _lane_polyline(lane, sample_interval)
            if not polyline.size:
                continue
            distance = float(np.linalg.norm(polyline[0] - previous_end))
            if distance < best_distance:
                best_distance = distance
                best_index = int(lane_index)
        if math.isfinite(best_distance):
            return best_index

    return int(len(lanes) // 2)


def extract_route_plot_segments(
    env: Any,
    *,
    sample_interval: float = 2.0,
    gap_threshold_m: float = 6.0,
) -> Dict[str, Any]:
    """Extract per-road route centerline segments for paper plotting.

    The older summary path joins all route centerlines into one polyline. That is useful for metrics,
    but it can draw artificial bridges at roundabouts and intersections. For paper figures we keep
    each road segment separate and report any discontinuities instead of hiding them.
    """
    agent = get_primary_agent(env)
    road_network = env.current_map.road_network
    checkpoints = list(agent.navigation.checkpoints)
    if len(checkpoints) < 2:
        return {
            "route_segments_world": [],
            "selected_route_lane_ids": [],
            "route_gap_count": 0,
            "max_route_gap_m": 0.0,
        }

    route_segments: List[np.ndarray] = []
    selected_lane_ids: List[str] = []
    previous_end: Optional[np.ndarray] = None
    route_gap_count = 0
    max_route_gap_m = 0.0
    gap_threshold = max(float(gap_threshold_m), 0.0)

    for segment_index, (start_node, end_node) in enumerate(zip(checkpoints[:-1], checkpoints[1:])):
        lanes = road_network.graph[start_node][end_node]
        lane_index = _select_route_lane_index(
            lanes,
            start_node=start_node,
            end_node=end_node,
            previous_end=previous_end,
            agent=agent,
            segment_index=segment_index,
            sample_interval=sample_interval,
        )
        lane = lanes[lane_index]
        polyline = _lane_polyline(lane, sample_interval)
        if not polyline.size:
            continue

        if previous_end is not None:
            gap = float(np.linalg.norm(polyline[0] - previous_end))
            max_route_gap_m = max(max_route_gap_m, gap)
            if gap > gap_threshold:
                route_gap_count += 1

        route_segments.append(polyline)
        selected_lane_ids.append(_route_lane_id(lane, start_node, end_node, lane_index))
        previous_end = polyline[-1]

    return {
        "route_segments_world": route_segments,
        "selected_route_lane_ids": selected_lane_ids,
        "route_gap_count": int(route_gap_count),
        "max_route_gap_m": float(max_route_gap_m),
    }


def resolve_paper_cmap(name: str):
    """Resolve the paper colormap while keeping NaN values transparent."""
    if name == "risk_legacy":
        return resolve_risk_cmap(name)
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name).copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    return cmap


def align_topdown_image_for_world_plot(base_image: np.ndarray) -> np.ndarray:
    """Flip MetaDrive top-down raster rows so Matplotlib world coordinates align with Y."""
    image = np.asarray(base_image)
    if image.ndim < 2:
        return image.copy()
    return np.flipud(image).copy()


def _style_planar_axis(ax: Any, title: str, label: str, extent: Sequence[float]) -> None:
    ax.set_title(title, pad=7.0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(float(extent[0]), float(extent[1]))
    ax.set_ylim(float(extent[2]), float(extent[3]))
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.grid(False)
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="black",
    )


def _style_surface_axis(
    ax: Any,
    title: str,
    label: str,
    *,
    floor_z: float,
    z_max: float,
    bounds: Sequence[float],
    z_label: str = "Risk Field Value",
) -> None:
    ax.set_title(title, pad=10.0)
    ax.set_xlabel("X (m)", labelpad=6.0)
    ax.set_ylabel("Y (m)", labelpad=8.0)
    ax.set_zlabel(z_label, labelpad=6.0)
    ax.set_xlim(float(bounds[0]), float(bounds[1]))
    ax.set_ylim(float(bounds[2]), float(bounds[3]))
    ax.set_zlim(float(floor_z), max(float(z_max), 1e-3))
    ax.tick_params(axis="both", which="major", pad=1.4)
    ax.text2D(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="black",
    )
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            axis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.10))
        except Exception:
            pass


def _normalize_route_segments(route_segments_world: Any) -> List[np.ndarray]:
    if isinstance(route_segments_world, np.ndarray):
        candidates = [route_segments_world]
    else:
        candidates = list(route_segments_world or [])
    segments: List[np.ndarray] = []
    for candidate in candidates:
        segment = np.asarray(candidate, dtype=float)
        if segment.ndim == 2 and segment.shape[0] >= 2 and segment.shape[1] >= 2:
            segments.append(segment[:, :2])
    return segments


def _plot_route_segments(ax: Any, route_segments_world: Any, *, zorder: int = 6) -> None:
    for route_segment in _normalize_route_segments(route_segments_world):
        ax.plot(
            route_segment[:, 0],
            route_segment[:, 1],
            color="#111827",
            linewidth=1.8,
            alpha=0.95,
            zorder=zorder,
        )


def _plot_ego(ax: Any, ego_position: np.ndarray, *, zorder: int = 7) -> None:
    ax.scatter(
        [float(ego_position[0])],
        [float(ego_position[1])],
        s=42,
        c="#111827",
        marker="o",
        edgecolors="white",
        linewidths=0.8,
        zorder=zorder,
    )


def _plot_route_and_ego(ax: Any, route_segments_world: Any, ego_position: np.ndarray) -> None:
    _plot_route_segments(ax, route_segments_world, zorder=6)
    _plot_ego(ax, ego_position, zorder=7)


def _route_surface_hit_ratios(
    route_segments_world: Any,
    surface: Any,
    road_pixel_mask: Optional[np.ndarray],
    base_image: np.ndarray,
    *,
    nonwhite_threshold: int = 245,
) -> Dict[str, Any]:
    """Measure whether plotted route samples land on the native road raster before image flipping."""
    road_mask = None if road_pixel_mask is None else np.asarray(road_pixel_mask, dtype=bool)
    image = np.asarray(base_image)
    image_height = int(image.shape[0]) if image.ndim >= 2 else 0
    image_width = int(image.shape[1]) if image.ndim >= 2 else 0
    mask_height, mask_width = road_mask.shape if road_mask is not None and road_mask.ndim == 2 else (0, 0)

    sample_count = 0
    road_hit_count = 0
    nonwhite_hit_count = 0
    valid_road_samples = 0
    valid_image_samples = 0

    for route_segment in _normalize_route_segments(route_segments_world):
        for point in route_segment:
            pixel_x, pixel_y = surface.pos2pix(float(point[0]), float(point[1]))
            sample_count += 1
            if road_mask is not None and 0 <= pixel_x < mask_width and 0 <= pixel_y < mask_height:
                valid_road_samples += 1
                road_hit_count += int(bool(road_mask[pixel_y, pixel_x]))
            if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
                valid_image_samples += 1
                pixel = image[pixel_y, pixel_x]
                nonwhite_hit_count += int(bool(np.any(np.asarray(pixel[:3]) < int(nonwhite_threshold))))

    return {
        "route_surface_sample_count": int(sample_count),
        "route_road_mask_hit_count": int(road_hit_count),
        "route_native_nonwhite_hit_count": int(nonwhite_hit_count),
        "route_road_mask_hit_ratio": float(road_hit_count / valid_road_samples) if valid_road_samples else 0.0,
        "route_native_nonwhite_hit_ratio": (
            float(nonwhite_hit_count / valid_image_samples) if valid_image_samples else 0.0
        ),
    }


def _plot_floor_texture(
    ax: Any,
    base_image: np.ndarray,
    surface_extent: Sequence[float],
    floor_z: float,
    *,
    max_texture_pixels: int = 220,
) -> None:
    image = np.asarray(base_image)
    if image.ndim != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        return
    stride = max(1, int(math.ceil(max(image.shape[0], image.shape[1]) / float(max_texture_pixels))))
    texture = image[::stride, ::stride, :].astype(float) / 255.0
    if texture.shape[2] == 3:
        alpha = np.ones((*texture.shape[:2], 1), dtype=float)
        texture = np.concatenate([texture, alpha], axis=2)
    elif texture.shape[2] > 4:
        texture = texture[:, :, :4]

    x_values = np.linspace(float(surface_extent[0]), float(surface_extent[1]), texture.shape[1])
    y_values = np.linspace(float(surface_extent[2]), float(surface_extent[3]), texture.shape[0])
    x_floor, y_floor = np.meshgrid(x_values, y_values)
    z_floor = np.full_like(x_floor, float(floor_z), dtype=float)
    ax.plot_surface(
        x_floor,
        y_floor,
        z_floor,
        facecolors=texture,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        zorder=0,
    )


def _plot_vertical_surface_projection(
    ax: Any,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    floor_z: float,
    *,
    max_lines: int = 90,
) -> int:
    """Draw sparse vertical guide lines from high lane-risk peaks down to the floor plane."""
    z_values = np.asarray(z_grid, dtype=float)
    finite_mask = np.isfinite(z_values)
    if not finite_mask.any():
        return 0

    finite_values = z_values[finite_mask]
    threshold = max(float(np.percentile(finite_values, 90.0)), float(np.max(finite_values)) * 0.55)
    candidate_rows, candidate_cols = np.where(finite_mask & (z_values >= threshold))
    if len(candidate_rows) == 0:
        return 0

    candidate_order = np.argsort(z_values[candidate_rows, candidate_cols])[::-1]
    if len(candidate_order) > max_lines:
        candidate_order = candidate_order[:max_lines]

    for candidate_index in candidate_order:
        row = int(candidate_rows[candidate_index])
        col = int(candidate_cols[candidate_index])
        x_value = float(x_grid[row, col])
        y_value = float(y_grid[row, col])
        z_value = float(z_values[row, col])
        ax.plot(
            [x_value, x_value],
            [y_value, y_value],
            [float(floor_z), z_value],
            color="#374151",
            linestyle=(0, (2.2, 2.2)),
            linewidth=0.45,
            alpha=0.40,
            zorder=5,
        )

    return int(len(candidate_order))


def build_paper_snapshot_figure(
    base_image: np.ndarray,
    surface_extent: Sequence[float],
    vehicle_surface_data: Dict[str, Any],
    lane_surface_data: Dict[str, Any],
    route_segments_world: Sequence[np.ndarray],
    ego_position: np.ndarray,
    *,
    title: str = "",
    show_title: bool = False,
    surface_elev: float = 30.0,
    surface_azim: float = -62.0,
    cmap_name: str = "turbo",
    z_aspect_ratio: float = 0.35,
) -> tuple[Any, Dict[str, Any]]:
    """Create the global IEEE-style paper figure."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    with plt.rc_context(get_paper_style_rcparams()):
        fig = plt.figure(figsize=(12.0, 10.5), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.35), hspace=0.10, wspace=0.08)

        if show_title and title:
            fig.suptitle(title, fontsize=13.0, fontweight="bold", y=1.01)

        ax_raw = fig.add_subplot(grid[0, 0])
        ax_overlay = fig.add_subplot(grid[0, 1])
        ax_surface = fig.add_subplot(grid[1, :], projection="3d")

        cmap = resolve_paper_cmap(cmap_name)
        vehicle_norm = Normalize(vmin=0.0, vmax=max(float(vehicle_surface_data["z_max"]), 1e-3))
        lane_z_max = max(float(lane_surface_data["z_max"]), 1e-3)
        lane_norm = Normalize(vmin=0.0, vmax=lane_z_max)
        floor_z = -0.18 * lane_z_max
        world_base_image = align_topdown_image_for_world_plot(base_image)

        ax_raw.imshow(world_base_image, origin="lower", extent=surface_extent, interpolation="nearest")
        _plot_route_and_ego(ax_raw, route_segments_world, ego_position)
        _style_planar_axis(ax_raw, "Raw Global BEV", "(a)", surface_extent)

        ax_overlay.imshow(world_base_image, origin="lower", extent=surface_extent, interpolation="nearest", alpha=0.80)
        overlay_risk = np.ma.masked_invalid(vehicle_surface_data["z_masked"])
        ax_overlay.imshow(
            overlay_risk,
            origin="lower",
            extent=vehicle_surface_data["extent"],
            cmap=cmap,
            norm=vehicle_norm,
            interpolation="bilinear",
            alpha=0.92,
        )
        _plot_ego(ax_overlay, ego_position, zorder=7)
        _style_planar_axis(ax_overlay, "Vehicle Risk Field", "(b)", surface_extent)

        x_grid = np.asarray(lane_surface_data["x_grid"], dtype=float)
        y_grid = np.asarray(lane_surface_data["y_grid"], dtype=float)
        z_masked = np.asarray(np.ma.filled(lane_surface_data["z_masked"], np.nan), dtype=float)
        z_max = lane_z_max
        visible = np.isfinite(z_masked).any()
        _plot_floor_texture(ax_surface, world_base_image, surface_extent, floor_z)
        if visible:
            surface_artist = ax_surface.plot_surface(
                x_grid,
                y_grid,
                z_masked,
                cmap=cmap,
                norm=lane_norm,
                linewidth=0.0,
                antialiased=True,
                shade=True,
                rcount=z_masked.shape[0],
                ccount=z_masked.shape[1],
            )
            contour_values = np.ma.masked_invalid(z_masked)
            ax_surface.contourf(
                x_grid,
                y_grid,
                contour_values,
                zdir="z",
                offset=floor_z + z_max * 0.006,
                levels=18,
                cmap=cmap,
                norm=lane_norm,
                alpha=0.82,
            )
        else:
            surface_artist = ScalarMappable(norm=lane_norm, cmap=cmap)
            surface_artist.set_array([])
        projection_line_count = _plot_vertical_surface_projection(
            ax_surface,
            x_grid,
            y_grid,
            z_masked,
            floor_z,
        )

        for route_segment in _normalize_route_segments(route_segments_world):
            ax_surface.plot(
                route_segment[:, 0],
                route_segment[:, 1],
                np.full(len(route_segment), floor_z + z_max * 0.025, dtype=float),
                color="#111827",
                linewidth=1.6,
                alpha=0.96,
                zorder=6,
            )
        ax_surface.scatter(
            [float(ego_position[0])],
            [float(ego_position[1])],
            [floor_z + z_max * 0.03],
            marker="o",
            s=32,
            c="#111827",
            depthshade=False,
            zorder=7,
        )

        x_span = max(float(surface_extent[1] - surface_extent[0]), 1.0)
        y_span = max(float(surface_extent[3] - surface_extent[2]), 1.0)
        visual_z_span = max(x_span, y_span) * max(float(z_aspect_ratio), 0.02)
        ax_surface.set_box_aspect((x_span, y_span, visual_z_span))
        ax_surface.view_init(elev=float(surface_elev), azim=float(surface_azim))
        ax_surface.grid(True)
        _style_surface_axis(
            ax_surface,
            "3D Lane-Line Risk Field",
            "(c)",
            floor_z=floor_z,
            z_max=z_max,
            bounds=surface_extent,
            z_label="Lane-Line Risk Value",
        )

        colorbar = fig.colorbar(surface_artist, ax=ax_surface, fraction=0.046, pad=0.05, shrink=0.84)
        colorbar.set_label("Lane-Line Risk Value")

        figure_meta = {
            "surface_zmax": z_max,
            "surface_floor_z": float(floor_z),
            "surface_max_risk": float(lane_surface_data["max_risk"]),
            "surface_has_visible_values": bool(visible),
            "vehicle_max_risk": float(vehicle_surface_data["max_risk"]),
            "vehicle_has_visible_values": bool(
                np.isfinite(np.ma.filled(vehicle_surface_data["z_masked"], np.nan)).any()
            ),
            "colorbar_label": "Lane-Line Risk Value",
            "surface_panel_component": str(lane_surface_data.get("component_key", "")),
            "vehicle_panel_component": str(vehicle_surface_data.get("component_key", "")),
            "z_aspect_ratio": float(z_aspect_ratio),
            "floor_texture": True,
            "base_image_y_aligned": True,
            "raw_panel_lane_risk_overlay": False,
            "raw_panel_component": "",
            "figure_layout": "ab_top_c_bottom",
            "surface_vertical_projection": True,
            "surface_projection_line_count": int(projection_line_count),
            "vehicle_requested_resolution": float(vehicle_surface_data.get("requested_resolution", math.nan)),
            "vehicle_effective_resolution": float(vehicle_surface_data.get("effective_resolution", math.nan)),
        }
        return fig, figure_meta


def main() -> None:
    from risk_field import RiskFieldCalculator

    args = parse_args()
    output_paths = resolve_output_paths(args.output, args.output_format)
    selected_scene = resolve_selected_scene(
        args.split,
        args.seed,
        args.complex_search_root,
        args.complex_rank,
    )

    env, _, resolved_seed = build_env(
        args.split,
        selected_scene["seed"],
        map_arg=None,
        traffic_density=args.traffic_density,
        accident_prob=args.accident_prob,
    )
    calculator = RiskFieldCalculator(getattr(env, "config", {}))

    try:
        env.reset()
        terminated_early = _warmup_env(env, args.warmup_steps)
        ego = get_primary_agent(env)
        ego_position = np.asarray(getattr(ego, "position", (0.0, 0.0)), dtype=float)

        route_summary = extract_route_geometry(
            env,
            sample_interval=args.route_sample_interval,
            turn_threshold_deg=args.turn_threshold_deg,
            curvature_threshold_deg=args.curvature_threshold_deg,
        )
        route_plot = extract_route_plot_segments(
            env,
            sample_interval=args.route_sample_interval,
            gap_threshold_m=args.route_gap_threshold_m,
        )
        route_segments_world = route_plot["route_segments_world"]

        padded_bbox = resolve_map_bbox(env, padding_m=args.map_padding_m)
        surface_scaling = resolve_surface_scaling(padded_bbox, args.film_size, args.scaling)
        surface, base_image, surface_extent = build_global_map_surface(
            env,
            film_size=args.film_size,
            scaling=surface_scaling,
            semantic_map=True,
            draw_center_line=False,
        )
        road_pixel_mask = build_surface_road_mask(env, surface, margin_px=args.road_display_margin_px)
        route_surface_diagnostics = _route_surface_hit_ratios(
            route_segments_world,
            surface,
            road_pixel_mask,
            base_image,
        )

        _, risk_info = calculator.calculate(env, ego)
        vehicle_surface_data = sample_risk_surface_world_bbox(
            env,
            ego,
            calculator=calculator,
            component="vehicle",
            bbox=surface_extent,
            resolution=args.vehicle_resolution,
            max_grid_points=args.max_grid_points,
            surface=surface,
            road_pixel_mask=road_pixel_mask,
            risk_min_visible=args.risk_min_visible,
            z_max=None,
        )
        lane_surface_data = sample_risk_surface_world_bbox(
            env,
            ego,
            calculator=calculator,
            component="lane",
            bbox=surface_extent,
            resolution=args.resolution,
            max_grid_points=args.max_grid_points,
            surface=surface,
            road_pixel_mask=road_pixel_mask,
            risk_min_visible=args.risk_min_visible,
            z_max=None if args.surface_zmax <= 0.0 else args.surface_zmax,
        )

        fig, figure_meta = build_paper_snapshot_figure(
            base_image,
            surface_extent,
            vehicle_surface_data,
            lane_surface_data,
            route_segments_world,
            ego_position,
            title=args.title,
            show_title=bool(args.show_title),
            surface_elev=args.surface_elev,
            surface_azim=args.surface_azim,
            cmap_name=args.paper_cmap,
            z_aspect_ratio=args.z_aspect_ratio,
        )
        saved_outputs = save_figure_outputs(fig, output_paths, args.dpi)

        import matplotlib.pyplot as plt

        plt.close(fig)

        sidecar = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "split": args.split,
            "selected_seed": int(resolved_seed),
            "auto_selected": bool(selected_scene["auto_selected"]),
            "complexity_score": selected_scene["complexity_score"],
            "manifest_path": selected_scene["manifest_path"],
            "warmup_steps": args.warmup_steps,
            "warmup_terminated_early": bool(terminated_early),
            "traffic_density": args.traffic_density,
            "accident_prob": args.accident_prob,
            "output_format": args.output_format,
            "dpi": args.dpi,
            "paper_cmap": args.paper_cmap,
            "title": args.title,
            "show_title": bool(args.show_title),
            "output_paths": {
                "png": saved_outputs["png"],
                "pdf": saved_outputs["pdf"],
                "json": str(output_paths["json"]) if output_paths["json"] is not None else None,
            },
            "map_bbox": [float(v) for v in surface_extent],
            "road_network_bbox": [float(v) for v in padded_bbox],
            "figure_layout": figure_meta["figure_layout"],
            "effective_resolution": float(lane_surface_data["effective_resolution"]),
            "requested_resolution": float(lane_surface_data["requested_resolution"]),
            "grid_shape": lane_surface_data["grid_shape"],
            "vehicle_requested_resolution": float(vehicle_surface_data["requested_resolution"]),
            "vehicle_effective_resolution": float(vehicle_surface_data["effective_resolution"]),
            "lane_requested_resolution": float(lane_surface_data["requested_resolution"]),
            "lane_effective_resolution": float(lane_surface_data["effective_resolution"]),
            "vehicle_grid_shape": vehicle_surface_data["grid_shape"],
            "lane_grid_shape": lane_surface_data["grid_shape"],
            "risk_min_visible": float(args.risk_min_visible),
            "road_display_margin_px": int(args.road_display_margin_px),
            "base_image_y_aligned": bool(figure_meta["base_image_y_aligned"]),
            "raw_panel_lane_risk_overlay": bool(figure_meta["raw_panel_lane_risk_overlay"]),
            "raw_panel_component": figure_meta["raw_panel_component"],
            "surface_vertical_projection": bool(figure_meta["surface_vertical_projection"]),
            "surface_projection_line_count": int(figure_meta["surface_projection_line_count"]),
            "route_road_mask_hit_ratio": float(route_surface_diagnostics["route_road_mask_hit_ratio"]),
            "route_native_nonwhite_hit_ratio": float(
                route_surface_diagnostics["route_native_nonwhite_hit_ratio"]
            ),
            "surface": {
                "surface_elev": args.surface_elev,
                "surface_azim": args.surface_azim,
                "surface_zmax": figure_meta["surface_zmax"],
                "surface_floor_z": figure_meta["surface_floor_z"],
                "max_risk": figure_meta["surface_max_risk"],
                "vehicle_max_risk": figure_meta["vehicle_max_risk"],
                "z_aspect_ratio": figure_meta["z_aspect_ratio"],
                "colorbar_label": figure_meta["colorbar_label"],
                "floor_texture": figure_meta["floor_texture"],
                "base_image_y_aligned": figure_meta["base_image_y_aligned"],
                "vertical_projection": figure_meta["surface_vertical_projection"],
                "projection_line_count": figure_meta["surface_projection_line_count"],
            },
            "route": {
                "segment_count": int(route_summary["route"].get("segment_count", 0)),
                "route_length_m": float(route_summary["route"].get("length_m", 0.0)),
                "curved_segment_count": int(route_summary["route"].get("curved_segment_count", 0)),
                "route_block_id_sequence": route_summary["route"].get("route_block_id_sequence_compressed", []),
                "plotted_segment_count": int(len(route_segments_world)),
                "route_gap_threshold_m": float(args.route_gap_threshold_m),
                "route_gap_count": int(route_plot["route_gap_count"]),
                "max_route_gap_m": float(route_plot["max_route_gap_m"]),
                "selected_route_lane_ids": list(route_plot["selected_route_lane_ids"]),
                "surface_sample_count": int(route_surface_diagnostics["route_surface_sample_count"]),
                "road_mask_hit_count": int(route_surface_diagnostics["route_road_mask_hit_count"]),
                "native_nonwhite_hit_count": int(route_surface_diagnostics["route_native_nonwhite_hit_count"]),
            },
            "risk_summary": {
                "total": float(risk_info.get("risk_field_cost", 0.0)),
                "road": float(risk_info.get("risk_field_road_cost", 0.0)),
                "lane": float(risk_info.get("risk_field_lane_cost", 0.0)),
                "boundary": float(risk_info.get("risk_field_boundary_cost", 0.0)),
                "offroad": float(risk_info.get("risk_field_offroad_cost", 0.0)),
                "vehicle": float(risk_info.get("risk_field_vehicle_cost", 0.0)),
                "object": float(risk_info.get("risk_field_object_cost", 0.0)),
                "headway": float(risk_info.get("risk_field_headway_cost", 0.0)),
                "ttc": float(risk_info.get("risk_field_ttc_cost", 0.0)),
            },
            "font_config": {
                "family": "serif",
                "serif_candidates": get_paper_style_rcparams()["font.serif"],
            },
        }

        json_path = output_paths["json"]
        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[paper-snapshot] seed={resolved_seed} auto_selected={int(bool(selected_scene['auto_selected']))}")
        print(f"[paper-snapshot] png={saved_outputs['png']}")
        print(f"[paper-snapshot] pdf={saved_outputs['pdf']}")
        print(f"[paper-snapshot] json={json_path}")
        print(
            "[paper-snapshot] complexity="
            f"{0.0 if selected_scene['complexity_score'] is None else selected_scene['complexity_score']:.3f}"
        )
        print(
            "[paper-snapshot] grid="
            f"{lane_surface_data['grid_shape']} effective_resolution={lane_surface_data['effective_resolution']:.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
