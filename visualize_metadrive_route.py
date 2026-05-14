#!/usr/bin/env python3
"""
导出 MetaDrive 整张地图与 ego 全局 route 的论文分析图。

默认行为：
1. 使用训练集或验证集的默认环境配置
2. 关闭交通流与事故物体，得到更干净的几何底图
3. 导出一张「地图 + route 高亮」PNG
4. 同时导出一份 JSON 摘要，包含地图规模、block 序列、route 长度、转弯统计等

示例：
    python visualize_metadrive_route.py --split train --seed 100
    python visualize_metadrive_route.py --split val --seed 1000 --map TXO
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from safe_metadrive_adapter.local_import import prefer_local_metadrive

prefer_local_metadrive()

Point2D = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 MetaDrive 整张地图与全局 route。")
    parser.add_argument("--split", choices=("train", "val"), default="train", help="使用训练集还是验证集预设。")
    parser.add_argument("--seed", type=int, default=None, help="场景 seed；默认使用对应 split 的 start_seed。")
    parser.add_argument(
        "--map",
        dest="map_arg",
        default=None,
        help="可选覆盖 map 参数，例如 3、5、TXO、CCCCC。",
    )
    parser.add_argument("--output", default=None, help="输出 PNG 路径；默认写入 debug/ 目录。")
    parser.add_argument(
        "--summary-path",
        default=None,
        help="输出 JSON 摘要路径；默认与 PNG 同名。",
    )
    parser.add_argument("--film-size", type=int, default=2400, help="底图画布大小（像素）。")
    parser.add_argument("--scaling", type=float, default=None, help="每米对应像素；默认自动缩放以覆盖整张地图。")
    parser.add_argument(
        "--semantic-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用语义色 top-down 底图。",
    )
    parser.add_argument(
        "--draw-center-line",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否绘制所有 lane 中心线。",
    )
    parser.add_argument(
        "--annotate-segments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否在图上标注 route segment 编号。",
    )
    parser.add_argument("--route-sample-interval", type=float, default=2.0, help="route 采样间隔（米）。")
    parser.add_argument("--turn-threshold-deg", type=float, default=15.0, help="路口转向分类阈值（度）。")
    parser.add_argument(
        "--curvature-threshold-deg",
        type=float,
        default=10.0,
        help="单 segment 判定为曲线路段的阈值（度）。",
    )
    parser.add_argument("--traffic-density", type=float, default=0.0, help="可视化时使用的交通密度。默认 0。")
    parser.add_argument("--accident-prob", type=float, default=0.0, help="可视化时使用的事故概率。默认 0。")
    parser.add_argument("--dpi", type=int, default=200, help="保存 PNG 的 DPI。")
    return parser.parse_args()


def parse_map_arg(raw: Optional[str]) -> Optional[Any]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def wrap_to_180(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0 and angle_deg > 0:
        return 180.0
    return wrapped


def compress_consecutive(values: Iterable[Any]) -> List[Any]:
    compressed: List[Any] = []
    for value in values:
        if not compressed or value != compressed[-1]:
            compressed.append(value)
    return compressed


def polyline_length(points: Sequence[Point2D]) -> float:
    if len(points) < 2:
        return 0.0
    array = np.asarray(points, dtype=float)
    diffs = array[1:] - array[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())


def _first_nonzero_direction(points: Sequence[Point2D], reverse: bool = False) -> Optional[np.ndarray]:
    if len(points) < 2:
        return None
    array = np.asarray(points, dtype=float)
    if reverse:
        anchor = array[-1]
        for idx in range(len(array) - 2, -1, -1):
            direction = anchor - array[idx]
            if float(np.linalg.norm(direction)) > 1e-8:
                return direction
        return None

    anchor = array[0]
    for idx in range(1, len(array)):
        direction = array[idx] - anchor
        if float(np.linalg.norm(direction)) > 1e-8:
            return direction
    return None


def polyline_endpoint_heading_deg(points: Sequence[Point2D], at_end: bool = False) -> float:
    direction = _first_nonzero_direction(points, reverse=at_end)
    if direction is None:
        return 0.0
    return math.degrees(math.atan2(float(direction[1]), float(direction[0])))


def classify_heading_change(angle_deg: float, threshold_deg: float = 15.0) -> str:
    if angle_deg > threshold_deg:
        return "left"
    if angle_deg < -threshold_deg:
        return "right"
    return "straight"


def _safe_float_list(point: Sequence[float]) -> List[float]:
    return [float(point[0]), float(point[1])]


def _safe_lane_index(index: Any) -> Any:
    if isinstance(index, tuple):
        return list(index)
    if isinstance(index, list):
        return index
    return index


def _default_output_paths(split: str, seed: int) -> Tuple[Path, Path]:
    stem = f"metadrive_route_{split}_seed{seed}"
    output_dir = Path("debug")
    return output_dir / f"{stem}.png", output_dir / f"{stem}.json"


def _load_env_builders():
    from env import TRAINING_CONFIG, VALIDATION_CONFIG, get_training_env, get_validation_env

    return {
        "TRAINING_CONFIG": TRAINING_CONFIG,
        "VALIDATION_CONFIG": VALIDATION_CONFIG,
        "get_training_env": get_training_env,
        "get_validation_env": get_validation_env,
    }


def get_default_start_seed(split: str) -> int:
    api = _load_env_builders()
    base_config = api["TRAINING_CONFIG"] if split == "train" else api["VALIDATION_CONFIG"]
    return int(base_config["start_seed"])


def build_env(split: str, seed: Optional[int], map_arg: Optional[Any], traffic_density: float, accident_prob: float):
    api = _load_env_builders()
    base_config = copy.deepcopy(api["TRAINING_CONFIG"] if split == "train" else api["VALIDATION_CONFIG"])
    builder = api["get_training_env"] if split == "train" else api["get_validation_env"]
    resolved_seed = int(base_config["start_seed"]) if seed is None else int(seed)
    env = builder(
        {
            "use_render": False,
            "manual_control": False,
            "num_scenarios": 1,
            "start_seed": resolved_seed,
            "traffic_density": float(traffic_density),
            "accident_prob": float(accident_prob),
            "log_level": 50,
        }
    )
    if map_arg is not None:
        env.close()
        env = builder(
            {
                "use_render": False,
                "manual_control": False,
                "num_scenarios": 1,
                "start_seed": resolved_seed,
                "traffic_density": float(traffic_density),
                "accident_prob": float(accident_prob),
                "log_level": 50,
                "map": map_arg,
            }
        )
    return env, base_config, resolved_seed


def get_primary_agent(env) -> Any:
    agent = getattr(env, "agent", None)
    if agent is not None:
        return agent
    agents = getattr(env, "agents", {})
    if agents:
        return next(iter(agents.values()))
    raise RuntimeError("No active ego agent found in env.")


def compute_complexity_metrics(map_summary: Dict[str, Any], route_summary: Dict[str, Any]) -> Dict[str, Any]:
    junction_turn_count = route_summary.get("junction_turn_count", {})
    left_turn_count = int(junction_turn_count.get("left", 0))
    right_turn_count = int(junction_turn_count.get("right", 0))
    straight_turn_count = int(junction_turn_count.get("straight", 0))
    non_straight_turn_count = left_turn_count + right_turn_count
    compressed_block_count = int(len(route_summary.get("route_block_type_sequence_compressed", [])))
    route_block_transition_count = max(compressed_block_count - 1, 0)
    curved_segment_count = int(route_summary.get("curved_segment_count", 0))
    cumulative_abs_segment_curvature_deg = float(route_summary.get("cumulative_abs_segment_curvature_deg", 0.0))
    mean_abs_junction_turn_change_deg = float(route_summary.get("mean_abs_junction_turn_change_deg", 0.0))
    segment_count = int(route_summary.get("segment_count", 0))
    map_num_blocks = int(map_summary.get("num_blocks", 0))

    composite_complexity_score = (
        curved_segment_count * 2.0
        + non_straight_turn_count * 3.0
        + route_block_transition_count * 1.5
        + cumulative_abs_segment_curvature_deg / 90.0
        + mean_abs_junction_turn_change_deg / 15.0
        + max(map_num_blocks - 1, 0) * 0.5
        + max(segment_count - 1, 0) * 0.1
    )

    return {
        "left_turn_count": left_turn_count,
        "right_turn_count": right_turn_count,
        "straight_turn_count": straight_turn_count,
        "non_straight_turn_count": non_straight_turn_count,
        "compressed_route_block_count": compressed_block_count,
        "route_block_transition_count": route_block_transition_count,
        "composite_complexity_score": float(composite_complexity_score),
    }


def extract_route_geometry(
    env,
    sample_interval: float,
    turn_threshold_deg: float,
    curvature_threshold_deg: float,
):
    from metadrive.component.road_network.road import Road

    agent = get_primary_agent(env)
    current_map = env.current_map
    road_network = current_map.road_network
    checkpoints = list(agent.navigation.checkpoints)
    if len(checkpoints) < 2:
        raise RuntimeError("Route checkpoints are empty or incomplete.")

    block_name_by_id: Dict[str, str] = {}
    block_sequence_ids: List[str] = []
    block_sequence_names: List[str] = []
    for block in current_map.blocks:
        block_id = getattr(block, "ID", "?")
        block_name = type(block).__name__
        block_name_by_id[block_id] = block_name
        block_sequence_ids.append(block_id)
        block_sequence_names.append(block_name)

    route_records: List[Dict[str, Any]] = []
    route_centerline_world: List[Point2D] = []

    for seg_idx, (start_node, end_node) in enumerate(zip(checkpoints[:-1], checkpoints[1:])):
        lanes = road_network.graph[start_node][end_node]
        ref_lane = lanes[len(lanes) // 2]
        centerline_world = [tuple(map(float, point)) for point in ref_lane.get_polyline(sample_interval)]
        if centerline_world:
            if route_centerline_world and np.allclose(route_centerline_world[-1], centerline_world[0], atol=1e-6):
                route_centerline_world.extend(centerline_world[1:])
            else:
                route_centerline_world.extend(centerline_world)

        start_heading_deg = polyline_endpoint_heading_deg(centerline_world, at_end=False)
        end_heading_deg = polyline_endpoint_heading_deg(centerline_world, at_end=True)
        segment_curvature_deg = wrap_to_180(end_heading_deg - start_heading_deg)
        road = Road(start_node, end_node)
        block_id = road.block_ID()

        route_records.append(
            {
                "segment_index": int(seg_idx),
                "from": str(start_node),
                "to": str(end_node),
                "lane_count": int(len(lanes)),
                "length_m": float(lanes[0].length),
                "centerline_length_m": polyline_length(centerline_world),
                "block_id": str(block_id),
                "block_type": block_name_by_id.get(block_id, "UnknownBlock"),
                "start_heading_deg": float(start_heading_deg),
                "end_heading_deg": float(end_heading_deg),
                "segment_curvature_deg": float(segment_curvature_deg),
                "is_curved_segment": bool(abs(segment_curvature_deg) >= curvature_threshold_deg),
            }
        )

    junction_changes_deg: List[float] = []
    junction_turn_types: List[str] = []
    for current_record, next_record in zip(route_records[:-1], route_records[1:]):
        junction_change = wrap_to_180(next_record["start_heading_deg"] - current_record["end_heading_deg"])
        turn_type = classify_heading_change(junction_change, threshold_deg=turn_threshold_deg)
        current_record["junction_turn_to_next_deg"] = float(junction_change)
        current_record["junction_turn_to_next_type"] = turn_type
        junction_changes_deg.append(float(junction_change))
        junction_turn_types.append(turn_type)

    if route_records:
        route_records[-1]["junction_turn_to_next_deg"] = None
        route_records[-1]["junction_turn_to_next_type"] = None

    summary = {
        "map": {
            "class_name": type(current_map).__name__,
            "map_parameter": env.config.get("map"),
            "num_blocks": int(current_map.num_blocks),
            "num_lanes": int(len(road_network.get_all_lanes())),
            "bounding_box_m": [float(v) for v in road_network.get_bounding_box()],
            "map_block_id_sequence": block_sequence_ids,
            "map_block_type_sequence": block_sequence_names,
            "map_block_type_histogram": dict(Counter(block_sequence_names)),
        },
        "route": {
            "spawn_lane_index": _safe_lane_index(getattr(agent, "lane_index", None)),
            "configured_spawn_lane_index": _safe_lane_index(agent.config.get("spawn_lane_index")),
            "destination": _safe_lane_index(agent.config.get("destination")),
            "checkpoints": [str(c) for c in checkpoints],
            "segment_count": int(len(route_records)),
            "length_m": float(sum(record["length_m"] for record in route_records)),
            "centerline_length_m": polyline_length(route_centerline_world),
            "route_lane_count_total": int(sum(record["lane_count"] for record in route_records)),
            "route_block_id_sequence": [record["block_id"] for record in route_records],
            "route_block_type_sequence": [record["block_type"] for record in route_records],
            "route_block_id_sequence_compressed": compress_consecutive(record["block_id"] for record in route_records),
            "route_block_type_sequence_compressed": compress_consecutive(
                record["block_type"] for record in route_records
            ),
            "curved_segment_count": int(sum(record["is_curved_segment"] for record in route_records)),
            "cumulative_abs_segment_curvature_deg": float(
                sum(abs(record["segment_curvature_deg"]) for record in route_records)
            ),
            "max_abs_segment_curvature_deg": float(
                max((abs(record["segment_curvature_deg"]) for record in route_records), default=0.0)
            ),
            "junction_turn_count": dict(Counter(junction_turn_types)),
            "junction_turn_changes_deg": [float(v) for v in junction_changes_deg],
            "mean_abs_junction_turn_change_deg": float(
                np.mean(np.abs(junction_changes_deg)) if junction_changes_deg else 0.0
            ),
            "segment_records": route_records,
            "sampled_centerline_world": [_safe_float_list(point) for point in route_centerline_world],
        },
    }
    summary["route"]["complexity_metrics"] = compute_complexity_metrics(summary["map"], summary["route"])

    return summary


def build_visual_layers(
    env,
    film_size: int,
    scaling: Optional[float],
    semantic_map: bool,
    draw_center_line: bool,
    sample_interval: float,
) -> Dict[str, Any]:
    from metadrive.component.road_network.road import Road
    from metadrive.engine.top_down_renderer import draw_top_down_map_native
    from metadrive.obs.top_down_obs_impl import WorldSurface

    agent = get_primary_agent(env)
    current_map = env.current_map
    road_network = current_map.road_network
    checkpoints = list(agent.navigation.checkpoints)

    surface = draw_top_down_map_native(
        current_map,
        semantic_map=semantic_map,
        draw_center_line=draw_center_line,
        return_surface=True,
        film_size=(film_size, film_size),
        scaling=scaling,
    )
    base_image = WorldSurface.to_cv2_image(surface).copy()

    route_centerline_world: List[Point2D] = []
    route_centerline_pixels: List[Tuple[int, int]] = []
    route_lane_polygons_pixels: List[List[Tuple[int, int]]] = []
    segment_anchor_pixels: List[Tuple[int, int]] = []

    for start_node, end_node in zip(checkpoints[:-1], checkpoints[1:]):
        lanes = road_network.graph[start_node][end_node]
        ref_lane = lanes[len(lanes) // 2]
        centerline_world = [tuple(map(float, point)) for point in ref_lane.get_polyline(sample_interval)]
        if centerline_world:
            if route_centerline_world and np.allclose(route_centerline_world[-1], centerline_world[0], atol=1e-6):
                route_centerline_world.extend(centerline_world[1:])
                centerline_for_pixels = centerline_world[1:]
            else:
                route_centerline_world.extend(centerline_world)
                centerline_for_pixels = centerline_world
            segment_pixels = [surface.pos2pix(float(point[0]), float(point[1])) for point in centerline_for_pixels]
            route_centerline_pixels.extend(segment_pixels)
            if segment_pixels:
                segment_anchor_pixels.append(segment_pixels[len(segment_pixels) // 2])

        for lane in lanes:
            polygon_pixels = [surface.pos2pix(float(point[0]), float(point[1])) for point in lane.polygon]
            route_lane_polygons_pixels.append(polygon_pixels)

    if route_centerline_world:
        start_world = route_centerline_world[0]
        end_world = route_centerline_world[-1]
    else:
        start_world = tuple(map(float, agent.position))
        end_world = start_world

    return {
        "base_image": base_image,
        "route_centerline_pixels": route_centerline_pixels,
        "route_lane_polygons_pixels": route_lane_polygons_pixels,
        "segment_anchor_pixels": segment_anchor_pixels,
        "start_pixel": surface.pos2pix(float(start_world[0]), float(start_world[1])),
        "end_pixel": surface.pos2pix(float(end_world[0]), float(end_world[1])),
    }


def save_route_figure(
    output_path: Path,
    dpi: int,
    base_image: np.ndarray,
    route_centerline_pixels: Sequence[Tuple[int, int]],
    route_lane_polygons_pixels: Sequence[Sequence[Tuple[int, int]]],
    segment_anchor_pixels: Sequence[Tuple[int, int]],
    start_pixel: Tuple[int, int],
    end_pixel: Tuple[int, int],
    annotate_segments: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    height, width = base_image.shape[:2]
    figure_width = width / dpi
    figure_height = height / dpi

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=dpi)
    ax.imshow(base_image)

    for polygon in route_lane_polygons_pixels:
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=(1.0, 0.55, 0.0, 0.16),
                edgecolor=(1.0, 0.45, 0.0, 0.28),
                linewidth=0.6,
            )
        )

    if route_centerline_pixels:
        xs = [point[0] for point in route_centerline_pixels]
        ys = [point[1] for point in route_centerline_pixels]
        ax.plot(xs, ys, color="#d62728", linewidth=3.2, solid_capstyle="round", zorder=5)

    ax.scatter(
        [start_pixel[0]],
        [start_pixel[1]],
        s=80,
        c="#2ca02c",
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        zorder=6,
    )
    ax.scatter(
        [end_pixel[0]],
        [end_pixel[1]],
        s=160,
        c="#1f77b4",
        marker="*",
        edgecolors="white",
        linewidths=1.0,
        zorder=6,
    )

    if annotate_segments:
        for idx, anchor in enumerate(segment_anchor_pixels):
            ax.text(
                anchor[0],
                anchor[1],
                str(idx),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                bbox={"boxstyle": "circle,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                zorder=7,
            )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def print_summary(split: str, seed: int, output_path: Path, summary_path: Path, summary: Dict[str, Any]) -> None:
    route = summary["route"]
    map_info = summary["map"]
    complexity = route.get("complexity_metrics", {})
    print(f"[route-vis] saved image: {output_path}")
    print(f"[route-vis] saved summary: {summary_path}")
    print(f"[route-vis] split={split} seed={seed}")
    print(
        "[route-vis] map: "
        f"{map_info['class_name']} blocks={map_info['num_blocks']} lanes={map_info['num_lanes']} "
        f"block_seq={''.join(map_info['map_block_id_sequence'])}"
    )
    print(
        "[route-vis] route: "
        f"segments={route['segment_count']} length={route['length_m']:.2f}m "
        f"curved_segments={route['curved_segment_count']} "
        f"block_seq={''.join(route['route_block_id_sequence_compressed'])} "
        f"complexity={complexity.get('composite_complexity_score', 0.0):.2f}"
    )
    print(
        "[route-vis] junction turns: "
        f"{json.dumps(route['junction_turn_count'], ensure_ascii=False, sort_keys=True)}"
    )


def generate_route_visualization(
    *,
    split: str,
    seed: Optional[int] = None,
    map_arg: Optional[Any] = None,
    output_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    film_size: int = 2400,
    scaling: Optional[float] = None,
    semantic_map: bool = True,
    draw_center_line: bool = False,
    annotate_segments: bool = False,
    route_sample_interval: float = 2.0,
    turn_threshold_deg: float = 15.0,
    curvature_threshold_deg: float = 10.0,
    traffic_density: float = 0.0,
    accident_prob: float = 0.0,
    dpi: int = 200,
    print_summary_log: bool = True,
) -> Dict[str, Any]:
    env = None
    try:
        env, _, resolved_seed = build_env(
            split=split,
            seed=seed,
            map_arg=map_arg,
            traffic_density=traffic_density,
            accident_prob=accident_prob,
        )
        env.reset()

        summary = extract_route_geometry(
            env,
            sample_interval=route_sample_interval,
            turn_threshold_deg=turn_threshold_deg,
            curvature_threshold_deg=curvature_threshold_deg,
        )
        layers = build_visual_layers(
            env,
            film_size=film_size,
            scaling=scaling,
            semantic_map=semantic_map,
            draw_center_line=draw_center_line,
            sample_interval=route_sample_interval,
        )

        default_output_path, default_summary_path = _default_output_paths(split, resolved_seed)
        resolved_output_path = Path(output_path).expanduser().resolve() if output_path else default_output_path.resolve()
        resolved_summary_path = (
            Path(summary_path).expanduser().resolve()
            if summary_path
            else default_summary_path.resolve()
        )

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "split": split,
            "seed": int(resolved_seed),
            "traffic_density": float(traffic_density),
            "accident_prob": float(accident_prob),
            "visualization": {
                "film_size": int(film_size),
                "scaling": None if scaling is None else float(scaling),
                "semantic_map": bool(semantic_map),
                "draw_center_line": bool(draw_center_line),
                "annotate_segments": bool(annotate_segments),
                "route_sample_interval_m": float(route_sample_interval),
                "turn_threshold_deg": float(turn_threshold_deg),
                "curvature_threshold_deg": float(curvature_threshold_deg),
                "dpi": int(dpi),
            },
            "files": {
                "image_path": str(resolved_output_path),
                "summary_path": str(resolved_summary_path),
            },
            "summary": summary,
        }

        save_route_figure(
            output_path=resolved_output_path,
            dpi=int(dpi),
            base_image=layers["base_image"],
            route_centerline_pixels=layers["route_centerline_pixels"],
            route_lane_polygons_pixels=layers["route_lane_polygons_pixels"],
            segment_anchor_pixels=layers["segment_anchor_pixels"],
            start_pixel=layers["start_pixel"],
            end_pixel=layers["end_pixel"],
            annotate_segments=bool(annotate_segments),
        )

        resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_summary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        if print_summary_log:
            print_summary(split, resolved_seed, resolved_output_path, resolved_summary_path, summary)

        return {
            "output_path": resolved_output_path,
            "summary_path": resolved_summary_path,
            "payload": payload,
        }
    finally:
        if env is not None:
            env.close()


def main() -> None:
    args = parse_args()
    map_arg = parse_map_arg(args.map_arg)

    try:
        generate_route_visualization(
            split=args.split,
            seed=args.seed,
            map_arg=map_arg,
            output_path=Path(args.output).expanduser().resolve() if args.output else None,
            summary_path=Path(args.summary_path).expanduser().resolve() if args.summary_path else None,
            film_size=args.film_size,
            scaling=args.scaling,
            semantic_map=bool(args.semantic_map),
            draw_center_line=bool(args.draw_center_line),
            annotate_segments=bool(args.annotate_segments),
            route_sample_interval=args.route_sample_interval,
            turn_threshold_deg=args.turn_threshold_deg,
            curvature_threshold_deg=args.curvature_threshold_deg,
            traffic_density=float(args.traffic_density),
            accident_prob=float(args.accident_prob),
            dpi=int(args.dpi),
            print_summary_log=True,
        )
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None) or str(exc)
        raise SystemExit(
            "运行失败：缺少依赖 {}。请先安装 MetaDrive 运行所需依赖（常见为 shapely、panda3d、matplotlib）。".format(
                missing_name
            )
        ) from exc


if __name__ == "__main__":
    main()
