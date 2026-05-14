"""快速导出单张风险场快照图。

默认输出左右对照图：
- 左侧：风险场（道路 / 周车 / 静态障碍物）
- 右侧：原始 topdown BEV

相比完整 eval / GIF 流程，这个脚本只 reset 一次、warmup 一小段、抓一帧，
适合高频调参时快速看图。
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from safe_metadrive_adapter.local_import import prefer_local_metadrive

prefer_local_metadrive()

from debug_risk_field_topdown_overlay_gif import (
    build_risk_overlay_args,
    render_risk_only_frame,
    render_risk_overlay_frame,
    render_risk_side_by_side_frame,
)
from env import get_validation_env
from risk_field import RiskFieldCalculator

try:
    from metadrive.policy.idm_policy import IDMPolicy
except ImportError:
    from metadrive.metadrive.policy.idm_policy import IDMPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a single fast risk-field snapshot PNG.")
    parser.add_argument("--output", default="debug/risk_snapshot.png", help="输出 PNG 路径")
    parser.add_argument("--seed", type=int, default=1000, help="场景 seed")
    parser.add_argument(
        "--auto-select-object-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否从 seed 开始自动寻找带静态障碍物/锥桶的场景；快照默认开启，方便调试障碍物风险场",
    )
    parser.add_argument("--object-search-seeds", type=int, default=60, help="自动寻找障碍物场景时最多尝试的 seed 数")
    parser.add_argument("--min-surrounding-objects", type=int, default=1, help="自动选场景时要求的最少近邻静态障碍物数")
    parser.add_argument(
        "--object-search-distance",
        type=float,
        default=80.0,
        help="自动选场景和风险/轮廓叠加时静态障碍物到 ego 的最大可视距离",
    )
    parser.add_argument(
        "--prefer-cone-scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="自动选场景时是否优先选择含 TrafficCone 的场景；默认开启",
    )
    parser.add_argument("--warmup-steps", type=int, default=20, help="抓图前步进次数")
    parser.add_argument("--traffic-density", type=float, default=0.05, help="交通密度")
    parser.add_argument("--accident-prob", type=float, default=0.8, help="静态障碍物事故概率")
    parser.add_argument("--width", type=int, default=800, help="图像宽度")
    parser.add_argument("--height", type=int, default=800, help="图像高度")
    parser.add_argument("--film-size", type=int, default=2000, help="topdown film size")
    parser.add_argument("--scaling", type=float, default=8.0, help="topdown scaling")
    parser.add_argument(
        "--view",
        choices=("side_by_side", "overlay", "risk_only"),
        default="side_by_side",
        help="输出视图类型",
    )
    parser.add_argument(
        "--road-risk-component",
        choices=("lane", "broken", "solid", "boundary", "oncoming", "total"),
        default="total",
        help="道路风险组件",
    )
    parser.add_argument(
        "--vehicle-risk-component",
        choices=("static", "dynamic", "total"),
        default="total",
        help="车辆风险组件",
    )
    parser.add_argument(
        "--road-risk-use-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="道路风险可视化是否乘入环境权重；快照默认开启，以贴近真实环境输出",
    )
    parser.add_argument(
        "--vehicle-risk-use-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="周车/障碍物风险可视化是否乘入环境权重；快照默认开启，以贴近真实环境输出",
    )
    parser.add_argument("--road-risk-min-visible", type=float, default=0.0, help="道路风险最小显示阈值")
    parser.add_argument("--vehicle-risk-min-visible", type=float, default=0.0, help="参与体风险最小显示阈值")
    parser.add_argument(
        "--show-risk-panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在图片中显示当前 ego 风险分解面板；快照默认开启",
    )
    parser.add_argument(
        "--show-config-panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在图片中显示当前风险场配置面板；快照默认开启",
    )
    parser.add_argument(
        "--show-surrounding-vehicle-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在风险场视图中叠加周车原始轮廓；快照默认开启",
    )
    parser.add_argument(
        "--show-static-object-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在风险场视图中叠加静态障碍物/锥桶原始轮廓;快照默认开启",
    )
    parser.add_argument(
        "--move-ego-to",
        type=str,
        default=None,
        help="在抓图前移动 ego 到指定位置,格式为 'dx,dy' (相对当前位置的偏移,单位米) 或 'x,y' (绝对世界坐标)",
    )
    parser.add_argument(
        "--move-ego-mode",
        choices=("relative", "absolute"),
        default="relative",
        help="移动模式:relative=相对偏移,absolute=绝对坐标;默认 relative",
    )
    return parser.parse_args()


def _move_ego_to_position(env, calculator: RiskFieldCalculator, ego, target_spec: str, mode: str):
    """移动 ego 到指定位置
    
    Args:
        target_spec: 位置描述,如 "10,5" 或 "-5,2"
        mode: "relative"(相对偏移) 或 "absolute"(绝对坐标)
    """
    try:
        parts = target_spec.split(",")
        if len(parts) != 2:
            print(f"[snapshot] ⚠️ 无效的位置格式: {target_spec},期望 'x,y'")
            return
        
        target_x = float(parts[0].strip())
        target_y = float(parts[1].strip())
        
        if mode == "relative":
            current_pos = getattr(ego, "position", (0.0, 0.0))
            target_x += current_pos[0]
            target_y += current_pos[1]
            print(f"[snapshot] 🚗 移动 ego: ({current_pos[0]:.1f},{current_pos[1]:.1f}) → ({target_x:.1f},{target_y:.1f})")
        else:
            print(f"[snapshot] 🚗 移动 ego 到绝对坐标: ({target_x:.1f},{target_y:.1f})")
        
        # 尝试多种设置位置的方法
        if hasattr(ego, "set_position"):
            ego.set_position((target_x, target_y))
        elif hasattr(ego, "body") and hasattr(ego.body, "setPos"):
            ego.body.setPos(target_x, target_y, 0)
        elif hasattr(ego, "physics_world"):
            # Panda3D 物理引擎直接设置
            from panda3d.core import LVector3
            ego.setPos(LVector3(target_x, target_y, 0))
        else:
            print(f"[snapshot] ⚠️ 无法找到设置位置的方法,跳过移动")
            return
        
        # 重置速度避免物理异常
        if hasattr(ego, "set_velocity"):
            ego.set_velocity((0.0, 0.0, 0.0))
        
        # 步进一帧让环境同步状态
        env.step([0, 0])
        print(f"[snapshot] ✅ ego 位置已更新")
        
    except Exception as e:
        print(f"[snapshot] ⚠️ 移动 ego 失败: {e}")


def _overlay_config_from_args(args: argparse.Namespace) -> dict:
    return {
        "risk_overlay_width": args.width,
        "risk_overlay_height": args.height,
        "risk_overlay_film_size": args.film_size,
        "risk_overlay_scaling": args.scaling,
        "risk_overlay_no_road": False,
        "risk_overlay_no_vehicle": False,
        "risk_overlay_road_component": args.road_risk_component,
        "risk_overlay_vehicle_component": args.vehicle_risk_component,
        "risk_overlay_road_use_weights": args.road_risk_use_weights,
        "risk_overlay_vehicle_use_weights": args.vehicle_risk_use_weights,
        "risk_overlay_road_min_visible": args.road_risk_min_visible,
        "risk_overlay_vehicle_min_visible": args.vehicle_risk_min_visible,
        "risk_overlay_risk_distance": args.object_search_distance,
        "risk_overlay_side_by_side": args.view == "side_by_side",
        "risk_overlay_show_surrounding_vehicle_shapes": args.show_surrounding_vehicle_shapes,
        "risk_overlay_show_static_object_shapes": args.show_static_object_shapes,
        # 快照图默认贴近“真实环境输出”观测：保留色条，并显示风险/配置面板。
        "risk_overlay_no_panel": not args.show_risk_panel,
        "risk_overlay_no_config_panel": not args.show_config_panel,
        "risk_overlay_no_colorbar": False,
    }


def _make_snapshot_frame(env, calculator: RiskFieldCalculator, overlay_args: argparse.Namespace, view: str):
    if view == "risk_only":
        return render_risk_only_frame(env, False, calculator, overlay_args)
    if view == "overlay":
        return render_risk_overlay_frame(env, False, calculator, overlay_args)
    return render_risk_side_by_side_frame(env, False, calculator, overlay_args)


def _env_config_from_args(args: argparse.Namespace, num_scenarios: int) -> dict:
    return {
        "num_scenarios": max(int(num_scenarios), 1),
        "start_seed": args.seed,
        "use_render": False,
        "manual_control": False,
        "agent_policy": IDMPolicy,
        "traffic_density": args.traffic_density,
        "accident_prob": args.accident_prob,
    }


def _visible_static_object_summary(env, calculator: RiskFieldCalculator, ego, max_distance: float) -> dict:
    ego_pos = calculator._xy(getattr(ego, "position", (0.0, 0.0)))
    max_distance = max(float(max_distance), 0.0)
    count = 0
    cone_count = 0
    nearest_distance = math.inf
    object_types = set()
    for obj in calculator._iter_static_objects(env):
        obj_pos = calculator._xy(getattr(obj, "position", (0.0, 0.0)))
        distance = float(np.linalg.norm(obj_pos - ego_pos))
        if distance > max_distance:
            continue
        count += 1
        nearest_distance = min(nearest_distance, distance)
        type_name = type(obj).__name__
        object_types.add(type_name)
        if "Cone" in type_name:
            cone_count += 1
    return {
        "visible_static_object_count": int(count),
        "visible_cone_count": int(cone_count),
        "nearest_visible_static_object_distance": nearest_distance if math.isfinite(nearest_distance) else math.nan,
        "visible_static_object_types": sorted(object_types),
    }


def _warmup_and_measure(
    env,
    calculator: RiskFieldCalculator,
    seed: int,
    warmup_steps: int,
    object_search_distance: float,
    move_ego_to: str = None,
    move_ego_mode: str = "relative",
) -> dict:
    env.reset(seed=seed)

    terminated_early = False
    for step_idx in range(max(int(warmup_steps), 0)):
        _, _, terminated, truncated, _ = env.step([0, 0])
        if terminated or truncated:
            terminated_early = True
            print(f"[snapshot] warmup terminated early at step {step_idx + 1} (seed={seed})")
            break

    agents = getattr(env, "agents", {})
    if not agents:
        raise RuntimeError("snapshot env has no active agents after reset/warmup")
    ego = next(iter(agents.values()))
    
    # 在抓图前移动 ego
    if move_ego_to:
        _move_ego_to_position(env, calculator, ego, move_ego_to, move_ego_mode)
    
    _, risk_info = calculator.calculate(env, ego)
    state = {
        "seed": int(seed),
        "ego": ego,
        "risk_info": risk_info,
        "terminated_early": terminated_early,
    }
    state.update(_visible_static_object_summary(env, calculator, ego, object_search_distance))
    return state


def _select_snapshot_state(args: argparse.Namespace):
    min_objects = max(int(args.min_surrounding_objects), 1)
    search_count = max(int(args.object_search_seeds), 1)
    search_enabled = bool(args.auto_select_object_scene) and float(args.accident_prob) > 0.0 and search_count > 1
    num_scenarios = search_count if search_enabled else 1

    env = get_validation_env(_env_config_from_args(args, num_scenarios))
    calculator = RiskFieldCalculator(getattr(env, "config", {}))

    try:
        best_seed = args.seed
        best_object_count = -1
        best_cone_count = -1
        searched = 0
        selected = None
        for offset in range(num_scenarios):
            seed = args.seed + offset
            state = _warmup_and_measure(
                env, 
                calculator, 
                seed, 
                args.warmup_steps, 
                args.object_search_distance,
                move_ego_to=args.move_ego_to,
                move_ego_mode=args.move_ego_mode,
            )
            searched += 1

            object_count = int(state["visible_static_object_count"])
            cone_count = int(state["visible_cone_count"])
            best_score = (cone_count, object_count) if args.prefer_cone_scenes else (0, object_count)
            current_best_score = (
                (best_cone_count, best_object_count) if args.prefer_cone_scenes else (0, best_object_count)
            )
            if best_score > current_best_score:
                best_seed = seed
                best_object_count = object_count
                best_cone_count = cone_count

            has_enough_objects = object_count >= min_objects
            has_preferred_cones = cone_count >= min_objects
            if not search_enabled or (has_preferred_cones if args.prefer_cone_scenes else has_enough_objects):
                selected = state
                break

        if selected is None:
            selected = _warmup_and_measure(
                env,
                calculator,
                best_seed,
                args.warmup_steps,
                args.object_search_distance,
                move_ego_to=args.move_ego_to,
                move_ego_mode=args.move_ego_mode,
            )

        selected["searched_seeds"] = searched
        selected["object_search_enabled"] = search_enabled
        selected["min_surrounding_objects"] = min_objects
        selected["best_object_count"] = max(best_object_count, 0)
        selected["best_cone_count"] = max(best_cone_count, 0)
        return env, calculator, selected
    except Exception:
        env.close()
        raise


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay_args = build_risk_overlay_args(_overlay_config_from_args(args))
    env, calculator, state = _select_snapshot_state(args)

    try:
        risk_info = state["risk_info"]
        terminated_early = bool(state["terminated_early"])
        selected_seed = int(state["seed"])
        reward_enabled = bool(getattr(env, "config", {}).get("use_risk_field_reward", False))
        reward_source_cost = math.nan
        reward_normalized_cost = math.nan
        reward_penalty = math.nan
        if hasattr(env, "_risk_field_reward_penalty"):
            reward_source_cost, reward_normalized_cost, reward_penalty = env._risk_field_reward_penalty(risk_info)

        frame = _make_snapshot_frame(env, calculator, overlay_args, args.view)
        Image.fromarray(frame).save(output_path)

        print(f"[snapshot] saved: {output_path}")
        print(f"[snapshot] view: {args.view}")
        print(f"[snapshot] seed: {selected_seed} (requested={args.seed})")
        print(
            "[snapshot] object scene search: "
            f"enabled={int(bool(state['object_search_enabled']))}, "
            f"searched={int(state['searched_seeds'])}, "
            f"min_objects={int(state['min_surrounding_objects'])}, "
            f"distance={float(args.object_search_distance):.1f}, "
            f"prefer_cones={int(bool(args.prefer_cone_scenes))}, "
            f"best_objects={int(state['best_object_count'])}, "
            f"best_cones={int(state['best_cone_count'])}"
        )
        print(f"[snapshot] warmup_steps: {args.warmup_steps} (terminated_early={int(terminated_early)})")
        print(
            "[snapshot] overlay mode: "
            f"road_weighted={int(bool(args.road_risk_use_weights))}, "
            f"vehicle_weighted={int(bool(args.vehicle_risk_use_weights))}, "
            f"risk_panel={int(bool(args.show_risk_panel))}, "
            f"config_panel={int(bool(args.show_config_panel))}, "
            f"surrounding_vehicle_shapes={int(bool(args.show_surrounding_vehicle_shapes))}, "
            f"static_object_shapes={int(bool(args.show_static_object_shapes))}"
        )
        print(f"[snapshot] surrounding vehicles: {int(risk_info.get('risk_field_surrounding_vehicle_count', 0))}")
        print(f"[snapshot] surrounding objects: {int(risk_info.get('risk_field_surrounding_object_count', 0))}")
        nearest_visible_object = state.get("nearest_visible_static_object_distance", math.nan)
        object_types = ",".join(state.get("visible_static_object_types", [])) or "none"
        print(
            "[snapshot] visible static objects: "
            f"count={int(state.get('visible_static_object_count', 0))}, "
            f"cones={int(state.get('visible_cone_count', 0))}, "
            f"nearest={0.0 if not math.isfinite(nearest_visible_object) else nearest_visible_object:.1f}, "
            f"types={object_types}"
        )
        print(
            "[snapshot] risk cost: "
            f"total={float(risk_info.get('risk_field_cost', 0.0)):.3f}, "
            f"road={float(risk_info.get('risk_field_road_cost', 0.0)):.3f}, "
            f"vehicle={float(risk_info.get('risk_field_vehicle_cost', 0.0)):.3f}, "
            f"object={float(risk_info.get('risk_field_object_cost', 0.0)):.3f}"
        )
        print(
            "[snapshot] road breakdown: "
            f"boundary={float(risk_info.get('risk_field_boundary_cost', 0.0)):.3f}, "
            f"lane={float(risk_info.get('risk_field_lane_cost', 0.0)):.3f}, "
            f"offroad={float(risk_info.get('risk_field_offroad_cost', 0.0)):.3f}, "
            f"lat={float(risk_info.get('risk_field_lateral_offset', 0.0)):.3f}, "
            f"lane_width={float(risk_info.get('risk_field_lane_width', 0.0)):.3f}, "
            f"right_boundary={float(risk_info.get('risk_field_dist_to_right_boundary', 0.0)):.3f}"
        )
        print(
            "[snapshot] reward shaping: "
            f"enabled={int(reward_enabled)}, "
            f"source_cost={0.0 if not math.isfinite(reward_source_cost) else reward_source_cost:.3f}, "
            f"normalized={0.0 if not math.isfinite(reward_normalized_cost) else reward_normalized_cost:.3f}, "
            f"penalty={0.0 if not math.isfinite(reward_penalty) else reward_penalty:.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
