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

from PIL import Image

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
    return parser.parse_args()


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
        "risk_overlay_side_by_side": args.view == "side_by_side",
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


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_config = {
        "num_scenarios": 1,
        "start_seed": args.seed,
        "use_render": False,
        "manual_control": False,
        "agent_policy": IDMPolicy,
        "traffic_density": args.traffic_density,
        "accident_prob": args.accident_prob,
    }
    overlay_args = build_risk_overlay_args(_overlay_config_from_args(args))

    env = get_validation_env(env_config)
    calculator = RiskFieldCalculator(getattr(env, "config", {}))

    try:
        env.reset()

        terminated_early = False
        for step_idx in range(max(args.warmup_steps, 0)):
            _, _, terminated, truncated, _ = env.step([0, 0])
            if terminated or truncated:
                terminated_early = True
                print(f"[snapshot] warmup terminated early at step {step_idx + 1}")
                break

        agents = getattr(env, "agents", {})
        if not agents:
            raise RuntimeError("snapshot env has no active agents after reset/warmup")
        ego = next(iter(agents.values()))
        _, risk_info = calculator.calculate(env, ego)
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
        print(f"[snapshot] seed: {args.seed}")
        print(f"[snapshot] warmup_steps: {args.warmup_steps} (terminated_early={int(terminated_early)})")
        print(
            "[snapshot] overlay mode: "
            f"road_weighted={int(bool(args.road_risk_use_weights))}, "
            f"vehicle_weighted={int(bool(args.vehicle_risk_use_weights))}, "
            f"risk_panel={int(bool(args.show_risk_panel))}, "
            f"config_panel={int(bool(args.show_config_panel))}"
        )
        print(f"[snapshot] surrounding vehicles: {int(risk_info.get('risk_field_surrounding_vehicle_count', 0))}")
        print(f"[snapshot] surrounding objects: {int(risk_info.get('risk_field_surrounding_object_count', 0))}")
        print(
            "[snapshot] risk cost: "
            f"total={float(risk_info.get('risk_field_cost', 0.0)):.3f}, "
            f"road={float(risk_info.get('risk_field_road_cost', 0.0)):.3f}, "
            f"vehicle={float(risk_info.get('risk_field_vehicle_cost', 0.0)):.3f}, "
            f"object={float(risk_info.get('risk_field_object_cost', 0.0)):.3f}"
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
