"""用IDM策略批量标定Safe RL的cost_limit。

该脚本让MetaDrive主车使用IDMPolicy行驶一批episode，并统计：
    - event_cost: MetaDrive原始离散事件成本
    - risk_field_cost: 风险场原始连续势场值
    - risk_field_event_equivalent_cost: 压缩到碰撞等价尺度后的风险成本
    - cost: 最终进入Safe RL算法的约束成本

推荐先用IDM作为“可接受安全驾驶参考线”，再根据episode总cost的分位数设置
PPO-Lag / SACL等算法里的cost_limit。

示例：
    python calibrate_cost_limit_idm.py --episodes 50 --num-scenarios 50 --start-seed 100
    python calibrate_cost_limit_idm.py --episodes 100 --recommend-percentile 90 --margin 1.1
"""

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List

import numpy as np

from safe_metadrive_adapter.local_import import prefer_local_metadrive

prefer_local_metadrive()

from env import get_training_env

try:
    from metadrive.policy.idm_policy import IDMPolicy
except ImportError:
    from metadrive.metadrive.policy.idm_policy import IDMPolicy


SUM_KEYS = (
    "cost",
    "event_cost",
    "risk_field_cost",
    "risk_field_weighted_cost",
    "risk_field_event_equivalent_cost",
    "risk_field_boundary_cost",
    "risk_field_lane_cost",
    "risk_field_offroad_cost",
    "risk_field_vehicle_cost",
    "risk_field_object_cost",
    "risk_field_headway_cost",
    "risk_field_ttc_cost",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run IDM episodes and calibrate a cost_limit for Safe RL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=50, help="标定episode数量")
    parser.add_argument("--num-scenarios", type=int, default=50, help="MetaDrive场景数量")
    parser.add_argument("--start-seed", type=int, default=100, help="场景起始seed")
    parser.add_argument("--max-steps", type=int, default=None, help="每个episode最多运行步数，默认使用环境horizon")
    parser.add_argument("--traffic-density", type=float, default=0.2, help="交通密度")
    parser.add_argument("--accident-prob", type=float, default=0.0, help="事故概率")
    parser.add_argument("--disable-idm-lane-change", action="store_true", help="禁用IDM换道")
    parser.add_argument("--disable-idm-deceleration", action="store_true", help="禁用IDM减速")
    parser.add_argument("--event-only", action="store_true", help="关闭风险场，只标定MetaDrive原始事件cost")
    parser.add_argument("--risk-only", action="store_true", help="只使用风险场cost，忽略原始事件cost")
    parser.add_argument("--risk-field-cost-scale", type=float, default=None, help="覆盖risk_field_cost_scale")
    parser.add_argument(
        "--risk-field-cost-mapping",
        choices=("linear_scale", "legacy"),
        default=None,
        help="覆盖风险场到事件等价cost的映射方式",
    )
    parser.add_argument("--risk-field-cost-weight", type=float, default=None, help="覆盖risk_field_cost_weight")
    parser.add_argument(
        "--risk-field-cost-transform",
        choices=("event_squash", "linear_clip"),
        default=None,
        help="覆盖风险场到碰撞等价cost的映射方式",
    )
    parser.add_argument(
        "--risk-field-cost-combine",
        choices=("max", "sum", "risk_only", "event_only"),
        default=None,
        help="覆盖event cost和风险场cost的组合方式",
    )
    parser.add_argument(
        "--risk-field-collision-equivalent-cost",
        type=float,
        default=None,
        help="覆盖风险场单步最大碰撞等价cost",
    )
    parser.add_argument("--risk-field-cost-clip", type=float, default=None, help="覆盖压缩后的风险场cost上限")
    parser.add_argument("--recommend-percentile", type=float, default=90.0, help="推荐cost_limit使用的episode cost分位数")
    parser.add_argument("--margin", type=float, default=1.1, help="推荐cost_limit的安全余量倍数")
    parser.add_argument("--output-json", default="debug/idm_cost_calibration.json", help="JSON汇总输出路径")
    parser.add_argument("--output-csv", default="debug/idm_cost_calibration.csv", help="逐episode CSV输出路径")
    parser.add_argument("--no-save", action="store_true", help="只打印结果，不保存JSON/CSV")
    return parser.parse_args()


def apply_overrides(config: Dict, args) -> None:
    config.update(
        {
            "num_scenarios": args.num_scenarios,
            "start_seed": args.start_seed,
            "use_render": False,
            "manual_control": False,
            "agent_policy": IDMPolicy,
            "traffic_density": args.traffic_density,
            "accident_prob": args.accident_prob,
            "enable_idm_lane_change": not args.disable_idm_lane_change,
            "disable_idm_deceleration": args.disable_idm_deceleration,
            "log_level": 50,
        }
    )
    if args.max_steps is not None:
        config["horizon"] = int(args.max_steps)
    if args.event_only:
        config["use_risk_field_cost"] = False
    if args.risk_only:
        config["risk_field_cost_combine"] = "risk_only"
        config["risk_field_event_cost_weight"] = 0.0

    optional_overrides = {
        "risk_field_cost_scale": args.risk_field_cost_scale,
        "risk_field_cost_mapping": args.risk_field_cost_mapping,
        "risk_field_cost_weight": args.risk_field_cost_weight,
        "risk_field_cost_transform": args.risk_field_cost_transform,
        "risk_field_cost_combine": args.risk_field_cost_combine,
        "risk_field_collision_equivalent_cost": args.risk_field_collision_equivalent_cost,
        "risk_field_cost_clip": args.risk_field_cost_clip,
    }
    for key, value in optional_overrides.items():
        if value is not None:
            config[key] = value


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def summarize(values: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return {key: 0.0 for key in ("mean", "std", "min", "p50", "p80", "p90", "p95", "max")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p80": float(np.percentile(values, 80)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def run_episode(env, episode_index: int, scenario_seed: int, max_steps: int = None) -> Dict[str, float]:
    env.reset(seed=scenario_seed)
    done = False
    step = 0
    totals = {key: 0.0 for key in SUM_KEYS}
    last_info = {}

    while not done:
        _, _, terminated, truncated, info = env.step([0, 0])
        last_info = info
        step += 1
        for key in SUM_KEYS:
            totals[key] += safe_float(info.get(key, 0.0))

        done = bool(terminated or truncated)
        if max_steps is not None and step >= max_steps:
            done = True

    row = {
        "episode": int(episode_index),
        "scenario_seed": int(scenario_seed),
        "steps": int(step),
        "terminated": bool(last_info.get("terminated", False)),
        "success": bool(last_info.get("arrive_dest", False) or last_info.get("success", False)),
        "crash_vehicle": bool(last_info.get("crash_vehicle", False)),
        "crash_object": bool(last_info.get("crash_object", False)),
        "out_of_road": bool(last_info.get("out_of_road", False)),
        "env_total_cost": safe_float(last_info.get("total_cost", totals["cost"])),
    }
    for key, value in totals.items():
        row[f"sum_{key}"] = float(value)
        row[f"mean_{key}"] = float(value / max(step, 1))
    return row


def save_outputs(rows: List[Dict], summary: Dict, args) -> None:
    if args.no_save:
        return

    for path in (args.output_json, args.output_csv):
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: Dict, args) -> None:
    cost = summary["episode_total_cost"]
    step_cost = summary["mean_step_cost"]
    recommended = summary["recommended_cost_limit"]

    print("\nIDM cost calibration")
    print("====================")
    print(f"episodes: {summary['episodes']}")
    print(f"mean episode length: {summary['episode_length']['mean']:.1f} steps")
    print(f"episode total cost mean/std: {cost['mean']:.4f} / {cost['std']:.4f}")
    print(f"episode total cost p50/p80/p90/p95/max: {cost['p50']:.4f} / {cost['p80']:.4f} / {cost['p90']:.4f} / {cost['p95']:.4f} / {cost['max']:.4f}")
    print(f"mean step cost mean/p90/max: {step_cost['mean']:.5f} / {step_cost['p90']:.5f} / {step_cost['max']:.5f}")
    print(f"event cost p90: {summary['event_cost']['p90']:.4f}")
    print(f"risk event-equivalent cost p90: {summary['risk_event_equivalent_cost']['p90']:.4f}")
    print(f"raw risk field cost p90: {summary['raw_risk_field_cost']['p90']:.4f}")
    print("")
    print(f"recommended cost_limit: {recommended:.4f}")
    print(f"rule: percentile={args.recommend_percentile:g}, margin={args.margin:g}")
    print("")
    print("Suggested reading:")
    print("- strict: use p80 if you want policy safer than IDM reference")
    print("- balanced: use p90 * margin as the default starting point")
    print("- loose: use p95 * margin if training collapses from over-tight constraints")
    if not args.no_save:
        print("")
        print(f"saved summary: {args.output_json}")
        print(f"saved episodes: {args.output_csv}")


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_scenarios <= 0:
        raise ValueError("--num-scenarios must be positive")

    env_config = {}
    apply_overrides(env_config, args)
    env = get_training_env(env_config)

    rows = []
    try:
        for episode_index in range(args.episodes):
            scenario_seed = args.start_seed + (episode_index % args.num_scenarios)
            row = run_episode(env, episode_index, scenario_seed, args.max_steps)
            rows.append(row)
            print(
                "episode {}/{} seed={} steps={} total_cost={:.4f} risk_equiv={:.4f}".format(
                    episode_index + 1,
                    args.episodes,
                    scenario_seed,
                    row["steps"],
                    row["sum_cost"],
                    row["sum_risk_field_event_equivalent_cost"],
                )
            )
    finally:
        env.close()

    episode_costs = [row["sum_cost"] for row in rows]
    percentile_cost = float(np.percentile(episode_costs, args.recommend_percentile))
    summary = {
        "episodes": int(args.episodes),
        "config": {
            "num_scenarios": int(args.num_scenarios),
            "start_seed": int(args.start_seed),
            "traffic_density": float(args.traffic_density),
            "accident_prob": float(args.accident_prob),
            "use_risk_field_cost": not bool(args.event_only),
            "risk_only": bool(args.risk_only),
            "risk_field_cost_scale": env_config.get("risk_field_cost_scale"),
            "risk_field_cost_mapping": env_config.get("risk_field_cost_mapping"),
            "risk_field_cost_weight": env_config.get("risk_field_cost_weight"),
            "risk_field_cost_transform": env_config.get("risk_field_cost_transform"),
            "risk_field_cost_combine": env_config.get("risk_field_cost_combine"),
            "risk_field_collision_equivalent_cost": env_config.get("risk_field_collision_equivalent_cost"),
            "risk_field_cost_clip": env_config.get("risk_field_cost_clip"),
        },
        "episode_length": summarize(row["steps"] for row in rows),
        "episode_total_cost": summarize(episode_costs),
        "mean_step_cost": summarize(row["mean_cost"] for row in rows),
        "event_cost": summarize(row["sum_event_cost"] for row in rows),
        "raw_risk_field_cost": summarize(row["sum_risk_field_cost"] for row in rows),
        "risk_event_equivalent_cost": summarize(row["sum_risk_field_event_equivalent_cost"] for row in rows),
        "component_episode_costs": {
            "boundary": summarize(row["sum_risk_field_boundary_cost"] for row in rows),
            "lane": summarize(row["sum_risk_field_lane_cost"] for row in rows),
            "offroad": summarize(row["sum_risk_field_offroad_cost"] for row in rows),
            "vehicle": summarize(row["sum_risk_field_vehicle_cost"] for row in rows),
            "object": summarize(row["sum_risk_field_object_cost"] for row in rows),
            "headway": summarize(row["sum_risk_field_headway_cost"] for row in rows),
            "ttc": summarize(row["sum_risk_field_ttc_cost"] for row in rows),
        },
        "recommended_cost_limit": float(percentile_cost * args.margin),
        "recommend_percentile": float(args.recommend_percentile),
        "margin": float(args.margin),
    }

    save_outputs(rows, summary, args)
    print_summary(summary, args)


if __name__ == "__main__":
    main()
