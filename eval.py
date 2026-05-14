"""
评估训练好的安全强化学习智能体在 MetaDrive 环境中的性能。

支持两种加载方式：
1. 使用 agents 目录中的封装策略：
   python eval.py --agent-name agent_sacl
2. 直接指定训练产物中的 checkpoint：
   python eval.py --model-path logs/.../checkpoint/model_best.pt
   python eval.py --model-path logs/.../periodic_eval/.../checkpoint/model.pt

如果提供 --gif-dir，则会在单进程评估时为每个 episode 保存一个 topdown GIF。
如果提供 --front-gif-dir，则会额外保存 MetaDrive 主摄像头前视角 GIF。

【便捷模式】可以直接在代码底部修改 EVAL_CONFIG 来配置评估参数，无需命令行传参！
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import gymnasium as gym
import numpy as np
import torch
import tqdm
import yaml

from agents import load_policies
from env import VALIDATION_CONFIG, get_validation_env
from resact_metadrive import RESACT_INFO_KEYS, wrap_residual_action_env
from risk_field import RiskFieldCalculator
from utils import compute_episode_success_metrics, make_envs, pretty_print, step_envs
from debug_risk_field_topdown_overlay_gif import (
    build_risk_overlay_args,
    _resolve_road_risk_vmax,
    _resolve_vehicle_risk_vmax,
    render_risk_overlay_frame,
    render_risk_side_by_side_frame,
)


RISK_FIELD_SNAPSHOT_KEYS = (
    "use_risk_field_cost",
    "risk_field_event_cost_weight",
    "risk_field_cost_scale",
    "risk_field_cost_mapping",
    "risk_field_cost_weight",
    "risk_field_cost_transform",
    "risk_field_collision_equivalent_cost",
    "risk_field_cost_clip",
    "risk_field_cost_combine",
    "risk_field_max_distance",
    "risk_field_boundary_weight",
    "risk_field_lane_weight",
    "risk_field_offroad_weight",
    "risk_field_vehicle_weight",
    "risk_field_object_weight",
    "risk_field_headway_weight",
    "risk_field_ttc_weight",
    "risk_field_boundary_sigma",
    "risk_field_lane_edge_sigma",
    "risk_field_broken_line_sigma",
    "risk_field_broken_line_factor",
    "risk_field_solid_line_factor",
    "risk_field_boundary_line_factor",
    "risk_field_oncoming_line_factor",
    "risk_field_offroad_cost",
    "risk_field_offroad_sigma",
    "risk_field_vehicle_longitudinal_sigma",
    "risk_field_vehicle_lateral_sigma",
    "risk_field_vehicle_beta",
    "risk_field_vehicle_dynamic_sigma_scale",
    "risk_field_vehicle_dynamic_alpha",
    "risk_field_vehicle_min_dynamic_sigma",
    "risk_field_object_longitudinal_sigma",
    "risk_field_object_lateral_sigma",
    "risk_field_object_beta",
    "risk_field_headway_time_threshold",
    "risk_field_ttc_threshold",
    "risk_field_raw_clip",
)


# ==================== 评估配置（便捷模式）====================
# 在这里直接配置评估参数。无命令行参数运行 `python eval.py` 时，会自动使用这组配置。
EVAL_CONFIG = {
    # ===== 模型加载配置 =====
    # 方式1: 使用agents目录中的智能体名称（推荐）
    # "agent_name": "agent_ppol",  # 可选: "agent_ppol", "agent_sacl" 等
    # 注意：如果同时设置了 model_path，则 model_path 优先，agent_name 会被忽略。
    "agent_name": None,  # 可选: "agent_ppol", "agent_sacl" 等
    # 方式2: 直接指定checkpoint路径（如果设置了此项，会覆盖agent_name）
    # "model_path": "logs/metadrive-fast-rl/scene-mixed_default/SAC-RESACT-RISK-LOG-s4_-steering_switch_penalty_weight0.03-field-use-lat-reward-done-n_step5_0.6-1-resact-steer0p6-throttle1-4065_scene-mixed_default/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "logs/metadrive-fast-rl/scene-mixed_default/SACL-EVENT_-New-jerk-density0.05-57ae_scene-mixed_default/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "agents/agent_ppol/checkpoint/model_best_lidar_50_0408.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "agents/agent_sacl/checkpoint/model_best_lidar_50_0408.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "logs/metadrive-Reward-v2/SAC-4bc0/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "logs/metadrive-Reward-v1/SAC-RA_-steer0.3-ra0p3-1-7d19/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "logs/metadrive-fast-rl/scene-mixed_default/SACL-RESACT-EVENT_-use-lat-reward-done-n_step5_0.8-1-steer0p15-throttle0p1-resact-steer0p6-throttle1-6630_scene-mixed_default/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "logs/metadrive-Reward-v3/1-SAC-RA_-steer0.8-ra0p8-1-5869/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    "model_path": "logs/metadrive-Reward-v3/SAC-RA_-steer0.8-ra0p8-1-5b01/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "logs/metadrive-Reward-v3/2-SAC-ce24/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/ppol-lidar-7e1f/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "logs/metadrive-fast-rl/scene-mixed_default/SACL-RESACT-EVENT_-resact-steer0p15-throttle0p1-df2b_scene-mixed_default/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "logs/metadrive-fast-rl/scene-mixed_default/SACL-RESACT-EVENT_-REAL-use-la  t-reward-done-n_step3_0.8-1-steer0p15-throttle0p1-resact-steer0p8-throttle1-4fe3_scene-mixed_default/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"
    # "model_path": "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/sacl-cost10-20260418-63d8/checkpoint/model_best.pt",  # 例如: "logs/metadrive_fast-safe-rl/SafeMetaDrive-cost-10/xxx/checkpoint/model_best.pt"

    # 算法类型（如果使用model_path且无法自动推断时才需要）
    "algo": "auto",  # "auto", "ppol", "sacl"
    # "algo": "sacl",  # "auto", "ppol", "sacl"

    # ===== 评估参数 =====
    "num_episodes": 50,  # 要评估的episode总数；这里等价于原命令的 10进程 * 每进程10条
    "seed": 0,  # 随机种子
    "repeat_eval": True,  # True时重复评估多轮，并统计每个指标的跨轮均值/标准差
    "num_eval_repeats": 5,  # repeat_eval=True时重复评估轮数；论文常用3~5个seed
    "repeat_eval_seed_step": 1000,  # 每轮评估seed递增步长，避免重复采样同一随机序列
    "repeat_eval_summary_path": "data/eval_repeat_summary.json",  # 重复评估汇总JSON保存路径；设为None则不保存
    "deterministic_eval": True,  # True 对齐常规最终评估，使用确定性动作
    "use_validation_seed_list": False,  # True 时固定使用验证集场景 seed 列表逐场景评估
    "scenario_seeds": None,  # 可显式指定场景 seed 列表，例如 [1000, 1001, 1002]

    # ===== 渲染配置 =====
    "render": True,  # 是否启用topdown渲染（设为True可以实时观看）
    "save_gif": False,  # 是否保存BEV/topdown GIF动画（设为True会保存到 gif_dir）
    "gif_dir": "data/gif_risk",  # BEV/topdown GIF保存目录
    "save_front_gif": False,  # 是否保存MetaDrive前视摄像头GIF（设为True会保存到 front_gif_dir）
    "front_gif_dir": "data/gif_front",  # MetaDrive前视摄像头GIF保存目录
    "front_gif_fps": 15.0,  # 前视摄像头GIF播放帧率
    "risk_overlay": True,  # 保存GIF时是否叠加风险场；仅在 save_gif=True 时生效
    "risk_overlay_fps": 10.0,  # 风险场GIF播放帧率
    "risk_overlay_width": 800,  # 风险场GIF宽度
    "risk_overlay_height": 800,  # 风险场GIF高度
    "risk_overlay_film_size": 2000,  # MetaDrive topdown film_size
    "risk_overlay_scaling": 8.0,  # topdown缩放比例（像素/米）
    "risk_overlay_vehicle_only": False,  # 仅显示车辆风险
    "risk_overlay_no_road": False,  # 隐藏道路/车道线风险
    "risk_overlay_no_vehicle": False,  # 隐藏车辆风险
    "risk_overlay_road_component": "total",  # lane/broken/solid/boundary/oncoming/total
    "risk_overlay_vehicle_component": "total",  # static/dynamic/total
    "risk_overlay_road_vmax": 0.0,  # <=0 自动按线型因子选择
    "risk_overlay_vehicle_vmax": 0.0,  # <=0 时 total=2.0, static/dynamic=1.0
    "risk_overlay_road_use_weights": True,  # 可视化时道路风险乘入env组件权重，更接近真实cost语义
    "risk_overlay_road_min_alpha": 28,  # 仅提高低风险车道线可见性，不改变风险值
    "risk_overlay_vehicle_use_weights": True,  # 可视化时车辆风险乘入env组件权重，更接近真实cost语义
    "risk_overlay_show_ego_shape": True,  # 在风险场视图中绘制自车形状
    "risk_overlay_no_panel": False,  # 隐藏右上角风险数值面板
    "risk_overlay_no_colorbar": False,  # 隐藏色条
    "risk_overlay_no_config_panel": False,  # 隐藏左下角风险场配置面板
    "risk_overlay_side_by_side": True,  # 左侧仅风险场，右侧原始BEV，不再直接叠加到同一张图上

    # ===== 并行配置（仅在render=False且save_gif=False且save_front_gif=False时有效）=====
    "num_processes": 16,  # 并行环境数量
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-config",
        action="store_true",
        help="显式使用文件顶部的 EVAL_CONFIG 配置。默认关闭，以保持与 eval_old.py 更一致的行为。",
    )
    parser.add_argument(
        "--agent-name",
        default=None,
        type=str,
        help="要评估的智能体名称,即 'agents/' 目录下的子文件夹名称。例如: agent_ppol 或 agent_sacl",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        type=str,
        help="直接指定要评估的 checkpoint 路径，例如 logs/.../checkpoint/model_best.pt",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        type=str,
        help="可选，显式指定与 checkpoint 对应的 config.yaml 路径。不提供时会自动向上查找。",
    )
    parser.add_argument(
        "--algo",
        default="auto",
        choices=["auto", "ppol", "sacl"],
        help="当使用 --model-path 时，指定算法类型。默认自动从 config 推断。",
    )
    parser.add_argument(
        "--log-dir",
        default="data/",
        type=str,
        help="评估产物输出目录的根路径。默认: ./data/",
    )
    parser.add_argument(
        "--gif-dir",
        default=None,
        type=str,
        help="若提供，则保存每个评估 episode 的 topdown GIF 到该目录。仅支持单进程评估。",
    )
    parser.add_argument(
        "--front-gif-dir",
        default=None,
        type=str,
        help="若提供，则保存每个评估 episode 的 MetaDrive 前视摄像头 GIF 到该目录。仅支持单进程评估。",
    )
    parser.add_argument(
        "--front-gif-fps",
        default=10.0,
        type=float,
        help="MetaDrive 前视摄像头 GIF 播放帧率。默认: 10。",
    )
    parser.add_argument(
        "--num-processes",
        default=10,
        type=int,
        help="并行运行的 RL 环境数量。默认: 10",
    )
    parser.add_argument(
        "--num-episodes-per-processes",
        default=10,
        type=int,
        help="每个进程评估的 episode 数量。默认: 10",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="随机种子。这与 MetaDrive 环境的场景种子无关。默认: 0",
    )
    parser.add_argument(
        "--repeat-eval",
        action="store_true",
        help="重复运行多轮完整评估，并输出每个指标的跨轮均值/标准差。",
    )
    parser.add_argument(
        "--num-eval-repeats",
        default=5,
        type=int,
        help="--repeat-eval 时重复评估轮数。默认: 5",
    )
    parser.add_argument(
        "--repeat-eval-seed-step",
        default=1000,
        type=int,
        help="--repeat-eval 时每轮评估seed递增步长。默认: 1000",
    )
    parser.add_argument(
        "--repeat-eval-summary-path",
        default=None,
        type=str,
        help="可选，保存重复评估汇总JSON的路径。",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="是否启用 topdown 渲染。默认: False。注意: 启用渲染时只能使用单进程。",
    )
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="将策略切到 eval 模式，使用确定性动作。默认关闭，以匹配 eval_old.py 的旧评估口径。",
    )
    parser.add_argument(
        "--use-validation-seed-list",
        action="store_true",
        help="固定使用验证集 seed 列表评估，而不是每次 reset 随机抽取场景。",
    )
    parser.add_argument(
        "--scenario-seeds",
        default=None,
        type=str,
        help="显式指定要评估的场景 seed 列表，逗号分隔，例如 1000,1001,1002。",
    )
    parser.add_argument(
        "--risk-overlay",
        action="store_true",
        help="保存GIF时叠加当前风险场，用于排查训练时cost来源。需要同时提供 --gif-dir。",
    )
    parser.add_argument("--risk-overlay-fps", default=10.0, type=float, help="风险场GIF播放帧率。")
    parser.add_argument("--risk-overlay-width", default=800, type=int, help="风险场GIF宽度。")
    parser.add_argument("--risk-overlay-height", default=800, type=int, help="风险场GIF高度。")
    parser.add_argument("--risk-overlay-film-size", default=2000, type=int, help="MetaDrive topdown film_size。")
    parser.add_argument("--risk-overlay-scaling", default=8.0, type=float, help="风险场GIF topdown缩放比例。")
    parser.add_argument("--risk-overlay-vehicle-only", action="store_true", help="风险GIF仅绘制车辆风险。")
    parser.add_argument("--risk-overlay-no-road", action="store_true", help="风险GIF隐藏道路/车道线风险。")
    parser.add_argument("--risk-overlay-no-vehicle", action="store_true", help="风险GIF隐藏车辆风险。")
    parser.add_argument(
        "--risk-overlay-road-component",
        choices=("lane", "broken", "solid", "boundary", "oncoming", "total"),
        default="total",
        help="风险GIF道路风险组件。",
    )
    parser.add_argument(
        "--risk-overlay-vehicle-component",
        choices=("static", "dynamic", "total"),
        default="total",
        help="风险GIF车辆风险组件。",
    )
    parser.add_argument("--risk-overlay-road-vmax", default=0.0, type=float, help="道路风险色条上限；<=0自动。")
    parser.add_argument("--risk-overlay-vehicle-vmax", default=0.0, type=float, help="车辆风险色条上限；<=0自动。")
    parser.add_argument(
        "--risk-overlay-road-min-alpha",
        default=28,
        type=int,
        help="道路/车道线风险可见像素的最小透明度；只影响可视化透明度。",
    )
    parser.add_argument("--risk-overlay-no-ego-shape", action="store_true", help="风险GIF不绘制自车形状。")
    parser.add_argument("--risk-overlay-no-panel", action="store_true", help="隐藏风险GIF右上角数值面板。")
    parser.add_argument("--risk-overlay-no-colorbar", action="store_true", help="隐藏风险GIF右上角色条。")
    parser.add_argument("--risk-overlay-no-config-panel", action="store_true", help="隐藏风险GIF左下角env风险场配置面板。")
    parser.add_argument(
        "--risk-overlay-side-by-side",
        action="store_true",
        help="将风险场GIF渲染为左右对比图：左边仅风险场，右边原始BEV。",
    )
    return parser


def resolve_config_path(model_path: Path) -> Path:
    for parent in [model_path.parent, *model_path.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"未能为 checkpoint 找到 config.yaml: {model_path}")


def infer_algo(config: dict, explicit_algo: str) -> str:
    if explicit_algo != "auto":
        return explicit_algo
    if "actor_lr" in config or "critic_lr" in config or "auto_alpha" in config:
        return "sacl"
    if "target_kl" in config or "repeat_per_collect" in config or "lr" in config:
        return "ppol"
    raise ValueError("无法从配置中自动判断算法类型，请显式传入 --algo ppol 或 --algo sacl")


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def get_policy_class_from_algo(algo: str):
    module_name = {
        "ppol": "agents.agent_ppol.agent",
        "sacl": "agents.agent_sacl.agent",
    }[algo]
    module = import_module(module_name)
    return module.Policy


def load_policy(
    agent_name: str | None,
    model_path: str | None,
    config_path: str | None,
    algo: str,
    deterministic_eval: bool,
):
    if model_path is None:
        all_policies = load_policies()
        if agent_name is None:
            raise ValueError("请至少提供 --agent-name 或 --model-path 中的一个")
        if agent_name not in all_policies:
            raise ValueError(f"在 agents 文件夹中未找到智能体 {agent_name}! 可用的智能体: {all_policies.keys()}")
        policy_class = all_policies[agent_name]
        policy = policy_class()
        selected_algo = "unknown"
        resolved_model_path = None
        resolved_config_path = None
        config = {}
        display_name = agent_name
    else:
        resolved_model_path = Path(model_path).expanduser().resolve()
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {resolved_model_path}")

        resolved_config_path = (
            Path(config_path).expanduser().resolve() if config_path else resolve_config_path(resolved_model_path)
        )
        if not resolved_config_path.exists():
            raise FileNotFoundError(f"config 不存在: {resolved_config_path}")

        config = load_yaml_config(resolved_config_path)
        selected_algo = infer_algo(config, algo)
        if agent_name is not None:
            print(
                f"⚠️  提示: 已提供 checkpoint，按 config 推断算法为 {selected_algo}，"
                f"将忽略 agent_name={agent_name!r}，避免策略结构和权重不匹配。"
            )
        policy_class = get_policy_class_from_algo(selected_algo)
        policy = policy_class(
            model_root=str(resolved_config_path.parent),
            model_path=str(resolved_model_path),
            config_path=str(resolved_config_path),
            best=False,
        )
        display_name = f"{selected_algo}:{resolved_model_path.name}"

    if hasattr(policy, "policy"):
        if deterministic_eval:
            policy.policy.eval()
        else:
            # Keep the legacy behavior of eval_old.py, which leaves the policy in
            # training mode and therefore samples actions stochastically.
            policy.policy.train()

    expected_obs_dim = getattr(policy, "expected_observation_dim", None)
    expected_base_obs_dim = getattr(policy, "expected_base_observation_dim", None)
    if expected_obs_dim is not None:
        config["_policy_expected_observation_dim"] = int(expected_obs_dim)
    if expected_base_obs_dim is not None:
        config["_policy_expected_base_observation_dim"] = int(expected_base_obs_dim)

    return policy, display_name, selected_algo, resolved_model_path, resolved_config_path, config


def is_resact_config(config: dict | None) -> bool:
    return bool(config and config.get("resact_enabled", False))


class ObservationDimCompatibilityWrapper(gym.ObservationWrapper):
    """Pad or truncate 1D observations to match older checkpoints."""

    def __init__(self, env, target_dim: int):
        super().__init__(env)
        self.target_dim = int(target_dim)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
            raise TypeError(f"ObservationDimCompatibilityWrapper requires 1D Box observations, got {space!r}")
        low = self._resize(space.low, fill=-np.inf)
        high = self._resize(space.high, fill=np.inf)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _resize(self, value, *, fill: float):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == self.target_dim:
            return arr
        if arr.size < self.target_dim:
            pad = np.full(self.target_dim - arr.size, fill, dtype=np.float32)
            return np.concatenate([arr, pad]).astype(np.float32)
        return arr[: self.target_dim].astype(np.float32)

    def observation(self, observation):
        return self._resize(observation, fill=0.0)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    @property
    def config(self):
        return getattr(self.env, "config", {})

    @property
    def top_down_renderer(self):
        return getattr(self.env, "top_down_renderer", None)

    @property
    def agents(self):
        return getattr(self.env, "agents", {})

    @property
    def engine(self):
        return getattr(self.env, "engine", None)

    @property
    def main_camera(self):
        return getattr(self.env, "main_camera", None)


def adapt_base_observation_dim(env, policy_config: dict | None):
    if not policy_config:
        return env
    target_dim = policy_config.get("_policy_expected_base_observation_dim")
    if target_dim is None:
        target_dim = policy_config.get("_policy_expected_observation_dim")
        if target_dim is not None and is_resact_config(policy_config):
            target_dim = int(target_dim) - 2
    if target_dim is None:
        return env
    target_dim = int(target_dim)
    current_shape = getattr(env.observation_space, "shape", None)
    if not current_shape or int(current_shape[0]) == target_dim:
        return env
    return ObservationDimCompatibilityWrapper(env, target_dim)


def wrap_eval_env_for_policy(env, policy_config: dict | None):
    env = adapt_base_observation_dim(env, policy_config)
    if not is_resact_config(policy_config):
        return env
    return wrap_residual_action_env(
        env,
        resact_enabled=True,
        resact_steer_delta_scale=float(policy_config.get("resact_steer_delta_scale", 0.15)),
        resact_throttle_delta_scale=float(policy_config.get("resact_throttle_delta_scale", 0.10)),
        resact_initial_action=policy_config.get("resact_initial_action", (0.0, 0.0)),
    )


def make_eval_validation_env(env_config_overrides: dict | None, policy_config: dict | None):
    env = get_validation_env(env_config_overrides or None)
    return wrap_eval_env_for_policy(env, policy_config)


def batch_obs(obs):
    if isinstance(obs, np.ndarray):
        return np.expand_dims(obs, axis=0)
    if isinstance(obs, tuple):
        return tuple(batch_obs(item) for item in obs)
    if isinstance(obs, dict):
        return {key: batch_obs(value) for key, value in obs.items()}
    return np.expand_dims(np.asarray(obs), axis=0)


def ensure_vector_action_batch(actions, num_envs: int):
    """Keep one action per env when num_envs == 1.

    The bundled agent wrappers squeeze singleton batch dimensions, which breaks
    the vector-env path for a single environment. We restore the env batch axis
    here so the stepping logic matches eval_old.py.
    """
    if num_envs != 1:
        return actions

    actions_arr = np.asarray(actions)
    if actions_arr.ndim == 0:
        return np.expand_dims(actions_arr, axis=0)
    if actions_arr.shape[0] != 1:
        return np.expand_dims(actions_arr, axis=0)
    return actions_arr


def parse_scenario_seeds(raw_value: str | None):
    if raw_value is None:
        return None
    seeds = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    return seeds or None


def get_validation_seed_list():
    start_seed = int(VALIDATION_CONFIG["start_seed"])
    num_scenarios = int(VALIDATION_CONFIG["num_scenarios"])
    return list(range(start_seed, start_seed + num_scenarios))


def resolve_scenario_seeds(
    explicit_seeds,
    use_validation_seed_list: bool,
    requested_episodes: int,
):
    if explicit_seeds is not None:
        seeds = list(explicit_seeds)
    elif use_validation_seed_list:
        seeds = get_validation_seed_list()
    else:
        return None

    if requested_episodes > len(seeds):
        raise ValueError(
            f"请求评估 {requested_episodes} 个 episode，但固定 seed 列表只有 {len(seeds)} 个场景。"
        )
    return seeds[:requested_episodes]


def extract_speed_km_h(info: dict) -> float | None:
    if "speed_km_h" in info:
        return float(info["speed_km_h"])
    if "velocity" in info:
        velocity = np.asarray(info["velocity"], dtype=np.float32)
        if velocity.ndim == 0:
            return float(velocity) * 3.6
        return float(np.linalg.norm(velocity)) * 3.6
    return None


def record_step_metrics(info: dict, result_recorder: defaultdict):
    if "crash_vehicle" in info:
        result_recorder["crash_vehicle_rate"].append(info["crash_vehicle"])
    if "crash_sidewalk" in info:
        result_recorder["crash_sidewalk_rate"].append(info["crash_sidewalk"])
    if "idle" in info:
        result_recorder["idle_rate"].append(info["idle"])
    speed_km_h = extract_speed_km_h(info)
    if speed_km_h is not None:
        result_recorder["speed_km_h"].append(speed_km_h)
    for key in RESACT_INFO_KEYS:
        if key in info:
            result_recorder[key].append(info[key])


def record_episode_metrics(
    info: dict,
    episode_reward: float,
    episode_cost: float,
    episode_length: int,
    result_recorder: defaultdict,
    episode_has_crash: bool = False,
    episode_out_of_road: bool = False,
):
    """
    记录每个episode的指标

    注意：这里定义了两种成功率：
    1. success_rate (旧版): 只要到达目的地就算成功，不管是否发生碰撞
    2. safe_success_rate (新版): 只有安全到达（无碰撞、未偏离道路）才算成功
    """
    result_recorder["episode_reward"].append(float(episode_reward))
    result_recorder["episode_cost"].append(float(episode_cost))
    if "route_completion" in info:
        result_recorder["route_completion"].append(info["route_completion"])

    success_metrics = compute_episode_success_metrics(
        info,
        episode_has_crash=episode_has_crash,
        episode_out_of_road=episode_out_of_road,
    )
    result_recorder["success_rate"].append(success_metrics["arrived"])
    result_recorder["safe_success_rate"].append(success_metrics["safe_success"])
    result_recorder["arrive_dest_rate"].append(success_metrics["arrived"])
    result_recorder["crash_free_rate"].append(not success_metrics["has_crash"])
    result_recorder["no_out_of_road_rate"].append(not success_metrics["out_of_road"])

    if "out_of_road" in info:
        result_recorder["out_of_road_rate"].append(info["out_of_road"])
    if "max_step" in info:
        result_recorder["max_step_rate"].append(info["max_step"])
    result_recorder["episode_length"].append(int(info.get("episode_length", episode_length)))


def render_topdown(env, render: bool, record: bool):
    kwargs = {"target_agent_heading_up": True}
    if record:
        kwargs.update(
            {
                "window": render,
                "screen_record": True,
                "screen_size": (800, 800),
                "film_size": (2000, 2000),
            }
        )
    env.render(mode="topdown", **kwargs)


def _json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_risk_overlay_snapshot(
    gif_dir: Path,
    env,
    calculator: RiskFieldCalculator,
    overlay_args: SimpleNamespace,
    metadata: dict | None,
) -> Path:
    snapshot = {
        "metadata": _json_safe(metadata or {}),
        "risk_config_source": "current env.py validation config",
        "risk_field_config": {
            key: _json_safe(getattr(env, "config", {}).get(key))
            for key in RISK_FIELD_SNAPSHOT_KEYS
            if key in getattr(env, "config", {})
        },
        "overlay_config": {
            "road_component": overlay_args.road_risk_component,
            "vehicle_component": overlay_args.vehicle_risk_component,
            "road_enabled": not overlay_args.vehicle_only and not overlay_args.no_road_edge_risk,
            "vehicle_enabled": not overlay_args.no_vehicle_risk,
            "side_by_side": bool(overlay_args.side_by_side),
            "road_vmax": _resolve_road_risk_vmax(overlay_args, calculator),
            "vehicle_vmax": _resolve_vehicle_risk_vmax(overlay_args, calculator),
            "road_use_weights": bool(overlay_args.road_risk_use_weights),
            "vehicle_use_weights": bool(overlay_args.vehicle_risk_use_weights),
            "road_min_alpha": int(getattr(overlay_args, "road_min_alpha", 0)),
            "show_ego_shape": bool(getattr(overlay_args, "show_ego_shape", False)),
            "width": overlay_args.width,
            "height": overlay_args.height,
            "scaling": overlay_args.scaling,
        },
    }
    output_path = gif_dir / "risk_overlay_config.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return output_path


def append_eval_gif_frame(
    env,
    *,
    render: bool,
    gif_writer,
    risk_overlay: bool,
    calculator: RiskFieldCalculator | None,
    overlay_args: SimpleNamespace | None,
    step_info: dict | None = None,
    episode_cost: float = 0.0,
    episode_length: int = 0,
) -> None:
    if risk_overlay:
        if getattr(overlay_args, "side_by_side", False):
            frame = render_risk_side_by_side_frame(
                env,
                render,
                calculator,
                overlay_args,
                step_info=step_info,
                episode_cost=episode_cost,
                episode_length=episode_length,
            )
        else:
            frame = render_risk_overlay_frame(
                env,
                render,
                calculator,
                overlay_args,
                step_info=step_info,
                episode_cost=episode_cost,
                episode_length=episode_length,
            )
        gif_writer.append_data(frame)
    else:
        render_topdown(env, render=render, record=True)


def open_risk_gif_writer(gif_path: Path, fps: float):
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(gif_path), mode="I", duration=1.0 / max(float(fps), 1e-6), loop=0)


def open_gif_writer(gif_path: Path, fps: float):
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(gif_path), mode="I", duration=1.0 / max(float(fps), 1e-6), loop=0)


def append_front_gif_frame(env, gif_writer) -> None:
    if getattr(env, "main_camera", None) is None:
        raise RuntimeError(
            "MetaDrive前视摄像头不可用。保存前视GIF时需要启用环境 use_render=True。"
        )
    # MetaDrive main_camera.perceive 返回 BGR，这里转成 imageio 需要的 RGB。
    frame = env.main_camera.perceive(to_float=False)[..., ::-1]
    gif_writer.append_data(frame)


def risk_overlay_config_from_args(args: argparse.Namespace) -> dict:
    """Collect risk-overlay CLI args using the same keys as EVAL_CONFIG."""

    return {
        "risk_overlay_fps": args.risk_overlay_fps,
        "risk_overlay_width": args.risk_overlay_width,
        "risk_overlay_height": args.risk_overlay_height,
        "risk_overlay_film_size": args.risk_overlay_film_size,
        "risk_overlay_scaling": args.risk_overlay_scaling,
        "risk_overlay_vehicle_only": args.risk_overlay_vehicle_only,
        "risk_overlay_no_road": args.risk_overlay_no_road,
        "risk_overlay_no_vehicle": args.risk_overlay_no_vehicle,
        "risk_overlay_road_component": args.risk_overlay_road_component,
        "risk_overlay_vehicle_component": args.risk_overlay_vehicle_component,
        "risk_overlay_road_vmax": args.risk_overlay_road_vmax,
        "risk_overlay_vehicle_vmax": args.risk_overlay_vehicle_vmax,
        "risk_overlay_road_use_weights": True,
        "risk_overlay_road_min_alpha": args.risk_overlay_road_min_alpha,
        "risk_overlay_vehicle_use_weights": True,
        "risk_overlay_show_ego_shape": not args.risk_overlay_no_ego_shape,
        "risk_overlay_no_panel": args.risk_overlay_no_panel,
        "risk_overlay_no_colorbar": args.risk_overlay_no_colorbar,
        "risk_overlay_no_config_panel": args.risk_overlay_no_config_panel,
        "risk_overlay_side_by_side": args.risk_overlay_side_by_side,
    }


def evaluate_single_env(
    policy,
    total_episodes_to_eval: int,
    render: bool,
    gif_dir: Path | None,
    front_gif_dir: Path | None = None,
    front_gif_fps: float = 10.0,
    scenario_seeds=None,
    risk_overlay: bool = False,
    risk_overlay_config: dict | None = None,
    env_config_overrides: dict | None = None,
    policy_config: dict | None = None,
    overlay_metadata: dict | None = None,
):
    env = make_eval_validation_env(env_config_overrides or None, policy_config)
    risk_overlay = bool(risk_overlay and gif_dir is not None)
    risk_calculator = RiskFieldCalculator(getattr(env, "config", {})) if risk_overlay else None
    overlay_args = build_risk_overlay_args(risk_overlay_config or {}) if risk_overlay else None
    risk_overlay_fps = float((risk_overlay_config or {}).get("risk_overlay_fps", 10.0))
    result_recorder = defaultdict(list)
    saved_gifs = []
    total_steps = 0
    total_episodes = 0

    if gif_dir is not None:
        gif_dir.mkdir(parents=True, exist_ok=True)
    if front_gif_dir is not None:
        front_gif_dir.mkdir(parents=True, exist_ok=True)
    if gif_dir is not None and risk_overlay:
        snapshot_path = save_risk_overlay_snapshot(gif_dir, env, risk_calculator, overlay_args, overlay_metadata)
        print(f"🌡️  风险场参数来源：当前 env_copy.py validation config")
        print(f"🧾 风险场配置快照: {snapshot_path}")

    if scenario_seeds is not None:
        with tqdm.tqdm(total=len(scenario_seeds)) as pbar:
            for episode_index, scenario_seed in enumerate(scenario_seeds, start=1):
                obs, _ = env.reset(seed=scenario_seed)
                if hasattr(policy, "reset"):
                    policy.reset()

                episode_reward = 0.0
                episode_cost = 0.0
                episode_length = 0
                episode_has_crash = False
                episode_out_of_road = False
                gif_writer = None
                gif_path = None
                front_gif_writer = None
                front_gif_path = None
                if gif_dir is not None and risk_overlay:
                    gif_path = gif_dir / f"seed_{scenario_seed}_episode_{episode_index:04d}_risk.gif"
                    gif_writer = open_risk_gif_writer(gif_path, risk_overlay_fps)
                if front_gif_dir is not None:
                    front_gif_path = front_gif_dir / f"seed_{scenario_seed}_episode_{episode_index:04d}_front.gif"
                    front_gif_writer = open_gif_writer(front_gif_path, front_gif_fps)

                if render or gif_dir is not None:
                    if gif_writer is not None:
                        append_eval_gif_frame(
                            env,
                            render=render,
                            gif_writer=gif_writer,
                            risk_overlay=True,
                            calculator=risk_calculator,
                            overlay_args=overlay_args,
                        )
                    else:
                        render_topdown(env, render=render, record=gif_dir is not None)
                if front_gif_writer is not None:
                    append_front_gif_frame(env, front_gif_writer)

                while True:
                    action = policy(batch_obs(obs))
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)

                    total_steps += 1
                    episode_reward += float(reward)
                    episode_cost += float(info.get("cost", 0.0))
                    episode_length += 1
                    episode_has_crash = bool(
                        episode_has_crash or info.get("crash_vehicle", False) or info.get("crash_object", False)
                    )
                    episode_out_of_road = bool(episode_out_of_road or info.get("out_of_road", False))
                    record_step_metrics(info, result_recorder)

                    if render or gif_dir is not None:
                        if gif_writer is not None:
                            append_eval_gif_frame(
                                env,
                                render=render,
                                gif_writer=gif_writer,
                                risk_overlay=True,
                                calculator=risk_calculator,
                                overlay_args=overlay_args,
                                step_info=info,
                                episode_cost=episode_cost,
                                episode_length=episode_length,
                            )
                        else:
                            render_topdown(env, render=render, record=gif_dir is not None)
                    if front_gif_writer is not None:
                        append_front_gif_frame(env, front_gif_writer)

                    if not done:
                        continue

                    record_episode_metrics(
                        info,
                        episode_reward,
                        episode_cost,
                        episode_length,
                        result_recorder,
                        episode_has_crash=episode_has_crash,
                        episode_out_of_road=episode_out_of_road,
                    )
                    total_episodes += 1
                    pbar.update(1)

                    if gif_writer is not None:
                        gif_writer.close()
                        saved_gifs.append(gif_path)
                        gif_writer = None
                        gif_path = None
                    elif gif_dir is not None and env.top_down_renderer is not None and env.top_down_renderer.screen_frames:
                        gif_path = gif_dir / f"seed_{scenario_seed}_episode_{episode_index:04d}.gif"
                        env.top_down_renderer.generate_gif(str(gif_path))
                        saved_gifs.append(gif_path)
                    if front_gif_writer is not None:
                        front_gif_writer.close()
                        saved_gifs.append(front_gif_path)
                        front_gif_writer = None
                        front_gif_path = None

                    if hasattr(policy, "reset"):
                        try:
                            policy.reset(done_batch=np.array([True]))
                        except TypeError:
                            policy.reset()
                    break
                if gif_writer is not None:
                    try:
                        gif_writer.close()
                    except Exception:
                        pass
                if front_gif_writer is not None:
                    try:
                        front_gif_writer.close()
                    except Exception:
                        pass
    else:
        obs, _ = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()

        episode_reward = 0.0
        episode_cost = 0.0
        episode_length = 0
        episode_has_crash = False
        episode_out_of_road = False
        gif_writer = None
        gif_path = None
        front_gif_writer = None
        front_gif_path = None
        if gif_dir is not None and risk_overlay:
            gif_path = gif_dir / f"episode_{total_episodes + 1:04d}_risk.gif"
            gif_writer = open_risk_gif_writer(gif_path, risk_overlay_fps)
        if front_gif_dir is not None:
            front_gif_path = front_gif_dir / f"episode_{total_episodes + 1:04d}_front.gif"
            front_gif_writer = open_gif_writer(front_gif_path, front_gif_fps)

        if render or gif_dir is not None:
            if gif_writer is not None:
                append_eval_gif_frame(
                    env,
                    render=render,
                    gif_writer=gif_writer,
                    risk_overlay=True,
                    calculator=risk_calculator,
                    overlay_args=overlay_args,
                )
            else:
                render_topdown(env, render=render, record=gif_dir is not None)
        if front_gif_writer is not None:
            append_front_gif_frame(env, front_gif_writer)

        with tqdm.tqdm(total=int(total_episodes_to_eval)) as pbar:
            while total_episodes < total_episodes_to_eval:
                action = policy(batch_obs(obs))
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                total_steps += 1
                episode_reward += float(reward)
                episode_cost += float(info.get("cost", 0.0))
                episode_length += 1
                episode_has_crash = bool(
                    episode_has_crash or info.get("crash_vehicle", False) or info.get("crash_object", False)
                )
                episode_out_of_road = bool(episode_out_of_road or info.get("out_of_road", False))
                record_step_metrics(info, result_recorder)

                if render or gif_dir is not None:
                    if gif_writer is not None:
                        append_eval_gif_frame(
                            env,
                            render=render,
                            gif_writer=gif_writer,
                            risk_overlay=True,
                            calculator=risk_calculator,
                            overlay_args=overlay_args,
                            step_info=info,
                            episode_cost=episode_cost,
                            episode_length=episode_length,
                        )
                    else:
                        render_topdown(env, render=render, record=gif_dir is not None)
                if front_gif_writer is not None:
                    append_front_gif_frame(env, front_gif_writer)

                if done:
                    record_episode_metrics(
                        info,
                        episode_reward,
                        episode_cost,
                        episode_length,
                        result_recorder,
                        episode_has_crash=episode_has_crash,
                        episode_out_of_road=episode_out_of_road,
                    )

                    total_episodes += 1
                    pbar.update(1)

                    if gif_writer is not None:
                        gif_writer.close()
                        saved_gifs.append(gif_path)
                        gif_writer = None
                        gif_path = None
                    elif gif_dir is not None and env.top_down_renderer is not None and env.top_down_renderer.screen_frames:
                        gif_path = gif_dir / f"episode_{total_episodes:04d}.gif"
                        env.top_down_renderer.generate_gif(str(gif_path))
                        saved_gifs.append(gif_path)
                    if front_gif_writer is not None:
                        front_gif_writer.close()
                        saved_gifs.append(front_gif_path)
                        front_gif_writer = None
                        front_gif_path = None

                    if hasattr(policy, "reset"):
                        try:
                            policy.reset(done_batch=np.array([True]))
                        except TypeError:
                            policy.reset()

                    if total_episodes >= total_episodes_to_eval:
                        break

                    obs, _ = env.reset()
                    episode_reward = 0.0
                    episode_cost = 0.0
                    episode_length = 0
                    episode_has_crash = False
                    episode_out_of_road = False
                    if gif_dir is not None and risk_overlay:
                        gif_path = gif_dir / f"episode_{total_episodes + 1:04d}_risk.gif"
                        gif_writer = open_risk_gif_writer(gif_path, risk_overlay_fps)
                    if front_gif_dir is not None:
                        front_gif_path = front_gif_dir / f"episode_{total_episodes + 1:04d}_front.gif"
                        front_gif_writer = open_gif_writer(front_gif_path, front_gif_fps)

                    if render or gif_dir is not None:
                        if gif_writer is not None:
                            append_eval_gif_frame(
                                env,
                                render=render,
                                gif_writer=gif_writer,
                                risk_overlay=True,
                                calculator=risk_calculator,
                                overlay_args=overlay_args,
                            )
                        else:
                            render_topdown(env, render=render, record=gif_dir is not None)
                    if front_gif_writer is not None:
                        append_front_gif_frame(env, front_gif_writer)

        if gif_writer is not None:
            try:
                gif_writer.close()
            except Exception:
                pass
        if front_gif_writer is not None:
            try:
                front_gif_writer.close()
            except Exception:
                pass

    env.close()
    return result_recorder, total_steps, saved_gifs


def evaluate_vector_env(
    policy,
    num_processes: int,
    total_episodes_to_eval: int,
    render: bool,
    env_config_overrides: dict | None = None,
    policy_config: dict | None = None,
):
    def single_env_factory():
        return make_eval_validation_env(env_config_overrides or None, policy_config)

    envs = make_envs(
        single_env_factory=single_env_factory,
        num_envs=num_processes,
        asynchronous=True,
    )

    episode_rewards = np.zeros([num_processes, 1], dtype=float)
    episode_costs = np.zeros([num_processes, 1], dtype=float)
    episode_has_crash = np.zeros(num_processes, dtype=bool)
    episode_out_of_road = np.zeros(num_processes, dtype=bool)
    total_episodes = 0
    total_steps = 0
    result_recorder = defaultdict(list)

    print("开始评估!")
    obs = envs.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    with tqdm.tqdm(total=int(total_episodes_to_eval)) as pbar:
        while True:
            cpu_actions = policy(obs)
            cpu_actions = ensure_vector_action_batch(cpu_actions, num_processes)
            obs, reward, done, info, masks, total_episodes, total_steps, episode_rewards, episode_costs = step_envs(
                cpu_actions=cpu_actions,
                envs=envs,
                episode_rewards=episode_rewards,
                episode_costs=episode_costs,
                result_recorder=result_recorder,
                total_steps=total_steps,
                total_episodes=total_episodes,
                device="cpu",
                episode_has_crash=episode_has_crash,
                episode_out_of_road=episode_out_of_road,
            )

            if hasattr(policy, "reset"):
                policy.reset(done_batch=done)

            if render:
                envs.render(mode="topdown")

            pbar.update(total_episodes - pbar.n)
            if total_episodes >= total_episodes_to_eval:
                break

    envs.close()
    return result_recorder, total_steps


def set_eval_seed(seed: int) -> None:
    """设置一次评估运行使用的随机种子。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def summarize_result_recorder(result_recorder: defaultdict) -> dict:
    """把单轮评估的原始记录压成每个指标一个均值。"""
    metrics = {}
    for key, values in result_recorder.items():
        if not values:
            continue
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.size == 0:
            continue
        metrics[key] = float(np.mean(array))
    return metrics


def aggregate_eval_runs(run_metrics: list[dict]) -> dict:
    """跨多轮评估统计 mean/std/min/max/sem。"""
    summary = {}
    metric_keys = sorted({key for metrics in run_metrics for key in metrics})
    for key in metric_keys:
        values = np.asarray([metrics[key] for metrics in run_metrics if key in metrics], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "sem": float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": int(values.size),
        }
    return summary


def save_repeat_eval_summary(
    output_path: Path | None,
    *,
    run_metrics: list[dict],
    summary: dict,
    metadata: dict,
) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": _json_safe(metadata),
        "runs": [_json_safe(metrics) for metrics in run_metrics],
        "summary": _json_safe(summary),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"🧾 重复评估汇总已保存: {output_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ==================== 使用EVAL_CONFIG配置（便捷模式）====================
    # 显式传入 --use-config，或直接运行 `python eval.py` 时启用，方便在文件顶部改默认评估参数。
    use_config_mode = args.use_config or len(sys.argv) == 1

    if use_config_mode:
        print("=" * 60)
        print("📌 使用 EVAL_CONFIG 配置进行评估（便捷模式）")
        print("=" * 60)

        # 从EVAL_CONFIG提取参数
        agent_name = EVAL_CONFIG["agent_name"]
        model_path = EVAL_CONFIG["model_path"]
        algo = EVAL_CONFIG["algo"]
        num_episodes = EVAL_CONFIG["num_episodes"]
        seed = EVAL_CONFIG["seed"]
        repeat_eval = bool(EVAL_CONFIG.get("repeat_eval", False))
        num_eval_repeats = int(EVAL_CONFIG.get("num_eval_repeats", 5))
        repeat_eval_seed_step = int(EVAL_CONFIG.get("repeat_eval_seed_step", 1000))
        repeat_eval_summary_path = EVAL_CONFIG.get("repeat_eval_summary_path")
        repeat_eval_summary_path = (
            Path(repeat_eval_summary_path).expanduser().resolve() if repeat_eval_summary_path else None
        )
        deterministic_eval = EVAL_CONFIG.get("deterministic_eval", False)
        render = EVAL_CONFIG["render"]
        save_gif = EVAL_CONFIG["save_gif"]
        gif_dir = Path(EVAL_CONFIG["gif_dir"]).expanduser().resolve() if save_gif else None
        save_front_gif = bool(EVAL_CONFIG.get("save_front_gif", False))
        front_gif_dir = Path(EVAL_CONFIG["front_gif_dir"]).expanduser().resolve() if save_front_gif else None
        front_gif_fps = float(EVAL_CONFIG.get("front_gif_fps", 10.0))
        risk_overlay = bool(EVAL_CONFIG.get("risk_overlay", False))
        risk_overlay_config = dict(EVAL_CONFIG)
        num_processes = EVAL_CONFIG["num_processes"]
        use_validation_seed_list = EVAL_CONFIG.get("use_validation_seed_list", False)
        explicit_scenario_seeds = EVAL_CONFIG.get("scenario_seeds")
        config_path = None  # EVAL_CONFIG中暂不支持

        # 验证配置
        if agent_name is None and model_path is None:
            raise ValueError("请在 EVAL_CONFIG 中设置 agent_name 或 model_path")

        if render and num_processes != 1:
            print("⚠️  警告: 启用渲染时自动切换到单进程模式")
            num_processes = 1

        if save_gif and num_processes != 1:
            print("⚠️  警告: 保存GIF时自动切换到单进程模式")
            num_processes = 1
        if save_front_gif and num_processes != 1:
            print("⚠️  警告: 保存前视GIF时自动切换到单进程模式")
            num_processes = 1
        if risk_overlay and not save_gif:
            print("⚠️  提示: risk_overlay=True 但 save_gif=False，本次不会保存风险场GIF")

        total_episodes_to_eval = num_episodes
    else:
        # ==================== 使用命令行参数（传统模式）====================
        print("=" * 60)
        print("📌 使用命令行参数进行评估（传统模式）")
        print("=" * 60)

        if args.agent_name is None and args.model_path is None:
            parser.error("请至少提供 --agent-name 或 --model-path 中的一个")

        agent_name = args.agent_name
        model_path = args.model_path
        algo = args.algo
        seed = args.seed
        repeat_eval = bool(args.repeat_eval)
        num_eval_repeats = int(args.num_eval_repeats)
        repeat_eval_seed_step = int(args.repeat_eval_seed_step)
        repeat_eval_summary_path = (
            Path(args.repeat_eval_summary_path).expanduser().resolve() if args.repeat_eval_summary_path else None
        )
        deterministic_eval = args.deterministic_eval
        render = args.render
        save_gif = args.gif_dir is not None
        gif_dir = Path(args.gif_dir).expanduser().resolve() if args.gif_dir else None
        save_front_gif = args.front_gif_dir is not None
        front_gif_dir = Path(args.front_gif_dir).expanduser().resolve() if args.front_gif_dir else None
        front_gif_fps = float(args.front_gif_fps)
        risk_overlay = bool(args.risk_overlay)
        risk_overlay_config = risk_overlay_config_from_args(args)
        num_processes = args.num_processes
        use_validation_seed_list = args.use_validation_seed_list
        explicit_scenario_seeds = parse_scenario_seeds(args.scenario_seeds)
        config_path = args.config_path
        total_episodes_to_eval = args.num_episodes_per_processes * args.num_processes
        if save_front_gif and num_processes != 1:
            print("⚠️  警告: 保存前视GIF时自动切换到单进程模式")
            num_processes = 1
        if risk_overlay and not save_gif:
            print("⚠️  提示: --risk-overlay 需要同时提供 --gif-dir，本次不会保存风险场GIF")

    scenario_seeds = resolve_scenario_seeds(
        explicit_seeds=explicit_scenario_seeds,
        use_validation_seed_list=use_validation_seed_list,
        requested_episodes=total_episodes_to_eval,
    )

    if scenario_seeds is not None and num_processes != 1:
        print("⚠️  警告: 固定场景 seed 列表评估需要顺序 reset，自动切换到单进程模式")
        num_processes = 1

    repeat_count = max(int(num_eval_repeats), 1) if repeat_eval else 1

    # 设置首次评估随机种子
    set_eval_seed(seed)
    torch.set_num_threads(1)

    # 加载策略
    policy, display_name, selected_algo, resolved_model_path, resolved_config_path, policy_config = load_policy(
        agent_name=agent_name,
        model_path=model_path,
        config_path=config_path,
        algo=algo,
        deterministic_eval=deterministic_eval,
    )
    # 策略网络仍按checkpoint/config加载；风险场评估刻意使用当前env_copy.py验证环境配置，
    # 方便调试“当前风险场设计”而不是复现旧日志中的历史风险参数。
    env_config_overrides = {}

    # 打印评估信息
    print("\n" + "=" * 60)
    print(f"🤖 正在评估智能体: {display_name}")
    if selected_algo != "unknown":
        print(f"🔧 算法类型: {selected_algo}")
    if resolved_config_path is not None:
        print(f"📄 配置文件: {resolved_config_path}")
    if resolved_model_path is not None:
        print(f"📦 模型文件: {resolved_model_path}")
    print("🌡️  风险场参数来源: 当前 env_copy.py validation config")
    print(f"👤 创建者: {policy.CREATOR_NAME}, UID: {policy.CREATOR_UID}")
    print(f"🎬 评估Episode数: {total_episodes_to_eval}")
    print(f"🎯 确定性策略: {'✅ 是' if deterministic_eval else '❌ 否（与 eval_old.py 一致）'}")
    print(f"🔁 重复评估: {'✅ 是' if repeat_count > 1 else '❌ 否'}")
    if repeat_count > 1:
        print(f"   repeats={repeat_count}, base_seed={seed}, seed_step={repeat_eval_seed_step}")
        if scenario_seeds is not None and deterministic_eval:
            print("   ⚠️ 固定场景 + 确定性策略下，多轮结果可能完全一致；论文表格通常改用不同训练seed。")
    if is_resact_config(policy_config):
        print(
            "🧩 ResAct残差动作: ✅ 是 "
            f"(steer_delta={float(policy_config.get('resact_steer_delta_scale', 0.15)):g}, "
            f"throttle_delta={float(policy_config.get('resact_throttle_delta_scale', 0.10)):g})"
        )
    else:
        print("🧩 ResAct残差动作: ❌ 否")
    print(f"🧭 固定场景列表: {'✅ 是' if scenario_seeds is not None else '❌ 否'}")
    if scenario_seeds is not None:
        print(f"🧩 场景Seeds: {scenario_seeds}")
    print(f"🎨 实时渲染: {'✅ 是' if render else '❌ 否'}")
    print(f"📹 保存BEV GIF: {'✅ 是' if save_gif else '❌ 否'}")
    if save_gif:
        print(f"💾 BEV GIF目录: {gif_dir}")
        print(f"🌡️  风险场叠加: {'✅ 是' if risk_overlay else '❌ 否'}")
        if risk_overlay:
            road_enabled = not risk_overlay_config.get("risk_overlay_vehicle_only", False) and not risk_overlay_config.get(
                "risk_overlay_no_road", False
            )
            vehicle_enabled = not risk_overlay_config.get("risk_overlay_no_vehicle", False)
            print(
                "   "
                f"road={risk_overlay_config.get('risk_overlay_road_component', 'total') if road_enabled else 'off'}, "
                f"vehicle={risk_overlay_config.get('risk_overlay_vehicle_component', 'total') if vehicle_enabled else 'off'}"
            )
            print(
                "   "
                f"view={'side_by_side' if risk_overlay_config.get('risk_overlay_side_by_side', False) else 'overlay'}"
            )
    print(f"🎥 保存前视GIF: {'✅ 是' if save_front_gif else '❌ 否'}")
    if save_front_gif:
        print(f"💾 前视GIF目录: {front_gif_dir}")
        print(f"🎞️  前视GIF FPS: {front_gif_fps:g}")
    print(f"⚡ 并行进程数: {num_processes}")
    print("=" * 60 + "\n")

    run_metrics = []
    run_steps = []
    saved_gifs_all = []

    # 执行评估
    for run_index in range(repeat_count):
        run_number = run_index + 1
        run_seed = int(seed) + run_index * int(repeat_eval_seed_step)
        set_eval_seed(run_seed)
        if hasattr(policy, "reset"):
            policy.reset()

        run_env_config_overrides = dict(env_config_overrides)
        if save_front_gif:
            run_env_config_overrides["use_render"] = True

        run_gif_dir = gif_dir
        run_front_gif_dir = front_gif_dir
        if repeat_count > 1:
            print("\n" + "=" * 60)
            print(f"🔁 重复评估 {run_number}/{repeat_count} (seed={run_seed})")
            print("=" * 60)
            if gif_dir is not None:
                run_gif_dir = gif_dir / f"run_{run_number:02d}_seed_{run_seed}"
            if front_gif_dir is not None:
                run_front_gif_dir = front_gif_dir / f"run_{run_number:02d}_seed_{run_seed}"

        if save_gif or save_front_gif or scenario_seeds is not None:
            print("🚀 开始顺序评估（支持固定场景 / GIF）...\n")
            result_recorder, total_steps, saved_gifs = evaluate_single_env(
                policy=policy,
                total_episodes_to_eval=total_episodes_to_eval,
                render=render,
                gif_dir=run_gif_dir,
                front_gif_dir=run_front_gif_dir,
                front_gif_fps=front_gif_fps,
                scenario_seeds=scenario_seeds,
                risk_overlay=risk_overlay,
                risk_overlay_config=risk_overlay_config,
                env_config_overrides=run_env_config_overrides,
                policy_config=policy_config,
                overlay_metadata={
                    "display_name": display_name,
                    "selected_algo": selected_algo,
                    "model_path": str(resolved_model_path) if resolved_model_path is not None else None,
                    "config_path": str(resolved_config_path) if resolved_config_path is not None else None,
                    "resact_enabled": is_resact_config(policy_config),
                    "deterministic_eval": deterministic_eval,
                    "scenario_seeds": scenario_seeds,
                    "total_episodes_to_eval": total_episodes_to_eval,
                    "repeat_eval": repeat_count > 1,
                    "repeat_run_index": run_index,
                    "repeat_run_seed": run_seed,
                },
            )
        else:
            print("🚀 开始向量环境评估（与 eval_old.py 口径一致）...\n")
            result_recorder, total_steps = evaluate_vector_env(
                policy=policy,
                num_processes=num_processes,
                total_episodes_to_eval=total_episodes_to_eval,
                render=render,
                env_config_overrides=run_env_config_overrides,
                policy_config=policy_config,
            )
            saved_gifs = []

        metrics_dict = summarize_result_recorder(result_recorder)
        run_metrics.append(metrics_dict)
        run_steps.append(int(total_steps))
        saved_gifs_all.extend(saved_gifs)

        if repeat_count > 1:
            print("\n" + "=" * 60)
            print(f"📊 第 {run_number}/{repeat_count} 轮指标:")
            print("=" * 60)
            pretty_print(metrics_dict)

    # 打印评估结果
    print("\n" + "=" * 60)
    print(f"📊 {display_name} 的性能指标:")
    print("=" * 60)

    if repeat_count == 1:
        metrics_dict = run_metrics[0]
        pretty_print(metrics_dict)
    else:
        metrics_dict = aggregate_eval_runs(run_metrics)
        pretty_print(metrics_dict)
        save_repeat_eval_summary(
            repeat_eval_summary_path,
            run_metrics=run_metrics,
            summary=metrics_dict,
            metadata={
                "display_name": display_name,
                "selected_algo": selected_algo,
                "model_path": str(resolved_model_path) if resolved_model_path is not None else None,
                "config_path": str(resolved_config_path) if resolved_config_path is not None else None,
                "deterministic_eval": deterministic_eval,
                "num_eval_repeats": repeat_count,
                "base_seed": int(seed),
                "repeat_eval_seed_step": int(repeat_eval_seed_step),
                "total_episodes_to_eval_per_run": int(total_episodes_to_eval),
                "scenario_seeds": scenario_seeds,
            },
        )

    # 特别提示新指标
    if repeat_count == 1 and "safe_success_rate" in metrics_dict:
        print("\n💡 关键指标说明:")
        print(f"   • success_rate (旧版): {metrics_dict.get('success_rate', 0):.4f} - 到达率（不管是否安全）")
        print(f"   • safe_success_rate (新版): {metrics_dict.get('safe_success_rate', 0):.4f} ⭐ - 安全到达率")
        print(f"   • crash_free_rate: {metrics_dict.get('crash_free_rate', 0):.4f} - 无碰撞率")
        print(f"   • no_out_of_road_rate: {metrics_dict.get('no_out_of_road_rate', 0):.4f} - 未偏离道路率")
        print(f"   • episode_cost: {metrics_dict.get('episode_cost', 0):.4f} - 平均代价")
    elif repeat_count > 1 and "safe_success_rate" in metrics_dict:
        print("\n💡 关键指标说明:")
        print(
            "   • safe_success_rate: "
            f"{metrics_dict['safe_success_rate']['mean']:.4f} ± {metrics_dict['safe_success_rate']['std']:.4f}"
        )
        if "episode_cost" in metrics_dict:
            print(
                "   • episode_cost: "
                f"{metrics_dict['episode_cost']['mean']:.4f} ± {metrics_dict['episode_cost']['std']:.4f}"
            )

    print("=" * 60)
    print(f"👣 总评估步数: {int(np.sum(run_steps))}")
    if repeat_count > 1:
        print(f"🔁 重复评估轮数: {repeat_count}")
        print(f"👣 每轮评估步数: {run_steps}")
    if saved_gifs_all:
        print(f"\n📹 已保存 {len(saved_gifs_all)} 个GIF文件:")
        for path in saved_gifs_all:
            print(f"   {path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
