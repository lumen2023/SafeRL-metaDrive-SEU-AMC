"""
PPO-Lag (Proximal Policy Optimization with Lagrangian) 安全强化学习算法训练脚本

该脚本用于在 SafeMetaDrive 环境中训练基于拉格朗日乘子法的 PPO 安全强化学习智能体。
PPO-Lag 通过将约束马尔可夫决策过程(CMDP)转化为无约束优化问题,动态调整拉格朗日乘子
来平衡奖励最大化与安全约束满足。

使用方法:
    python3 train_ppol.py --task SafeMetaDrive

可选参数可通过命令行传入,或在代码中修改配置。
"""
import os
import shlex
import subprocess
import sys
import time
import types
from dataclasses import asdict

import numpy as np

try:
    import bullet_safety_gym  # noqa: F401
except ImportError:
    bullet_safety_gym = None
try:
    import safety_gymnasium  # noqa: F401
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from safe_metadrive_adapter.local_import import prefer_local_metadrive

prefer_local_metadrive()

from env import DEFAULT_CONFIG, SafeMetaDriveEnv_mini, SafeMetaDriveSingleSceneEnv, build_single_scene_config
from fsrl.fsrl.agent import PPOLagAgent
from fsrl.fsrl.data import FastCollector
from fsrl.fsrl.config.ppol_cfg import (
    Bullet1MCfg,
    Bullet5MCfg,
    Bullet10MCfg,
    Mujoco2MCfg,
    Mujoco10MCfg,
    Mujoco20MCfg,
    MujocoBaseCfg,
    TrainCfg,
)
from fsrl.fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.fsrl.utils.exp_util import DEFAULT_SKIP_KEY, auto_name
from safe_metadrive_sweep import (
    SAFE_METADRIVE_PROJECT,
    SAFE_METADRIVE_SWEEP_SCENES,
    append_scene_tag_to_run_name,
    build_safe_metadrive_child_command,
    is_mixed_default_scene,
    normalize_safe_metadrive_scene,
    safe_metadrive_group,
)
from safe_metadrive_adapter.factory import make_safe_metadrive_env as make_adapter_safe_metadrive_env

import gymnasium as gym
try:
    from metadrive.policy.idm_policy import IDMPolicy
except ImportError:
    from metadrive.metadrive.policy.idm_policy import IDMPolicy

TRAIN_ENV_CONFIG = {"num_scenarios": 10, "start_seed": 1000}
VALIDATION_ENV_CONFIG = {"num_scenarios": 10, "start_seed": 1000}

COLLECTOR_OPTIONAL_LOG_KEYS = (
    "success_rate",
    "safe_success_rate",
    "crash_free_rate",
    "no_out_of_road_rate",
    "avg_speed_km_h",
    "event_cost",
    "risk_field_cost",
    "risk_field_road_cost",
    "risk_field_vehicle_cost",
    "risk_field_object_cost",
    "base_reward",
    "driving_component_reward",
    "speed_component_reward",
    "normalized_speed_reward",
    "longitudinal_progress",
    "positive_road",
    "reward_override_active",
    "reward_override_delta",
    "final_reward",
    "smoothness_penalty",
    "smoothness_penalty_raw",
    "action_smoothness_penalty",
    "action_smoothness_penalty_raw",
    "steering_delta",
    "steering_switch",
    "steering_switch_count",
    "steering_absolute_value",
    "steering_jerk",
    "steering_smoothness_penalty",
    "steering_switch_penalty",
    "steering_switch_count_penalty",
    "steering_absolute_penalty",
    "steering_penalty",
    "steering_jerk_smoothness_penalty",
    "throttle_smoothness_penalty",
    "jerk_smoothness_penalty",
    "lateral_now",
    "lateral_norm",
    "lateral_score",
    "lateral_speed_gate",
    "lateral_forward_gate",
    "lateral_broken_line_gate",
    "lateral_reward",
    "lateral_velocity_m_s",
    "lateral_acceleration_m_s2",
    "lateral_jerk_m_s3",
    "yaw_rate_rad_s",
    "yaw_acceleration_rad_s2",
    "base_reward_step_mean",
    "driving_component_reward_step_mean",
    "speed_component_reward_step_mean",
    "normalized_speed_reward_step_mean",
    "longitudinal_progress_step_mean",
    "positive_road_step_mean",
    "reward_override_active_step_mean",
    "reward_override_delta_step_mean",
    "final_reward_step_mean",
    "smoothness_penalty_step_mean",
    "smoothness_penalty_raw_step_mean",
    "action_smoothness_penalty_step_mean",
    "action_smoothness_penalty_raw_step_mean",
    "steering_delta_step_mean",
    "steering_switch_step_mean",
    "steering_switch_count_step_mean",
    "steering_absolute_value_step_mean",
    "steering_jerk_step_mean",
    "steering_smoothness_penalty_step_mean",
    "steering_switch_penalty_step_mean",
    "steering_switch_count_penalty_step_mean",
    "steering_absolute_penalty_step_mean",
    "steering_penalty_step_mean",
    "steering_jerk_smoothness_penalty_step_mean",
    "throttle_smoothness_penalty_step_mean",
    "jerk_smoothness_penalty_step_mean",
    "lateral_now_step_mean",
    "lateral_norm_step_mean",
    "lateral_score_step_mean",
    "lateral_speed_gate_step_mean",
    "lateral_forward_gate_step_mean",
    "lateral_broken_line_gate_step_mean",
    "lateral_reward_step_mean",
    "risk_field_cost_step_mean",
    "risk_field_road_cost_step_mean",
    "risk_field_vehicle_cost_step_mean",
    "risk_field_object_cost_step_mean",
    "lateral_velocity_m_s_step_mean",
    "lateral_acceleration_m_s2_step_mean",
    "lateral_jerk_m_s3_step_mean",
    "yaw_rate_rad_s_step_mean",
    "yaw_acceleration_rad_s2_step_mean",
)

# ===== 注册自定义 SafeMetaDrive 环境 =====
# 注册训练环境:使用50个场景,种子范围[1000, 1050)
gym.register(
    id="SafeMetaDrive-training",
    entry_point="env:SafeMetaDriveEnv_mini",  # 使用 env.py 中定义的简化环境类
    max_episode_steps=1000,  # 每个episode最大步数
    kwargs={"config": dict(TRAIN_ENV_CONFIG)},  # 环境配置
)

# 注册验证环境:使用相同的场景数量但不同的种子范围
gym.register(
    id="SafeMetaDrive-validation",
    entry_point="env:SafeMetaDriveEnv_mini",
    max_episode_steps=1000,
    kwargs={"config": dict(VALIDATION_ENV_CONFIG)},
)


# ===== 任务到配置的映射表 =====
# 将不同的安全强化学习任务映射到对应的默认配置类
TASK_TO_CFG = {
    # ===== MetaDrive 安全驾驶任务 =====
    "SafeMetaDrive": TrainCfg,

    # ===== Bullet Safety Gym 任务 =====
    "SafetyCarRun-v0": Bullet1MCfg,      # 小车直线行驶
    "SafetyBallRun-v0": Bullet1MCfg,     # 球体直线行驶
    "SafetyBallCircle-v0": Bullet1MCfg,  # 球体环形行驶
    "SafetyCarCircle-v0": TrainCfg,      # 小车环形行驶
    "SafetyDroneRun-v0": TrainCfg,       # 无人机直线飞行
    "SafetyAntRun-v0": TrainCfg,         # Ant机器人直线行走
    "SafetyDroneCircle-v0": Bullet5MCfg, # 无人机环形飞行
    "SafetyAntCircle-v0": Bullet10MCfg,  # Ant机器人环形行走

    # ===== Safety Gymnasium 任务 =====
    "SafetyPointCircle1Gymnasium-v0": Mujoco2MCfg,
    "SafetyPointCircle2Gymnasium-v0": Mujoco2MCfg,
    "SafetyCarCircle1Gymnasium-v0": Mujoco2MCfg,
    "SafetyCarCircle2Gymnasium-v0": Mujoco2MCfg,
    "SafetyPointGoal1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointGoal2Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointButton1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointButton2Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointPush1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointPush2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarGoal1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarGoal2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarButton1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarButton2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarPush1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarPush2Gymnasium-v0": MujocoBaseCfg,
    "SafetyHalfCheetahVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetyHopperVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetySwimmerVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetyWalker2dVelocityGymnasium-v1": Mujoco10MCfg,
    "SafetyAntVelocityGymnasium-v1": Mujoco10MCfg,
    "SafetyHumanoidVelocityGymnasium-v1": Mujoco20MCfg,
}


def _compact_float_tag(value):
    value = float(value)
    text = str(int(value)) if value.is_integer() else f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def effective_risk_field_enabled(args):
    value = getattr(args, "use_risk_field_cost", None)
    if value is None:
        return bool(DEFAULT_CONFIG.get("use_risk_field_cost", False))
    return bool(value)


def effective_risk_field_scale(args):
    value = getattr(args, "risk_field_cost_scale", None)
    if value is None:
        return float(DEFAULT_CONFIG.get("risk_field_cost_scale", 1.0))
    return float(value)


def effective_traffic_density(args):
    value = getattr(args, "traffic_density", None)
    if value is None:
        return float(DEFAULT_CONFIG.get("traffic_density", 0.0))
    return float(value)


def resolve_experiment_prefix(args):
    """Return table-aligned PPO/PPOL/IDM experiment labels."""
    if getattr(args, "use_idm_policy", False):
        return "IDM"
    risk_enabled = effective_risk_field_enabled(args)
    if getattr(args, "use_lagrangian", True):
        prefix = "PPOL-RISK" if risk_enabled else "PPOL"
    else:
        prefix = "PPO-RISK" if risk_enabled else "PPO"
    return prefix


def build_safe_metadrive_config(base_config, args, *, artifact=False):
    """根据当前命令行参数构造SafeMetaDrive配置。

    如果开启use_idm_policy，MetaDrive官方IDMPolicy会接管自车运动。
    此时PPO仍会产生动作，但这些动作不再决定车辆动力学，适合做IDM
    baseline/debug，而不是正常PPO训练。
    """
    config = dict(base_config)
    if getattr(args, "use_risk_field_cost", None) is not None:
        config["use_risk_field_cost"] = bool(args.use_risk_field_cost)
    if getattr(args, "risk_field_cost_scale", None) is not None:
        config["risk_field_cost_scale"] = float(args.risk_field_cost_scale)
    if getattr(args, "traffic_density", None) is not None:
        config["traffic_density"] = float(args.traffic_density)
    for key in (
        "resact_enabled",
        "resact_steer_delta_scale",
        "resact_throttle_delta_scale",
        "resact_initial_action",
    ):
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if getattr(args, "use_idm_policy", False):
        config.update(
            {
                "manual_control": False,
                "agent_policy": IDMPolicy,
                "enable_idm_lane_change": not getattr(args, "disable_idm_lane_change", False),
                "disable_idm_deceleration": getattr(args, "disable_idm_deceleration", False),
            }
        )
        if getattr(args, "idm_rollout_max_steps", None) is not None:
            config["horizon"] = int(args.idm_rollout_max_steps)
    if artifact:
        config.update(
            {
                "disable_model_compression": True,
                "show_interface": False,
                "use_render": True,
            }
        )
    return config


def make_safe_metadrive_env(base_config, args, *, artifact=False):
    return make_adapter_safe_metadrive_env(
        split="train",
        config=build_safe_metadrive_config(base_config, args, artifact=artifact),
    )


def make_safe_metadrive_scene_env(split: str, args, *, artifact=False):
    scene = normalize_safe_metadrive_scene(getattr(args, "safe_metadrive_scene", "mixed_default"))
    if is_mixed_default_scene(scene):
        base_config = TRAIN_ENV_CONFIG if split == "train" else VALIDATION_ENV_CONFIG
        return make_safe_metadrive_env(base_config, args, artifact=artifact)

    if SafeMetaDriveSingleSceneEnv is None or build_single_scene_config is None:
        raise RuntimeError(
            "SafeMetaDrive env module does not provide single-scene helpers. "
            "Use --safe_metadrive_scene mixed_default, or restore a single-scene env module for this experiment."
        )

    runtime_overrides = build_safe_metadrive_config({}, args, artifact=artifact)
    scene_config = build_single_scene_config(scene, split=split, extra_config=runtime_overrides)
    return SafeMetaDriveSingleSceneEnv(scene_config)


def maybe_run_safe_metadrive_scene_sweep(args):
    if args.task != "SafeMetaDrive" or not bool(getattr(args, "safe_metadrive_sweep", False)):
        return False

    script_path = os.path.abspath(__file__)
    parent_argv = sys.argv[1:]
    total = len(SAFE_METADRIVE_SWEEP_SCENES)
    print("SafeMetaDrive scene sweep: {}".format(", ".join(SAFE_METADRIVE_SWEEP_SCENES)))
    for index, scene in enumerate(SAFE_METADRIVE_SWEEP_SCENES, start=1):
        command = build_safe_metadrive_child_command(script_path, parent_argv, scene)
        print("[SafeMetaDrive][{}/{}] {}".format(index, total, shlex.join(command)))
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    return True


def store_collect_stats(logger, prefix, stats):
    """Store collector metrics using the same keys for PPO and IDM baselines."""
    logger.store(
        **{
            f"{prefix}/reward": stats["rew"],
            f"{prefix}/cost": stats["cost"],
            f"{prefix}/length": int(stats["len"]),
        }
    )
    for key in COLLECTOR_OPTIONAL_LOG_KEYS:
        if key in stats:
            logger.store(**{f"{prefix}/{key}": stats[key]})


def close_envs(*envs):
    for env in envs:
        try:
            env.close()
        except Exception:
            pass


def run_idm_policy_collect(args, agent, train_envs, test_envs, logger):
    """Collect official MetaDrive IDMPolicy rollouts with PPO-compatible logging."""
    train_collector = FastCollector(agent.policy, train_envs)
    test_collector = FastCollector(agent.policy, test_envs)
    train_collector.reset_stat()
    test_collector.reset_stat()

    total_episodes = max(int(args.idm_rollout_episodes), 1)
    collect_batch = max(int(args.episode_per_collect), 1)
    test_interval = max(int(args.test_every_episode), 0)
    next_test_episode = test_interval if test_interval > 0 else None

    env_step = 0
    cum_episode = 0
    cum_cost = 0.0
    start_time = time.time()

    agent.policy.train()
    while cum_episode < total_episodes:
        n_episode = min(collect_batch, total_episodes - cum_episode)
        stats_train = train_collector.collect(n_episode=n_episode)

        env_step += int(stats_train["n/st"])
        cum_episode += int(stats_train["n/ep"])
        cum_cost += float(stats_train["total_cost"])

        logger.store(
            **{
                "update/episode": cum_episode,
                "update/cum_cost": cum_cost,
                "update/env_step": env_step,
            }
        )
        store_collect_stats(logger, "train", stats_train)

        print(
            "[IDMPolicy] episodes={}/{} env_step={} reward={:.3f} cost={:.3f} length={:.1f}".format(
                cum_episode,
                total_episodes,
                env_step,
                float(stats_train["rew"]),
                float(stats_train["cost"]),
                float(stats_train["len"]),
            )
        )

        if next_test_episode is not None and cum_episode >= next_test_episode:
            while next_test_episode is not None and cum_episode >= next_test_episode:
                next_test_episode += test_interval
            agent.policy.eval()
            test_collector.reset_env()
            test_collector.reset_buffer()
            stats_test = test_collector.collect(n_episode=max(int(args.testing_num), 1))
            store_collect_stats(logger, "test", stats_test)
            logger.store(tab="update", test_trigger_episode=cum_episode)
            agent.policy.train()

        duration = max(time.time() - start_time, 1e-9)
        logger.store(
            tab="update",
            duration=duration,
            train_collector_time=train_collector.collect_time,
            test_time=test_collector.collect_time,
            train_speed=train_collector.collect_step / duration,
            test_speed=(
                test_collector.collect_step / test_collector.collect_time
                if test_collector.collect_time > 0
                else 0.0
            ),
        )
        logger.write(step=env_step, display=args.verbose)

    print("\nIDMPolicy collector baseline summary")
    print("====================================")
    print("episodes: {}".format(cum_episode))
    print("env_step: {}".format(env_step))
    print("mean episode cost over all collected batches is logged as train/cost in WandB/progress.txt")


@pyrallis.wrap()  # 使用 pyrallis 装饰器自动解析命令行参数到 TrainCfg
def train(args: TrainCfg):
    """
    训练 PPO-Lag 安全强化学习智能体的主函数

    Args:
        args: 训练配置参数,包含算法超参数、环境设置、日志配置等
    """

    # ===== 更新配置 =====
    # 比较当前参数与默认参数的差异,只保留用户自定义的参数
    cfg, old_cfg = asdict(args), asdict(TrainCfg())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}

    # 根据任务类型加载对应的默认配置
    cfg = asdict(TASK_TO_CFG[args.task]())
    # 用用户自定义的参数覆盖默认配置
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    if maybe_run_safe_metadrive_scene_sweep(args):
        return

    # ===== 设置日志记录器 =====
    default_cfg = asdict(TASK_TO_CFG[args.task]())

    if args.prefix is None:
        args.prefix = resolve_experiment_prefix(args)
        cfg["prefix"] = args.prefix
    cfg["effective_use_risk_field_cost"] = effective_risk_field_enabled(args)
    cfg["effective_risk_field_cost_scale"] = effective_risk_field_scale(args)

    # 自动生成实验名称(如果未指定)
    if args.name is None:
        skip_keys = DEFAULT_SKIP_KEY + [
            "use_lagrangian",
            "use_risk_field_cost",
            "risk_field_cost_scale",
            "use_idm_policy",
            "traffic_density",
            "safe_metadrive_sweep",
            "safe_metadrive_scene",
        ]
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix, skip_keys=skip_keys)
    if args.task == "SafeMetaDrive":
        args.safe_metadrive_scene = normalize_safe_metadrive_scene(args.safe_metadrive_scene)
        args.traffic_density = effective_traffic_density(args)
        args.project = SAFE_METADRIVE_PROJECT
        args.group = safe_metadrive_group(args.safe_metadrive_scene)
        args.name = append_scene_tag_to_run_name(args.name, args.safe_metadrive_scene, args.traffic_density)
        cfg["project"] = args.project
        cfg["group"] = args.group
        cfg["safe_metadrive_scene"] = args.safe_metadrive_scene
        cfg["safe_metadrive_sweep"] = bool(args.safe_metadrive_sweep)
        cfg["traffic_density"] = args.traffic_density

    # 设置实验分组(任务名称 + 成本限制)
    if args.task != "SafeMetaDrive" and args.group is None:
        args.group = args.task + "-cost-" + str(args.cost_limit)

    # 设置日志目录
    if args.logdir is not None:
        path_parts = [args.logdir, args.project]
        if args.group:
            path_parts.append(args.group)
        args.logdir = os.path.join(*path_parts)

    # 创建 WandB logger 用于实验追踪和可视化
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # 也可以使用 TensorBoard logger
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)

    # 保存配置文件
    logger.save_config(cfg, verbose=args.verbose)

    # 创建示例环境(用于初始化智能体)
    if args.task == "SafeMetaDrive":
        demo_env = make_safe_metadrive_scene_env("train", args)
    else:
        demo_env = make_safe_metadrive_env(TRAIN_ENV_CONFIG, args)

    # ===== 创建 PPO-Lag 智能体 =====
    agent = PPOLagAgent(
        env=demo_env,
        logger=logger,
        # ===== 基础配置 =====
        device=args.device,              # 计算设备(cpu/cuda)
        thread=args.thread,              # PyTorch线程数
        seed=args.seed,                  # 随机种子

        # ===== 网络结构配置 =====
        lr=args.lr,                      # 学习率
        hidden_sizes=args.hidden_sizes,  # 隐藏层大小
        unbounded=args.unbounded,        # 是否使用无界动作空间
        last_layer_scale=args.last_layer_scale,  # 最后一层权重缩放

        # ===== PPO 算法超参数 =====
        target_kl=args.target_kl,        # 目标 KL 散度(用于早停)
        vf_coef=args.vf_coef,            # 价值函数损失系数
        max_grad_norm=args.max_grad_norm,  # 梯度裁剪范数
        gae_lambda=args.gae_lambda,      # GAE(广义优势估计)的lambda参数
        eps_clip=args.eps_clip,          # PPO 裁剪范围
        dual_clip=args.dual_clip,        # 双重裁剪阈值
        value_clip=args.value_clip,      # 价值函数裁剪
        advantage_normalization=args.norm_adv,  # 是否标准化优势函数
        recompute_advantage=args.recompute_adv, # 是否重新计算优势

        # ===== 拉格朗日方法配置 =====
        use_lagrangian=args.use_lagrangian,   # 是否使用拉格朗日乘子法
        lagrangian_pid=args.lagrangian_pid,   # PID 控制器参数(用于调整拉格朗日乘子)
        cost_limit=args.cost_limit,           # 成本约束上限
        rescaling=args.rescaling,             # 成本重缩放

        # ===== 其他配置 =====
        gamma=args.gamma,                # 折扣因子
        max_batchsize=args.max_batchsize,  # 最大批次大小
        reward_normalization=args.rew_norm,  # 是否标准化奖励
        deterministic_eval=args.deterministic_eval,  # 评估时是否使用确定性策略
        action_scaling=args.action_scaling,  # 动作缩放
        action_bound_method=args.action_bound_method,  # 动作边界处理方法
    )

    # ===== 准备并行训练环境 =====
    # 训练进程数取配置值和每轮收集episode数的较小值
    training_num = min(args.training_num, args.episode_per_collect)

    # 选择环境worker类型(通过eval解析字符串)
    worker = eval(args.worker)

    # 创建训练环境向量(并行运行多个环境实例)
    if args.task == "SafeMetaDrive":
        train_envs = worker([lambda: make_safe_metadrive_scene_env("train", args) for _ in range(training_num)])
    else:
        train_envs = worker([lambda: make_safe_metadrive_env(TRAIN_ENV_CONFIG, args) for _ in range(training_num)])

    # 创建测试/验证环境向量(并行运行多个环境实例)
    if args.task == "SafeMetaDrive":
        test_envs = worker([lambda: make_safe_metadrive_scene_env("val", args) for _ in range(args.testing_num)])
    else:
        test_envs = worker([lambda: make_safe_metadrive_env(VALIDATION_ENV_CONFIG, args) for _ in range(args.testing_num)])

    # 创建用于生成 BEV GIF 的专用环境工厂。
    # 这里直接返回原生 MetaDrive env，避免 Gym wrapper 拦截自定义的
    # render(mode="topdown", ...) 调用。
    def artifact_env_factory():
        if args.task == "SafeMetaDrive":
            return make_safe_metadrive_scene_env("val", args, artifact=True)
        return make_safe_metadrive_env(VALIDATION_ENV_CONFIG, args, artifact=True)

    if getattr(args, "use_idm_policy", False):
        print(
            "[IDMPolicy] SafeMetaDrive ego is controlled by MetaDrive official IDMPolicy. "
            "FastCollector will log PPO-compatible train/test metrics, but PPO updates and checkpoints are skipped."
        )
        try:
            run_idm_policy_collect(args, agent, train_envs, test_envs, logger)
        finally:
            close_envs(train_envs, test_envs, demo_env)
        return

    # ===== 开始训练 =====
    agent.learn(
        train_envs=train_envs,           # 训练环境
        test_envs=test_envs,             # 测试环境
        epoch=args.epoch,                # 训练总轮数
        episode_per_collect=args.episode_per_collect,  # 每轮收集的episode数
        step_per_epoch=args.step_per_epoch,  # 每轮的步数
        repeat_per_collect=args.repeat_per_collect,  # 每次收集后重复更新次数
        buffer_size=args.buffer_size,    # 经验回放缓冲区大小
        testing_num=args.testing_num,    # 测试环境数量
        batch_size=args.batch_size,      # 训练批次大小
        reward_threshold=args.reward_threshold,  # 奖励阈值(用于早停)
        save_interval=args.save_interval,  # 模型保存间隔
        test_every_episode=args.test_every_episode,  # 周期测试间隔
        save_test_artifacts=args.save_test_artifacts,  # 是否保存测试产物
        artifact_env_factory=artifact_env_factory,  # 用于生成调试 GIF 的测试环境
        resume=args.resume,              # 是否从检查点恢复
        save_ckpt=args.save_ckpt,        # 是否保存检查点
        verbose=args.verbose,            # 是否显示详细日志
    )

    # ===== 训练完成后评估性能 =====
    if __name__ == "__main__":
        from fsrl.data import FastCollector

        # 创建验证环境
        if args.task == "SafeMetaDrive":
            env = make_safe_metadrive_scene_env("val", args)
        else:
            env = make_safe_metadrive_env(VALIDATION_ENV_CONFIG, args)

        # 设置为评估模式(使用确定性策略)
        agent.policy.eval()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final eval reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")

        # 设置为训练模式(使用随机策略)
        agent.policy.train()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final train reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")


if __name__ == "__main__":
    train()
