"""
SAC-Lag (Soft Actor-Critic with Lagrangian) 安全强化学习算法训练脚本

该脚本用于在 SafeMetaDrive 环境中训练基于拉格朗日乘子法的 SAC 安全强化学习智能体。
SAC-Lag 是一种 off-policy 算法，结合了最大熵强化学习和拉格朗日约束优化方法，
能够在保证安全性的同时实现高效的样本利用和稳定的训练过程。

使用方法:
    python3 train_sacl.py --task SafeMetaDrive

可选参数可通过命令行传入，或在代码中修改配置。
"""
import os
import shlex
import subprocess
import sys
import types
import importlib
from dataclasses import asdict

try:
    import bullet_safety_gym  # noqa: F401
except ImportError:
    bullet_safety_gym = None

# try:
#     import safety_gymnasium
# except ImportError:
#     print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from safe_metadrive_adapter.local_import import prefer_local_metadrive

prefer_local_metadrive()

# SAFE_METADRIVE_ENV_MODULE = os.environ.get("SAFE_METADRIVE_ENV_MODULE", "env_copy")
SAFE_METADRIVE_ENV_MODULE = os.environ.get("SAFE_METADRIVE_ENV_MODULE", "env")
_safe_metadrive_env_module = importlib.import_module(SAFE_METADRIVE_ENV_MODULE)
DEFAULT_CONFIG = _safe_metadrive_env_module.DEFAULT_CONFIG
SafeMetaDriveEnv_mini = _safe_metadrive_env_module.SafeMetaDriveEnv_mini
SafeMetaDriveSingleSceneEnv = getattr(_safe_metadrive_env_module, "SafeMetaDriveSingleSceneEnv", None)
build_single_scene_config = getattr(_safe_metadrive_env_module, "build_single_scene_config", None)
from fsrl.fsrl.agent import SACLagAgent
from fsrl.fsrl.config.sacl_cfg import (
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

TRAIN_ENV_CONFIG = {"num_scenarios": 50, "start_seed": 1000}
VALIDATION_ENV_CONFIG = {"num_scenarios": 50, "start_seed": 1000}

# ===== 注册自定义 SafeMetaDrive 环境 =====
# 注册训练环境：使用50个场景，种子范围[1000, 1050)
gym.register(
    id="SafeMetaDrive-training",
    entry_point="env:SafeMetaDriveEnv_mini",  # 使用 env.py 中定义的简化环境类
    max_episode_steps=1000,  # 每个episode最大步数
    kwargs={"config": dict(TRAIN_ENV_CONFIG)},  # 环境配置
)

# 注册验证环境：使用相同的场景数量但不同的种子范围
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


def effective_risk_field_reward_enabled(args):
    value = getattr(args, "use_risk_field_reward", None)
    if value is None:
        return bool(DEFAULT_CONFIG.get("use_risk_field_reward", False))
    return bool(value)


def effective_risk_field_reward_scale(args):
    value = getattr(args, "risk_field_reward_scale", None)
    if value is None:
        return float(DEFAULT_CONFIG.get("risk_field_reward_scale", 0.15))
    return float(value)


def resolve_experiment_prefix(args):
    """Return table-aligned SAC/SACL experiment labels."""
    risk_enabled = effective_risk_field_enabled(args)
    if getattr(args, "use_lagrangian", True):
        prefix = "SACL-RISK" if risk_enabled else "SACL"
    else:
        prefix = "SAC-RISK" if risk_enabled else "SAC"
    return prefix


def build_safe_metadrive_config(base_config, args, *, artifact=False):
    """Build a MetaDrive config using training-script risk-field overrides."""
    config = dict(base_config)
    if getattr(args, "use_risk_field_cost", None) is not None:
        config["use_risk_field_cost"] = bool(args.use_risk_field_cost)
    if getattr(args, "risk_field_cost_scale", None) is not None:
        config["risk_field_cost_scale"] = float(args.risk_field_cost_scale)
    if getattr(args, "use_risk_field_reward", None) is not None:
        config["use_risk_field_reward"] = bool(args.use_risk_field_reward)
    if getattr(args, "risk_field_reward_scale", None) is not None:
        config["risk_field_reward_scale"] = float(args.risk_field_reward_scale)
    for key in (
        "resact_enabled",
        "resact_steer_delta_scale",
        "resact_throttle_delta_scale",
        "resact_initial_action",
    ):
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
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
            "SafeMetaDrive env module '{}' does not provide single-scene helpers. "
            "Use --safe_metadrive_scene mixed_default to reproduce the old env behavior, "
            "or switch SAFE_METADRIVE_ENV_MODULE/env import back to 'env' for single-scene experiments.".format(
                SAFE_METADRIVE_ENV_MODULE
            )
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


@pyrallis.wrap()  # 使用 pyrallis 装饰器自动解析命令行参数到 TrainCfg
def train(args: TrainCfg):
    """
    训练 SAC-Lag 安全强化学习智能体的主函数

    Args:
        args: 训练配置参数，包含算法超参数、环境设置、日志配置等
    """

    # ===== 更新配置 =====
    # 比较当前参数与默认参数的差异，只保留用户自定义的参数
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
    cfg["effective_use_risk_field_reward"] = effective_risk_field_reward_enabled(args)
    cfg["effective_risk_field_reward_scale"] = effective_risk_field_reward_scale(args)
    cfg["safe_metadrive_env_module"] = SAFE_METADRIVE_ENV_MODULE

    # 自动生成实验名称（如果未指定）
    if args.name is None:
        skip_keys = DEFAULT_SKIP_KEY + [
            "use_lagrangian",
            "use_risk_field_cost",
            "risk_field_cost_scale",
            "safe_metadrive_sweep",
            "safe_metadrive_scene",
        ]
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix, skip_keys=skip_keys)
    if args.task == "SafeMetaDrive":
        args.safe_metadrive_scene = normalize_safe_metadrive_scene(args.safe_metadrive_scene)
        args.project = SAFE_METADRIVE_PROJECT
        args.group = safe_metadrive_group(args.safe_metadrive_scene)
        args.name = append_scene_tag_to_run_name(args.name, args.safe_metadrive_scene)
        cfg["project"] = args.project
        cfg["group"] = args.group
        cfg["safe_metadrive_scene"] = args.safe_metadrive_scene
        cfg["safe_metadrive_sweep"] = bool(args.safe_metadrive_sweep)

    # 设置实验分组（任务名称 + 成本限制）
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

    # 创建示例环境（用于初始化智能体）
    if args.task == "SafeMetaDrive":
        demo_env = make_safe_metadrive_scene_env("train", args)
    else:
        demo_env = make_safe_metadrive_env(TRAIN_ENV_CONFIG, args)

    # ===== 创建 SAC-Lag 智能体 =====
    agent = SACLagAgent(
        env=demo_env,
        logger=logger,

        # ===== 基础配置 =====
        device=args.device,              # 计算设备(cpu/cuda)
        thread=args.thread,              # PyTorch线程数
        seed=args.seed,                  # 随机种子

        # ===== SAC 算法超参数 =====
        actor_lr=args.actor_lr,          # Actor网络学习率
        critic_lr=args.critic_lr,        # Critic网络学习率
        hidden_sizes=args.hidden_sizes,  # 隐藏层大小
        auto_alpha=args.auto_alpha,      # 是否自动调整温度系数alpha
        alpha_lr=args.alpha_lr,          # Alpha学习率
        alpha=args.alpha,                # 初始温度系数（控制探索程度）
        tau=args.tau,                    # 目标网络软更新系数
        n_step=args.n_step,              # N步回报

        # ===== 拉格朗日方法配置 =====
        use_lagrangian=args.use_lagrangian,   # 是否使用拉格朗日乘子法
        lagrangian_pid=args.lagrangian_pid,   # PID控制器参数（用于调整拉格朗日乘子）
        cost_limit=args.cost_limit,           # 成本约束上限
        rescaling=args.rescaling,             # 成本重缩放

        # ===== 策略网络配置 =====
        gamma=args.gamma,                # 折扣因子
        conditioned_sigma=args.conditioned_sigma,  # sigma是否依赖于状态
        unbounded=args.unbounded,        # 是否使用无界动作空间
        last_layer_scale=args.last_layer_scale,  # 最后一层权重缩放
        deterministic_eval=args.deterministic_eval,  # 评估时是否使用确定性策略
        action_scaling=args.action_scaling,  # 动作缩放
        action_bound_method=args.action_bound_method,  # 动作边界处理方法
        lr_scheduler=None                # 学习率调度器（此处未使用）
    )

    # ===== 准备并行训练环境 =====
    # 训练进程数取配置值和每轮收集episode数的较小值
    training_num = min(args.training_num, args.episode_per_collect)

    # 选择环境worker类型（通过eval解析字符串）
    worker = eval(args.worker)

    # 创建训练环境向量（并行运行多个环境实例）
    if args.task == "SafeMetaDrive":
        train_envs = worker([lambda: make_safe_metadrive_scene_env("train", args) for _ in range(training_num)])
    else:
        train_envs = worker([lambda: make_safe_metadrive_env(TRAIN_ENV_CONFIG, args) for _ in range(training_num)])

    # 创建测试/验证环境向量
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

    # ===== 开始训练 =====
    agent.learn(
        train_envs=train_envs,           # 训练环境
        test_envs=test_envs,             # 测试环境
        epoch=args.epoch,                # 训练总轮数
        episode_per_collect=args.episode_per_collect,  # 每轮收集的episode数
        step_per_epoch=args.step_per_epoch,  # 每轮的步数
        update_per_step=args.update_per_step,  # 每步更新次数（off-policy特性）
        buffer_size=args.buffer_size,    # 经验回放缓冲区大小
        testing_num=args.testing_num,    # 测试环境数量
        batch_size=args.batch_size,      # 训练批次大小
        reward_threshold=args.reward_threshold,  # 奖励阈值（用于早停）
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
            env = gym.make('SafeMetaDrive-validation')

        # 设置为评估模式（使用确定性策略）
        agent.policy.eval()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final eval reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")

        # 设置为训练模式（使用随机策略）
        agent.policy.train()
        collector = FastCollector(agent.policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final train reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")


if __name__ == "__main__":
    train()
