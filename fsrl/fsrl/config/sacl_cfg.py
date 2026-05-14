from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainCfg:
    # general task params
    task: str = "SafetyCarCircle-v0"
    cost_limit: float = 1
    # device: str = "cpu"
    device: str = "cuda"
    thread: int = 8
    seed: int = 10
    # SAC arguments
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    hidden_sizes: Tuple[int, ...] = (128, 128)
    auto_alpha: bool = True
    alpha_lr: float = 3e-4
    alpha: float = 0.005
    tau: float = 0.05
    n_step: int = 2
    conditioned_sigma: bool = True
    unbounded: bool = False
    last_layer_scale: bool = False
    # Lagrangian specific arguments
    # use_lagrangian: bool = True
    use_lagrangian: bool = False
    lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1)
    rescaling: bool = True
    # Base policy common arguments
    gamma: float = 0.97
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: str = "clip"
    # collecting params
    epoch: int = 100
    episode_per_collect: int = 2
    step_per_epoch: int = 10000
    update_per_step: float = 0.2
    buffer_size: int = 100000
    worker: str = "ShmemVectorEnv"
    training_num: int = 50
    testing_num: int = 50
    # general train params
    batch_size: int = 256
    reward_threshold: float = 10000
    save_interval: int = 4
    test_every_episode: int = 2000
    save_test_artifacts: bool = True
    resume: bool = False
    save_ckpt: bool = True
    verbose: bool = True
    render: bool = False
    # MetaDrive risk-field experiment switches.
    # None means following env.py DEFAULT_CONFIG; set True/False to override per run.
    use_risk_field_cost: Optional[bool] = None
    risk_field_cost_scale: Optional[float] = None
    use_risk_field_reward: Optional[bool] = None
    risk_field_reward_scale: Optional[float] = None
    # SafeMetaDrive multi-scene orchestration.
    # When task == SafeMetaDrive, default to sweeping mixed + 5 single scenes.
    safe_metadrive_sweep: bool = False  # ⭐ 关闭多场景扫描，只训练单一场景
    # safe_metadrive_sweep: bool = True  # ⭐ 开启多场景扫描，训练多场景
    safe_metadrive_scene: str = "mixed_default"  # ⭐ 指定训练场景为混合场景
    # logger params
    logdir: str = "logs"
    project: str = "metadrive-Reward-step2"
    group: Optional[str] = None
    name: Optional[str] = None
    # ===== 实验命名前缀规范 =====
    # SAC: 普通SAC，无Lagrange约束，env.py中关闭风险场cost
    # SAC-RISK-LOG: 普通SAC，无Lagrange约束，env.py中开启风险场，只用于记录风险指标
    # SACL: SAC-Lagrangian 默认前缀
    # SACL-RISK: SAC-Lagrangian + RiskField cost
    # None means generated automatically:
    # SACL / SACL-RISK / SAC / SAC-RISK
    prefix: Optional[str] = None
    suffix: Optional[str] = ""


# bullet-safety-gym task default configs


@dataclass
class Bullet1MCfg(TrainCfg):
    epoch: int = 100


@dataclass
class Bullet5MCfg(TrainCfg):
    epoch: int = 500


@dataclass
class Bullet10MCfg(TrainCfg):
    epoch: int = 1000


# safety gymnasium task default configs


@dataclass
class MujocoBaseCfg(TrainCfg):
    task: str = "SafetyPointCircle1Gymnasium-v0"
    epoch: int = 250
    cost_limit: float = 25
    gamma: float = 0.99
    n_step: int = 3
    # collecting params
    step_per_epoch: int = 20000
    episode_per_collect: int = 5
    buffer_size: int = 800000


@dataclass
class Mujoco2MCfg(MujocoBaseCfg):
    epoch: int = 100


@dataclass
class Mujoco20MCfg(MujocoBaseCfg):
    epoch: int = 1000


@dataclass
class Mujoco10MCfg(MujocoBaseCfg):
    epoch: int = 500
