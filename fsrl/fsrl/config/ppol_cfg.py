from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainCfg:
    # general task params
    task: str = "SafetyCarCircle-v0"
    cost_limit: float = 1
    # device: str = "cpu"
    device: str = "cuda"
    thread: int = 8  # if use "cpu" to train
    seed: int = 10
    # algorithm params
    lr: float = 5e-4
    hidden_sizes: Tuple[int, ...] = (128, 128)
    unbounded: bool = False
    last_layer_scale: bool = False
    # PPO specific arguments
    target_kl: float = 0.02
    vf_coef: float = 0.25
    max_grad_norm: Optional[float] = 0.5
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    dual_clip: Optional[float] = None
    value_clip: bool = False  # no need
    norm_adv: bool = True  # good for improving training stability
    recompute_adv: bool = False
    # Lagrangian specific arguments
    # use_lagrangian: bool = True
    use_lagrangian: bool = False
    lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1)
    rescaling: bool = True
    # Base policy common arguments
    gamma: float = 0.99
    max_batchsize: int = 100000
    rew_norm: bool = False  # no need, it will slow down training and decrease final perf
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: str = "clip"
    # collecting params
    epoch: int = 200
    episode_per_collect: int = 20
    step_per_epoch: int = 10000
    repeat_per_collect: int = 4  # increasing this can improve efficiency, but less stability
    buffer_size: int = 100000
    worker: str = "ShmemVectorEnv"
    training_num: int = 50
    testing_num: int = 50
    # general params
    batch_size: int = 256
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 4
    test_every_episode: int = 100
    save_test_artifacts: bool = True
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = True
    render: bool = False
    # MetaDrive risk-field experiment switches.
    # None means following env.py DEFAULT_CONFIG; set True/False to override per run.
    use_risk_field_cost: Optional[bool] = None
    risk_field_cost_scale: Optional[float] = None
    # MetaDrive debug/baseline policy switch
    # use_idm_policy: bool = True  # if True, MetaDrive ego is controlled by official IDMPolicy instead of PPO actions
    use_idm_policy: bool = False  # if True, MetaDrive ego is controlled by official IDMPolicy instead of PPO actions
    disable_idm_lane_change: bool = False
    disable_idm_deceleration: bool = False
    idm_rollout_episodes: int = 1000
    idm_rollout_max_steps: Optional[int] = None
    # SafeMetaDrive multi-scene orchestration.
    # When task == SafeMetaDrive, default to sweeping mixed + 5 single scenes.
    safe_metadrive_sweep: bool = False  # ⭐ 关闭多场景扫描，只训练单一场景
    # safe_metadrive_sweep: bool = True  # ⭐ 关闭多场景扫描，只训练单一场景
    safe_metadrive_scene: str = "mixed_default"  # ⭐ 指定训练场景为混合场景
    traffic_density: Optional[float] = None  # None 表示使用 safe_metadrive_adapter/config.py 中的默认车辆密度
    # logger params
    logdir: str = "logs"
    project: str = "metadrive-Reward"
    group: Optional[str] = None
    name: Optional[str] = None
    # ===== 实验命名前缀规范 =====
    # PPO: 普通PPO，无Lagrange约束，env.py中关闭风险场cost
    # PPO-RISK-LOG: 普通PPO，无Lagrange约束，env.py中开启风险场，只用于记录风险指标
    # PPOL: PPO-Lagrangian 默认前缀
    # PPOL-RISK: PPO-Lagrangian + RiskField cost
    # IDM: MetaDrive官方IDM策略baseline，不更新PPO
    # None means generated automatically:
    # PPOL / PPOL-RISK / PPO / PPO-RISK / IDM
    prefix: Optional[str] = None
    suffix: Optional[str] = "-d0.05"


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
    # collecting params
    episode_per_collect: int = 20
    step_per_epoch: int = 20000
    repeat_per_collect: int = 4  # increasing this can improve efficiency, but less stability


@dataclass
class Mujoco2MCfg(MujocoBaseCfg):
    epoch: int = 100


@dataclass
class Mujoco20MCfg(MujocoBaseCfg):
    epoch: int = 1000


@dataclass
class Mujoco10MCfg(MujocoBaseCfg):
    epoch: int = 500
