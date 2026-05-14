"""
SAC-Lag + ResAct residual-action training entrypoint for SafeMetaDrive.

This script keeps the existing SACL policy, collectors, replay buffer, and env reward/cost
semantics untouched. It only wraps MetaDrive env instances so the policy output is
interpreted as a residual action.
"""
import importlib
import os
import shlex
import subprocess
import sys
import types
from dataclasses import asdict

import gymnasium as gym
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

import train_sacl as sacl_base
from env import DEFAULT_CONFIG
from fsrl.fsrl.agent import SACLagAgent
from fsrl.fsrl.config.sacl_resact_cfg import (
    Bullet1MCfg,
    Bullet5MCfg,
    Bullet10MCfg,
    Mujoco2MCfg,
    Mujoco10MCfg,
    Mujoco20MCfg,
    MujocoBaseCfg,
    TrainCfg,
)
from fsrl.fsrl.utils import WandbLogger
from fsrl.fsrl.utils.exp_util import DEFAULT_SKIP_KEY, auto_name
from resact_metadrive import RESACT_INFO_KEYS, wrap_residual_action_env
from safe_metadrive_sweep import (
    SAFE_METADRIVE_PROJECT,
    SAFE_METADRIVE_SWEEP_SCENES,
    append_scene_tag_to_run_name,
    build_safe_metadrive_child_command,
    normalize_safe_metadrive_scene,
    safe_metadrive_group,
)


TASK_TO_CFG = {
    "SafeMetaDrive": TrainCfg,
    "SafetyCarRun-v0": Bullet1MCfg,
    "SafetyBallRun-v0": Bullet1MCfg,
    "SafetyBallCircle-v0": Bullet1MCfg,
    "SafetyCarCircle-v0": TrainCfg,
    "SafetyDroneRun-v0": TrainCfg,
    "SafetyAntRun-v0": TrainCfg,
    "SafetyDroneCircle-v0": Bullet5MCfg,
    "SafetyAntCircle-v0": Bullet10MCfg,
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
    return sacl_base._compact_float_tag(value)


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
    risk_enabled = effective_risk_field_enabled(args)
    if getattr(args, "use_lagrangian", True):
        prefix = "SACL-RA-RISK" if risk_enabled else "SACL-RA"
    else:
        prefix = "SAC-RA-RISK" if risk_enabled else "SAC-RA"
    return prefix


def resact_suffix(args):
    if not getattr(args, "resact_enabled", True):
        return "no-resact"
    steer = _compact_float_tag(getattr(args, "resact_steer_delta_scale", 0.15))
    throttle = _compact_float_tag(getattr(args, "resact_throttle_delta_scale", 0.10))
    return f"ra{steer}-{throttle}"


def ensure_resact_suffix(args, cfg):
    tag = resact_suffix(args)
    suffix = getattr(args, "suffix", None) or ""
    if tag not in suffix:
        suffix = f"{suffix}-{tag}" if suffix else f"-{tag}"
    args.suffix = suffix
    cfg["suffix"] = suffix


def extend_metric_allowlists():
    info_keys = tuple(RESACT_INFO_KEYS)
    optional_keys = info_keys + tuple(f"{key}_step_mean" for key in info_keys)

    for module_name in ("fsrl.data.fast_collector", "fsrl.fsrl.data.fast_collector"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        current = tuple(getattr(module, "INFO_COST_KEYS", ()))
        module.INFO_COST_KEYS = current + tuple(key for key in info_keys if key not in current)

    for module_name in ("fsrl.trainer.base_trainer", "fsrl.fsrl.trainer.base_trainer"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        current = tuple(getattr(module, "COLLECTOR_OPTIONAL_LOG_KEYS", ()))
        module.COLLECTOR_OPTIONAL_LOG_KEYS = current + tuple(key for key in optional_keys if key not in current)


def wrap_env_for_resact(env: gym.Env, args) -> gym.Env:
    return wrap_residual_action_env(
        env,
        resact_enabled=bool(getattr(args, "resact_enabled", True)),
        resact_steer_delta_scale=float(getattr(args, "resact_steer_delta_scale", 0.15)),
        resact_throttle_delta_scale=float(getattr(args, "resact_throttle_delta_scale", 0.10)),
        resact_initial_action=getattr(args, "resact_initial_action", (0.0, 0.0)),
    )


def make_safe_metadrive_scene_resact_env(split: str, args, *, artifact=False):
    env = sacl_base.make_safe_metadrive_scene_env(split, args, artifact=artifact)
    return wrap_env_for_resact(env, args)


def maybe_run_safe_metadrive_scene_sweep(args):
    if args.task != "SafeMetaDrive" or not bool(getattr(args, "safe_metadrive_sweep", False)):
        return False

    script_path = os.path.abspath(__file__)
    parent_argv = sys.argv[1:]
    total = len(SAFE_METADRIVE_SWEEP_SCENES)
    print("SafeMetaDrive ResAct scene sweep: {}".format(", ".join(SAFE_METADRIVE_SWEEP_SCENES)))
    for index, scene in enumerate(SAFE_METADRIVE_SWEEP_SCENES, start=1):
        command = build_safe_metadrive_child_command(script_path, parent_argv, scene)
        print("[SafeMetaDrive-ResAct][{}/{}] {}".format(index, total, shlex.join(command)))
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    return True


@pyrallis.wrap()
def train(args: TrainCfg):
    extend_metric_allowlists()

    cfg, old_cfg = asdict(args), asdict(TrainCfg())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(TASK_TO_CFG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    if args.task != "SafeMetaDrive":
        raise ValueError("train_sacl_resact.py is intended for task=SafeMetaDrive.")

    ensure_resact_suffix(args, cfg)

    if maybe_run_safe_metadrive_scene_sweep(args):
        return

    default_cfg = asdict(TASK_TO_CFG[args.task]())

    if args.prefix is None:
        args.prefix = resolve_experiment_prefix(args)
        cfg["prefix"] = args.prefix
    cfg["effective_use_risk_field_cost"] = effective_risk_field_enabled(args)
    cfg["effective_risk_field_cost_scale"] = effective_risk_field_scale(args)
    cfg["effective_use_risk_field_reward"] = effective_risk_field_reward_enabled(args)
    cfg["effective_risk_field_reward_scale"] = effective_risk_field_reward_scale(args)
    cfg["effective_resact_enabled"] = bool(args.resact_enabled)

    if args.name is None:
        skip_keys = DEFAULT_SKIP_KEY + [
            "use_lagrangian",
            "use_risk_field_cost",
            "risk_field_cost_scale",
            "safe_metadrive_sweep",
            "safe_metadrive_scene",
            "resact_enabled",
            "resact_steer_delta_scale",
            "resact_throttle_delta_scale",
            "resact_initial_action",
        ]
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix, skip_keys=skip_keys)

    args.safe_metadrive_scene = normalize_safe_metadrive_scene(args.safe_metadrive_scene)
    args.project = SAFE_METADRIVE_PROJECT
    args.group = safe_metadrive_group(args.safe_metadrive_scene)
    args.name = append_scene_tag_to_run_name(args.name, args.safe_metadrive_scene)
    cfg["project"] = args.project
    cfg["group"] = args.group
    cfg["safe_metadrive_scene"] = args.safe_metadrive_scene
    cfg["safe_metadrive_sweep"] = bool(args.safe_metadrive_sweep)

    if args.logdir is not None:
        path_parts = [args.logdir, args.project]
        if args.group:
            path_parts.append(args.group)
        args.logdir = os.path.join(*path_parts)

    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    demo_env = make_safe_metadrive_scene_resact_env("train", args)
    agent = SACLagAgent(
        env=demo_env,
        logger=logger,
        device=args.device,
        thread=args.thread,
        seed=args.seed,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_sizes=args.hidden_sizes,
        auto_alpha=args.auto_alpha,
        alpha_lr=args.alpha_lr,
        alpha=args.alpha,
        tau=args.tau,
        n_step=args.n_step,
        use_lagrangian=args.use_lagrangian,
        lagrangian_pid=args.lagrangian_pid,
        cost_limit=args.cost_limit,
        rescaling=args.rescaling,
        gamma=args.gamma,
        conditioned_sigma=args.conditioned_sigma,
        unbounded=args.unbounded,
        last_layer_scale=args.last_layer_scale,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        lr_scheduler=None,
    )

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)
    train_envs = worker([
        lambda: make_safe_metadrive_scene_resact_env("train", args)
        for _ in range(training_num)
    ])
    test_envs = worker([
        lambda: make_safe_metadrive_scene_resact_env("val", args)
        for _ in range(args.testing_num)
    ])

    def artifact_env_factory():
        return make_safe_metadrive_scene_resact_env("val", args, artifact=True)

    try:
        agent.learn(
            train_envs=train_envs,
            test_envs=test_envs,
            epoch=args.epoch,
            episode_per_collect=args.episode_per_collect,
            step_per_epoch=args.step_per_epoch,
            update_per_step=args.update_per_step,
            buffer_size=args.buffer_size,
            testing_num=args.testing_num,
            batch_size=args.batch_size,
            reward_threshold=args.reward_threshold,
            save_interval=args.save_interval,
            test_every_episode=args.test_every_episode,
            save_test_artifacts=args.save_test_artifacts,
            artifact_env_factory=artifact_env_factory,
            resume=args.resume,
            save_ckpt=args.save_ckpt,
            verbose=args.verbose,
        )
    finally:
        for env in (demo_env, train_envs, test_envs):
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    train()
