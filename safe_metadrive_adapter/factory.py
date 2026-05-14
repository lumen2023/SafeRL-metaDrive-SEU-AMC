"""Factory helpers for training, validation, and custom algorithm integration."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .config import TRAINING_CONFIG, VALIDATION_CONFIG, merged_config
from .env import SafeMetaDriveAdapterEnv, SafeMetaDriveEnv_mini
from .resact import wrap_residual_action_env


def _normalize_resact_config(resact: Dict[str, Any] | bool | None) -> Dict[str, Any]:
    if resact is None or resact is False:
        return {"enabled": False}
    if resact is True:
        return {"enabled": True}
    normalized = dict(resact)
    if "enabled" not in normalized:
        normalized["enabled"] = bool(normalized.get("resact_enabled", True))
    return normalized


def _apply_resact(env, resact: Dict[str, Any] | bool | None):
    options = _normalize_resact_config(resact)
    if not bool(options.get("enabled", False)):
        return env
    return wrap_residual_action_env(
        env,
        resact_enabled=True,
        resact_steer_delta_scale=float(options.get("steer_delta_scale", options.get("resact_steer_delta_scale", 0.15))),
        resact_throttle_delta_scale=float(
            options.get("throttle_delta_scale", options.get("resact_throttle_delta_scale", 0.10))
        ),
        resact_initial_action=options.get("initial_action", options.get("resact_initial_action", (0.0, 0.0))),
    )


def make_safe_metadrive_env(
    split: str = "train",
    config: Dict[str, Any] | None = None,
    resact: Dict[str, Any] | bool | None = None,
):
    split_key = str(split).lower()
    if split_key in {"train", "training"}:
        base_config = TRAINING_CONFIG
    elif split_key in {"val", "valid", "validation", "eval", "test"}:
        base_config = VALIDATION_CONFIG
    else:
        raise ValueError("split must be one of train/training or val/validation/eval/test")
    env_config = merged_config(base_config, config)
    if resact is None and bool(env_config.get("resact_enabled", False)):
        resact = {
            "enabled": True,
            "resact_steer_delta_scale": env_config.get("resact_steer_delta_scale", 0.15),
            "resact_throttle_delta_scale": env_config.get("resact_throttle_delta_scale", 0.10),
            "resact_initial_action": env_config.get("resact_initial_action", (0.0, 0.0)),
        }
    env = SafeMetaDriveAdapterEnv(env_config)
    return _apply_resact(env, resact)


def make_env_factory(
    split: str = "train",
    config: Dict[str, Any] | None = None,
    resact: Dict[str, Any] | bool | None = None,
) -> Callable[[], Any]:
    def factory():
        return make_safe_metadrive_env(split=split, config=config, resact=resact)

    return factory


def get_training_env(extra_config: Dict[str, Any] | None = None):
    return make_safe_metadrive_env(split="train", config=extra_config)


def get_validation_env(extra_config: Dict[str, Any] | None = None):
    return make_safe_metadrive_env(split="val", config=extra_config)


__all__ = [
    "SafeMetaDriveAdapterEnv",
    "SafeMetaDriveEnv_mini",
    "make_safe_metadrive_env",
    "make_env_factory",
    "get_training_env",
    "get_validation_env",
]
