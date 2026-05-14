"""Backward-compatible SafeMetaDrive entrypoint.

New code should prefer importing from ``safe_metadrive_adapter`` directly. This
module keeps the historical project API intact for existing training,
evaluation, and debugging scripts.
"""

from safe_metadrive_adapter import (
    DEFAULT_CONFIG,
    TRAINING_CONFIG,
    VALIDATION_CONFIG,
    SafeMetaDriveAdapterEnv,
    SafeMetaDriveEnv_mini,
    get_training_env,
    get_validation_env,
    make_env_factory,
    make_safe_metadrive_env,
)
from safe_metadrive_adapter.env import SafeMetaDriveEnv

SafeMetaDriveSingleSceneEnv = None
build_single_scene_config = None


def get_single_scene_training_env(*args, **kwargs):
    raise RuntimeError("Single-scene helpers are not available in this adapter build.")


def get_single_scene_validation_env(*args, **kwargs):
    raise RuntimeError("Single-scene helpers are not available in this adapter build.")

__all__ = [
    "DEFAULT_CONFIG",
    "TRAINING_CONFIG",
    "VALIDATION_CONFIG",
    "SafeMetaDriveEnv",
    "SafeMetaDriveAdapterEnv",
    "SafeMetaDriveEnv_mini",
    "get_training_env",
    "get_validation_env",
    "make_env_factory",
    "make_safe_metadrive_env",
    "SafeMetaDriveSingleSceneEnv",
    "build_single_scene_config",
    "get_single_scene_training_env",
    "get_single_scene_validation_env",
]


if __name__ == "__main__":
    env = get_training_env(
        {
            "manual_control": True,
            "use_render": True,
        }
    )
    try:
        env.reset()
        env.engine.toggle_help_message()
        while True:
            _, _, terminated, truncated, _ = env.step([0, 0])
            env.render(mode="topdown", target_agent_heading_up=True)
            if terminated or truncated:
                env.reset()
    finally:
        env.close()
