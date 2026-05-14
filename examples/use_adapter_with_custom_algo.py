"""Minimal example for using the portable SafeMetaDrive adapter in any algorithm."""

from __future__ import annotations

import numpy as np

from safe_metadrive_adapter import make_safe_metadrive_env


def main() -> None:
    env = make_safe_metadrive_env(
        split="train",
        config={
            "log_level": 50,
            "use_render": False,
            "traffic_density": 0.05,
            "use_risk_field_reward": True,
        },
        resact={
            "enabled": True,
            "steer_delta_scale": 0.8,
            "throttle_delta_scale": 1.0,
        },
    )
    try:
        obs, info = env.reset()
        done = False
        while not done:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            print(
                "reward={:.3f} cost={:.3f} risk={:.3f} resact_steer={:.3f}".format(
                    float(reward),
                    float(info.get("cost", 0.0)),
                    float(info.get("risk_field_cost", 0.0)),
                    float(info.get("resact_executed_steer", 0.0)),
                )
            )
            break
    finally:
        env.close()


if __name__ == "__main__":
    main()
