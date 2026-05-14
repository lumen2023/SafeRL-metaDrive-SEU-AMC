from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np


RESACT_INFO_KEYS: Tuple[str, ...] = (
    "resact_residual_steer",
    "resact_residual_throttle",
    "resact_executed_steer",
    "resact_executed_throttle",
    "resact_delta_steer_abs",
    "resact_delta_throttle_abs",
    "resact_clip_fraction",
    "resact_action_distance",
)


def _as_action_vector(value: Sequence[float], *, default: Tuple[float, float]) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        arr = np.asarray(default, dtype=np.float32)
    if arr.size == 0:
        arr = np.asarray(default, dtype=np.float32)
    if arr.size == 1:
        arr = np.asarray([arr[0], default[1]], dtype=np.float32)
    return arr[:2].astype(np.float32, copy=False)


class ResidualActionWrapper(gym.Wrapper):
    """Interpret policy actions as residuals added to the previous executed action.

    The wrapped policy still sees a 2D continuous action space in [-1, 1]. The wrapper
    converts that residual into a bounded, physically smoother MetaDrive action and
    appends the previous executed action to the observation.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        resact_enabled: bool = True,
        resact_steer_delta_scale: float = 0.15,
        resact_throttle_delta_scale: float = 0.10,
        resact_initial_action: Sequence[float] = (0.0, 0.0),
    ) -> None:
        super().__init__(env)
        self.resact_enabled = bool(resact_enabled)
        self.delta_scale = np.asarray(
            [float(resact_steer_delta_scale), float(resact_throttle_delta_scale)],
            dtype=np.float32,
        )
        self.initial_action = _as_action_vector(resact_initial_action, default=(0.0, 0.0))
        self.prev_executed_action = self.initial_action.copy()
        self._last_info: Dict[str, float] = {}

        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise TypeError("ResidualActionWrapper requires a continuous Box action space.")
        self._base_action_low = _as_action_vector(self.env.action_space.low, default=(-1.0, -1.0))
        self._base_action_high = _as_action_vector(self.env.action_space.high, default=(1.0, 1.0))
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        if self.resact_enabled:
            self.observation_space = self._build_augmented_observation_space(self.env.observation_space)
        else:
            self.observation_space = self.env.observation_space

    def _build_augmented_observation_space(self, space: gym.Space) -> gym.spaces.Box:
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
            raise TypeError(
                "ResidualActionWrapper currently supports 1D Box observations only; "
                f"got {space!r}."
            )
        obs_low = np.asarray(space.low, dtype=np.float32).reshape(-1)
        obs_high = np.asarray(space.high, dtype=np.float32).reshape(-1)
        low = np.concatenate([obs_low, self._base_action_low]).astype(np.float32)
        high = np.concatenate([obs_high, self._base_action_high]).astype(np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment_obs(self, obs: Any) -> Any:
        if not self.resact_enabled:
            return obs
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        return np.concatenate([obs_arr, self.prev_executed_action]).astype(np.float32)

    def _reset_resact_state(self) -> None:
        self.prev_executed_action = np.clip(
            self.initial_action,
            self._base_action_low,
            self._base_action_high,
        ).astype(np.float32)
        self._last_info = {
            key: 0.0
            for key in RESACT_INFO_KEYS
        }
        self._last_info["resact_executed_steer"] = float(self.prev_executed_action[0])
        self._last_info["resact_executed_throttle"] = float(self.prev_executed_action[1])

    def _residual_to_executed_action(self, action: Sequence[float]) -> Tuple[np.ndarray, Dict[str, float]]:
        previous = self.prev_executed_action.copy()
        action_arr = _as_action_vector(action, default=(0.0, 0.0))
        residual = np.clip(action_arr, -1.0, 1.0).astype(np.float32)
        delta = residual * self.delta_scale
        unclipped_executed = previous + delta
        executed = np.clip(unclipped_executed, self._base_action_low, self._base_action_high).astype(np.float32)

        residual_clip = np.not_equal(action_arr, residual)
        action_clip = np.not_equal(unclipped_executed, executed)
        clip_fraction = float(np.logical_or(residual_clip, action_clip).mean())
        executed_delta = executed - previous

        info = {
            "resact_residual_steer": float(residual[0]),
            "resact_residual_throttle": float(residual[1]),
            "resact_executed_steer": float(executed[0]),
            "resact_executed_throttle": float(executed[1]),
            "resact_delta_steer_abs": float(abs(executed_delta[0])),
            "resact_delta_throttle_abs": float(abs(executed_delta[1])),
            "resact_clip_fraction": clip_fraction,
            "resact_action_distance": float(np.linalg.norm(executed_delta)),
        }
        return executed, info

    def reset(self, *args: Any, **kwargs: Any):
        self._reset_resact_state()
        result = self.env.reset(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            if isinstance(info, dict):
                info.update(self._last_info)
            return self._augment_obs(obs), info
        return self._augment_obs(result)

    def step(self, action: Sequence[float]):
        if self.resact_enabled:
            executed_action, resact_info = self._residual_to_executed_action(action)
        else:
            executed_action = _as_action_vector(action, default=(0.0, 0.0))
            resact_info = dict(self._last_info)

        result = self.env.step(executed_action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done_payload = (terminated, truncated)
        elif len(result) == 4:
            obs, reward, done, info = result
            done_payload = (done,)
        else:
            raise ValueError(f"Unsupported env.step result length: {len(result)}")

        self.prev_executed_action = executed_action.copy()
        self._last_info = resact_info
        if isinstance(info, dict):
            info.update(resact_info)

        obs = self._augment_obs(obs)
        if len(done_payload) == 2:
            return obs, reward, done_payload[0], done_payload[1], info
        return obs, reward, done_payload[0], info

    def get_episode_stats(self) -> Dict[str, float]:
        return dict(self._last_info)

    def render(self, *args: Any, **kwargs: Any):
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


def wrap_residual_action_env(
    env: gym.Env,
    *,
    resact_enabled: bool = True,
    resact_steer_delta_scale: float = 0.15,
    resact_throttle_delta_scale: float = 0.10,
    resact_initial_action: Sequence[float] = (0.0, 0.0),
) -> gym.Env:
    if not resact_enabled:
        return env
    return ResidualActionWrapper(
        env,
        resact_enabled=resact_enabled,
        resact_steer_delta_scale=resact_steer_delta_scale,
        resact_throttle_delta_scale=resact_throttle_delta_scale,
        resact_initial_action=resact_initial_action,
    )
