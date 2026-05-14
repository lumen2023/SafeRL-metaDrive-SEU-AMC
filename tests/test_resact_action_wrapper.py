import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from resact_metadrive import RESACT_INFO_KEYS, ResidualActionWrapper


class DummyContinuousEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_action = None
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}


def test_reset_augments_observation_with_previous_action():
    env = ResidualActionWrapper(DummyContinuousEnv())

    obs, info = env.reset()

    assert obs.shape == (5,)
    np.testing.assert_allclose(obs[-2:], [0.0, 0.0])
    for key in RESACT_INFO_KEYS:
        assert key in info


def test_steering_residual_changes_by_configured_step():
    base_env = DummyContinuousEnv()
    env = ResidualActionWrapper(base_env, resact_steer_delta_scale=0.15)
    env.reset()

    _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

    np.testing.assert_allclose(base_env.last_action, [0.15, 0.0], atol=1e-6)
    assert np.isclose(info["resact_delta_steer_abs"], 0.15)


def test_opposite_residual_does_not_jump_across_full_range():
    base_env = DummyContinuousEnv()
    env = ResidualActionWrapper(base_env, resact_steer_delta_scale=0.15)
    env.reset()
    env.step(np.array([1.0, 0.0], dtype=np.float32))

    _, _, _, _, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))

    np.testing.assert_allclose(base_env.last_action, [0.0, 0.0], atol=1e-6)
    assert np.isclose(info["resact_delta_steer_abs"], 0.15)


def test_throttle_residual_changes_by_configured_step():
    base_env = DummyContinuousEnv()
    env = ResidualActionWrapper(base_env, resact_throttle_delta_scale=0.10)
    env.reset()

    _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))

    np.testing.assert_allclose(base_env.last_action, [0.0, 0.10], atol=1e-6)
    assert np.isclose(info["resact_delta_throttle_abs"], 0.10)


def test_info_contains_resact_metrics():
    env = ResidualActionWrapper(DummyContinuousEnv())
    env.reset()

    _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))

    for key in RESACT_INFO_KEYS:
        assert key in info
    assert info["resact_action_distance"] > 0.0
