import pytest


def _make_env_or_skip(*args, **kwargs):
    try:
        from safe_metadrive_adapter import make_safe_metadrive_env

        return make_safe_metadrive_env(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on local MetaDrive runtime.
        pytest.skip(f"MetaDrive runtime is not available: {exc}")


def test_training_env_reset_step_and_risk_field_info():
    env = _make_env_or_skip("train", {"log_level": 50, "num_scenarios": 1, "start_seed": 100})
    try:
        obs, reset_info = env.reset()
        assert obs is not None
        assert isinstance(reset_info, dict)

        _, _, _, _, info = env.step([0.0, 0.0])
        for key in (
            "risk_field_cost",
            "risk_field_lane_cost",
            "risk_field_vehicle_cost",
            "risk_field_object_cost",
            "risk_field_reward_penalty",
        ):
            assert key in info

        from risk_field import RiskFieldCalculator

        _, risk_info = RiskFieldCalculator(env.config).calculate(env, env.agent)
        for key in (
            "risk_field_cost",
            "risk_field_lane_cost",
            "risk_field_vehicle_cost",
            "risk_field_object_cost",
        ):
            assert key in risk_info
    finally:
        env.close()


def test_resact_factory_reset_step_info():
    env = _make_env_or_skip(
        "train",
        {"log_level": 50, "num_scenarios": 1, "start_seed": 100},
        resact={"enabled": True},
    )
    try:
        obs, _ = env.reset()
        assert obs is not None

        _, _, _, _, info = env.step([0.0, 0.0])
        for key in ("resact_delta_steer_abs", "resact_delta_throttle_abs", "resact_action_distance"):
            assert key in info
    finally:
        env.close()
