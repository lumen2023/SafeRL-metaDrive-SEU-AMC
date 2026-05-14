"""SafeMetaDrive environment subclass with portable reward/cost hooks."""

from __future__ import annotations

from typing import Any, Dict

from .config import DEFAULT_CONFIG
from .local_import import prefer_local_metadrive
from .reward import (
    aggregate_risk_field_reward_penalty,
    combine_event_and_risk_cost,
    compute_reward,
    compute_steering_penalty,
    event_cost,
    risk_field_event_equivalent_cost,
    risk_field_reward_enabled,
)
from .risk_field import RiskFieldCalculator


prefer_local_metadrive()

try:
    from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
except ImportError:  # pragma: no cover - compatibility with unusual editable installs.
    from metadrive.metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv


class SafeMetaDriveAdapterEnv(SafeMetaDriveEnv):
    """SafeMetaDrive env with adapter-owned reward, cost, and risk-field logic."""

    def default_config(self):
        config = super(SafeMetaDriveAdapterEnv, self).default_config()
        config.update(DEFAULT_CONFIG, allow_add_new_key=True)
        return config

    def _get_risk_field_calculator(self) -> RiskFieldCalculator:
        calculator = getattr(self, "_risk_field_calculator", None)
        if calculator is None:
            calculator = RiskFieldCalculator(self.config)
            self._risk_field_calculator = calculator
        return calculator

    def _risk_field_reward_enabled(self) -> bool:
        return risk_field_reward_enabled(self)

    def _risk_field_reward_penalty(self, risk_info: Dict[str, Any]):
        return aggregate_risk_field_reward_penalty(self.config, risk_info)

    def _compute_steering_penalty(self, vehicle, first_step: bool = False):
        return compute_steering_penalty(self, vehicle, first_step=first_step)

    def reward_function(self, vehicle_id: str):
        return compute_reward(self, vehicle_id)

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        cost, step_info = event_cost(self, vehicle)

        risk_equivalent = 0.0
        if bool(self.config.get("use_risk_field_cost", False)):
            _, risk_info = self._get_risk_field_calculator().calculate(self, vehicle)
            step_info.update(risk_info)
            risk_equivalent = risk_field_event_equivalent_cost(self.config, risk_info.get("risk_field_cost", 0.0))
            cost = combine_event_and_risk_cost(self.config, step_info.get("event_cost", 0.0), risk_equivalent)
            step_info["cost"] = float(cost)

        step_info["risk_field_event_equivalent_cost"] = float(risk_equivalent)
        self.episode_cost = getattr(self, "episode_cost", 0.0) + float(cost)
        step_info["total_cost"] = self.episode_cost
        return float(cost), step_info


SafeMetaDriveEnv_mini = SafeMetaDriveAdapterEnv
