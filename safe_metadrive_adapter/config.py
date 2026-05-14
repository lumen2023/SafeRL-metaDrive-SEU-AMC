"""Shared configuration for the portable SafeMetaDrive adapter."""

from __future__ import annotations

import copy
from typing import Any, Dict

from .risk_field import RiskFieldCalculator


DEFAULT_CONFIG: Dict[str, Any] = {
    # Environment difficulty.
    "accident_prob": 0.0,
    "traffic_density": 0.2,
    "remove_crashed_traffic_vehicle": False,
    # Termination switches.
    "crash_vehicle_done": False,
    "crash_object_done": True,
    "out_of_road_done": True,
    "out_of_route_done": False,
    "on_continuous_line_done": True,
    "on_broken_line_done": False,
    # Event costs for safe RL.
    "crash_vehicle_cost": 1.0,
    "crash_object_cost": 1.0,
    "out_of_road_cost": 1.0,
    # Base reward and event reward overrides.
    "success_reward": 10.0,
    "out_of_road_penalty": 8.0,
    "crash_vehicle_penalty": 1.0,
    "crash_object_penalty": 1.0,
    "crash_sidewalk_penalty": 0.0,
    "driving_reward": 1.0,
    "speed_reward": 0.1,
    "reward_speed_range": [10, 30],
    # Lateral reward shaping.
    "use_lateral_reward": True,
    "lateral_reward_weight": 1.5,
    "lateral_reward_safe_band": 0.25,
    "lateral_reward_min_speed": 8.0,
    "lateral_reward_full_speed": 20.0,
    "lateral_reward_zero_on_broken_line": False,
    # Steering smoothness and absolute steering penalties.
    "use_steering_penalty": False,
    "use_absolute_steering_penalty": True,
    "steering_smoothness_weight": 0.0,
    "steering_switch_penalty_weight": 100.0,
    "steering_switch_count_penalty_weight": 0.0,
    "steering_absolute_penalty_weight": 1.2,
    # Dense risk-field reward shaping.
    "use_risk_field_reward": True,
    "risk_field_reward_scale": 25.0,
    # Optional dense risk-field cost signal for algorithms that want it.
    "use_risk_field_cost": False,
    "risk_field_cost_scale": 1.0,
    "risk_field_cost_mapping": "linear_scale",
    "risk_field_cost_combine": "event_only",
    "risk_field_cost_clip": 1.0,
    "risk_field_event_cost_weight": 1.0,
    "risk_field_collision_equivalent_cost": 1.0,
    "risk_field_cost_weight": 1.0,
    # Residual action wrapper defaults. The wrapper is applied by the factory,
    # not by the env class itself.
    "resact_enabled": False,
    "resact_steer_delta_scale": 0.15,
    "resact_throttle_delta_scale": 0.10,
    "resact_initial_action": (0.0, 0.0),
}

DEFAULT_CONFIG.update(RiskFieldCalculator.DEFAULTS)

TRAINING_CONFIG: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
TRAINING_CONFIG.update(
    {
        "num_scenarios": 50,
        "start_seed": 100,
    }
)

VALIDATION_CONFIG: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
VALIDATION_CONFIG.update(
    {
        "num_scenarios": 50,
        "start_seed": 1000,
    }
)


def merged_config(base_config: Dict[str, Any], extra_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    if extra_config:
        config.update(extra_config)
    return config
