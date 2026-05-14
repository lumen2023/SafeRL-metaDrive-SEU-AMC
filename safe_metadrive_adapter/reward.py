"""Reward, cost, and risk-shaping helpers for SafeMetaDriveAdapterEnv."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def lmap(value: float, old_range, new_range) -> float:
    old_min, old_max = old_range
    new_min, new_max = new_range
    if old_max == old_min:
        return float(new_min)
    return float(new_min + (float(value) - old_min) * (new_max - new_min) / (old_max - old_min))


def risk_field_reward_enabled(env: Any) -> bool:
    return bool(env.config.get("use_risk_field_reward", False))


def aggregate_risk_field_reward_penalty(config: Dict[str, Any], risk_info: Dict[str, Any]) -> Tuple[float, float, float]:
    source_cost = (
        float(config.get("risk_field_boundary_weight", 0.0)) * float(risk_info.get("risk_field_boundary_cost", 0.0))
        + float(config.get("risk_field_lane_weight", 0.0)) * float(risk_info.get("risk_field_lane_cost", 0.0))
        + float(config.get("risk_field_offroad_weight", 0.0)) * float(risk_info.get("risk_field_offroad_cost", 0.0))
        + float(config.get("risk_field_vehicle_weight", 0.0)) * float(risk_info.get("risk_field_vehicle_cost", 0.0))
        + float(config.get("risk_field_object_weight", 0.0)) * float(risk_info.get("risk_field_object_cost", 0.0))
        + float(config.get("risk_field_headway_weight", 0.0)) * float(risk_info.get("risk_field_headway_cost", 0.0))
        + float(config.get("risk_field_ttc_weight", 0.0)) * float(risk_info.get("risk_field_ttc_cost", 0.0))
    )
    normalized_cost = clip(
        source_cost / max(float(config.get("risk_field_raw_clip", 10.0)), 1e-6),
        0.0,
        1.0,
    )
    penalty = float(config.get("risk_field_reward_scale", 0.0)) * normalized_cost
    return float(source_cost), float(normalized_cost), float(penalty)


def default_risk_info() -> Dict[str, float]:
    return {
        "risk_field_cost": 0.0,
        "risk_field_road_cost": 0.0,
        "risk_field_boundary_cost": 0.0,
        "risk_field_lane_cost": 0.0,
        "risk_field_offroad_cost": 0.0,
        "risk_field_vehicle_cost": 0.0,
        "risk_field_object_cost": 0.0,
        "risk_field_headway_cost": 0.0,
        "risk_field_ttc_cost": 0.0,
    }


def compute_steering_penalty(env: Any, vehicle: Any, *, first_step: bool = False) -> Dict[str, float]:
    current_steering = float(getattr(vehicle, "steering", 0.0))
    previous_steering = getattr(vehicle, "_last_reward_steering", current_steering)
    switch_eps = 1e-3

    if first_step:
        steering_delta = 0.0
        steering_switch = 0.0
        steering_switch_count = 0.0
    else:
        steering_delta = abs(current_steering - previous_steering)
        steering_switch = max(0.0, -current_steering * previous_steering)
        current_sign = 0 if abs(current_steering) <= switch_eps else (1 if current_steering > 0.0 else -1)
        previous_sign = 0 if abs(previous_steering) <= switch_eps else (1 if previous_steering > 0.0 else -1)
        steering_switch_count = 1.0 if current_sign * previous_sign < 0 else 0.0

    use_smooth_switch_penalty = bool(env.config.get("use_steering_penalty", False))
    steering_smoothness_penalty = 0.0
    steering_switch_penalty = 0.0
    steering_switch_count_penalty = 0.0
    if use_smooth_switch_penalty:
        steering_smoothness_penalty = float(env.config.get("steering_smoothness_weight", 0.0)) * steering_delta
        steering_switch_penalty = float(env.config.get("steering_switch_penalty_weight", 0.0)) * steering_switch
        steering_switch_count_penalty = (
            float(env.config.get("steering_switch_count_penalty_weight", 0.0)) * steering_switch_count
        )

    steering_absolute_penalty = 0.0
    if env.config.get("use_absolute_steering_penalty", False):
        steering_absolute_penalty = float(env.config.get("steering_absolute_penalty_weight", 0.0)) * (
            current_steering ** 2
        )

    steering_penalty = (
        steering_smoothness_penalty
        + steering_switch_penalty
        + steering_switch_count_penalty
        + steering_absolute_penalty
    )
    vehicle._last_reward_steering = current_steering
    return {
        "action_steering": current_steering,
        "steering_absolute_value": abs(current_steering),
        "steering_delta": float(steering_delta),
        "steering_switch": float(steering_switch),
        "steering_switch_count": float(steering_switch_count),
        "steering_smoothness_penalty": float(steering_smoothness_penalty),
        "steering_switch_penalty": float(steering_switch_penalty),
        "steering_switch_count_penalty": float(steering_switch_count_penalty),
        "steering_absolute_penalty": float(steering_absolute_penalty),
        "steering_penalty": float(steering_penalty),
    }


def compute_reward(env: Any, vehicle_id: str) -> Tuple[float, Dict[str, Any]]:
    vehicle = env.agents[vehicle_id]
    step_info: Dict[str, Any] = {}

    if vehicle.lane in vehicle.navigation.current_ref_lanes:
        current_lane = vehicle.lane
        positive_road = 1
    else:
        current_lane = vehicle.navigation.current_ref_lanes[0]
        current_road = vehicle.navigation.current_road
        positive_road = 1 if not current_road.is_negative_road() else -1

    long_last, _ = current_lane.local_coordinates(vehicle.last_position)
    long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

    lateral_bonus = 0.0
    lateral_norm = 0.0
    lateral_score = 0.0
    lateral_speed_gate = 0.0
    lateral_forward_gate = 0.0
    lateral_broken_line_gate = 1.0
    if env.config.get("use_lateral_reward", False):
        lane_width = max(float(vehicle.navigation.get_current_lane_width()), 1e-6)
        half_lane_width = 0.5 * lane_width
        lateral_norm = abs(lateral_now) / half_lane_width
        safe_band = clip(env.config.get("lateral_reward_safe_band", 0.25), 0.0, 0.95)
        if lateral_norm <= safe_band:
            lateral_score = 1.0
        else:
            edge_excess = (lateral_norm - safe_band) / max(1.0 - safe_band, 1e-6)
            lateral_score = 1.0 - clip(edge_excess, 0.0, 1.0) ** 2

        min_speed = float(env.config.get("lateral_reward_min_speed", 8.0))
        full_speed = max(float(env.config.get("lateral_reward_full_speed", 20.0)), min_speed + 1e-6)
        lateral_speed_gate = clip((vehicle.speed_km_h - min_speed) / (full_speed - min_speed), 0.0, 1.0)
        lateral_forward_gate = 1.0 if (long_now - long_last) > 0.0 else 0.0
        if env.config.get("lateral_reward_zero_on_broken_line", False) and vehicle.on_broken_line:
            lateral_broken_line_gate = 0.0

        lateral_bonus = (
            float(env.config.get("lateral_reward_weight", 0.0))
            * lateral_score
            * lateral_speed_gate
            * lateral_forward_gate
            * lateral_broken_line_gate
        )

    longitudinal_progress = float(long_now - long_last)
    driving_component = float(env.config.get("driving_reward", 0.0)) * longitudinal_progress * positive_road
    scaled_speed = lmap(vehicle.speed_km_h, env.config.get("reward_speed_range", [10, 30]), [-1, 1])
    normalized_speed_reward = clip(scaled_speed, -1, 1) * positive_road
    speed_component = float(env.config.get("speed_reward", 0.0)) * normalized_speed_reward

    reward = lateral_bonus + driving_component + speed_component
    base_reward = float(reward)

    steering_info = {
        "action_steering": float(getattr(vehicle, "steering", 0.0)),
        "steering_absolute_value": abs(float(getattr(vehicle, "steering", 0.0))),
        "steering_delta": 0.0,
        "steering_switch": 0.0,
        "steering_switch_count": 0.0,
        "steering_smoothness_penalty": 0.0,
        "steering_switch_penalty": 0.0,
        "steering_switch_count_penalty": 0.0,
        "steering_absolute_penalty": 0.0,
        "steering_penalty": 0.0,
    }
    if env.config.get("use_steering_penalty", False) or env.config.get("use_absolute_steering_penalty", False):
        steering_info = compute_steering_penalty(
            env,
            vehicle,
            first_step=env.episode_lengths.get(vehicle_id, 0) == 0,
        )
        reward = base_reward - steering_info["steering_penalty"]

    reward_after_steering = float(reward)
    risk_info = default_risk_info()
    risk_source_cost = 0.0
    risk_normalized_cost = 0.0
    risk_penalty = 0.0
    if risk_field_reward_enabled(env):
        _, risk_info = env._get_risk_field_calculator().calculate(env, vehicle)
        risk_source_cost, risk_normalized_cost, risk_penalty = aggregate_risk_field_reward_penalty(env.config, risk_info)
        reward = reward_after_steering - risk_penalty

    shaped_reward = float(reward)
    step_info.update(risk_info)
    step_info.update(
        {
            "step_reward": shaped_reward,
            "base_reward": base_reward,
            "driving_component_reward": float(driving_component),
            "speed_component_reward": float(speed_component),
            "normalized_speed_reward": float(normalized_speed_reward),
            "longitudinal_progress": longitudinal_progress,
            "positive_road": float(positive_road),
            "speed_km_h": vehicle.speed_km_h,
            "scaled_speed": scaled_speed,
            "lateral_now": abs(lateral_now),
            "lateral_norm": lateral_norm,
            "lateral_score": lateral_score,
            "lateral_speed_gate": lateral_speed_gate,
            "lateral_forward_gate": lateral_forward_gate,
            "lateral_broken_line_gate": lateral_broken_line_gate,
            "lateral_reward": lateral_bonus,
            "reward_after_steering_penalty": reward_after_steering,
            "risk_field_reward_source_cost": risk_source_cost,
            "risk_field_reward_normalized_cost": risk_normalized_cost,
            "risk_field_reward_penalty": risk_penalty,
        }
    )
    step_info.update(steering_info)

    reward_override_active = 0.0
    if env._is_arrive_destination(vehicle):
        reward = float(env.config.get("success_reward", 10.0))
        reward_override_active = 1.0
    elif env._is_out_of_road(vehicle):
        reward = -float(env.config.get("out_of_road_penalty", 5.0))
        reward_override_active = 1.0
    elif getattr(vehicle, "crash_vehicle", False):
        reward = -float(env.config.get("crash_vehicle_penalty", 5.0))
        reward_override_active = 1.0
    elif getattr(vehicle, "crash_object", False):
        reward = -float(env.config.get("crash_object_penalty", 5.0))
        reward_override_active = 1.0
    elif getattr(vehicle, "crash_sidewalk", False):
        reward = -float(env.config.get("crash_sidewalk_penalty", 0.0))
        reward_override_active = 1.0

    step_info["reward_override_active"] = reward_override_active
    step_info["reward_override_delta"] = float(reward - shaped_reward)
    step_info["final_reward"] = float(reward)
    step_info["route_completion"] = vehicle.navigation.route_completion
    return float(reward), step_info


def event_cost(env: Any, vehicle: Any) -> Tuple[float, Dict[str, Any]]:
    info: Dict[str, Any] = {"cost": 0.0, "event_cost": 0.0}
    if env._is_out_of_road(vehicle):
        info["event_cost"] = float(env.config.get("out_of_road_cost", 1.0))
    elif getattr(vehicle, "crash_vehicle", False):
        info["event_cost"] = float(env.config.get("crash_vehicle_cost", 1.0))
    elif getattr(vehicle, "crash_object", False):
        info["event_cost"] = float(env.config.get("crash_object_cost", 1.0))
    info["cost"] = info["event_cost"]
    return float(info["cost"]), info


def risk_field_event_equivalent_cost(config: Dict[str, Any], risk_cost: float) -> float:
    scaled = float(config.get("risk_field_cost_scale", 1.0)) * float(risk_cost)
    mapping = str(config.get("risk_field_cost_mapping", "linear_scale"))
    if mapping == "risk_only":
        value = float(risk_cost)
    else:
        value = scaled
    upper = min(
        float(config.get("risk_field_collision_equivalent_cost", 1.0)),
        float(config.get("risk_field_cost_clip", 1.0)),
    )
    return clip(value, 0.0, upper)


def combine_event_and_risk_cost(config: Dict[str, Any], event_value: float, risk_value: float) -> float:
    mode = str(config.get("risk_field_cost_combine", "event_only"))
    weighted_event = float(config.get("risk_field_event_cost_weight", 1.0)) * float(event_value)
    if mode == "risk_only":
        return float(risk_value)
    if mode == "max":
        return max(weighted_event, float(risk_value))
    if mode == "add":
        return weighted_event + float(risk_value)
    return weighted_event
