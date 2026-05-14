import importlib.util
import math

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("shapely") is None or importlib.util.find_spec("panda3d") is None,
    reason="MetaDrive runtime dependencies shapely/panda3d are not installed",
)


def _env_api():
    from env import (
        SafeMetaDriveEnv_mini,
        SafeMetaDriveSingleSceneEnv,
        get_single_scene_training_env,
        get_training_env,
        get_validation_env,
    )

    return {
        "SafeMetaDriveEnv_mini": SafeMetaDriveEnv_mini,
        "SafeMetaDriveSingleSceneEnv": SafeMetaDriveSingleSceneEnv,
        "get_single_scene_training_env": get_single_scene_training_env,
        "get_training_env": get_training_env,
        "get_validation_env": get_validation_env,
    }


def _ramp_merge_no_accident_config():
    return {
        "log_level": 50,
        "use_render": False,
        "single_scene": {
            "traffic": {
                "ramp_merge": {
                    "accident_prob": 0.0,
                }
            }
        },
    }


def _straight_no_accident_config():
    return {
        "log_level": 50,
        "use_render": False,
        "single_scene": {
            "traffic": {
                "straight_4lane": {
                    "accident_prob": 0.0,
                }
            }
        },
    }


def _straight_official_config():
    config = _straight_no_accident_config()
    config.setdefault("single_scene", {})["traffic_backend"] = "official"
    return config


def _traffic_signature(env):
    traffic_manager = env.engine.traffic_manager
    slots = getattr(traffic_manager, "single_scene_slot_by_vehicle", {})
    signature = []
    for vehicle in traffic_manager.traffic_vehicles:
        slot = slots.get(vehicle.id, {})
        lane_index = tuple(getattr(vehicle, "lane_index", ()))
        lane = env.engine.map_manager.current_map.road_network.get_lane(lane_index)
        longitude = env._vehicle_longitude_on_lane(vehicle, lane)
        signature.append(
            (
                slot.get("road"),
                lane_index,
                round(float(longitude), 2),
                round(float(getattr(vehicle, "speed", 0.0)), 2),
            )
        )
    return tuple(sorted(signature))


def _scene_signature(env):
    agent = env.agent
    ego_lane = tuple(agent.lane_index)
    ego_lane_obj = env.engine.map_manager.current_map.road_network.get_lane(ego_lane)
    ego_longitude = env._vehicle_longitude_on_lane(agent, ego_lane_obj)
    return (
        ego_lane,
        round(float(ego_longitude), 2),
        agent.config.get("destination"),
        int(env._single_scene_target_traffic_count),
        _traffic_signature(env),
    )


def test_straight_4lane_splits_long_road_into_short_segments():
    api = _env_api()
    env = api["get_single_scene_training_env"]("straight_4lane", extra_config=_straight_no_accident_config())
    try:
        env.reset(seed=10000)
        scene_map = env.engine.map_manager.current_map
        aliases = scene_map.scene_metadata["road_aliases"]
        geometry = env.config["single_scene"]["geometry"]["straight_4lane"]
        straight_length = float(geometry["straight_length"])
        segment_length = float(geometry["straight_segment_length"])
        expected_segment_count = int(math.ceil(straight_length / segment_length))
        expected_segment_lengths = [segment_length] * expected_segment_count
        expected_segment_lengths[-1] = straight_length - segment_length * (expected_segment_count - 1)
        segment_aliases = sorted(
            name for name in aliases if name.startswith("straight_") and name != "straight"
        )

        assert segment_aliases == ["straight_{}".format(index) for index in range(expected_segment_count)]

        segment_lengths = []
        for alias in segment_aliases:
            road = aliases[alias]
            lane = scene_map.road_network.get_lane(road.lane_index(0))
            segment_lengths.append(round(float(lane.length), 2))

        assert segment_lengths == pytest.approx(expected_segment_lengths)
        assert sum(segment_lengths) == pytest.approx(straight_length)
        assert aliases["straight"].start_node == aliases["straight_0"].start_node
        assert aliases["straight"].end_node == aliases["straight_0"].end_node
    finally:
        env.close()


def test_straight_4lane_traffic_survives_past_midroad():
    api = _env_api()
    config = _straight_no_accident_config()
    config["single_scene"]["randomization"] = {
        "traffic_speed_range": (12.0, 12.0),
        "traffic_slot_longitude_jitter": (0.0, 0.0),
        "traffic_slot_lateral_jitter": (0.0, 0.0),
    }
    env = api["get_single_scene_training_env"]("straight_4lane", extra_config=config)
    try:
        env.reset(seed=10000)
        traffic_manager = env.engine.traffic_manager
        initial_vehicle_count = len(traffic_manager.traffic_vehicles)
        assert initial_vehicle_count > 0

        max_step_removed_count = 0
        min_vehicle_count = initial_vehicle_count
        max_vehicle_x = -float("inf")
        for _ in range(260):
            _, _, terminated, truncated, info = env.step([0.0, 0.0])
            step_removed_count = int(info.get("single_scene_traffic_removed", 0))
            max_step_removed_count = max(max_step_removed_count, step_removed_count)
            min_vehicle_count = min(min_vehicle_count, len(traffic_manager.traffic_vehicles))
            for vehicle in traffic_manager.traffic_vehicles:
                max_vehicle_x = max(max_vehicle_x, float(vehicle.position[0]))
            assert not terminated
            assert not truncated

        assert max_step_removed_count < initial_vehicle_count
        assert min_vehicle_count >= initial_vehicle_count
        assert len(traffic_manager.traffic_vehicles) == initial_vehicle_count
        assert max_vehicle_x >= 250.0
    finally:
        env.close()


def test_straight_4lane_traffic_density_zero_disables_traffic():
    api = _env_api()
    config = _straight_no_accident_config()
    config["traffic_density"] = 0.0
    env = api["get_single_scene_training_env"]("straight_4lane", extra_config=config)
    try:
        env.reset(seed=10000)
        assert env._single_scene_target_traffic_count == 0
        assert len(env.engine.traffic_manager.traffic_vehicles) == 0
    finally:
        env.close()


def test_straight_4lane_traffic_density_scales_generated_candidates():
    api = _env_api()
    counts = []
    for density in (0.05, 0.2):
        config = _straight_no_accident_config()
        config["traffic_density"] = density
        env = api["get_single_scene_training_env"]("straight_4lane", extra_config=config)
        try:
            env.reset(seed=10000)
            counts.append(len(env.engine.traffic_manager.traffic_vehicles))
        finally:
            env.close()

    low_count, high_count = counts
    assert low_count > 0
    assert high_count > low_count
    assert high_count > 8


def test_straight_4lane_range_mode_keeps_legacy_count_range():
    api = _env_api()
    config = _straight_no_accident_config()
    config["traffic_density"] = 0.2
    config["single_scene"]["randomization"] = {
        "traffic_count_mode": "range",
        "traffic_vehicle_count_range": (2, 2),
    }
    env = api["get_single_scene_training_env"]("straight_4lane", extra_config=config)
    try:
        env.reset(seed=10000)
        assert env._single_scene_target_traffic_count == 2
        assert len(env.engine.traffic_manager.traffic_vehicles) == 2
    finally:
        env.close()


def test_straight_4lane_default_profile_keeps_front_vehicle_on_same_physical_lane():
    api = _env_api()
    env = api["get_single_scene_training_env"]("straight_4lane", extra_config=_straight_no_accident_config())
    try:
        for seed in range(10000, 10010):
            env.reset(seed=seed)
            ego_lane = env.engine.map_manager.current_map.road_network.get_lane(env.agent.lane_index)
            ego_longitude = env._vehicle_longitude_on_lane(env.agent, ego_lane)
            ego_corridor_longitude = env._lane_corridor_longitude(ego_lane, ego_longitude)
            traffic_longs = env._same_lane_corridor_traffic_longitudes(ego_lane)
            front_gaps = sorted(long - ego_corridor_longitude for long in traffic_longs if long > ego_corridor_longitude)

            assert front_gaps
            lead_low, lead_high = env.config["single_scene"]["randomization"]["ego_lead_gap_range"]
            assert lead_low <= front_gaps[0] <= lead_high
    finally:
        env.close()


def test_straight_4lane_official_backend_uses_pg_traffic_manager():
    api = _env_api()
    config = _straight_official_config()
    env = api["get_single_scene_training_env"]("straight_4lane", extra_config=config)
    try:
        from env import PGTrafficManager

        env.reset(seed=10000)
        assert isinstance(env.engine.traffic_manager, PGTrafficManager)
    finally:
        env.close()


def test_straight_4lane_official_backend_density_scales_vehicle_count():
    api = _env_api()
    counts = []
    for density in (0.0, 0.05, 0.2):
        config = _straight_official_config()
        config["traffic_density"] = density
        env = api["get_single_scene_training_env"]("straight_4lane", extra_config=config)
        try:
            env.reset(seed=10000)
            counts.append(len(env.engine.traffic_manager.traffic_vehicles))
        finally:
            env.close()

    zero_count, low_count, high_count = counts
    assert zero_count == 0
    assert low_count > 0
    assert high_count > low_count


def test_straight_4lane_official_backend_spawns_beyond_entry_segment():
    api = _env_api()
    config = _straight_official_config()
    config["traffic_density"] = 0.2
    env = api["get_single_scene_training_env"]("straight_4lane", extra_config=config)
    try:
        env.reset(seed=10000)
        scene_map = env.engine.map_manager.current_map
        road_network = scene_map.road_network
        straight_lane_indices = {
            road.lane_index(lane_idx)
            for alias, road in scene_map.scene_metadata["road_aliases"].items()
            if alias.startswith("straight_")
            for lane_idx, _ in enumerate(road.get_lanes(road_network))
        }

        assert any(tuple(vehicle.lane_index) in straight_lane_indices for vehicle in env.engine.traffic_manager.traffic_vehicles)
    finally:
        env.close()


def test_straight_4lane_official_trigger_backend_triggers_block_vehicles():
    api = _env_api()
    config = _straight_official_config()
    config.update({"traffic_density": 0.2, "traffic_mode": "trigger"})
    env = api["get_single_scene_training_env"]("straight_4lane", extra_config=config)
    try:
        env.reset(seed=10000)
        traffic_manager = env.engine.traffic_manager
        pending_before = sum(len(block_vehicles.vehicles) for block_vehicles in traffic_manager.block_triggered_vehicles)
        active_before = len(traffic_manager.traffic_vehicles)
        assert pending_before > 0

        triggered = False
        for _ in range(10):
            env.step([0.0, 0.0])
            pending_after = sum(len(block_vehicles.vehicles) for block_vehicles in traffic_manager.block_triggered_vehicles)
            active_after = len(traffic_manager.traffic_vehicles)
            if active_after > active_before or pending_after < pending_before:
                triggered = True
                break

        assert triggered
    finally:
        env.close()


def test_ramp_merge_random_reset_varies_and_explicit_seed_reproduces():
    api = _env_api()
    env = api["get_single_scene_training_env"]("ramp_merge", extra_config=_ramp_merge_no_accident_config())
    try:
        signatures = []
        for _ in range(3):
            env.reset()
            signatures.append(_scene_signature(env))
        assert len(set(signatures)) > 1

        env.reset(seed=12345)
        first = _scene_signature(env)
        env.reset(seed=12345)
        second = _scene_signature(env)
        assert first == second
    finally:
        env.close()


def test_ramp_merge_respects_road_min_counts_and_ego_gap():
    api = _env_api()
    env = api["get_single_scene_training_env"]("ramp_merge", extra_config=_ramp_merge_no_accident_config())
    try:
        env.reset(seed=2026)

        road_counts = env._single_scene_active_road_counts()
        road_min_counts = env.config["single_scene"]["traffic"]["ramp_merge"]["road_min_counts"]
        for road, min_count in road_min_counts.items():
            assert road_counts.get(road, 0) >= min_count

        ego_lane = env.engine.map_manager.current_map.road_network.get_lane(env.agent.lane_index)
        ego_longitude = env._vehicle_longitude_on_lane(env.agent, ego_lane)
        traffic_longs = env._same_lane_traffic_longitudes(ego_lane)
        front_gaps = sorted(long - ego_longitude for long in traffic_longs if long > ego_longitude)
        rear_gaps = sorted(ego_longitude - long for long in traffic_longs if long <= ego_longitude)
        lead_low, lead_high = env.config["single_scene"]["randomization"]["ego_lead_gap_range"]
        rear_min = env.config["single_scene"]["randomization"]["ego_rear_gap_min"]

        assert front_gaps
        assert lead_low <= front_gaps[0] <= lead_high
        if rear_gaps:
            assert rear_gaps[0] >= rear_min
    finally:
        env.close()


def test_ramp_merge_spawns_more_ramp_traffic_by_default():
    api = _env_api()
    env = api["get_single_scene_training_env"]("ramp_merge", extra_config=_ramp_merge_no_accident_config())
    try:
        for seed in range(10000, 10005):
            env.reset(seed=seed)
            road_counts = env._single_scene_active_road_counts()
            assert road_counts.get("ramp", 0) >= 2
    finally:
        env.close()


def test_ramp_merge_replenishes_toward_target_after_vehicle_removal():
    api = _env_api()
    env = api["get_single_scene_training_env"]("ramp_merge", extra_config=_ramp_merge_no_accident_config())
    try:
        env.reset(seed=7)
        traffic_manager = env.engine.traffic_manager
        first_vehicle = traffic_manager.traffic_vehicles[0]
        traffic_manager.clear_objects([first_vehicle.id])
        traffic_manager._traffic_vehicles.remove(first_vehicle)
        traffic_manager.single_scene_slot_by_vehicle.pop(first_vehicle.id, None)

        before = len(traffic_manager.traffic_vehicles)
        env._replenish_single_scene_traffic()
        after = len(traffic_manager.traffic_vehicles)
        assert after >= before
        assert after <= env._single_scene_target_traffic_count
    finally:
        env.close()


def test_t_intersection_defaults_to_left_only_ego_destination():
    api = _env_api()
    env = api["get_single_scene_training_env"](
        "t_intersection",
        extra_config={
            "log_level": 50,
            "use_render": False,
            "single_scene": {"traffic": {"t_intersection": {"accident_prob": 0.0}}},
        },
    )
    try:
        for seed in (11, 12, 13):
            env.reset(seed=seed)
            left_destinations = set(env.engine.map_manager.current_map.scene_metadata["turn_destinations"]["left"])
            assert env.agent.config.get("destination") in left_destinations
    finally:
        env.close()


def test_generic_env_factories_do_not_use_single_scene_env():
    api = _env_api()
    train_env = api["get_training_env"]({"log_level": 50})
    val_env = api["get_validation_env"]({"log_level": 50})
    try:
        assert isinstance(train_env, api["SafeMetaDriveEnv_mini"])
        assert isinstance(val_env, api["SafeMetaDriveEnv_mini"])
        assert not isinstance(train_env, api["SafeMetaDriveSingleSceneEnv"])
        assert not isinstance(val_env, api["SafeMetaDriveSingleSceneEnv"])
    finally:
        train_env.close()
        val_env.close()
