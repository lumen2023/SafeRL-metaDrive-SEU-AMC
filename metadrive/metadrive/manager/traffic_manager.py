import copy
import logging
from collections import namedtuple
from typing import Dict

import math
import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network import Road
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from metadrive.manager.base_manager import BaseManager
from metadrive.utils import merge_dicts

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


class TrafficMode:
    # 基础模式：一次性生成所有交通车辆
    Basic = "basic"

    # 重生模式：车辆到达目的地后重新生成
    Respawn = "respawn"

    # 触发模式：自车靠近时才激活该路段的交通车辆
    Trigger = "trigger"

    # 混合模式：部分车辆常驻，部分车辆触发式生成
    Hybrid = "hybrid"


class PGTrafficManager(BaseManager):
    VEHICLE_GAP = 12  # 车辆间距（米）

    def __init__(self):
        """
        控制整体交通流
        """
        super(PGTrafficManager, self).__init__()

        self._traffic_vehicles = []  # 当前活跃的交通车辆列表

        # 按路段分组的待触发车辆（触发模式下使用）
        self.block_triggered_vehicles = []

        # 交通配置参数
        self.mode = self.engine.global_config["traffic_mode"]  # 交通模式
        self.random_traffic = self.engine.global_config["random_traffic"]  # 是否随机交通
        self.density = self.engine.global_config["traffic_density"]  # 交通密度
        self.respawn_lanes = None  # 可重生车道列表

    def reset(self):
        """
        根据模式和密度在地图上生成交通流
        :return: 交通车辆列表
        """
        map = self.current_map
        logging.debug("load scene {}".format("Use random traffic" if self.random_traffic else ""))

        # 重置待触发车辆列表
        self.block_triggered_vehicles = []

        traffic_density = self.density
        if abs(traffic_density) < 1e-2:
            return
        self.respawn_lanes = self._get_available_respawn_lanes(map)

        logging.debug(f"Resetting Traffic Manager with mode {self.mode} and density {traffic_density}")

        # 根据模式选择生成策略
        if self.mode in {TrafficMode.Basic, TrafficMode.Respawn}:
            self._create_basic_vehicles(map, traffic_density)
        elif self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            self._create_trigger_vehicles(map, traffic_density)
        else:
            raise ValueError(f"No such mode named {self.mode}")

    def before_step(self):
        """
        所有交通车辆在此做出驾驶决策
        :return: None
        """
        engine = self.engine
        
        # 【触发模式核心逻辑】检查自车位置，激活前方路段的待触发车辆
        if self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            for v in engine.agent_manager.active_agents.values():
                if len(self.block_triggered_vehicles) > 0:
                    # 获取自车当前所在道路
                    ego_lane_idx = v.lane_index[:-1]
                    ego_road = Road(ego_lane_idx[0], ego_lane_idx[1])
                    
                    # 如果自车进入待触发路段，则激活该路段的所有车辆
                    if ego_road == self.block_triggered_vehicles[-1].trigger_road:
                        block_vehicles = self.block_triggered_vehicles.pop()
                        self._traffic_vehicles += list(self.get_objects(block_vehicles.vehicles).values())

        # 执行所有活跃交通车辆的策略
        for v in self._traffic_vehicles:
            p = self.engine.get_policy(v.name)
            v.before_step(p.act())
        return dict()

    def after_step(self, *args, **kwargs):
        """
        更新所有交通车辆状态，处理移除和重生
        """
        v_to_remove = []
        remove_crashed_traffic_vehicle = self.engine.global_config.get("remove_crashed_traffic_vehicle", False)
        
        # 检查每辆车的状态
        for v in self._traffic_vehicles:
            v.after_step()
            # 标记需要移除的车辆（偏离车道或碰撞）
            if (not v.on_lane) or (remove_crashed_traffic_vehicle and v.crash_vehicle):
                v_to_remove.append(v)

        # 处理需要移除的车辆
        for v in v_to_remove:
            vehicle_type = type(v)
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)

            # 【重生模式】生成新车辆替换被移除的车辆
            if self.mode in {TrafficMode.Respawn, TrafficMode.Hybrid}:
                lane = self.respawn_lanes[self.np_random.randint(0, len(self.respawn_lanes))]
                lane_idx = lane.index
                long = self.np_random.rand() * lane.length / 2
                traffic_v_config = {"spawn_lane_index": lane_idx, "spawn_longitude": long}
                new_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                from metadrive.policy.idm_policy import IDMPolicy
                self.add_policy(new_v.id, IDMPolicy, new_v, self.generate_seed())
                self._traffic_vehicles.append(new_v)

        return dict()

    def before_reset(self) -> None:
        """
        清空场景并重置
        :return: None
        """
        super(PGTrafficManager, self).before_reset()
        self.density = self.engine.global_config["traffic_density"]
        self.block_triggered_vehicles = []
        self._traffic_vehicles = []

    def get_vehicle_num(self):
        """
        获取道路上的车辆数量
        :return: 车辆数
        """
        if self.mode in {TrafficMode.Basic, TrafficMode.Respawn}:
            return len(self._traffic_vehicles)
        # 触发模式下统计待触发的车辆总数
        return sum(len(block_vehicle_set.vehicles) for block_vehicle_set in self.block_triggered_vehicles)

    def get_global_states(self) -> Dict:
        """
        返回所有交通车辆的状态
        :return: 所有车辆状态字典
        """
        states = dict()
        traffic_states = dict()
        for vehicle in self._traffic_vehicles:
            traffic_states[vehicle.index] = vehicle.get_state()

        # 收集待触发的其他车辆（触发/混合模式）
        if self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    traffic_states[vehicle.index] = vehicle.get_state()
        states[TRAFFIC_VEHICLES] = traffic_states
        active_obj = copy.copy(self.engine.agent_manager._active_objects)
        pending_obj = copy.copy(self.engine.agent_manager._pending_objects)
        dying_obj = copy.copy(self.engine.agent_manager._dying_objects)
        states[TARGET_VEHICLES] = {k: v.get_state() for k, v in active_obj.items()}
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v.get_state()
                                      for k, v in pending_obj.items()}, allow_new_keys=True
        )
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v_count[0].get_state()
                                      for k, v_count in dying_obj.items()},
            allow_new_keys=True
        )

        states[OBJECT_TO_AGENT] = copy.deepcopy(self.engine.agent_manager._object_to_agent)
        states[AGENT_TO_OBJECT] = copy.deepcopy(self.engine.agent_manager._agent_to_object)
        return states

    def get_global_init_states(self) -> Dict:
        """
        特殊处理交通车辆的初始状态
        :return: 所有车辆初始状态
        """
        vehicles = dict()
        for vehicle in self._traffic_vehicles:
            init_state = vehicle.get_state()
            init_state["index"] = vehicle.index
            init_state["type"] = vehicle.class_name
            init_state["enable_respawn"] = vehicle.enable_respawn
            vehicles[vehicle.index] = init_state

        # 收集其他车辆的初始状态（触发/混合模式）
        if self.mode in {TrafficMode.Trigger, TrafficMode.Hybrid}:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    init_state = vehicle.get_state()
                    init_state["type"] = vehicle.class_name
                    init_state["index"] = vehicle.index
                    init_state["enable_respawn"] = vehicle.enable_respawn
                    vehicles[vehicle.index] = init_state
        return vehicles

    def _propose_vehicle_configs(self, lane: AbstractLane):
        """为给定车道生成候选车辆配置"""
        potential_vehicle_configs = []
        total_num = int(lane.length / self.VEHICLE_GAP)
        vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
        # 按固定间距生成候选位置
        for long in vehicle_longs:
            random_vehicle_config = {"spawn_lane_index": lane.index, "spawn_longitude": long, "enable_reverse": False}
            potential_vehicle_configs.append(random_vehicle_config)
        return potential_vehicle_configs

    def _create_basic_vehicles(self, map: BaseMap, traffic_density: float):
        """基础模式：在所有可重生车道上生成车辆"""
        total_num = len(self.respawn_lanes)
        for lane in self.respawn_lanes:
            _traffic_vehicles = []
            total_num = int(lane.length / self.VEHICLE_GAP)
            vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
            self.np_random.shuffle(vehicle_longs)
            # 根据密度选择生成数量
            for long in vehicle_longs[:int(np.ceil(traffic_density * len(vehicle_longs)))]:
                vehicle_type = self.random_vehicle_type()
                traffic_v_config = {"spawn_lane_index": lane.index, "spawn_longitude": long}
                traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                from metadrive.policy.idm_policy import IDMPolicy
                self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                self._traffic_vehicles.append(random_v)

    def _create_trigger_vehicles(self, map: BaseMap, traffic_density: float) -> None:
        """
        【触发模式核心方法】按路段分组预生成车辆，等待自车靠近时激活
        
        工作流程：
        1. 遍历地图每个路段块（block）
        2. 在该路段的中间车道上预生成指定密度的车辆
        3. 将车辆按路段分组存储到 block_triggered_vehicles
        4. 反转列表，确保从远到近的顺序触发
        
        :param map: 地图对象
        :param traffic_density: 交通密度（可在每回合调整）
        :return: None
        """
        vehicle_num = 0
        # 遍历除起点外的所有路段块
        for block in map.blocks[1:]:

            # 获取该路段块的候选生成车道
            trigger_lanes = block.get_intermediate_spawn_lanes()
            
            # 如果需要反向交通流，添加负向车道
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            
            # 生成所有候选车辆配置
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    # 跳过事故车道
                    if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(l)

            # 计算该路段应生成的车辆数量
            total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            total_vehicles = int(math.floor(total_spawn_points * traffic_density))

            # 随机选择指定数量的配置并生成车辆
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]

            from metadrive.policy.idm_policy import IDMPolicy
            for v_config in selected:
                vehicle_type = self.random_vehicle_type()
                v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
                seed = self.generate_seed()
                self.add_policy(random_v.id, IDMPolicy, random_v, seed)
                vehicles_on_block.append(random_v.name)

            # 将该路段的车辆打包成 BlockVehicles 对象
            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

            # 添加到待触发列表
            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        
        # 反转列表，使远处的路段先被触发（栈结构）
        self.block_triggered_vehicles.reverse()

    def _get_available_respawn_lanes(self, map: BaseMap) -> list:
        """
        查找可用的重生车道
        :param map: 从中选择重生车道的地图
        :return: 重生车道列表
        """
        respawn_lanes = []
        respawn_roads = []
        for block in map.blocks:
            roads = block.get_respawn_roads()
            for road in roads:
                if road in respawn_roads:
                    respawn_roads.remove(road)
                else:
                    respawn_roads.append(road)
        for road in respawn_roads:
            respawn_lanes += road.get_lanes(map.road_network)
        return respawn_lanes

    def random_vehicle_type(self):
        """随机选择车辆类型"""
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random, [0.2, 0.3, 0.3, 0.2, 0.0])
        return vehicle_type

    def destroy(self) -> None:
        """
        销毁函数，释放资源
        :return: None
        """
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []
        # current map

        # traffic vehicle list
        self.block_triggered_vehicles = []

        # traffic property
        self.mode = None
        self.random_traffic = None
        self.density = None

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def __repr__(self):
        return self._traffic_vehicles.__repr__()

    @property
    def vehicles(self):
        """获取所有车辆（包括自车和交通车）"""
        return list(self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())

    @property
    def traffic_vehicles(self):
        """获取所有交通车辆"""
        return list(self._traffic_vehicles)

    def seed(self, random_seed):
        """设置随机种子"""
        if not self.random_traffic:
            super(PGTrafficManager, self).seed(random_seed)

    @property
    def current_map(self):
        """获取当前地图"""
        return self.engine.map_manager.current_map

    def get_state(self):
        """获取管理器状态（用于存档）"""
        ret = super(PGTrafficManager, self).get_state()
        ret["_traffic_vehicles"] = [v.name for v in self._traffic_vehicles]
        flat = []
        for b_v in self.block_triggered_vehicles:
            flat.append((b_v.trigger_road.start_node, b_v.trigger_road.end_node, b_v.vehicles))
        ret["block_triggered_vehicles"] = flat
        return ret

    def set_state(self, state: dict, old_name_to_current=None):
        """从存档恢复管理器状态"""
        super(PGTrafficManager, self).set_state(state, old_name_to_current)
        self._traffic_vehicles = list(
            self.get_objects([old_name_to_current[name] for name in state["_traffic_vehicles"]]).values()
        )
        self.block_triggered_vehicles = [
            BlockVehicles(trigger_road=Road(s, e), vehicles=[old_name_to_current[name] for name in v])
            for s, e, v in state["block_triggered_vehicles"]
        ]


# For compatibility check
TrafficManager = PGTrafficManager


class MixedPGTrafficManager(PGTrafficManager):
    """混合交通管理器：部分车辆使用RL策略，部分使用IDM策略"""
    
    def _create_basic_vehicles(self, *args, **kwargs):
        raise NotImplementedError()

    def _create_trigger_vehicles(self, map: BaseMap, traffic_density: float) -> None:
        """
        【混合模式触发方法】与父类类似，但支持为部分车辆分配RL专家策略
        
        区别：根据 rl_agent_ratio 概率决定使用 ExpertPolicy 还是 IDMPolicy
        """
        vehicle_num = 0
        for block in map.blocks[1:]:

            # 获取候选生成车道
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(l)

            # 计算应生成的车辆数量
            total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            total_vehicles = int(math.floor(total_spawn_points * traffic_density))

            # 生成车辆
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]

            from metadrive.policy.idm_policy import IDMPolicy
            from metadrive.policy.expert_policy import ExpertPolicy
            for v_config in selected:
                vehicle_type = self.random_vehicle_type()
                v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
                
                # 根据比例分配RL专家策略或IDM策略
                if self.np_random.random() < self.engine.global_config["rl_agent_ratio"]:
                    self.add_policy(random_v.id, ExpertPolicy, random_v, self.generate_seed())
                else:
                    self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                vehicles_on_block.append(random_v.name)

            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()
