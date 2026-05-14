"""
MetaDrive标准环境类 - 提供完整的驾驶仿真环境实现

该模块实现了MetaDriveEnv类,继承自BaseEnv,是MetaDrive的核心环境实现。
主要功能包括:
- 程序化地图生成配置(Procedural Generation Map)
- 交通流管理(车辆密度、生成模式)
- 自定义奖励函数(前进奖励、速度奖励、惩罚项)
- 安全代价函数(碰撞、越界等安全约束信号)
- 终止条件判断(成功到达、碰撞、越界、超时等)
- 目标到达检测
- 道路偏离检测(连续线、虚线、路肩等)

此环境专为强化学习训练设计,特别适用于安全强化学习(Safe RL)研究。
"""

import copy
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from typing import Union

import numpy as np
from risk_field import RiskFieldCalculator

from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import clip, Config

# ==================== 线性映射工具函数 ====================
def lmap(value, old_range, new_range):
    """
    线性映射函数：将 value 从 old_range 映射到 new_range
    
    Args:
        value: 待映射的值
        old_range: [old_min, old_max] 原始范围
        new_range: [new_min, new_max] 目标范围
        
    Returns:
        映射后的值
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    
    # 防止除零
    if old_max == old_min:
        return new_min
    
    # 线性映射公式
    mapped = new_min + (value - old_min) / (old_max - old_min) * (new_max - new_min)
    return mapped

# ==================== MetaDrive默认配置 ====================
# 定义了MetaDrive环境的所有可配置参数及其默认值
METADRIVE_DEFAULT_CONFIG = dict(
    # ===== 泛化配置 =====
    start_seed=0,  # 起始场景种子索引
    num_scenarios=1,  # 场景总数

    # ===== 程序化生成(PG)地图配置 =====
    map=3,  # int或string: 填充map_config的简便方式,表示地图复杂度
    block_dist_config=PGBlockDistConfig,  # 区块分布配置类
    random_lane_width=False,  # 是否随机化车道宽度
    random_lane_num=False,  # 是否随机化车道数量
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,  # 地图生成类型: 大区块数量
        BaseMap.GENERATE_CONFIG: None,  # 可以是文件路径/区块数量/区块ID序列
        BaseMap.LANE_WIDTH: 3.5,  # 车道宽度(米)
        BaseMap.LANE_NUM: 3,  # 车道数量
        "exit_length": 50,  # 出口长度(米)
        "start_position": [0, 0],  # 起始位置坐标
    },
    store_map=True,  # 是否缓存地图以提高性能

    # ===== 交通流配置 =====
    traffic_density=0.1,  # 交通密度(0-1),控制道路上其他车辆的数量
    need_inverse_traffic=False,  # 是否需要反向交通流
    traffic_mode=TrafficMode.Trigger,  # 交通模式: "Respawn"(重生)或"Trigger"(触发)
    random_traffic=False,  # 交通是否随机化(默认为False)
    remove_crashed_traffic_vehicle=True,  # 是否移除 crash_vehicle=True 的周车,避免非终止碰撞后堵在场景中
    # 这将更新车辆配置并应用到交通车辆
    traffic_vehicle_config=dict(
        show_navi_mark=False,  # 不显示导航标记
        show_dest_mark=False,  # 不显示目标标记
        enable_reverse=False,  # 禁止倒车
        show_lidar=False,  # 不显示激光雷达
        show_lane_line_detector=False,  # 不显示车道线检测器
        show_side_detector=False,  # 不显示侧向检测器
    ),

    # ===== 障碍物配置 =====
    accident_prob=0.05,  # 每个区块发生事故的概率,多出口区块除外
    static_traffic_object=True,  # 交通物体不会对任何碰撞做出反应(静态障碍物)

    # ===== 其他配置 =====
    use_AI_protector=False,  # 是否使用AI保护器防止极端行为
    save_level=0.5,  # 保存级别
    horizon=1000,  # episode最大步数

    # ===== 智能体配置 =====
    random_spawn_lane_index=True,  # 是否随机生成车道索引
    vehicle_config=dict(navigation_module=NodeNetworkNavigation),  # 使用节点网络导航模块
    agent_configs={
        DEFAULT_AGENT: dict(
            use_special_color=True,  # 使用特殊颜色(绿色)标识主车
            spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),  # 初始生成车道
        )
    },

    # ===== 奖励方案配置 =====
    # 详见: https://github.com/metadriverse/metadrive/issues/283
    success_reward=10.0,  # 成功到达目标的奖励
    out_of_road_penalty=5.0,  # 偏离道路的惩罚
    crash_vehicle_penalty=5.0,  # 碰撞车辆的惩罚
    crash_object_penalty=5.0,  # 碰撞障碍物的惩罚
    crash_sidewalk_penalty=0.0,  # 碰撞人行道的惩罚
    driving_reward=1.0,  # 前进距离奖励系数
    speed_reward=0.2,  # 速度奖励系数
    # 横向正奖励总开关:
    # - False: 完全关闭该项, reward 退回到“前进奖励 + 速度奖励 + 事件惩罚”的原始结构
    # - True: 启用“小权重、平台型、带速度门控”的车道保持正奖励
    
    use_lateral_reward=True,
    # 横向正奖励的单步最大附加值。建议保持在 0.01~0.03 之间,
    # 避免量级过大而盖过现有的速度奖励/前进奖励。
    lateral_reward_weight=1.5,
    # 平台宽度(归一化半车道宽坐标):
    # - lateral_norm = |lateral_now| / (lane_width / 2)
    # - 当 lateral_norm <= safe_band 时给满额正奖励
    # - 这样能鼓励“别太贴边”,但不会逼着策略死贴车道中心
    lateral_reward_safe_band=0.25,
    # 低速不发放横向正奖励,避免策略通过低速/原地小抖动“蹭横向分”
    lateral_reward_min_speed=8.0,   # km/h, 低于该速度门控为0
    lateral_reward_full_speed=20.0, # km/h, 高于该速度门控为1
    # 压虚线/合法变道时暂时关闭横向正奖励,降低它对超车/变道的干扰
    lateral_reward_zero_on_broken_line=False,
    # 轻量转向惩罚总开关:
    # - False: 不额外约束左右抖动
    # - True: 对“转向变化过快”和“左右反复换向”施加轻量惩罚
    use_steering_penalty=False,
    # 师兄同款的绝对转向惩罚总开关:
    # - False: 不惩罚转向幅值本身
    # - True: 对真实执行转向施加 w * steer^2 惩罚
    # 这里使用 vehicle.steering, 对 ResAct 链路等价于惩罚最终执行后的真实转向。
    use_absolute_steering_penalty=True,
    # 一阶转向变化惩罚,主要抑制“每步都大幅改方向”
    steering_smoothness_weight=0,
    # 左右快速换向惩罚,专门针对“左一下右一下”的高频切换
    steering_switch_penalty_weight=100.0,
    steering_switch_count_penalty_weight = 0.0,
    # 绝对转向惩罚系数,直接复现 reward -= w * steer^2 的效果
    steering_absolute_penalty_weight=1.2,
    reward_speed_range=[10, 30], # 奖励速度范围 km/h
   
   
    # 风险场 reward shaping 总开关:
    # - False: 完全关闭风险场 reward, 行为与当前 reward 分解保持一致
    # - True: 复用 risk_field.py 的原始风险分量,对基础 reward 施加一个轻量惩罚
    use_risk_field_reward=True,
    # 风险场 reward 的缩放系数。保持较小量级,避免盖过前进奖励和速度奖励。
    risk_field_reward_scale=25.0,   
    # reward 侧使用的风险场权重:
    # 仅将道路边界/车道线/冲出车道/周车风险接入 reward。
    risk_field_boundary_weight=2.0,
    risk_field_lane_weight=0.1,
    risk_field_offroad_weight=2.0,
    risk_field_vehicle_weight=2.5,
    # 这些分量保留在原始风险信息中,默认不参与 reward 聚合。
    risk_field_object_weight=1.0,
    risk_field_headway_weight=0.0,
    risk_field_ttc_weight=0.0,
    # 用于将原始风险和归一化到 [0, 1] 再缩放为轻量 reward 惩罚。
    risk_field_raw_clip=10.0,

    # ===== 代价方案配置 (用于安全强化学习) =====
    crash_vehicle_cost=1.0,  # 碰撞车辆的代价值
    crash_object_cost=1.0,  # 碰撞障碍物的代价值
    out_of_road_cost=1.0,  # 偏离道路的代价值

    # ===== 终止方案配置 =====
    out_of_route_done=False,  # 偏离路线是否终止
    out_of_road_done=True,  # 偏离道路是否终止
    on_continuous_line_done=True,  # 压在连续线(黄线/白线)上是否终止
    on_broken_line_done=False,  # 压在虚线上是否终止
    crash_vehicle_done=True,  # 碰撞车辆是否终止
    crash_object_done=True,  # 碰撞障碍物是否终止
    crash_human_done=True,  # 碰撞行人是否终止
)


class MetaDriveEnv(BaseEnv):
    """
    MetaDrive标准环境类
    
    继承自BaseEnv,实现了完整的驾驶仿真环境,包括:
    - 程序化地图生成
    - 动态交通流
    - 自定义奖励/代价/终止逻辑
    - 目标检测和道路偏离检测
    
    适用于单智能体和多智能体强化学习训练,特别适合安全强化学习研究。
    """
    
    @classmethod
    def default_config(cls) -> Config:
        """
        获取默认配置
        
        Returns:
            Config对象,包含BaseEnv和MetaDriveEnv的所有默认配置
        """
        config = super(MetaDriveEnv, cls).default_config()
        config.update(METADRIVE_DEFAULT_CONFIG)
        config.register_type("map", str, int)  # 注册map参数类型为str或int
        config["map_config"].register_type("config", None)  # 注册map_config.config类型
        return config

    def __init__(self, config: Union[dict, None] = None):
        """
        初始化MetaDrive环境
        
        Args:
            config: 环境配置字典,将与默认配置合并
        """
        # 保存不可变的默认配置副本,用于后续解析
        print("本地MetaDrive环境")
        # print("本地MetaDrive环境")
        # print("本地MetaDrive环境")
        # print("本地MetaDrive环境")
        # print("本地MetaDrive环境")
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(MetaDriveEnv, self).__init__(config)

        # 场景设置
        self.start_seed = self.start_index = self.config["start_seed"]  # 起始种子索引
        self.env_num = self.num_scenarios  # 环境数量(场景数)

    def _post_process_config(self, config):
        """
        对配置进行后处理,解析地图配置并合并车辆配置
        
        Args:
            config: 合并后的配置字典
            
        Returns:
            处理后的配置字典
        """
        config = super(MetaDriveEnv, self)._post_process_config(config)
        
        # 检查像素归一化设置
        if not config["norm_pixel"]:
            self.logger.warning(
                "您已设置 norm_pixel = False,这意味着观测值将是[0, 255]范围内的uint8值。"
                "请确保在输入网络之前对其进行解析!"
            )

        # 解析地图配置: 将简化的map参数转换为完整的map_config
        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], 
            new_map_config=config["map_config"], 
            default_config=self.default_config_copy
        )
        
        # 将全局配置传递到车辆配置中
        config["vehicle_config"]["norm_pixel"] = config["norm_pixel"]
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        
        # 对于单智能体环境,将agent_configs合并到vehicle_config
        target_v_config = copy.deepcopy(config["vehicle_config"])
        if not config["is_multi_agent"]:
            target_v_config.update(config["agent_configs"][DEFAULT_AGENT])
            config["agent_configs"][DEFAULT_AGENT] = target_v_config
        return config

    def done_function(self, vehicle_id: str):
        """
        终止函数 - 判断智能体是否应该终止episode
        
        检查多种终止条件:
        - 成功到达目的地
        - 偏离道路
        - 碰撞车辆/障碍物/建筑/行人/人行道
        - 达到最大步数
        
        Args:
            vehicle_id: 车辆ID
            
        Returns:
            done: 是否终止
            done_info: 包含各种终止原因的详细信息字典
        """
        vehicle = self.agents[vehicle_id]
        done = False
        
        # 检查是否达到最大步数
        max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        
        # 构建终止信息字典,记录所有可能的终止原因
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,  # 碰撞车辆
            TerminationState.CRASH_OBJECT: vehicle.crash_object,  # 碰撞障碍物
            TerminationState.CRASH_BUILDING: vehicle.crash_building,  # 碰撞建筑
            TerminationState.CRASH_HUMAN: vehicle.crash_human,  # 碰撞行人
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,  # 碰撞人行道
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),  # 偏离道路
            TerminationState.SUCCESS: self._is_arrive_destination(vehicle),  # 成功到达
            TerminationState.MAX_STEP: max_step,  # 达到最大步数
            TerminationState.ENV_SEED: self.current_seed,  # 环境种子
            # TerminationState.CURRENT_BLOCK: self.agent.navigation.current_road.block_ID(),
        }

        # 兼容性处理: crash几乎等同于与车辆碰撞
        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING] or done_info[TerminationState.CRASH_SIDEWALK]
            or done_info[TerminationState.CRASH_HUMAN]
        )

        # 根据配置确定环境返回的终止状态
        if done_info[TerminationState.SUCCESS]:
            done = True
            self.logger.debug(
                "Episode结束! 场景索引: {} 原因: 到达目的地".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.OUT_OF_ROAD] and self.config["out_of_road_done"]:
            done = True
            self.logger.debug(
                "Episode结束! 场景索引: {} 原因: 偏离道路".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.debug(
                "Episode结束! 场景索引: {} 原因: 碰撞车辆".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.debug(
                "Episode结束! 场景索引: {} 原因: 碰撞障碍物".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_BUILDING]:
            done = True
            self.logger.debug(
                "Episode结束! 场景索引: {} 原因: 碰撞建筑".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.debug(
                "Episode结束! 场景索引: {} 原因: 碰撞行人".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.MAX_STEP]:
            # 单智能体的horizon与max_step_per_agent含义相同
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.debug(
                "Episode结束! 场景索引: {} 原因: 达到最大步数".format(self.current_seed),
                extra={"log_once": True}
            )
        return done, done_info

    def cost_function(self, vehicle_id: str):
        """
        代价函数 - 用于安全强化学习的约束信号
        
        计算当前步骤的安全代价,优先级:
        1. 偏离道路 > 2. 碰撞车辆 > 3. 碰撞障碍物
        
        Args:
            vehicle_id: 车辆ID
            
        Returns:
            cost: 代价值(标量)
            step_info: 代价详细信息字典
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        
        # 按优先级检查各种不安全事件
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        
        return step_info['cost'], step_info

    def _get_risk_field_calculator(self):
        """懒加载风险场计算器,避免每步重复构造。"""
        calculator = getattr(self, "_risk_field_calculator", None)
        if calculator is None:
            calculator = RiskFieldCalculator(self.config)
            self._risk_field_calculator = calculator
        return calculator

    def _risk_field_reward_enabled(self) -> bool:
        """风险场 reward 总开关。"""
        return bool(self.config.get("use_risk_field_reward", False))

    def _risk_field_reward_penalty(self, risk_info):
        """将原始风险分量聚合成一个轻量 reward 惩罚。"""
        reward_source_cost = (
            float(self.config["risk_field_boundary_weight"]) * float(risk_info.get("risk_field_boundary_cost", 0.0))
            + float(self.config["risk_field_lane_weight"]) * float(risk_info.get("risk_field_lane_cost", 0.0))
            + float(self.config["risk_field_offroad_weight"]) * float(risk_info.get("risk_field_offroad_cost", 0.0))
            + float(self.config["risk_field_vehicle_weight"]) * float(risk_info.get("risk_field_vehicle_cost", 0.0))
            + float(self.config["risk_field_object_weight"]) * float(risk_info.get("risk_field_object_cost", 0.0))        )
        normalized_cost = clip(
            reward_source_cost / max(float(self.config["risk_field_raw_clip"]), 1e-6),
            0.0,
            1.0,
        )
        reward_penalty = float(self.config["risk_field_reward_scale"]) * normalized_cost
        return float(reward_source_cost), float(normalized_cost), float(reward_penalty)

    def _compute_steering_penalty(self, vehicle, first_step=False):
        """计算转向惩罚及诊断项。

        两类开关相互独立:
        - use_steering_penalty: 控制平滑/切换类惩罚
        - use_absolute_steering_penalty: 控制师兄同款 steer^2 惩罚
        """
        current_steering = float(vehicle.steering)
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

        use_smooth_switch_penalty = bool(self.config.get("use_steering_penalty", False))
        steering_smoothness_penalty = 0.0
        steering_switch_penalty = 0.0
        steering_switch_count_penalty = 0.0
        if use_smooth_switch_penalty:
            steering_smoothness_penalty = float(self.config["steering_smoothness_weight"]) * steering_delta
            steering_switch_penalty = float(self.config["steering_switch_penalty_weight"]) * steering_switch
            steering_switch_count_penalty = (
                float(self.config["steering_switch_count_penalty_weight"]) * steering_switch_count
            )
        steering_absolute_value = abs(current_steering)
        steering_absolute_penalty = 0.0
        if self.config.get("use_absolute_steering_penalty", False):
            steering_absolute_penalty = (
                float(self.config["steering_absolute_penalty_weight"])
                * (current_steering ** 2)
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
            "steering_absolute_value": float(steering_absolute_value),
            "steering_delta": float(steering_delta),
            "steering_switch": float(steering_switch),
            "steering_switch_count": float(steering_switch_count),
            "steering_smoothness_penalty": float(steering_smoothness_penalty),
            "steering_switch_penalty": float(steering_switch_penalty),
            "steering_switch_count_penalty": float(steering_switch_count_penalty),
            "steering_absolute_penalty": float(steering_absolute_penalty),
            "steering_penalty": float(steering_penalty),
        }

    @staticmethod
    def _is_arrive_destination(vehicle):
        """
        判断车辆是否到达目的地
        
        检查条件:
        1. 纵向位置在终点前后5米范围内
        2. 横向位置在当前车道内
        
        Args:
            vehicle: BaseVehicle实例
            
        Returns:
            flag: 车辆是否到达目的地
        """
        # 获取车辆在最终车道上的局部坐标(纵向, 横向)
        long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
        
        # 检查是否在终点附近且在车道内
        flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
            vehicle.navigation.get_current_lane_width() / 2 >= lat >=
            (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
        )
        return flag

    def _is_out_of_road(self, vehicle):
        """
        判断车辆是否偏离道路
        
        根据配置检查多种偏离情况:
        - 不在车道上
        - 偏离路线(如果启用out_of_route_done)
        - 压在连续线上(黄线/白线)或碰撞人行道(如果启用on_continuous_line_done)
        - 压在虚线上(如果启用on_broken_line_done)
        
        Args:
            vehicle: 车辆对象
            
        Returns:
            ret: 是否偏离道路
        """
        # 基本检查: 是否在车道上
        ret = not vehicle.on_lane
        
        # 如果启用偏离路线检测
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        # 否则检查连续线和人行道
        elif self.config["on_continuous_line_done"]:
            ret = ret or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or vehicle.crash_sidewalk
        
        # 检查虚线
        if self.config["on_broken_line_done"]:
            ret = ret or vehicle.on_broken_line
        
        return ret

    def reward_function(self, vehicle_id: str):
        """
        奖励函数 - 计算当前步骤的奖励值
        
        奖励组成:
        1. 前进奖励: 基于在当前车道上的前进距离
        2. 速度奖励: 基于当前速度比例（使用线性映射归一化）
        3. 横向保持奖励(可选): 小权重、平台型正奖励,鼓励不要长期贴近车道边缘
        4. 事件奖励/惩罚: 到达目标、偏离道路、碰撞等
        
        Args:
            vehicle_id: 车辆ID
            
        Returns:
            reward: 奖励值(标量)
            step_info: 奖励详细信息字典
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # 确定当前参考车道
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1  # 正向道路
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1  # 反向道路为-1
        
        # 计算上一位置和当前位置在车道上的局部坐标
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)
        
        # ===== 分解后的奖励子项 =====
        # 把 reward 拆成几项单独记录，便于在日志里判断:
        # - 是前进奖励没起来
        # - 还是速度奖励长期为负/过低
        # - 还是横向正奖励基本没激活
        reward = 0.0

        # 横向正奖励（可选）
        # 设计目标:
        # 1) 给“别太贴边”一个轻微正反馈;
        # 2) 不把车辆强行吸到车道中心;
        # 3) 低速和变道时尽量不干扰策略的自然行为。
        lateral_bonus = 0.0
        lateral_norm = 0.0
        lateral_score = 0.0
        lateral_speed_gate = 0.0
        lateral_forward_gate = 0.0
        lateral_broken_line_gate = 1.0
        if self.config["use_lateral_reward"]:
            lane_width = max(vehicle.navigation.get_current_lane_width(), 1e-6)
            half_lane_width = 0.5 * lane_width
            lateral_norm = abs(lateral_now) / half_lane_width  # 0 在车道中心, 1 在车道边缘

            # 平台型车道保持得分:
            # - 在 safe_band 内统一给满分,不追求“越居中越高”
            # - 超出 safe_band 后再平方衰减,对贴边更敏感
            safe_band = clip(self.config["lateral_reward_safe_band"], 0.0, 0.95)
            if lateral_norm <= safe_band:
                lateral_score = 1.0
            else:
                edge_excess = (lateral_norm - safe_band) / max(1.0 - safe_band, 1e-6)
                lateral_score = 1.0 - clip(edge_excess, 0.0, 1.0) ** 2

            # 速度门控:
            # - 低于 min_speed 时不给横向正奖励
            # - 高于 full_speed 时门控拉满
            # 这样能减少“低速蹭横向正奖励”的现象
            min_speed = float(self.config["lateral_reward_min_speed"])
            full_speed = max(float(self.config["lateral_reward_full_speed"]), min_speed + 1e-6)
            lateral_speed_gate = clip((vehicle.speed_km_h - min_speed) / (full_speed - min_speed), 0.0, 1.0)

            # 只有当前步确实在向前推进时才发放该正奖励
            lateral_forward_gate = 1.0 if (long_now - long_last) > 0.0 else 0.0

            # 合法压虚线变道时暂时关掉横向正奖励,避免影响超车/变道
            if self.config["lateral_reward_zero_on_broken_line"] and vehicle.on_broken_line:
                lateral_broken_line_gate = 0.0

            lateral_bonus = (
                self.config["lateral_reward_weight"]
                * lateral_score
                * lateral_speed_gate
                * lateral_forward_gate
                * lateral_broken_line_gate
            )

        longitudinal_progress = float(long_now - long_last)
        driving_component = self.config["driving_reward"] * longitudinal_progress * positive_road
        
        # ⭐ 速度奖励: 使用线性映射归一化到 [-1, 1]
        # 将速度从 reward_speed_range 映射到 [-1, 1]
        scaled_speed = lmap(vehicle.speed_km_h, self.config["reward_speed_range"], [-1, 1])
        # 裁剪到合法范围并考虑道路方向
        normalized_speed_reward = clip(scaled_speed, -1, 1) * positive_road
        speed_component = self.config["speed_reward"] * normalized_speed_reward

        # reward += self.config["speed_reward"] * (vehicle.speed_km_h / (35.0) * positive_road)
        reward = lateral_bonus + driving_component + speed_component
        base_reward = float(reward)
        steering_info = {
            "action_steering": float(vehicle.steering),
            "steering_absolute_value": abs(float(vehicle.steering)),
            "steering_delta": 0.0,
            "steering_switch": 0.0,
            "steering_switch_count": 0.0,
            "steering_smoothness_penalty": 0.0,
            "steering_switch_penalty": 0.0,
            "steering_switch_count_penalty": 0.0,
            "steering_absolute_penalty": 0.0,
            "steering_penalty": 0.0,
        }
        if self.config["use_steering_penalty"] or self.config.get("use_absolute_steering_penalty", False):
            steering_info = self._compute_steering_penalty(
                vehicle,
                first_step=self.episode_lengths.get(vehicle_id, 0) == 0,
            )
            reward = base_reward - steering_info["steering_penalty"]

        steering_penalty = float(steering_info["steering_penalty"])
        reward_after_steering = float(reward)
        risk_field_reward_source_cost = 0.0
        risk_field_reward_normalized_cost = 0.0
        risk_field_reward_penalty = 0.0
        step_info["risk_field_cost"] = 0.0
        step_info["risk_field_road_cost"] = 0.0
        step_info["risk_field_boundary_cost"] = 0.0
        step_info["risk_field_lane_cost"] = 0.0
        step_info["risk_field_offroad_cost"] = 0.0
        step_info["risk_field_vehicle_cost"] = 0.0

        if self._risk_field_reward_enabled():
            _, risk_info = self._get_risk_field_calculator().calculate(self, vehicle)
            step_info.update(risk_info)
            (
                risk_field_reward_source_cost,
                risk_field_reward_normalized_cost,
                risk_field_reward_penalty,
            ) = self._risk_field_reward_penalty(risk_info)
            reward = reward_after_steering - risk_field_reward_penalty

        shaped_reward = float(reward)
        
        step_info["step_reward"] = shaped_reward
        step_info["base_reward"] = base_reward
        step_info["driving_component_reward"] = float(driving_component)
        step_info["speed_component_reward"] = float(speed_component)
        step_info["normalized_speed_reward"] = float(normalized_speed_reward)
        step_info["longitudinal_progress"] = longitudinal_progress
        step_info["positive_road"] = float(positive_road)
        step_info["speed_km_h"] = vehicle.speed_km_h
        step_info["scaled_speed"] = scaled_speed  # 记录映射后的速度值
        # 横向奖励相关信息全部留在 info 里,方便做可视化/调参/回滚
        step_info["lateral_now"] = abs(lateral_now)
        step_info["lateral_norm"] = lateral_norm
        step_info["lateral_score"] = lateral_score
        step_info["lateral_speed_gate"] = lateral_speed_gate
        step_info["lateral_forward_gate"] = lateral_forward_gate
        step_info["lateral_broken_line_gate"] = lateral_broken_line_gate
        step_info["lateral_reward"] = lateral_bonus
        step_info.update(steering_info)
        step_info["reward_after_steering_penalty"] = reward_after_steering
        step_info["risk_field_reward_source_cost"] = risk_field_reward_source_cost
        step_info["risk_field_reward_normalized_cost"] = risk_field_reward_normalized_cost
        step_info["risk_field_reward_penalty"] = risk_field_reward_penalty

        reward_override_active = 0.0
        # 检查是否发生重要事件,覆盖基础奖励
        if self._is_arrive_destination(vehicle):
            # 成功到达目的地: 大额正奖励
            reward = +self.config["success_reward"]
            reward_override_active = 1.0
        elif self._is_out_of_road(vehicle):
            # 偏离道路: 惩罚
            reward = -self.config["out_of_road_penalty"]
            reward_override_active = 1.0
        elif vehicle.crash_vehicle:
            # 碰撞车辆: 惩罚
            reward = -self.config["crash_vehicle_penalty"]
            reward_override_active = 1.0
        elif vehicle.crash_object:
            # 碰撞障碍物: 惩罚
            reward = -self.config["crash_object_penalty"]
            reward_override_active = 1.0
        elif vehicle.crash_sidewalk:
            # 碰撞人行道: 惩罚
            reward = -self.config["crash_sidewalk_penalty"]
            reward_override_active = 1.0
        
        # 记录路线完成度
        step_info["reward_override_active"] = reward_override_active
        step_info["reward_override_delta"] = float(reward - shaped_reward)
        step_info["final_reward"] = float(reward)
        step_info["route_completion"] = vehicle.navigation.route_completion

        return reward, step_info

    def setup_engine(self):
        """
        设置引擎管理器
        
        注册以下管理器:
        - PGMapManager: 程序化生成地图管理器
        - PGTrafficManager: 程序化生成交通流管理器
        - TrafficObjectManager: 交通障碍物管理器(仅在accident_prob > 0时注册)
        """
        super(MetaDriveEnv, self).setup_engine()
        from metadrive.manager.traffic_manager import PGTrafficManager
        from metadrive.manager.pg_map_manager import PGMapManager
        from metadrive.manager.object_manager import TrafficObjectManager
        
        # 注册地图管理器
        self.engine.register_manager("map_manager", PGMapManager())
        # 注册交通流管理器
        self.engine.register_manager("traffic_manager", PGTrafficManager())
        # 仅在事故概率大于0时注册障碍物管理器
        if abs(self.config["accident_prob"] - 0) > 1e-2:
            self.engine.register_manager("object_manager", TrafficObjectManager())

if __name__ == '__main__':
    """测试代码 - 验证环境的基本功能"""

    def _act(env, action):
        """执行动作并验证返回值格式"""
        assert env.action_space.contains(action), "动作不在动作空间内"
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), "观测不在观测空间内"
        assert np.isscalar(reward), "奖励应该是标量"
        assert isinstance(info, dict), "info应该是字典"

    # 创建环境实例
    env = MetaDriveEnv()
    try:
        # 重置环境并验证初始观测
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        
        # 测试随机动作
        _act(env, env.action_space.sample())
        
        # 测试不同方向和转向的动作组合
        for x in [-1, 0, 1]:  # 转向: 左、直、右
            env.reset()
            for y in [-1, 0, 1]:  # 油门/刹车: 刹、怠、油
                _act(env, [x, y])
    finally:
        env.close()
