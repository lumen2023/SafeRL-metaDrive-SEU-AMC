"""
车辆智能体管理器 - 管理环境中活跃智能体与底层对象实例的映射关系

核心概念:
- agent name: 环境中的智能体名称，如 default_agent, agent0, agent1...
- object name: 每个对象的唯一名称，通常为随机字符串
"""
from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.logger import get_logger
from metadrive.manager.base_manager import BaseAgentManager
from metadrive.policy.AI_protect_policy import AIProtectPolicy
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy, TakeoverPolicy, TakeoverPolicyWithoutBrake
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy

logger = get_logger()


class VehicleAgentManager(BaseAgentManager):
    """
    车辆智能体管理器
    
    职责:
    - 维护智能体名称与车辆对象实例的映射关系
    - 管理车辆的创建、销毁和重生逻辑
    - 处理策略动作执行和观测空间管理
    """
    INITIALIZED = False  # 车辆实例创建后设为 True

    def __init__(self, init_observations):
        """
        初始化智能体管理器
        
        参数:
            init_observations: 初始观测空间配置字典
        """
        super(VehicleAgentManager, self).__init__(init_observations)
        
        # 多智能体环境配置（在 init() 中更新）
        self._allow_respawn = None      # 是否允许重生
        self._delay_done = None         # 延迟终止倒计时步数
        self._infinite_agents = None    # 是否无限智能体模式
        
        # 车辆状态管理
        self._dying_objects = {}                    # 待回收车辆字典 {vehicle_name: [vehicle, countdown]}
        self._agents_finished_this_frame = dict()   # 本帧终止的智能体 {agent_name: vehicle_name}
        self.next_agent_count = 0                   # 下一个智能体计数器

    def _create_agents(self, config_dict: dict):
        """
        根据配置字典创建智能体车辆
        
        流程:
        1. 选择车辆类型（随机或指定）
        2. 生成车辆对象并绑定策略
        3. 返回 {agent_id: vehicle} 映射
        
        参数:
            config_dict: 智能体配置字典 {agent_id: vehicle_config}
            
        返回:
            dict: {agent_id: vehicle_instance}
        """
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        for agent_id, v_config in config_dict.items():
            # 选择车辆类型：随机模型或使用配置指定
            v_type = random_vehicle_type(self.np_random) if self.engine.global_config["random_agent_model"] else \
                vehicle_type[v_config["vehicle_model"] if v_config.get("vehicle_model", False) else "default"]

            # 生成车辆对象（强制复用名称或自动生成）
            obj_name = agent_id if self.engine.global_config["force_reuse_object_name"] else None
            obj = self.spawn_object(v_type, vehicle_config=v_config, name=obj_name)
            ret[agent_id] = obj
            
            # 绑定控制策略
            policy_cls = self.agent_policy
            args = [obj, self.generate_seed()]
            if policy_cls == TrajectoryIDMPolicy or issubclass(policy_cls, TrajectoryIDMPolicy):
                args.append(self.engine.map_manager.current_sdc_route)
            self.add_policy(obj.id, policy_cls, *args)
        return ret

    @property
    def agent_policy(self):
        """
        获取智能体策略类
        
        优先级:
        1. 接管策略 (TakeoverPolicy)
        2. 手控策略 (ManualControlPolicy / AIProtectPolicy)
        3. 默认策略 (从全局配置读取)
        
        返回:
            Policy class: 策略类对象
        """
        from metadrive.engine.engine_utils import get_global_config
        
        # 接管模式：共享 RL 智能体与外部设备的控制权
        if get_global_config()["agent_policy"] in [TakeoverPolicy, TakeoverPolicyWithoutBrake]:
            return get_global_config()["agent_policy"]
        
        # 手控模式
        if get_global_config()["manual_control"]:
            if get_global_config().get("use_AI_protector", False):
                policy = AIProtectPolicy  # AI 保护模式
            else:
                policy = ManualControlPolicy  # 纯手控
        else:
            policy = get_global_config()["agent_policy"]  # 默认策略
        
        return policy

    def before_reset(self):
        """
        重置前清理操作
        
        流程:
        1. 首次初始化基类
        2. 移除所有"濒死"车辆
        3. 调用所有活动车辆的 before_reset()
        4. 调用父类 before_reset() 清空对象
        """
        if not self.INITIALIZED:
            super(BaseAgentManager, self).__init__()
            self.INITIALIZED = True

        self.episode_created_agents = None

        # 非回放模式下，移除所有待回收车辆
        if not self.engine.replay_episode:
            for v in self.dying_agents.values():
                self._remove_vehicle(v)

        # 调用所有活动车辆和濒死车辆的 before_reset()
        for v in list(self._active_objects.values()) + [v for (v, _) in self._dying_objects.values()]:
            if hasattr(v, "before_reset"):
                v.before_reset()

        super(VehicleAgentManager, self).before_reset()

    def reset(self):
        """
        重置智能体管理器
        
        注意:
        - 在车辆实例创建后才真正初始化
        - 从全局配置读取重生相关参数
        """
        self.random_spawn_lane_in_single_agent()  # 单智能体模式下随机化出生车道
        config = self.engine.global_config
        self._delay_done = config["delay_done"]           # 延迟终止步数
        self._infinite_agents = config["num_agents"] == -1  # 无限智能体标志
        self._allow_respawn = config["allow_respawn"]       # 允许重生标志
        super(VehicleAgentManager, self).reset()

    def after_reset(self):
        """
        重置后初始化状态
        
        清空:
        - 濒死车辆字典
        - 本帧终止智能体记录
        - 更新下一智能体计数
        """
        super(VehicleAgentManager, self).after_reset()
        self._dying_objects = {}
        self._agents_finished_this_frame = dict()
        self.next_agent_count = len(self.episode_created_agents)

    def random_spawn_lane_in_single_agent(self):
        """
        单智能体模式下随机化出生车道索引
        
        条件:
        - 非多智能体环境
        - 启用了 random_spawn_lane_index 配置
        - 地图已加载
        """
        if not self.engine.global_config["is_multi_agent"] and \
                self.engine.global_config.get("random_spawn_lane_index", False) and self.engine.current_map is not None:
            spawn_road_start = self.engine.global_config["agent_configs"][DEFAULT_AGENT]["spawn_lane_index"][0]
            spawn_road_end = self.engine.global_config["agent_configs"][DEFAULT_AGENT]["spawn_lane_index"][1]
            index = self.np_random.randint(self.engine.current_map.config["lane_num"])
            self.engine.global_config["agent_configs"][DEFAULT_AGENT]["spawn_lane_index"] = (
                spawn_road_start, spawn_road_end, index
            )

    def _finish(self, agent_name, ignore_delay_done=False):
        """
        终止指定智能体
        
        流程:
        1. 从活动对象中弹出车辆
        2. 若启用延迟终止，加入濒死队列；否则直接移除
        3. 记录到本帧终止列表
        
        参数:
            agent_name: 智能体名称
            ignore_delay_done: 是否忽略延迟终止（成功到达终点时设为 True）
        """
        if not self.engine.replay_episode:
            vehicle_name = self._agent_to_object[agent_name]
            v = self._active_objects.pop(vehicle_name)
            
            # 延迟终止或直接移除
            if (not ignore_delay_done) and (self._delay_done > 0):
                self._put_to_dying_queue(v)  # 加入濒死队列
            else:
                self._remove_vehicle(v)  # 立即移除
            
            self._agents_finished_this_frame[agent_name] = v.name
            self._check()  # 调试检查

    def _check(self):
        """
        调试模式下的完整性检查
        
        验证:
        - 活动对象 + 濒死对象 的键集合 == 对象到智能体的映射键集合
        """
        if self._debug:
            current_keys = sorted(list(self._active_objects.keys()) + list(self._dying_objects.keys()))
            exist_keys = sorted(list(self._object_to_agent.keys()))
            assert current_keys == exist_keys, "You should confirm_respawn() after request for propose_new_vehicle()!"

    def propose_new_vehicle(self):
        """
        提议生成新车辆（多智能体重生机制）
        
        流程:
        1. 生成下一个智能体 ID
        2. 使用 agent0 的配置创建新车辆
        3. 建立智能体-对象双向映射
        4. 初始化观测空间和动作空间
        5. 激活车辆并返回相关信息
        
        返回:
            tuple: (agent_name, vehicle, step_info)
        """
        # 创建新车辆
        agent_name = self._next_agent_id()
        next_config = self.engine.global_config["agent_configs"]["agent0"]
        vehicle = self._create_agents({agent_name: next_config})[agent_name]
        new_v_name = vehicle.name
        
        # 建立双向映射
        self._agent_to_object[agent_name] = new_v_name
        self._object_to_agent[new_v_name] = agent_name
        
        # 初始化观测空间和动作空间（复用 agent0 的配置）
        self.observations[new_v_name] = self._init_observations["agent0"]
        self.observations[new_v_name].reset(vehicle)
        self.observation_spaces[new_v_name] = self._init_observation_spaces["agent0"]
        self.action_spaces[new_v_name] = self._init_action_spaces["agent0"]
        
        # 激活车辆
        self._active_objects[vehicle.name] = vehicle
        self._check()
        
        # 初始化车辆状态
        step_info = vehicle.before_step([0, 0])
        vehicle.set_static(False)  # 解除静态模式
        
        return agent_name, vehicle, step_info

    def _next_agent_id(self):
        """
        生成下一个智能体 ID
        
        格式: "agent0", "agent1", "agent2"...
        
        返回:
            str: 智能体名称
        """
        ret = "agent{}".format(self.next_agent_count)
        self.next_agent_count += 1
        return ret

    def set_allow_respawn(self, flag: bool):
        """
        设置是否允许重生
        
        参数:
            flag: True 允许重生，False 禁止
        """
        self._allow_respawn = flag

    def try_actuate_agent(self, step_infos, stage="before_step"):
        """
        尝试执行智能体策略动作
        
        两种阶段:
        - before_step: 决策型策略（如 RL 策略），在物理仿真前执行
        - after_step: 回放型策略（如 ReplayPolicy），在物理仿真后执行
        
        参数:
            step_infos: 步骤信息字典 {agent_id: info}
            stage: 执行阶段 ("before_step" 或 "after_step")
            
        返回:
            dict: 更新后的 step_infos
        """
        assert stage == "before_step" or stage == "after_step"
        
        for agent_id in self.active_agents.keys():
            policy = self.get_policy(self._agent_to_object[agent_id])
            is_replay = isinstance(policy, ReplayTrafficParticipantPolicy)
            assert policy is not None, "No policy is set for agent {}".format(agent_id)
            
            if is_replay:
                # 回放策略：在 after_step 阶段执行
                if stage == "after_step":
                    policy.act(agent_id)
                    step_infos[agent_id] = policy.get_action_info()
                else:
                    step_infos[agent_id] = self.get_agent(agent_id).before_step([0, 0])
            else:
                # 决策策略：在 before_step 阶段执行
                if stage == "before_step":
                    action = policy.act(agent_id)
                    step_infos[agent_id] = policy.get_action_info()
                    step_infos[agent_id].update(self.get_agent(agent_id).before_step(action))

        return step_infos

    def before_step(self):
        """
        每步执行前的处理
        
        主要任务:
        - 更新濒死车辆倒计时
        - 移除倒计时归零的车辆
        
        返回:
            dict: 步骤信息
        """
        step_infos = super(VehicleAgentManager, self).before_step()
        self._agents_finished_this_frame = dict()
        
        finished = set()
        for v_name in self._dying_objects.keys():
            self._dying_objects[v_name][1] -= 1  # 倒计时减 1
            if self._dying_objects[v_name][1] <= 0:  # 倒计时归零，移除车辆
                v = self._dying_objects[v_name][0]
                self._remove_vehicle(v)
                finished.add(v_name)
        
        for v_name in finished:
            self._dying_objects.pop(v_name)
        
        return step_infos

    def get_observations(self):
        """
        获取所有智能体的观测值
        
        包含:
        - 本帧终止智能体的最终观测
        - 所有活动智能体的当前观测
        
        返回:
            dict: {agent_id: observation}
        """
        if hasattr(self, "engine") and self.engine.replay_episode:
            return self.engine.replay_manager.get_replay_agent_observations()
        else:
            # 本帧终止智能体的观测
            ret = {
                old_agent_id: self.observations[v_name]
                for old_agent_id, v_name in self._agents_finished_this_frame.items()
            }
            # 活动智能体的观测
            for obj_id, observation in self.observations.items():
                if self.is_active_object(obj_id):
                    ret[self.object_to_agent(obj_id)] = observation
            return ret

    def get_observation_spaces(self):
        """
        获取所有智能体的观测空间
        
        返回:
            dict: {agent_id: observation_space}
        """
        ret = {
            old_agent_id: self.observation_spaces[v_name]
            for old_agent_id, v_name in self._agents_finished_this_frame.items()
        }
        for obj_id, space in self.observation_spaces.items():
            if self.is_active_object(obj_id):
                ret[self.object_to_agent(obj_id)] = space
        return ret

    @property
    def dying_agents(self):
        """
        获取所有濒死智能体
        
        返回:
            dict: {agent_name: vehicle}
        """
        assert not self.engine.replay_episode
        return {self._object_to_agent[k]: v for k, (v, _) in self._dying_objects.items()}

    @property
    def just_terminated_agents(self):
        """
        获取本帧刚终止的智能体
        
        返回:
            dict: {agent_name: vehicle}
        """
        assert not self.engine.replay_episode
        ret = {}
        for agent_name, v_name in self._agents_finished_this_frame.items():
            v = self.get_agent(v_name, raise_error=False)
            ret[agent_name] = v
        return ret

    def get_agent(self, agent_name, raise_error=True):
        """
        获取指定智能体对应的车辆对象
        
        搜索顺序:
        1. 活动对象字典
        2. 濒死对象字典
        
        参数:
            agent_name: 智能体名称
            raise_error: 未找到时是否抛出异常
            
        返回:
            BaseVehicle: 车辆对象实例，未找到返回 None
        """
        try:
            object_name = self.agent_to_object(agent_name)
        except KeyError:
            if raise_error:
                raise ValueError("Object {} not found!".format(object_name))
            else:
                return None
        
        # 在活动对象中查找
        if object_name in self._active_objects:
            return self._active_objects[object_name]
        # 在濒死对象中查找
        elif object_name in self._dying_objects:
            return self._dying_objects[object_name][0]
        else:
            if raise_error:
                raise ValueError("Object {} not found!".format(object_name))
            else:
                return None

    def destroy(self):
        """
        销毁管理器，清理资源
        
        清空:
        - 濒死车辆字典
        - 智能体计数器
        - 本帧终止记录
        """
        super(VehicleAgentManager, self).destroy()
        self._dying_objects = {}
        self.next_agent_count = 0
        self._agents_finished_this_frame = dict()

    def _put_to_dying_queue(self, v, ignore_delay_done=False):
        """
        将车辆加入濒死队列
        
        作用:
        - 设置车辆为静态（停止物理仿真）
        - 记录倒计时步数
        
        参数:
            v: 车辆对象
            ignore_delay_done: 是否忽略延迟（设为 0 立即移除）
        """
        vehicle_name = v.name
        v.set_static(True)  # 设为静态，停止运动
        self._dying_objects[vehicle_name] = [v, 0 if ignore_delay_done else self._delay_done]

    def _remove_vehicle(self, vehicle):
        """
        彻底移除车辆对象
        
        流程:
        1. 从引擎中清除对象
        2. 删除智能体-对象双向映射
        
        参数:
            vehicle: 待移除的车辆对象
        """
        vehicle_name = vehicle.name
        assert vehicle_name not in self._active_objects  # 确保不在活动列表中
        
        self.clear_objects([vehicle_name])  # 从引擎清除
        self._agent_to_object.pop(self._object_to_agent[vehicle_name])  # 删除映射
        self._object_to_agent.pop(vehicle_name)

    @property
    def allow_respawn(self):
        """
        判断是否允许生成新智能体
        
        条件:
        1. _allow_respawn 标志为 True
        2. 当前智能体数量 < 最大数量 或 无限智能体模式
        
        返回:
            bool: 是否允许重生
        """
        if not self._allow_respawn:
            return False
        if len(self._active_objects) + len(self._dying_objects) < self.engine.global_config["num_agents"] \
                or self._infinite_agents:
            return True
        else:
            return False