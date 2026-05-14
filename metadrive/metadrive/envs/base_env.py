"""
MetaDrive基础环境类 - 提供强化学习环境的完整接口实现

该模块实现了MetaDrive仿真环境的核心基类BaseEnv,继承自gymnasium.Env。
主要功能包括:
- 环境配置管理与初始化
- 智能体(Agent)管理(单智能体/多智能体)
- 传感器系统配置
- 观测空间、动作空间定义
- 步进(step)、重置(reset)、渲染(render)等核心方法
- 奖励函数、代价函数、终止条件定义
- 场景录制与回放功能
- 相机控制与视角切换
"""

import logging
import time
from collections import defaultdict
from typing import Union, Dict, AnyStr, Optional, Tuple, Callable

import gymnasium as gym
import numpy as np
from panda3d.core import PNMImage

from metadrive import constants
from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.component.sensors.dashboard import DashBoard
from metadrive.component.sensors.distance_detector import LaneLineDetector, SideDetector
from metadrive.component.sensors.lidar import Lidar
from metadrive.constants import DEFAULT_SENSOR_HPR, DEFAULT_SENSOR_OFFSET
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT
from metadrive.constants import RENDER_MODE_ONSCREEN, RENDER_MODE_OFFSCREEN
from metadrive.constants import TerminationState, TerrainProperty
from metadrive.engine.engine_utils import initialize_engine, close_engine, \
    engine_initialized, set_global_random_seed, initialize_global_config, get_global_config
from metadrive.engine.logger import get_logger, set_log_level
from metadrive.manager.agent_manager import VehicleAgentManager
from metadrive.manager.record_manager import RecordManager
from metadrive.manager.replay_manager import ReplayManager
from metadrive.obs.image_obs import ImageStateObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.observation_base import DummyObservation
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.scenario.utils import convert_recorded_scenario_exported
from metadrive.utils import Config, merge_dicts, get_np_random, concat_step_infos
from metadrive.version import VERSION

# ==================== 默认配置字典 ====================
# 定义了MetaDrive环境的所有可配置参数及其默认值
BASE_DEFAULT_CONFIG = dict(

    # ===== 智能体配置 =====
    # 是否随机化智能体的车辆模型,从4种车型中随机选择
    random_agent_model=False,
    # 主车配置: env_config["vehicle_config"]会与env_config["agent_configs"]["default_agent"]合并
    agent_configs={DEFAULT_AGENT: dict(use_special_color=True, spawn_lane_index=None)},

    # ===== 多智能体配置 =====
    # 在多智能体环境中应设置为>1,或设置为-1以生成尽可能多的车辆
    num_agents=1,
    # 启用此项以通知模拟器这是多智能体环境
    is_multi_agent=False,
    # 如果设置为False,智能体数量将在episode开始时固定并确定
    allow_respawn=False,
    # 智能体在死亡位置保持静止的子步数(多智能体默认为25)
    delay_done=0,

    # ===== 动作/控制配置 =====
    # 详见文档: Action and Policy
    # 用于控制智能体的策略类型
    agent_policy=EnvInputPolicy,
    # 如果设置为True,agent_policy将被覆盖并更改为ManualControlPolicy(手动控制策略)
    manual_control=False,
    # 手动控制的接口类型,可选: "steering_wheel"(方向盘)、"keyboard"(键盘)或"xbox"(手柄)
    controller="keyboard",
    # 与EnvInputPolicy配合使用。如果设置为True,env.action_space将是离散动作空间
    discrete_action=False,
    # 如果为True,使用MultiDiscrete动作空间;否则使用Discrete
    use_multi_discrete=False,
    # 转向维度的离散动作数量
    discrete_steering_dim=5,
    # 油门/刹车维度的离散动作数量
    discrete_throttle_dim=5,
    # 检查动作是否在gym.space范围内。通常关闭以加速仿真
    action_check=False,

    # ===== 观测配置 =====
    # 详见文档: Observation
    # 是否将像素值从0-255归一化到0-1
    norm_pixel=True,
    # 图像观测堆叠的时间步数
    stack_size=3,
    # 是否使用图像观测还是激光雷达观测。在get_single_observation中生效
    image_observation=False,
    # 类似agent_policy,用户可以通过此字段使用自定义的观测类
    agent_observation=None,

    # ===== 终止条件配置 =====
    # 每个智能体episode的最大长度。设置为None以移除此限制
    horizon=None,
    # 如果设置为True,当智能体episode长度超过horizon时,terminated也将为True
    truncate_as_terminate=False,

    # ===== 主相机配置 =====
    # True值使相机跟随参考线而非车辆,使移动更平滑
    use_chase_camera_follow_lane=False,
    # 主相机高度
    camera_height=2.2,
    # 相机与车辆的距离。这是投影到x-y平面的距离
    camera_dist=7.5,
    # 主相机的俯仰角。如果为None,将自动计算
    camera_pitch=None,  # 单位:度
    # 平滑相机移动
    camera_smooth=True,
    # 用于平滑相机的帧数
    camera_smooth_buffer_size=20,
    # 主相机的视场角(FOV)
    camera_fov=65,
    # 仅适用于多智能体设置,选择跟踪哪个智能体。值应为"agent0"、"agent1"等
    prefer_track_agent=None,
    # 设置3D查看器俯视相机的初始位置(按"B"键激活)
    top_down_camera_initial_x=0,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=200,

    # ===== 车辆配置 =====
    vehicle_config=dict(
        # 车辆模型。候选: "s"、"m"、"l"、"xl"、"default"。random_agent_model会使此配置无效
        vehicle_model="default",
        # 如果设置为True,车辆可以通过油门/刹车 < -1进行倒车
        enable_reverse=False,
        # 是否显示导航点标记框
        show_navi_mark=True,
        # 是否在目的地显示标记框
        show_dest_mark=False,
        # 是否从当前车辆位置到目标点绘制一条线
        show_line_to_dest=False,
        # 是否从当前车辆位置到下一个导航点绘制一条线
        show_line_to_navi_mark=False,
        # 是否在界面中绘制左/右箭头表示导航方向
        show_navigation_arrow=True,
        # 如果设置为True,车辆在俯视渲染器或多智能体设置中将显示为绿色
        use_special_color=False,
        # 清除车轮摩擦力,使其无法通过设置转向和油门/刹车移动。用于ReplayPolicy
        no_wheel_friction=False,

        # ===== 图像捕获配置 =====
        # 用于图像观测的相机。应该是传感器配置中注册的传感器
        image_source="rgb_camera",

        # ===== 车辆生成与导航配置 =====
        # BaseNavigation实例。应与道路网络类型匹配
        navigation_module=None,
        # 指定在哪个车道生成此车辆的车道ID
        spawn_lane_index=None,
        # 目标车道ID。仅在navigation_module不为None时需要
        destination=None,
        # 在生成车道上的纵向和横向位置
        spawn_longitude=5.0,
        spawn_lateral=0.0,

        # 如果分配了以下项,车辆将以特定速度在指定位置生成
        spawn_position_heading=None,
        spawn_velocity=None,  # 单位: m/s
        spawn_velocity_car_frame=False,

        # ==== 其他配置 ====
        # 车辆已超车的车辆数量。由于bug已被弃用
        overtake_stat=False,
        # 如果设置为True,车辆的默认纹理将被替换为纯色
        random_color=False,
        # 车辆形状由其类预定义。但在特殊场景(WaymoVehicle)中可能需要设置为任意形状
        width=None,
        length=None,
        height=None,
        mass=None,
        scale=None,  # 三元组 (x, y, z)

        # 仅为pygame俯视渲染器设置车辆大小。不影响物理尺寸!
        top_down_width=None,
        top_down_length=None,

        # ===== 车辆模块配置 =====
        lidar=dict(
            num_lasers=240, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0, add_others_navi=False
        ),
        side_detector=dict(num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
        lane_line_detector=dict(num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0),
        show_lidar=False,
        show_side_detector=False,
        show_lane_line_detector=False,
        # 是否开启车灯,仅在启用render-pipeline时可用
        light=False,
    ),

    # ===== 传感器配置 =====
    sensors=dict(lidar=(Lidar, ), side_detector=(SideDetector, ), lane_line_detector=(LaneLineDetector, )),

    # ===== 引擎核心配置 =====
    # 如果为true,弹出窗口进行渲染
    use_render=False,
    # (宽度, 高度),如果设置为None,将自动确定
    window_size=(1200, 900),
    # 物理世界步长为0.02秒,每次env.step()将重复decision_repeat次
    physics_world_step_size=2e-2,
    decision_repeat=5,
    # 高级功能:无需将图像移动到RAM即可访问!
    image_on_cuda=False,
    # 不要设置此配置。我们将自动确定渲染模式,默认以纯物理模式运行
    _render_mode=RENDER_MODE_NONE,
    # 如果设置为None:程序将尽可能快地运行。否则,fps将限制在此值以下
    force_render_fps=None,
    # 我们将在引擎中维护一组缓冲区来存储已使用的对象,并在可能时重用它们以提高效率
    # 如果设置为True,调用clear()时将强制销毁所有对象
    force_destroy=False,
    # 每类对象的缓冲数量
    num_buffering_objects=200,
    # 启用它以使用渲染管线,提供高级渲染效果(Beta版)
    render_pipeline=False,
    # daytime仅在启用render-pipeline时可用
    daytime="19:00",  # 使用如"13:40"的字符串,我们通常在工具包中通过编辑器设置
    # 阴影范围,单位: [米]
    shadow_range=50,
    # 是否使用多线程渲染
    multi_thread_render=True,
    multi_thread_render_mode="Cull",  # 或 "Cull/Draw"
    # 模型加载优化。预加载行人以避免首次创建时卡顿
    preload_models=True,
    # 模型压缩会增加启动时间
    disable_model_compression=True,
    # 是否禁用碰撞检测(对调试/回放记录的场景有用)
    disable_collision=False,

    # ===== 地形配置 =====
    # 方形地图区域的大小,以[0, 0]为中心。其外的地图对象将被剔除
    map_region_size=2048,
    # 是否移除地图区域外的车道。如果为True,车道局部化仅应用于地图区域
    cull_lanes_outside_map=False,
    # 道路将有一个平坦的路肩,其宽度由此值决定,单位: [米]
    drivable_area_extension=7,
    # 山脉的高度比例,单位: [米]。0高度使地形平坦
    height_scale=50,
    # 如果使用网格碰撞,山脉将具有物理体并与车辆交互
    use_mesh_terrain=False,
    # 如果设置为False,只有地形的中心区域具有物理体
    full_size_mesh=True,
    # 是否显示人行横道
    show_crosswalk=True,
    # 是否显示人行道
    show_sidewalk=True,

    # ===== 调试配置 =====
    # 详见文档: Debug
    pstats=False,  # 启用以分析效率
    debug=False,  # 调试模式,输出更多消息
    debug_panda3d=False,  # 调试panda3d
    debug_physics_world=False,  # 仅渲染物理世界而不渲染模型,特殊的调试选项
    debug_static_world=False,  # 调试静态世界
    log_level=logging.INFO,  # 日志级别。logging.DEBUG/logging.CRITICAL等
    show_coordinates=False,  # 为地图和对象显示坐标以进行调试

    # ===== GUI配置 =====
    # 详见文档: GUI
    # 是否在3D场景中显示这些元素
    show_fps=True,
    show_logo=True,
    show_mouse=True,
    show_skybox=True,
    show_terrain=True,
    show_interface=True,
    # 为调试多策略设置显示策略标记
    show_policy_mark=False,
    # 显示箭头标记以提供导航信息
    show_interface_navi_mark=True,
    # 在窗口上显示传感器输出的列表。其元素从sensors.keys() + "dashboard"中选择
    interface_panel=["dashboard"],

    # ===== 录制/回放元数据配置 =====
    # 详见文档: Record and Replay
    # 当replay_episode为True时,将记录episode元数据
    record_episode=False,
    # 值应为None或日志数据。如果是后者,模拟器将回放记录的场景
    replay_episode=None,
    # 当设置为True时,回放系统将仅从记录的场景元数据重建第一帧
    only_reset_when_replay=False,
    # 如果为True,在创建和回放对象轨迹时使用与数据集中相同的ID
    force_reuse_object_name=False,

    # ===== 随机化配置 =====
    num_scenarios=1  # 此环境中的场景数量
)


class BaseEnv(gym.Env):
    """
    MetaDrive基础环境类
    
    继承自gymnasium.Env,提供完整的强化学习环境接口。
    支持单智能体和多智能体(MARL)设置,集成了Panda3D 3D渲染引擎。
    
    主要特性:
    - 灵活的环境配置系统
    - 延迟初始化机制(lazy initialization)
    - 多传感器支持(激光雷达、相机、仪表盘等)
    - 场景录制与回放
    - 多种渲染模式(屏幕渲染、离屏渲染、俯视渲染)
    - 键盘快捷键控制(重置、截图、暂停、视角切换等)
    """
    
    # 必要时强制使用此种子。注意,强制种子的接收者应被显式实现
    _DEBUG_RANDOM_SEED: Union[int, None] = None

    @classmethod
    def default_config(cls) -> Config:
        """返回默认配置对象"""
        return Config(BASE_DEFAULT_CONFIG)

    # ===== 初始化阶段 =====
    def __init__(self, config: dict = None):
        """
        初始化MetaDrive环境
        
        Args:
            config: 环境配置字典,将与默认配置合并
        """
        if config is None:
            config = {}
        
        # 初始化日志系统
        self.logger = get_logger()
        set_log_level(config.get("log_level", logging.DEBUG if config.get("debug", False) else logging.INFO))
        
        # 合并用户配置与默认配置
        merged_config = self.default_config().update(config, False, ["agent_configs", "sensors"])
        global_config = self._post_process_config(merged_config)

        self.config = global_config
        initialize_global_config(self.config)

        # 智能体配置检查
        self.num_agents = self.config["num_agents"]
        self.is_multi_agent = self.config["is_multi_agent"]
        if not self.is_multi_agent:
            assert self.num_agents == 1, "单智能体环境下num_agents必须为1"
        else:
            assert not self.config["image_on_cuda"], "CUDA上的图像不支持多智能体!"
        assert isinstance(self.num_agents, int) and (self.num_agents > 0 or self.num_agents == -1), \
            "num_agents必须是正整数或-1"

        # 观测和动作空间
        self.agent_manager = self._get_agent_manager()

        # 延迟初始化,在lazy_init()函数中创建主仿真
        # self.engine: Optional[BaseEngine] = None

        # 在具有重生机制的多智能体环境中,episodes长度可能不同
        self.episode_rewards = defaultdict(float)  # 记录每个智能体的累积奖励
        self.episode_lengths = defaultdict(int)    # 记录每个智能体的episode步数

        # 按p键暂停
        self.in_stop = False

        # 场景索引
        self.start_index = 0

    def _post_process_config(self, config):
        """
        对合并后的配置进行后处理,添加特殊处理逻辑
        
        Args:
            config: 合并后的配置字典
            
        Returns:
            处理后的配置字典
        """
        # 取消界面板
        self.logger.info("Environment: {}".format(self.__class__.__name__))
        self.logger.info("MetaDrive version: {}".format(VERSION))
        if not config["show_interface"]:
            config["interface_panel"] = []

        # 调整地形配置
        n = config["map_region_size"]
        assert (n & (n - 1)) == 0 and 512 <= n <= 4096, "map_region_size应为2的幂且< 2048"
        TerrainProperty.map_region_size = config["map_region_size"]

        # 多线程渲染
        # if config["image_on_cuda"]:
        #     self.logger.info("由于image_on_cuda=True,关闭多线程渲染")
        #     config["multi_thread_render"] = False

        # 在无屏幕模式下优化传感器创建
        if not config["use_render"] and not config["image_observation"]:
            filtered = {}
            for id, cfg in config["sensors"].items():
                if len(cfg) > 0 and not issubclass(cfg[0], BaseCamera) and id != "main_camera":
                    filtered[id] = cfg
            config["sensors"] = filtered
            config["interface_panel"] = []

        # 检查传感器存在性
        if config["use_render"] or "main_camera" in config["sensors"]:
            config["sensors"]["main_camera"] = ("MainCamera", *config["window_size"])

        # 合并仪表盘配置与传感器
        to_use = []
        if not config["render_pipeline"] and config["show_interface"] and "main_camera" in config["sensors"]:
            for panel in config["interface_panel"]:
                if panel == "dashboard":
                    config["sensors"]["dashboard"] = (DashBoard, )
                if panel not in config["sensors"]:
                    self.logger.warning(
                        "无法将传感器: {} 添加到界面。从面板列表中移除!".format(panel)
                    )
                elif panel == "main_camera":
                    self.logger.warning("main_camera不能添加到interface_panel,已移除")
                else:
                    to_use.append(panel)
        config["interface_panel"] = to_use

        # 合并默认传感器到列表
        sensor_cfg = self.default_config()["sensors"].update(config["sensors"])
        config["sensors"] = sensor_cfg

        # 显示传感器列表
        _str = "Sensors: [{}]"
        sensors_str = ""
        for _id, cfg in config["sensors"].items():
            sensors_str += "{}: {}{}, ".format(_id, cfg[0] if isinstance(cfg[0], str) else cfg[0].__name__, cfg[1:])
        self.logger.info(_str.format(sensors_str[:-2]))

        # 自动确定渲染模式
        if config["use_render"]:
            assert "main_camera" in config["sensors"]
            config["_render_mode"] = RENDER_MODE_ONSCREEN
        else:
            config["_render_mode"] = RENDER_MODE_NONE
            for sensor in config["sensors"].values():
                if sensor[0] == "MainCamera" or (issubclass(sensor[0], BaseCamera) and sensor[0] != DashBoard):
                    config["_render_mode"] = RENDER_MODE_OFFSCREEN
                    break
        self.logger.info("Render Mode: {}".format(config["_render_mode"]))
        self.logger.info("Horizon (Max steps per agent): {}".format(config["horizon"]))
        if config["truncate_as_terminate"]:
            self.logger.warning(
                "当达到最大步数时,'terminate'和'truncate'都将为True。"
                "通常,只有`truncate`应该为`True`。"
            )
        return config

    def _get_observations(self) -> Dict[str, "BaseObservation"]:
        """获取观测字典,默认为单个智能体"""
        return {DEFAULT_AGENT: self.get_single_observation()}

    def _get_agent_manager(self):
        """创建并返回智能体管理器"""
        return VehicleAgentManager(init_observations=self._get_observations())

    def lazy_init(self):
        """
        延迟初始化方法 - 仅在运行时初始化一次,变量存在直到调用close_env
        
        这是真正的init()函数,用于创建主车辆及其模块,以避免与ray不兼容
        """
        if engine_initialized():
            return
        initialize_engine(self.config)
        # 引擎设置
        self.setup_engine()
        # 其他可选初始化
        self._after_lazy_init()
        self.logger.info(
            "Start Scenario Index: {}, Num Scenarios : {}".format(
                self.engine.gets_start_index(self.config), self.config.get("num_scenarios", 1)
            )
        )

    @property
    def engine(self):
        """获取引擎实例的属性方法"""
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    def _after_lazy_init(self):
        """延迟初始化后的钩子方法,子类可重写"""
        pass

    # ===== 运行时核心方法 =====
    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        """
        执行一步仿真
        
        Args:
            actions: 动作输入,可以是数组、列表或字典(多智能体)
            
        Returns:
            observation: 观测
            reward: 奖励
            terminated: 是否终止(智能体自身原因)
            truncated: 是否截断(环境原因,如达到最大步数)
            info: 额外信息字典
        """
        actions = self._preprocess_actions(actions)  # 预处理环境输入
        engine_info = self._step_simulator(actions)  # 步进仿真器
        while self.in_stop:
            self.engine.taskMgr.step()  # 暂停仿真
        return self._get_step_return(actions, engine_info=engine_info)  # 收集观测、奖励、终止状态

    def _preprocess_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray], int]) \
            -> Union[np.ndarray, Dict[AnyStr, np.ndarray], int]:
        """
        预处理动作输入
        
        对于单智能体环境,将动作转换为字典格式;
        对于多智能体环境,验证动作键的完整性或过滤多余动作
        
        Args:
            actions: 原始动作输入
            
        Returns:
            处理后的动作字典
        """
        if not self.is_multi_agent:
            # 单智能体:将动作广播给唯一的智能体
            actions = {v_id: actions for v_id in self.agents.keys()}
        else:
            if self.config["action_check"]:
                # 检查是否缺少某些动作
                given_keys = set(actions.keys())
                have_keys = set(self.agents.keys())
                assert given_keys == have_keys, "输入动作: {} 的键与现有的 {} 不兼容!".format(
                    given_keys, have_keys
                )
            else:
                # 如果给出额外的动作也可以接受。这是因为,当评估策略时,
                # "终止观测"仍会在T=t-1时给出。在T=t时,当你从policy(last_obs)收集动作而
                # 不进行掩码时,"终止观测"的动作仍会被计算。我们在这里将其过滤掉。
                actions = {v_id: actions[v_id] for v_id in self.agents.keys()}
        return actions

    def _step_simulator(self, actions):
        """
        步进仿真器核心逻辑
        
        执行顺序:
        1. 准备步进前的场景管理器状态
        2. 步进所有实体和仿真器
        3. 更新步进后的状态
        4. 合并前后状态信息
        
        Args:
            actions: 智能体动作字典
            
        Returns:
            合并后的引擎信息字典
        """
        # 准备步进仿真
        scene_manager_before_step_infos = self.engine.before_step(actions)
        # 步进所有实体和仿真器
        self.engine.step(self.config["decision_repeat"])
        # 更新状态,如果从episode数据恢复,位置和航向将在update_state()函数中强制设置
        scene_manager_after_step_infos = self.engine.after_step()

        # 注意:我们在此函数中对info字典使用浅层更新!这将加速系统
        return merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )

    def reward_function(self, object_id: str) -> Tuple[float, Dict]:
        """
        奖励函数 - 子类应重写此方法以实现自定义奖励逻辑
        
        Args:
            object_id: 对象ID
            
        Returns:
            reward: 奖励值
            reward_info: 奖励详细信息字典
        """
        self.logger.warning("Reward function is not implemented. Return reward = 0", extra={"log_once": True})
        return 0, {}

    def cost_function(self, object_id: str) -> Tuple[float, Dict]:
        """
        代价函数 - 用于安全强化学习的约束信号
        
        Args:
            object_id: 对象ID
            
        Returns:
            cost: 代价值
            cost_info: 代价详细信息字典
        """
        self.logger.warning("Cost function is not implemented. Return cost = 0", extra={"log_once": True})
        return 0, {}

    def done_function(self, object_id: str) -> Tuple[bool, Dict]:
        """
        终止函数 - 判断智能体是否应终止
        
        Args:
            object_id: 对象ID
            
        Returns:
            done: 是否终止
            done_info: 终止原因详细信息
        """
        self.logger.warning("Done function is not implemented. Return Done = False", extra={"log_once": True})
        return False, {}

    def render(self, text: Optional[Union[dict, str]] = None, mode=None, *args, **kwargs) -> Optional[np.ndarray]:
        """
        渲染函数 - 伪渲染函数,仅用于在使用panda3d后端时更新屏幕消息
        
        Args:
            text: 要显示的文本
            mode: 渲染模式,可选: ["top_down", "topdown", "bev", "birdview"] 用于俯视渲染
            
        Returns:
            None 或 俯视图图像数组
        """
        if mode in ["top_down", "topdown", "bev", "birdview"]:
            ret = self._render_topdown(text=text, *args, **kwargs)
            return ret
        if self.config["use_render"] or self.engine.mode != RENDER_MODE_NONE:
            self.engine.render_frame(text)
        else:
            self.logger.warning(
                "Panda渲染已关闭,无法渲染。请设置config['use_render'] = True!",
                exc_info={"log_once": True}
            )
        return None

    def reset(self, seed: Union[None, int] = None):
        """
        重置环境
        
        可以重置环境或通过提供episode_data来恢复和回放场景
        
        Args:
            seed: 设置环境的种子。实际上是想要选择的场景索引
            
        Returns:
            observation: 初始观测
            info: 初始信息字典
        """
        if self.logger is None:
            self.logger = get_logger()
            log_level = self.config.get("log_level", logging.DEBUG if self.config.get("debug", False) else logging.INFO)
            set_log_level(log_level)
        
        # 延迟初始化 - 仅在第一次调用reset()时工作,以避免渲染时的错误
        self.lazy_init()
        self._reset_global_seed(seed)
        
        if self.engine is None:
            raise ValueError(
                "当前MetaDrive实例已损坏。请确保一个进程中只有一个活动的MetaDrive "
                "环境存在。您可以尝试调用env.close()然后调用 "
                "env.reset()来挽救此环境。然而,更好且更安全的解决方案是检查 "
                "MetaDrive的单例模式并重新启动程序。"
            )
        
        # 重置引擎
        reset_info = self.engine.reset()
        # 重置传感器
        self.reset_sensors()
        # 渲染场景
        self.engine.taskMgr.step()
        if self.top_down_renderer is not None:
            self.top_down_renderer.clear()
            self.engine.top_down_renderer = None

        # 重置智能体状态
        self.dones = {agent_id: False for agent_id in self.agents.keys()}
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        assert (len(self.agents) == self.num_agents) or (self.num_agents == -1), \
            "Agents: {} != Num_agents: {}".format(len(self.agents), self.num_agents)
        assert self.config is self.engine.global_config is get_global_config(), "不一致的配置可能导致错误!"
        return self._get_reset_return(reset_info)

    def reset_sensors(self):
        """
        重置传感器 - 开发者API
        
        重写此方法可以确定如何在场景中放置传感器。可以将其安装在对象上或在整个episode中固定在给定位置。
        """
        # 在episode开始时重置相机
        if self.main_camera is not None:
            self.main_camera.reset()
            if hasattr(self, "agent_manager"):
                bev_cam = self.main_camera.is_bird_view_camera() and self.main_camera.current_track_agent is not None
                agents = list(self.engine.agents.values())
                current_track_agent = agents[0]
                self.main_camera.set_follow_lane(self.config["use_chase_camera_follow_lane"])
                self.main_camera.track(current_track_agent)
                if bev_cam:
                    self.main_camera.stop_track()
                    self.main_camera.set_bird_view_pos_hpr(current_track_agent.position)
                for name, sensor in self.engine.sensors.items():
                    if hasattr(sensor, "track") and name != "main_camera":
                        sensor.track(current_track_agent.origin, DEFAULT_SENSOR_OFFSET, DEFAULT_SENSOR_HPR)
        # 步进环境以避免第一帧黑屏
        self.engine.taskMgr.step()

    def _get_reset_return(self, reset_info):
        """
        获取重置返回值
        
        收集初始观测、奖励、终止状态和代价信息
        
        Args:
            reset_info: 引擎重置信息
            
        Returns:
            单智能体: (observation, info)
            多智能体: (observations_dict, infos_dict)
        """
        # TODO: 弄清楚如何获取步进前的信息
        scene_manager_before_step_infos = reset_info
        scene_manager_after_step_infos = self.engine.after_step()

        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        engine_info = merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )
        for v_id, v in self.agents.items():
            self.observations[v_id].reset(self, v)
            obses[v_id] = self.observations[v_id].observe(v)
            _, reward_infos[v_id] = self.reward_function(v_id)
            _, done_infos[v_id] = self.done_function(v_id)
            _, cost_infos[v_id] = self.cost_function(v_id)

        step_infos = concat_step_infos([engine_info, done_infos, reward_infos, cost_infos])

        if self.is_multi_agent:
            return obses, step_infos
        else:
            return self._wrap_as_single_agent(obses), self._wrap_info_as_single_agent(step_infos)

    def _wrap_info_as_single_agent(self, data):
        """
        将信息包装为单智能体格式
        
        Args:
            data: 多智能体信息字典
            
        Returns:
            单智能体信息字典
        """
        agent_info = data.pop(next(iter(self.agents.keys())))
        data.update(agent_info)
        return data

    def _get_step_return(self, actions, engine_info):
        """
        获取步进返回值
        
        更新观测、终止状态、奖励、代价,首先计算done!
        
        Args:
            actions: 执行的动作
            engine_info: 引擎信息
            
        Returns:
            单智能体: (obs, reward, terminated, truncated, info)
            多智能体: (obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict)
        """
        # 更新obs、dones、rewards、costs,首先计算done!
        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        rewards = {}
        for v_id, v in self.agents.items():
            self.episode_lengths[v_id] += 1
            rewards[v_id], reward_infos[v_id] = self.reward_function(v_id)
            self.episode_rewards[v_id] += rewards[v_id]
            done_function_result, done_infos[v_id] = self.done_function(v_id)
            _, cost_infos[v_id] = self.cost_function(v_id)
            self.dones[v_id] = done_function_result or self.dones[v_id]
            o = self.observations[v_id].observe(v)
            obses[v_id] = o

        step_infos = concat_step_infos([engine_info, done_infos, reward_infos, cost_infos])
        truncateds = {k: step_infos[k].get(TerminationState.MAX_STEP, False) for k in self.agents.keys()}
        terminateds = {k: self.dones[k] for k in self.agents.keys()}

        # 仅用于极端情况。如果环境步数超过horizon的5倍,强制终止所有智能体
        if self.config["horizon"] and self.episode_step > 5 * self.config["horizon"]:
            for k in truncateds:
                truncateds[k] = True
                if self.config["truncate_as_terminate"]:
                    self.dones[k] = terminateds[k] = True

        # 添加episode累积奖励和长度到info
        for v_id, r in rewards.items():
            step_infos[v_id]["episode_reward"] = self.episode_rewards[v_id]
            step_infos[v_id]["episode_length"] = self.episode_lengths[v_id]

        if not self.is_multi_agent:
            return self._wrap_as_single_agent(obses), self._wrap_as_single_agent(rewards), \
                self._wrap_as_single_agent(terminateds), self._wrap_as_single_agent(
                truncateds), self._wrap_info_as_single_agent(step_infos)
        else:
            return obses, rewards, terminateds, truncateds, step_infos

    def close(self):
        """关闭环境,释放资源"""
        if self.engine is not None:
            close_engine()

    def force_close(self):
        """强制关闭环境 - 用于紧急退出"""
        print("Closing environment ... Please wait")
        self.close()
        time.sleep(2)  # 睡眠两秒
        raise KeyboardInterrupt("'Esc' is pressed. MetaDrive exits now.")

    def capture(self, file_name=None):
        """
        捕获当前屏幕截图并保存为图片
        
        Args:
            file_name: 保存的文件名,如果为None则自动生成
        """
        if not hasattr(self, "_capture_img"):
            self._capture_img = PNMImage()
        self.engine.win.getScreenshot(self._capture_img)
        if file_name is None:
            file_name = "main_index_{}_step_{}_{}.png".format(self.current_seed, self.engine.episode_step, time.time())
        self._capture_img.write(file_name)
        self.logger.info("Image is saved at: {}".format(file_name))

    def for_each_agent(self, func, *args, **kwargs):
        """对所有活动智能体执行函数"""
        return self.agent_manager.for_each_active_agents(func, *args, **kwargs)

    def get_single_observation(self):
        """
        获取单个对象的观测器
        
        Returns:
            BaseObservation实例
        """
        if self.__class__ is BaseEnv:
            o = DummyObservation()
        else:
            if self.config["agent_observation"]:
                o = self.config["agent_observation"](self.config)
            else:
                img_obs = self.config["image_observation"]
                o = ImageStateObservation(self.config) if img_obs else LidarStateObservation(self.config)
        return o

    def _wrap_as_single_agent(self, data):
        """将数据包装为单智能体格式"""
        return data[next(iter(self.agents.keys()))]

    def seed(self, seed=None):
        """设置全局随机种子"""
        if seed is not None:
            set_global_random_seed(seed)

    @property
    def current_seed(self):
        """获取当前种子"""
        return self.engine.global_random_seed

    @property
    def num_scenarios(self):
        """获取场景数量"""
        return self.config["num_scenarios"]

    @property
    def observations(self):
        """
        返回活动和可控智能体的观测器
        
        Returns:
            观测器字典
        """
        return self.agent_manager.get_observations()

    @property
    def observation_space(self) -> gym.Space:
        """
        返回活动和可控智能体的观测空间
        
        Returns:
            gym.Space对象
        """
        ret = self.agent_manager.get_observation_spaces()
        if not self.is_multi_agent:
            return next(iter(ret.values()))
        else:
            return gym.spaces.Dict(ret)

    @property
    def action_space(self) -> gym.Space:
        """
        返回活动和可控智能体的动作空间。通常在AgentManager中定义,但您仍可重写此函数为环境定义动作空间
        
        Returns:
            gym.Space对象
        """
        ret = self.agent_manager.get_action_spaces()
        if not self.is_multi_agent:
            return next(iter(ret.values()))
        else:
            return gym.spaces.Dict(ret)

    @property
    def vehicles(self):
        """
        返回所有活动车辆(已弃用,请使用agents)
        
        Returns:
            车辆字典
        """
        self.logger.warning("env.vehicles will be deprecated soon. Use env.agents instead", extra={"log_once": True})
        return self.agents

    @property
    def vehicle(self):
        """返回单个车辆(已弃用,请使用agent)"""
        self.logger.warning("env.vehicle will be deprecated soon. Use env.agent instead", extra={"log_once": True})
        return self.agent

    @property
    def agents(self):
        """
        返回所有活动智能体
        
        Returns:
            智能体字典 {agent_id: agent}
        """
        return self.agent_manager.active_agents

    @property
    def agent(self):
        """辅助方法,仅在单智能体环境中返回智能体!"""
        assert len(self.agents) == 1, (
            "env.agent仅在单智能体环境中支持!"
            if len(self.agents) > 1 else "请先初始化环境!"
        )
        return self.agents[DEFAULT_AGENT]

    @property
    def agents_including_just_terminated(self):
        """
        返回当前环境中占据空间的所有智能体(包括刚终止的)
        
        Returns:
            智能体字典
        """
        ret = self.agent_manager.active_agents
        ret.update(self.agent_manager.just_terminated_agents)
        return ret

    def setup_engine(self):
        """
        启动后的引擎设置
        
        注册键盘快捷键和管理器
        """
        self.engine.accept("r", self.reset)  # R键: 重置环境
        self.engine.accept("c", self.capture)  # C键: 截图
        self.engine.accept("p", self.stop)  # P键: 暂停/继续
        self.engine.accept("b", self.switch_to_top_down_view)  # B键: 切换到俯视视角
        self.engine.accept("q", self.switch_to_third_person_view)  # Q键: 切换到第三人称视角
        self.engine.accept("]", self.next_seed_reset)  # ]键: 下一个场景
        self.engine.accept("[", self.last_seed_reset)  # [键: 上一个场景
        self.engine.register_manager("agent_manager", self.agent_manager)
        self.engine.register_manager("record_manager", RecordManager())
        self.engine.register_manager("replay_manager", ReplayManager())

    @property
    def current_map(self):
        """获取当前地图"""
        return self.engine.current_map

    @property
    def maps(self):
        """获取所有地图"""
        return self.engine.map_manager.maps

    def _render_topdown(self, text, *args, **kwargs):
        """渲染俯视图"""
        return self.engine.render_topdown(text, *args, **kwargs)

    @property
    def main_camera(self):
        """获取主相机"""
        return self.engine.main_camera

    @property
    def current_track_agent(self):
        """获取当前跟踪的智能体"""
        return self.engine.current_track_agent

    @property
    def top_down_renderer(self):
        """获取俯视渲染器"""
        return self.engine.top_down_renderer

    @property
    def episode_step(self):
        """获取当前episode步数"""
        return self.engine.episode_step if self.engine is not None else 0

    def export_scenarios(
        self,
        policies: Union[dict, Callable],
        scenario_index: Union[list, int],
        max_episode_length=None,
        verbose=False,
        suppress_warning=False,
        render_topdown=False,
        return_done_info=True,
        to_dict=True
    ):
        """
        导出场景数据 - 以10hz采样率将场景导出为统一格式
        
        Args:
            policies: 策略,单智能体为Callable,多智能体为dict
            scenario_index: 要导出的场景索引列表或单个索引
            max_episode_length: 最大episode长度
            verbose: 是否详细输出
            suppress_warning: 是否抑制警告
            render_topdown: 是否渲染俯视图
            return_done_info: 是否返回完成信息
            to_dict: 是否转换为字典格式
            
        Returns:
            scenarios_to_export: 导出的场景字典
            done_info: 完成信息字典(如果return_done_info为True)
        """
        def _act(observation):
            """根据策略生成动作"""
            if isinstance(policies, dict):
                ret = {}
                for id, o in observation.items():
                    ret[id] = policies[id](o)
            else:
                ret = policies(observation)
            return ret

        if self.is_multi_agent:
            assert isinstance(policies, dict), "在多智能体设置中,policies应根据id映射到智能体"
        else:
            assert isinstance(policies, Callable), "在单智能体情况下,policy应该是可调用对象,接受观测作为输入"
        
        scenarios_to_export = dict()
        if isinstance(scenario_index, int):
            scenario_index = [scenario_index]
        
        self.config["record_episode"] = True
        done_info = {}
        for index in scenario_index:
            obs = self.reset(seed=index)
            done = False
            count = 0
            info = None
            while not done:
                obs, reward, terminated, truncated, info = self.step(_act(obs))
                done = terminated or truncated
                count += 1
                if max_episode_length is not None and count > max_episode_length:
                    done = True
                    info[TerminationState.MAX_STEP] = True
                if count > 10000 and not suppress_warning:
                    self.logger.warning(
                        "Episode长度太长!如果这是预期行为, "
                        "设置suppress_warning=True以禁用此消息"
                    )
                if render_topdown:
                    self.render("topdown")
            episode = self.engine.dump_episode()
            if verbose:
                self.logger.info("Finish scenario {} with {} steps.".format(index, count))
            scenarios_to_export[index] = convert_recorded_scenario_exported(episode, to_dict=to_dict)
            done_info[index] = info
        self.config["record_episode"] = False
        if return_done_info:
            return scenarios_to_export, done_info
        else:
            return scenarios_to_export

    def stop(self):
        """暂停/继续仿真"""
        self.in_stop = not self.in_stop

    def switch_to_top_down_view(self):
        """切换到俯视视角"""
        self.main_camera.stop_track()

    def switch_to_third_person_view(self):
        """切换到第三人称视角"""
        if self.main_camera is None:
            return
        self.main_camera.reset()
        if self.config["prefer_track_agent"] is not None and self.config["prefer_track_agent"] in self.agents.keys():
            new_v = self.agents[self.config["prefer_track_agent"]]
            current_track_agent = new_v
        else:
            if self.main_camera.is_bird_view_camera():
                current_track_agent = self.current_track_agent
            else:
                agents = list(self.engine.agents.values())
                if len(agents) <= 1:
                    return
                if self.current_track_agent in agents:
                    agents.remove(self.current_track_agent)
                new_v = get_np_random().choice(agents)
                current_track_agent = new_v
        self.main_camera.track(current_track_agent)
        for name, sensor in self.engine.sensors.items():
            if hasattr(sensor, "track") and name != "main_camera":
                sensor.track(current_track_agent.origin, constants.DEFAULT_SENSOR_OFFSET, DEFAULT_SENSOR_HPR)
        return

    def next_seed_reset(self):
        """重置到下一个场景种子"""
        if self.current_seed + 1 < self.start_index + self.num_scenarios:
            self.reset(self.current_seed + 1)
        else:
            self.logger.warning(
                "无法加载下一个场景!当前种子已经是最大场景索引。"
                "允许的索引范围: {}-{}".format(self.start_index, self.start_index + self.num_scenarios - 1)
            )

    def last_seed_reset(self):
        """重置到上一个场景种子"""
        if self.current_seed - 1 >= self.start_index:
            self.reset(self.current_seed - 1)
        else:
            self.logger.warning(
                "无法加载上一个场景!当前种子已经是最小场景索引"
                "允许的索引范围: {}-{}".format(self.start_index, self.start_index + self.num_scenarios - 1)
            )

    def _reset_global_seed(self, force_seed=None):
        """
        重置全局随机种子
        
        Args:
            force_seed: 强制使用的种子值,如果为None则随机选择
        """
        current_seed = force_seed if force_seed is not None else \
            get_np_random(self._DEBUG_RANDOM_SEED).randint(self.start_index, self.start_index + self.num_scenarios)
        assert self.start_index <= current_seed < self.start_index + self.num_scenarios, \
            "scenario_index (seed) 应该在 [{}:{}) 范围内".format(self.start_index, self.start_index + self.num_scenarios)
        self.seed(current_seed)


if __name__ == '__main__':
    # 测试代码 - 创建环境并运行
    cfg = {"use_render": True}
    env = BaseEnv(cfg)
    env.reset()
    while True:
        env.step(env.action_space.sample())
