"""
MetaDrive 参数空间定义模块

本模块定义了 MetaDrive 环境中使用的各种参数空间类型，主要用于：
1. 程序化地图生成（PG Map）的参数随机化
2. 车辆动力学参数的随机化配置
3. 观测空间和动作空间的定义

主要组件：
- Space: 基础空间类，继承自 gym.Space
- Dict: 字典空间，用于组合多个子空间
- ParameterSpace: 参数空间，将自定义的 Box/Discrete/Constant 映射到 gym.Box
- VehicleParameterSpace: 车辆参数字典，定义不同车型的动力学参数范围
- BlockParameterSpace: 地图块参数字典，定义各种道路几何结构的参数范围

注意：本模块大量复用自 gym==0.17.2，但针对 MetaDrive 的需求进行了扩展。
"""

import logging
import typing as tp
from collections import namedtuple, OrderedDict

import numpy as np

from metadrive.utils import get_np_random

# ==================== 命名元组定义 ====================
# 用于简洁地表示不同类型的参数空间

# 连续区间空间：[min, max] 范围内的连续值
BoxSpace = namedtuple("BoxSpace", "max min")

# 离散空间：[min, max) 范围内的整数值
DiscreteSpace = namedtuple("DiscreteSpace", "max min")

# 常量空间：固定值，不可随机化
ConstantSpace = namedtuple("ConstantSpace", "value")


class Space:
    """
    基础空间类（复制自 gym.spaces.Space）
    
    定义观测空间和动作空间的基类，使得可以编写通用的环境代码。
    例如，可以基于此空间选择随机动作或验证数据的有效性。
    
    主要功能：
    - sample(): 从空间中随机采样一个元素
    - seed(): 设置随机种子以保证可复现性
    - contains(): 检查某个值是否属于该空间
    """
    def __init__(self, shape=None, dtype=None):
        """
        初始化空间对象
        
        Args:
            shape: 数据的形状，如 (3,) 表示三维向量
            dtype: 数据类型，如 np.float32, np.int64
        """
        import numpy as np  # 延迟导入 numpy，因为导入耗时约 300-400ms
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self.np_random = None
        self.seed()

    def sample(self):
        """
        从该空间中随机采样一个元素
        
        根据空间的有界性，可以是均匀分布或非均匀分布采样。
        子类必须实现此方法。
        
        Returns:
            采样得到的数据点
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """
        设置此空间的伪随机数生成器（PRNG）种子
        
        Args:
            seed: 随机种子，None 表示使用系统时间
            
        Returns:
            包含种子的列表
        """
        self.np_random, seed = get_np_random(seed, return_seed=True)
        return [seed]

    def contains(self, x):
        """
        检查 x 是否是该空间的有效成员
        
        Args:
            x: 待检查的值
            
        Returns:
            bool: x 是否属于该空间
        """
        raise NotImplementedError

    def __contains__(self, x):
        """支持 'x in space' 语法"""
        return self.contains(x)

    def to_jsonable(self, sample_n):
        """
        将一批样本转换为可 JSON 序列化的数据类型
        
        Args:
            sample_n: 样本列表
            
        Returns:
            可 JSON 序列化的数据
        """
        # 默认情况下，假设数据本身即可 JSON 序列化
        return sample_n

    def from_jsonable(self, sample_n):
        """
        将 JSON 数据转换回样本批次
        
        Args:
            sample_n: JSON 格式的数据
            
        Returns:
            恢复后的样本列表
        """
        # 默认情况下，假设数据本身就是样本
        return sample_n

    def destroy(self):
        """
        清理内存，释放随机数生成器
        """
        self.np_random = None


class Dict(Space):
    """
    字典空间（复制自 gym.spaces.Dict）
    
    由多个简单空间组成的字典结构，每个键对应一个子空间。
    
    使用场景：
    - 多传感器观测：{"position": Box, "velocity": Box, "image": Box}
    - 嵌套结构：支持多层嵌套的 Dict
    
    示例用法：
    ```python
    # 简单字典
    observation_space = spaces.Dict({
        "position": spaces.Discrete(2), 
        "velocity": spaces.Discrete(3)
    })
    
    # 嵌套字典
    nested_observation_space = spaces.Dict({
        'sensors': spaces.Dict({
            'position': spaces.Box(low=-100, high=100, shape=(3,)),
            'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
        }),
        'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
    })
    ```
    """
    def __init__(self, spaces=None, **spaces_kwargs):
        """
        初始化字典空间
        
        Args:
            spaces: 字典或有序字典，键为字符串，值为 Space 实例
            **spaces_kwargs: 关键字参数形式的空间定义（不能与 spaces 同时使用）
            
        Raises:
            AssertionError: 如果同时提供 spaces 和 spaces_kwargs
        """
        assert (spaces is None) or (not spaces_kwargs), 'Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)'
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, dict) and not isinstance(spaces, OrderedDict):
            spaces = OrderedDict(sorted(list(spaces.items())))
        if isinstance(spaces, list):
            spaces = OrderedDict(spaces)
        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(space, Space), 'Values of the dict should be instances of gym.Space'
        super(Dict, self).__init__(None, None)  # shape 和 dtype 需要特殊处理，设为 None

    def seed(self, seed=None):
        """为所有子空间设置相同的随机种子"""
        for space in self.spaces.values():
            space.seed(seed)

    def sample(self):
        """
        从字典空间中采样，对每个子空间独立采样
        
        Returns:
            OrderedDict: 键与子空间对应，值为采样结果
        """
        return OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])

    def contains(self, x):
        """
        检查 x 是否属于该字典空间
        
        Args:
            x: 待检查的字典
            
        Returns:
            bool: x 的键和值是否都与空间定义匹配
        """
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def __getitem__(self, key):
        """通过键访问子空间"""
        return self.spaces[key]

    def __repr__(self):
        """字符串表示"""
        return "Dict(" + ", ".join([str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def to_jsonable(self, sample_n):
        """
        将样本批次转换为 JSON 格式
        
        Args:
            sample_n: 样本列表，每个样本是字典
            
        Returns:
            字典，键为子空间名，值为对应的 JSON 数据列表
        """
        return {key: space.to_jsonable([sample[key] for sample in sample_n]) \
                for key, space in self.spaces.items()}

    def from_jsonable(self, sample_n):
        """
        从 JSON 格式恢复样本批次
        
        Args:
            sample_n: JSON 格式的字典
            
        Returns:
            恢复后的样本列表
        """
        dict_of_list = {}
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        for i, _ in enumerate(dict_of_list[key]):
            entry = {}
            for key, value in dict_of_list.items():
                entry[key] = value[i]
            ret.append(entry)
        return ret

    def __eq__(self, other):
        """判断两个字典空间是否相等"""
        return isinstance(other, Dict) and self.spaces == other.spaces


class ParameterSpace(Dict):
    """
    参数空间：用于程序化地图和车辆参数的随机化
    
    将自定义的空间类型（BoxSpace, DiscreteSpace, ConstantSpace）包装为 gym.Box，
    以便利用 gym 的采样和验证功能。
    
    使用示例：
    ```python
    # 定义长度参数：10~50 米之间的连续值
    length = PGSpace(name="length", max=50.0, min=10.0)
    
    # 创建参数空间
    param_space = ParameterSpace({"lane_length": length})
    
    # 采样
    sample = param_space.sample()  # {"lane_length": array([35.2])}
    ```
    """
    def __init__(self, our_config: tp.Dict[str, tp.Union[BoxSpace, DiscreteSpace, ConstantSpace]]):
        """
        初始化参数空间
        
        Args:
            our_config: 参数字典，键为参数名，值为 BoxSpace/DiscreteSpace/ConstantSpace
        """
        super(ParameterSpace, self).__init__(ParameterSpace.wrap2gym_space(our_config))
        self.parameters = set(our_config.keys())

    @staticmethod
    def wrap2gym_space(our_config):
        """
        将自定义空间类型包装为 gym.Box
        
        转换规则：
        - BoxSpace(min, max) -> gym.Box(low=min, high=max, shape=(1,))
        - DiscreteSpace(min, max) -> gym.Box(low=min, high=max, shape=(1,), dtype=int64)
        - ConstantSpace(value) -> gym.Box(low=value, high=value, shape=(1,))
        
        Args:
            our_config: 自定义配置字典
            
        Returns:
            dict: 包装后的 gym 空间字典
            
        Raises:
            ValueError: 如果遇到不支持的空间类型
        """
        ret = dict()
        for key, value in our_config.items():
            if isinstance(value, BoxSpace):
                # 连续区间：转换为浮点数 Box
                ret[key] = Box(low=value.min, high=value.max, shape=(1, ))
            elif isinstance(value, DiscreteSpace):
                # 离散区间：转换为整数 Box
                ret[key] = Box(low=value.min, high=value.max, shape=(1, ), dtype=np.int64)
            elif isinstance(value, ConstantSpace):
                # 常量：上下界相同的 Box
                ret[key] = Box(low=value.value, high=value.value, shape=(1, ))
            else:
                raise ValueError("{} can not be wrapped in gym space".format(key))
        return ret


class Parameter:
    """
    参数名称常量定义
    
    集中管理所有可用的参数名称，避免硬编码字符串。
    分为三类：
    1. 地图块参数（Block Parameters）
    2. 车辆参数（Vehicle Parameters）
    3. 车辆可视化参数（Visualization Parameters）
    """
    # ==================== 地图块参数 ====================
    length = "length"              # 路段长度（米）
    radius = "radius"              # 曲线半径（米）
    angle = "angle"                # 曲线角度（度）
    goal = "goal"                  # 目标位置
    dir = "dir"                    # 方向（0=左转，1=右转）
    radius_inner = "inner_radius"  # 内环半径（仅环岛使用）
    radius_exit = "exit_radius"    # 出口半径
    exit_length = "exit_length"    # 出口直道长度（仅环岛使用，米）
    t_intersection_type = "t_type" # T型路口类型（0/1/2 三种形态）
    lane_num = "lane_num"          # 车道数量
    change_lane_num = "change_lane_num"      # 变道增加/减少的车道数
    decrease_increase = "decrease_increase"  # 0=减少车道，1=增加车道
    one_side_vehicle_num = "one_side_vehicle_number"  # 单侧停车数量
    extension_length = "extension_length"  # 延伸段长度

    # ==================== 车辆动力学参数 ====================
    # vehicle_length = "v_len"     # 车辆长度（已弃用）
    # vehicle_width = "v_width"    # 车辆宽度（已弃用）
    vehicle_height = "v_height"    # 车辆高度（米）
    front_tire_longitude = "f_tire_long"  # 前轮胎纵向位置
    rear_tire_longitude = "r_tire_long"   # 后轮胎纵向位置
    tire_lateral = "tire_lateral"         # 轮胎横向间距
    tire_axis_height = "tire_axis_height" # 车轴高度
    tire_radius = "tire_radius"           # 轮胎半径
    mass = "mass"                         # 车辆质量（kg）
    heading = "heading"                   # 朝向角（弧度）
    # steering_max = "steering_max"  # 最大转向角（已弃用，改用 max_steering）
    # engine_force_max = "e_f_max"   # 最大引擎力（已弃用，改用 max_engine_force）
    # brake_force_max = "b_f_max"    # 最大刹车力（已弃用，改用 max_brake_force）
    # speed_max = "s_max"            # 最大速度（已弃用，改用 max_speed_km_h）

    # ==================== 车辆可视化参数 ====================
    vehicle_vis_z = "vis_z"        # 渲染 Z 轴偏移
    vehicle_vis_y = "vis_y"        # 渲染 Y 轴偏移
    vehicle_vis_h = "vis_h"        # 渲染高度缩放
    vehicle_vis_scale = "vis_scale" # 渲染整体缩放比例


class VehicleParameterSpace:
    """
    车辆参数空间定义
    
    定义了不同车型的动力学参数范围，支持两种模式：
    1. STATIC_*: 静态参数（ConstantSpace），所有episode使用相同值
    2. 非STATIC: 动态参数（BoxSpace），每次重置时从区间内随机采样
    
    支持的车型：
    - BASE_VEHICLE / DEFAULT_VEHICLE: 标准轿车（默认）
    - S_VEHICLE: 小型车（动力弱，转向灵活）
    - M_VEHICLE: 中型车（均衡性能）
    - L_VEHICLE: 大型车（动力中等，制动较弱）
    - XL_VEHICLE: 超大型车（动力弱，制动弱，转向笨重）
    
    关键参数说明：
    - wheel_friction: 轮胎摩擦系数（影响抓地力和操控性）
    - max_engine_force: 最大引擎力（牛顿，决定加速能力）
    - max_brake_force: 最大刹车力（牛顿，决定制动距离）
    - max_steering: 最大转向角（度，决定转弯半径）
    - max_speed_km_h: 最大速度（km/h，电子限速）
    """
    
    # ==================== 静态基准车型（参数固定）====================
    STATIC_BASE_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.9),      # 轮胎摩擦系数：0.9（高抓地力）
        max_engine_force=ConstantSpace(800),    # 最大引擎力：800 N
        max_brake_force=ConstantSpace(150),     # 最大刹车力：150 N
        max_steering=ConstantSpace(40),         # 最大转向角：40 度
        max_speed_km_h=ConstantSpace(80),       # 最大速度：80 km/h
    )
    STATIC_DEFAULT_VEHICLE = STATIC_BASE_VEHICLE  # 静态默认车型与基准车型相同

    # ==================== 动态基准车型（参数随机化）====================
    BASE_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.9),              # 轮胎摩擦系数：固定 0.9
        max_engine_force=BoxSpace(750, 850),            # 最大引擎力：750~850 N 随机
        max_brake_force=BoxSpace(80, 180),              # 最大刹车力：80~180 N 随机
        max_steering=ConstantSpace(40),                 # 最大转向角：固定 40 度
        max_speed_km_h=ConstantSpace(80),               # 最大速度：固定 80 km/h
    )
    DEFAULT_VEHICLE = BASE_VEHICLE  # 默认车型使用基准配置

    # ==================== 小型车（S型）====================
    S_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.9),              # 轮胎摩擦系数：0.9
        max_engine_force=BoxSpace(350, 550),            # 最大引擎力：350~550 N（较弱）
        max_brake_force=BoxSpace(35, 80),               # 最大刹车力：35~80 N（较弱）
        max_steering=ConstantSpace(50),                 # 最大转向角：50 度（更灵活）
        max_speed_km_h=ConstantSpace(80),               # 最大速度：80 km/h
    )

    # ==================== 中型车（M型）====================
    M_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.75),             # 轮胎摩擦系数：0.75（较低抓地力）
        max_engine_force=BoxSpace(650, 850),            # 最大引擎力：650~850 N
        max_brake_force=BoxSpace(60, 150),              # 最大刹车力：60~150 N
        max_steering=ConstantSpace(45),                 # 最大转向角：45 度
        max_speed_km_h=ConstantSpace(80),               # 最大速度：80 km/h
    )

    # ==================== 大型车（L型）====================
    L_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.8),              # 轮胎摩擦系数：0.8
        max_engine_force=BoxSpace(450, 650),            # 最大引擎力：450~650 N（中等）
        max_brake_force=BoxSpace(60, 120),              # 最大刹车力：60~120 N（较弱）
        max_steering=ConstantSpace(40),                 # 最大转向角：40 度
        max_speed_km_h=ConstantSpace(80),               # 最大速度：80 km/h
    )

    # ==================== 超大型车（XL型）====================
    XL_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.7),              # 轮胎摩擦系数：0.7（低抓地力）
        max_engine_force=BoxSpace(500, 700),            # 最大引擎力：500~700 N
        max_brake_force=BoxSpace(50, 100),              # 最大刹车力：50~100 N（很弱）
        max_steering=ConstantSpace(35),                 # 最大转向角：35 度（转向笨重）
        max_speed_km_h=ConstantSpace(80),               # 最大速度：80 km/h
    )


class BlockParameterSpace:
    """
    地图块参数空间定义
    
    定义了程序化地图（Procedural Generation）中各种道路几何结构的参数范围。
    这些参数用于在运行时随机生成多样化的道路场景，增强智能体的泛化能力。
    
    重要提示：
    - 曲线参数的范围必须覆盖其他所有块的参数空间，否则导航信息归一化可能出错
    - 所有长度单位均为米（m），角度单位为度（°）
    
    支持的地图块类型：
    1. STRAIGHT: 直道路段
    2. BIDIRECTION: 双向道路
    3. CURVE: 曲线路段
    4. INTERSECTION: 十字路口
    5. ROUNDABOUT: 环岛
    6. T_INTERSECTION: T型路口
    7. RAMP_PARAMETER: 匝道（汇入/汇出）
    8. FORK_PARAMETER: 分叉路口
    9. BOTTLENECK_PARAMETER: 瓶颈路段（车道数变化）
    10. TOLLGATE_PARAMETER: 收费站
    11. PARKING_LOT_PARAMETER: 停车场
    """
    
    # ==================== 直道路段 ====================
    STRAIGHT = {
        Parameter.length: BoxSpace(min=40.0, max=80.0)  # 长度：40~80 米
    }
    
    # ==================== 双向道路 ====================
    BIDIRECTION = {
        Parameter.length: BoxSpace(min=40.0, max=80.0)  # 长度：40~80 米
    }

    # ==================== 曲线路段 ====================
    CURVE = {
        Parameter.length: BoxSpace(min=40.0, max=80.0),   # 弧长：40~80 米
        Parameter.radius: BoxSpace(min=25.0, max=60.0),   # 曲率半径：25~60 米
        Parameter.angle: BoxSpace(min=45, max=135),       # 圆心角：45~135 度
        Parameter.dir: DiscreteSpace(min=0, max=1)        # 方向：0=左转，1=右转
    }
    
    # ==================== 十字路口 ====================
    INTERSECTION = {
        Parameter.radius: ConstantSpace(10),                        # 路口圆角半径：固定 10 米
        Parameter.change_lane_num: DiscreteSpace(min=0, max=1),     # 变道数：0 或 1
        Parameter.decrease_increase: DiscreteSpace(min=0, max=1)    # 0=车道减少，1=车道增加
    }
    
    # ==================== 环岛 ====================
    ROUNDABOUT = {
        Parameter.radius_exit: BoxSpace(min=5, max=15),   # 出口半径：5~15 米
        Parameter.radius_inner: BoxSpace(min=15, max=45), # 内环半径：15~45 米
        Parameter.angle: ConstantSpace(60)                # 出口间隔角：固定 60 度
    }
    
    # ==================== T型路口 ====================
    T_INTERSECTION = {
        Parameter.radius: ConstantSpace(10),                        # 路口圆角半径：固定 10 米
        Parameter.t_intersection_type: DiscreteSpace(min=0, max=2), # T型形态：0/1/2 三种
        Parameter.change_lane_num: DiscreteSpace(min=0, max=1),     # 变道数：0 或 1
        Parameter.decrease_increase: DiscreteSpace(min=0, max=1)    # 0=车道减少，1=车道增加
    }
    
    # ==================== 匝道参数 ====================
    RAMP_PARAMETER = {
        Parameter.length: BoxSpace(min=20, max=40),         # 加减速段长度：20~40 米
        Parameter.extension_length: BoxSpace(min=20, max=40) # 延伸段长度：20~40 米
    }
    
    # ==================== 分叉路口参数 ====================
    FORK_PARAMETER = {
        Parameter.length: BoxSpace(min=20, max=40),  # 分叉段长度：20~40 米
        Parameter.lane_num: DiscreteSpace(min=0, max=1)  # 分叉车道数：0 或 1
    }
    
    # ==================== 瓶颈路段参数 ====================
    BOTTLENECK_PARAMETER = {
        Parameter.length: BoxSpace(min=20, max=50),           # 直道部分长度：20~50 米
        Parameter.lane_num: DiscreteSpace(min=1, max=2),      # 增减车道数：1→2 或 2→1
        "bottle_len": ConstantSpace(20),                      # 瓶颈段长度：固定 20 米
        "solid_center_line": ConstantSpace(0)                 # 中心黄线：0=虚线，1=实线（bool）
    }
    
    # ==================== 收费站参数 ====================
    TOLLGATE_PARAMETER = {
        Parameter.length: ConstantSpace(20),  # 直道部分长度：固定 20 米
    }
    
    # ==================== 停车场参数 ====================
    PARKING_LOT_PARAMETER = {
        Parameter.one_side_vehicle_num: DiscreteSpace(min=2, max=10),  # 单侧停车数量：2~10 辆
        Parameter.radius: ConstantSpace(value=4),                       # 停车位半径：固定 4 米
        Parameter.length: ConstantSpace(value=8)                        # 停车位长度：固定 8 米
    }


class Discrete(Space):
    r"""
    离散空间（复制自 gym.spaces.Discrete）
    
    表示集合 {0, 1, ..., n-1} 中的离散值。
    
    示例：
    ```python
    >>> Discrete(2)  # 表示 {0, 1} 两个值
    ```
    
    应用场景：
    - 离散动作空间（如：左转/直行/右转）
    - 分类标签
    - 枚举类型的参数
    """
    def __init__(self, n):
        """
        初始化离散空间
        
        Args:
            n: 离散值的数量，空间为 {0, 1, ..., n-1}
            
        Raises:
            AssertionError: 如果 n < 0
        """
        assert n >= 0
        self.n = n
        super(Discrete, self).__init__((), np.int64)

    def sample(self):
        """从 {0, 1, ..., n-1} 中均匀随机采样一个整数"""
        return self.np_random.randint(self.n)

    def contains(self, x):
        """
        检查 x 是否是有效的离散值
        
        Args:
            x: 待检查的值
            
        Returns:
            bool: x 是否在 [0, n) 范围内且为整数
        """
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        """字符串表示"""
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        """判断两个离散空间是否相等"""
        return isinstance(other, Discrete) and self.n == other.n


class Box(Space):
    """
    连续盒子空间（复制自 gym.spaces.Box）
    
    表示 R^n 中的一个（可能有界的）超矩形区域。
    具体来说，Box 表示 n 个闭区间的笛卡尔积。
    每个区间可以是以下形式之一：
    - [a, b]: 有界区间
    - (-∞, b]: 上界区间
    - [a, ∞): 下界区间
    - (-∞, ∞): 无界区间
    
    两种常见使用场景：
    
    1. 所有维度具有相同的边界：
    ```python
    >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
    Box(3, 4)
    ```
    
    2. 每个维度独立边界：
    ```python
    >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
    Box(2,)
    ```
    
    采样策略（根据区间类型）：
    - [a, b]: 均匀分布 U(a, b)
    - [a, ∞): 移位指数分布 a + Exp(1)
    - (-∞, b]: 移位负指数分布 b - Exp(1)
    - (-∞, ∞): 标准正态分布 N(0, 1)
    """
    def __init__(self, low, high, shape=None, dtype=np.float32):
        """
        初始化盒子空间
        
        Args:
            low: 下界，可以是标量或与 shape 匹配的数组
            high: 上界，可以是标量或与 shape 匹配的数组
            shape: 数据形状，如未提供则从 low/high 推断
            dtype: 数据类型，默认为 np.float32
            
        Raises:
            AssertionError: 如果 low/high 的 shape 不匹配
            ValueError: 如果 shape 未提供且无法从 low/high 推断
        """
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = np.dtype(dtype)

        # 如果未直接提供 shape，则尝试推断
        if shape is not None:
            shape = tuple(shape)
            assert np.isscalar(low) or low.shape == shape, "low.shape doesn't match provided shape"
            assert np.isscalar(high) or high.shape == shape, "high.shape doesn't match provided shape"
        elif not np.isscalar(low):
            shape = low.shape
            assert np.isscalar(high) or high.shape == shape, "high.shape doesn't match low.shape"
        elif not np.isscalar(high):
            shape = high.shape
            assert np.isscalar(low) or low.shape == shape, "low.shape doesn't match high.shape"
        else:
            raise ValueError("shape must be provided or inferred from the shapes of low or high")

        # 如果 low/high 是标量，则广播为与 shape 匹配的数组
        if np.isscalar(low):
            low = np.full(shape, low, dtype=dtype)

        if np.isscalar(high):
            high = np.full(shape, high, dtype=dtype)

        self.shape = shape
        self.low = low
        self.high = high

        def _get_precision(dtype):
            """获取数据类型的精度"""
            if np.issubdtype(dtype, np.floating):
                return np.finfo(dtype).precision
            else:
                return np.inf

        low_precision = _get_precision(self.low.dtype)
        high_precision = _get_precision(self.high.dtype)
        dtype_precision = _get_precision(self.dtype)
        if min(low_precision, high_precision) > dtype_precision:
            logging.warning("Box bound precision lowered by casting to {}".format(self.dtype))
        self.low = self.low.astype(self.dtype)
        self.high = self.high.astype(self.dtype)

        # 布尔数组，指示每个坐标的区间类型
        self.bounded_below = -np.inf < self.low  # 是否有下界
        self.bounded_above = np.inf > self.high  # 是否有上界

        super(Box, self).__init__(self.shape, self.dtype)

    def is_bounded(self, manner="both"):
        """
        检查空间是否有界
        
        Args:
            manner: 检查方式
                - "both": 检查是否有上下界
                - "below": 仅检查是否有下界
                - "above": 仅检查是否有上界
                
        Returns:
            bool: 是否满足指定的有界条件
        """
        below = np.all(self.bounded_below)
        above = np.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
        """
        在 Box 内生成单个随机样本
        
        采样策略根据区间类型：
        - [a, b]: 均匀分布 U(a, b)
        - [a, ∞): 移位指数分布 a + Exp(1)
        - (-∞, b]: 移位负指数分布 b - Exp(1)
        - (-∞, ∞): 标准正态分布 N(0, 1)
        
        Returns:
            np.ndarray: 采样得到的数据点
        """
        high = self.high if self.dtype.kind == 'f' \
            else self.high.astype('int64') + 1
        sample = np.empty(self.shape)

        # 掩码数组，根据区间类型分类坐标
        unbounded = ~self.bounded_below & ~self.bounded_above  # 无界：(-∞, ∞)
        upp_bounded = ~self.bounded_below & self.bounded_above  # 上界：(-∞, b]
        low_bounded = self.bounded_below & ~self.bounded_above  # 下界：[a, ∞)
        bounded = self.bounded_below & self.bounded_above       # 有界：[a, b]

        # 向量化采样
        sample[unbounded] = self.np_random.normal(size=unbounded[unbounded].shape)
        sample[low_bounded] = self.np_random.exponential(size=low_bounded[low_bounded].shape) + self.low[low_bounded]
        sample[upp_bounded] = -self.np_random.exponential(size=upp_bounded[upp_bounded].shape) + self.high[upp_bounded]
        sample[bounded] = self.np_random.uniform(low=self.low[bounded], high=high[bounded], size=bounded[bounded].shape)
        
        # 如果是整数类型，向下取整
        if self.dtype.kind == 'i':
            sample = np.floor(sample)

        return sample.astype(self.dtype)

    def contains(self, x):
        """
        检查 x 是否在该 Box 空间内
        
        Args:
            x: 待检查的数组或列表
            
        Returns:
            bool: x 的 shape 是否匹配且所有元素都在 [low, high] 范围内
        """
        if isinstance(x, list):
            x = np.array(x)  # 将列表提升为数组以便检查
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def to_jsonable(self, sample_n):
        """将样本批次转换为 JSON 格式（嵌套列表）"""
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        """从 JSON 格式（嵌套列表）恢复样本批次"""
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        """字符串表示"""
        return "Box" + str(self.shape)

    def __eq__(self, other):
        """判断两个 Box 空间是否相等（shape 和边界都相同）"""
        return isinstance(other, Box) and \
               (self.shape == other.shape) and \
               np.allclose(self.low, other.low) and \
               np.allclose(self.high, other.high)


if __name__ == "__main__":
    """
    测试代码：验证 ParameterSpace 的功能
    """
    # 定义配置：包含连续、离散和混合类型的参数
    config = {
        "length": BoxSpace(min=10.0, max=80.0),   # 连续参数：10~80
        "angle": BoxSpace(min=50.0, max=360.0),   # 连续参数：50~360
        "goal": DiscreteSpace(min=0, max=2)       # 离散参数：0, 1, 2
    }
    
    # 创建参数空间
    config = ParameterSpace(config)
    
    # 测试采样
    print("随机采样 1:", config.sample())
    
    # 设置种子并采样（可复现）
    config.seed(1)
    print("种子=1 采样 1:", config.sample())
    print("种子=1 采样 2:", config.sample())
    
    # 重置种子并验证一致性
    config.seed(1)
    print("重置种子=1 采样:", *config.sample()["length"])
