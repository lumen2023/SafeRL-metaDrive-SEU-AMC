#!/usr/bin/env python3
"""
单场景 MetaDrive 环境的命令行启动器。

该脚本用于快速测试和调试 env.py 中定义的单场景环境，支持多种控制器模式
（手控、IDM策略、PPO训练好的模型），并提供丰富的可视化选项。

使用示例:
    # 启动环岛场景，启用渲染窗口
    python run_single_scene.py --scene roundabout --render
    
    # 使用验证集种子运行T型交叉口
    python run_single_scene.py --scene t_intersection --render --split val
    
    # 运行3个episode后自动退出
    python run_single_scene.py --scene straight_4lane --episodes 3
    
    # 通过JSON字符串覆盖配置参数（降低交通流速度）
    python run_single_scene.py --scene lane_change_bottleneck --extra-config-json '{"single_scene":{"randomization":{"traffic_speed_range":[3,8]}}}'
"""
import argparse  # 命令行参数解析库
import json      # JSON数据处理
import os        # 操作系统接口
import sys       # 系统相关参数和函数
from typing import Any, Dict  # 类型提示

import numpy as np  # 数值计算库

from safe_metadrive_adapter.local_import import prefer_local_metadrive

prefer_local_metadrive()


# ==================== 常量定义 ====================

# 支持的5种单场景名称列表
AVAILABLE_SINGLE_SCENES = (
    "straight_4lane",         # 四车道直道
    "ramp_merge",             # 匝道汇入
    "t_intersection",         # T型交叉口
    "roundabout",             # 环岛
    "lane_change_bottleneck", # 变道瓶颈路段
)

# 默认参数配置
DEFAULT_SCENE = "ramp_merge"  # 默认场景：四车道直道
DEFAULT_SPLIT = "train"           # 默认数据分割：训练集
DEFAULT_RENDER = True             # 默认启用渲染
DEFAULT_EGO_CONTROLLER = "idm"    # 默认自车控制器：IDM智能驾驶员模型
DEFAULT_PPO_DEVICE = "cpu"        # 默认PPO推理设备：CPU


# ==================== 工具函数 ====================

def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并两个字典，override中的值会覆盖base中的值。
    
    对于嵌套字典，会递归合并而不是直接替换，确保配置的灵活性。
    
    Args:
        base: 基础配置字典
        override: 覆盖配置字典
        
    Returns:
        合并后的新字典（不修改原字典）
    """
    result = dict(base)  # 创建浅拷贝，避免修改原始字典
    for key, value in override.items():
        # 如果两个字典中同一键对应的值都是字典，则递归合并
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            # 否则直接用override的值覆盖
            result[key] = value
    return result


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    
    定义了所有可用的命令行选项，包括场景选择、控制器类型、渲染设置等。
    
    Returns:
        解析后的命名空间对象，包含所有参数值
    """
    parser = argparse.ArgumentParser(description="运行单个固定几何的MetaDrive场景环境。")
    
    # 场景选择参数
    parser.add_argument(
        "--scene",
        choices=AVAILABLE_SINGLE_SCENES,  # 限制只能选择预定义的5种场景
        default=DEFAULT_SCENE,
        help="要启动的场景名称。默认: %(default)s",
    )
    
    # 数据分割选择（训练集或验证集，影响随机化种子）
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default=DEFAULT_SPLIT,
        help="使用训练集还是验证集预设。默认: %(default)s",
    )
    
    # 渲染开关（支持 --render 和 --no-render）
    parser.add_argument("--render", dest="render", action="store_true", help="启用MetaDrive渲染窗口。")
    parser.add_argument("--no-render", dest="render", action="store_false", help="禁用渲染。")
    parser.set_defaults(render=DEFAULT_RENDER)
    
    # 自车控制器类型选择
    parser.add_argument(
        "--ego-controller",
        choices=("manual", "idm", "ppo"),  # manual=手控, idm=智能驾驶员, ppo=训练好的模型
        default=DEFAULT_EGO_CONTROLLER,
        help="循环中使用的自车控制器。默认: %(default)s",
    )
    
    # 渲染模式选择
    parser.add_argument(
        "--render-mode",
        choices=("topdown", "none"),  # topdown=俯视鸟瞰图, none=不渲染
        default="topdown",
        help="主循环中使用的渲染模式。",
    )
    
    # 手控模式开关
    parser.add_argument(
        "--manual-control",
        dest="manual_control",
        action="store_true",
        help="使用键盘控制。仅在 --ego-controller=manual 时有意义。",
    )
    parser.add_argument(
        "--no-manual-control",
        dest="manual_control",
        action="store_false",
        help="禁用键盘控制。仅在 --ego-controller=manual 时有意义。",
    )
    parser.set_defaults(manual_control=None)  # 默认为None，根据--ego-controller自动决定
    
    # 动作模式（当手控关闭时的备选方案）
    parser.add_argument(
        "--action-mode",
        choices=("idle", "forward", "random"),  # idle=静止, forward=前进, random=随机
        default=None,
        help="当 ego-controller=manual 且手动控制关闭时的动作来源。",
    )
    
    # PPO模型路径（加载训练好的策略）
    parser.add_argument(
        "--ppo-path",
        type=str,
        default=None,
        help="PPO实验目录或checkpoint/model.pt路径。当 --ego-controller=ppo 时必需。",
    )
    
    # 是否加载最佳模型
    parser.add_argument(
        "--ppo-best",
        action="store_true",
        help="当 --ppo-path 指向实验目录时，加载 checkpoint/model_best.pt 而非 model.pt。",
    )
    
    # PPO推理设备
    parser.add_argument(
        "--ppo-device",
        choices=("cpu", "cuda"),
        default=DEFAULT_PPO_DEVICE,
        help="PPO推理使用的设备。默认: %(default)s",
    )
    
    # IDM策略调试选项：禁用车道变换
    parser.add_argument(
        "--disable-idm-lane-change",
        action="store_true",
        help="禁用MetaDrive IDMPolicy内部的车道变换行为。",
    )
    
    # IDM策略调试选项：禁用减速逻辑
    parser.add_argument(
        "--disable-idm-deceleration",
        action="store_true",
        help="禁用IDM减速逻辑以进行调试。",
    )

    # 终止条件配置
    parser.add_argument(
        "--crash-vehicle-done",
        dest="crash_vehicle_done",
        action="store_true",
        default=False,
        help="车辆碰撞后立即结束episode。",
    )
    parser.add_argument(
        "--crash-object-done",
        dest="crash_object_done",
        action="store_true",
        default=False,
        help="碰撞障碍物后立即结束episode。",
    )
    parser.add_argument(
        "--no-out-of-road-done",
        dest="out_of_road_done",
        action="store_false",
        help="偏离道路后不结束episode（默认偏离即结束）。",
    )
    parser.set_defaults(out_of_road_done=False)

    # 周车密度配置
    parser.add_argument(
        "--traffic-density",
        type=float,
        default=None,
        help="覆盖单场景周车密度，按MetaDrive官方语义使用0-1范围。",
    )
    parser.add_argument(
        "--traffic-backend",
        choices=("single_scene", "official"),
        default=None,
        help="单场景周车后端：single_scene 使用当前自定义安全补车，official 使用MetaDrive官方 PGTrafficManager。",
    )

    # 运行episode数量限制
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="要运行的完整episode数量。0表示一直运行直到Ctrl+C中断。",
    )
    
    # 全局步数上限
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="所有episode的全局步数上限。0表示无限制。",
    )
    
    # 额外配置（通过JSON字符串传入）
    parser.add_argument(
        "--extra-config-json",
        type=str,
        default=None,
        help="额外的环境配置，以JSON字符串形式传入。将递归合并到场景配置中。",
    )
    
    # 重置时打印的交通车辆数量限制
    parser.add_argument(
        "--print-traffic-limit",
        type=int,
        default=5,
        help="在重置摘要中包含多少辆交通车辆的信息。",
    )
    
    return parser.parse_args()


def load_env_api():
    """
    动态导入环境API函数。
    
    处理可能的依赖缺失问题（如shapely库），提供友好的错误提示。
    
    Returns:
        元组：(get_single_scene_training_env, get_single_scene_validation_env)
    """
    try:
        from env import get_single_scene_training_env, get_single_scene_validation_env
    except ModuleNotFoundError as exc:
        # 特别处理shapely库缺失的情况（MetaDrive的常见依赖问题）
        if exc.name == "shapely":
            print("缺少依赖库: shapely", file=sys.stderr)
            print("请先安装: pip install shapely", file=sys.stderr)
            raise SystemExit(1) from exc
        raise  # 其他导入错误直接抛出
    return get_single_scene_training_env, get_single_scene_validation_env


def load_idm_policy():
    """
    加载IDM（Intelligent Driver Model，智能驾驶员模型）策略类。
    
    IDM是一种经典的跟车和换道模型，常用于生成合理的背景交通流。
    
    Returns:
        IDMPolicy类引用
    """
    # 兼容两种导入路径（取决于metadrive的安装方式）
    try:
        from metadrive.policy.idm_policy import IDMPolicy
    except ImportError:
        from metadrive.metadrive.policy.idm_policy import IDMPolicy
    return IDMPolicy


def batchify_observation(obs: Any) -> Any:
    """
    将观测数据批量化的辅助函数。
    
    Tianshou框架要求输入是batch维度（即第一维为batch size），
    此函数将单个观测扩展为batch_size=1的形式。
    
    Args:
        obs: 原始观测数据（可以是字典或数组）
        
    Returns:
        添加了batch维度的观测数据
    """
    if isinstance(obs, dict):
        # 如果观测是字典，对每个字段单独扩展维度
        return {key: np.expand_dims(np.asarray(value), axis=0) for key, value in obs.items()}
    # 如果是数组，直接在第0维扩展
    return np.expand_dims(np.asarray(obs), axis=0)


def resolve_ppo_artifacts(path: str, best: bool) -> Dict[str, str]:
    """
    解析PPO模型相关文件路径。
    
    支持两种输入形式：
    1. 实验目录路径：自动查找 checkpoint/model.pt 或 checkpoint/model_best.pt
    2. 模型文件直接路径：自动推导config.yaml的位置
    
    Args:
        path: 用户提供的路径（目录或文件）
        best: 是否加载最佳模型（model_best.pt）
        
    Returns:
        包含三个路径的字典：
        - run_dir: 实验根目录
        - config_path: 配置文件路径
        - model_path: 模型权重文件路径
        
    Raises:
        FileNotFoundError: 如果找不到config.yaml或模型文件
    """
    expanded_path = os.path.abspath(os.path.expanduser(path))  # 展开~符号并转为绝对路径
    
    if os.path.isdir(expanded_path):
        # 情况1：用户提供的是实验目录
        run_dir = expanded_path
        model_name = "model_best.pt" if best else "model.pt"
        model_path = os.path.join(run_dir, "checkpoint", model_name)
    else:
        # 情况2：用户提供的是模型文件路径
        model_path = expanded_path
        checkpoint_dir = os.path.dirname(model_path)
        # 判断是否在checkpoint子目录下
        if os.path.basename(checkpoint_dir) == "checkpoint":
            run_dir = os.path.dirname(checkpoint_dir)  # 上一级目录是实验根目录
        else:
            run_dir = os.path.dirname(model_path)  # 同级目录是实验根目录
    
    # 配置文件始终在实验根目录下
    config_path = os.path.join(run_dir, "config.yaml")
    
    # 验证文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError("缺少PPO配置文件: {}".format(config_path))
    if not os.path.exists(model_path):
        raise FileNotFoundError("缺少PPO检查点文件: {}".format(model_path))
    
    return {
        "run_dir": run_dir,
        "config_path": config_path,
        "model_path": model_path,
    }


# ==================== PPO控制器类 ====================

class PPOEgoController:
    """
    PPO策略控制器，用于加载和执行训练好的PPO-Lag模型。
    
    该类封装了FSRL框架的PPOLagAgent，负责：
    1. 从配置文件和检查点恢复训练好的策略
    2. 将环境观测转换为动作输出
    3. 管理推理设备（CPU/GPU）
    """
    
    def __init__(self, env, config: Dict[str, Any], model_path: str, device: str):
        """
        初始化PPO控制器。
        
        Args:
            env: MetaDrive环境实例（用于获取观察空间和动作空间）
            config: PPO训练时的配置字典（从config.yaml加载）
            model_path: 模型权重文件路径
            device: 推理设备（"cpu"或"cuda"）
        """
        # 延迟导入，避免不必要的依赖检查
        try:
            import torch
            from tianshou.data import Batch, to_numpy
            
            from fsrl.fsrl.agent import PPOLagAgent
            from fsrl.fsrl.utils import BaseLogger
        except ModuleNotFoundError as exc:
            print("缺少PPO依赖: {}".format(exc.name), file=sys.stderr)
            raise SystemExit(1) from exc
        
        # 保存模块引用，避免重复导入
        self._torch = torch
        self._Batch = Batch
        self._to_numpy = to_numpy
        self.device = device
        self.config = config
        self.model_path = model_path
        
        # 创建PPO-Lag智能体，使用训练时的超参数
        agent = PPOLagAgent(
            env=env,
            logger=BaseLogger(),  # 空日志记录器（推理阶段不需要）
            device=device,
            thread=int(config.get("thread", 4)),  # 并行线程数
            seed=int(config.get("seed", 10)),     # 随机种子
            lr=float(config.get("lr", 5e-4)),     # 学习率（推理时不使用）
            hidden_sizes=tuple(config.get("hidden_sizes", (128, 128))),  # 网络隐藏层尺寸
            unbounded=bool(config.get("unbounded", False)),  # 无界动作空间
            last_layer_scale=bool(config.get("last_layer_scale", False)),  # 最后一层缩放
            target_kl=float(config.get("target_kl", 0.02)),  # KL散度阈值（训练时使用）
            vf_coef=float(config.get("vf_coef", 0.25)),  # 价值函数损失系数
            max_grad_norm=config.get("max_grad_norm", 0.5),  # 梯度裁剪范数
            gae_lambda=float(config.get("gae_lambda", 0.95)),  # GAE优势估计lambda参数
            eps_clip=float(config.get("eps_clip", 0.2)),  # PPO裁剪范围
            dual_clip=config.get("dual_clip"),  # 双重裁剪（可选）
            value_clip=bool(config.get("value_clip", False)),  # 价值函数裁剪
            advantage_normalization=bool(config.get("norm_adv", config.get("advantage_normalization", True))),  # 优势归一化
            recompute_advantage=bool(config.get("recompute_adv", config.get("recompute_advantage", False))),  # 重新计算优势
            use_lagrangian=bool(config.get("use_lagrangian", True)),  # 是否使用拉格朗日乘子法
            lagrangian_pid=tuple(config.get("lagrangian_pid", (0.05, 0.0005, 0.1))),  # PID控制器参数
            cost_limit=config.get("cost_limit", 10.0),  # 成本约束上限
            rescaling=bool(config.get("rescaling", True)),  # 奖励重缩放
            gamma=float(config.get("gamma", 0.99)),  # 折扣因子
            max_batchsize=int(config.get("max_batchsize", 100000)),  # 最大批次大小
            reward_normalization=bool(config.get("rew_norm", config.get("reward_normalization", False))),  # 奖励归一化
            deterministic_eval=bool(config.get("deterministic_eval", True)),  # 确定性评估（推理时设为True）
            action_scaling=bool(config.get("action_scaling", True)),  # 动作缩放
            action_bound_method=config.get("action_bound_method", "clip"),  # 动作边界处理方法
        )
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        # 兼容不同的检查点格式
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        agent.policy.load_state_dict(state_dict)
        agent.policy.eval()  # 切换到评估模式（禁用dropout等）
        self.policy = agent.policy
    
    def act(self, obs: Any) -> np.ndarray:
        """
        根据观测输出动作。
        
        Args:
            obs: 环境观测数据
            
        Returns:
            动作数组（numpy.ndarray）
        """
        # 将观测批量化（batch_size=1）
        batch = self._Batch(obs=batchify_observation(obs), info={})
        
        # 前向传播（禁用梯度计算以提高效率）
        with self._torch.no_grad():
            policy_output = self.policy(batch)
        
        # 提取动作并转换回numpy数组
        action = self._to_numpy(policy_output.act)
        # 映射动作到环境要求的范围
        action = self.policy.map_action(action)
        # 返回第一个（也是唯一一个）样本的动作
        return np.asarray(action[0], dtype=np.float32)


# ==================== 配置构建函数 ====================

def build_extra_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    根据命令行参数构建额外的环境配置。
    
    该函数将命令行参数转换为环境可识别的配置字典，
    并根据控制器类型添加特定的配置项。
    
    Args:
        args: 解析后的命令行参数
        
    Returns:
        额外配置字典
    """
    # 确定是否启用手控模式
    if args.ego_controller == "manual":
        # 手控模式下，如果未明确指定，则根据--render自动决定
        manual_control = args.render if args.manual_control is None else args.manual_control
    else:
        # 非手控模式（IDM或PPO），强制关闭手控
        manual_control = False
    
    # 基础配置
    extra_config: Dict[str, Any] = {
        "use_render": args.render,
        "manual_control": manual_control,
        "crash_vehicle_done": args.crash_vehicle_done,
        "crash_object_done": args.crash_object_done,
        "out_of_road_done": args.out_of_road_done,
    }
    if args.traffic_density is not None:
        extra_config["traffic_density"] = float(args.traffic_density)
    if args.traffic_backend is not None:
        extra_config.setdefault("single_scene", {})["traffic_backend"] = str(args.traffic_backend)
    
    # 如果使用IDM控制器，添加IDM特定配置
    if args.ego_controller == "idm":
        extra_config.update(
            {
                "agent_policy": load_idm_policy(),  # IDM策略类
                "enable_idm_lane_change": not args.disable_idm_lane_change,  # 是否允许IDM变道
                "disable_idm_deceleration": args.disable_idm_deceleration,   # 是否禁用IDM减速
            }
        )
    
    # 如果提供了额外的JSON配置，递归合并
    if args.extra_config_json:
        user_extra = json.loads(args.extra_config_json)
        if not isinstance(user_extra, dict):
            raise ValueError("--extra-config-json 必须解码为JSON对象")
        extra_config = deep_merge_dict(extra_config, user_extra)
    
    return extra_config


def get_action(env, action_mode: str):
    """
    根据动作模式生成动作。
    
    Args:
        env: 环境实例
        action_mode: 动作模式（"idle"/"forward"/"random"）
        
    Returns:
        动作列表或从动作空间采样
        
    Raises:
        ValueError: 不支持的动作模式
    """
    if action_mode == "idle":
        return [0.0, 0.0]  # [加速度, 转向角]，全零表示静止
    if action_mode == "forward":
        return [0.0, 1.0]  # 保持当前速度，不转向
    if action_mode == "random":
        return env.action_space.sample()  # 从动作空间随机采样
    raise ValueError("不支持的动作模式 {!r}".format(action_mode))


def format_spawn_velocity(vehicle) -> Any:
    """
    格式化车辆的初始速度信息。
    
    Args:
        vehicle: 车辆对象
        
    Returns:
        格式化的速度列表 [纵向速度, 横向速度]，保留两位小数
    """
    spawn_velocity = vehicle.config.get("spawn_velocity")
    if spawn_velocity is None:
        return None
    return [round(float(v), 2) for v in spawn_velocity]


def collect_initialized_traffic(env) -> list:
    """
    统一收集当前episode已初始化的交通车辆。

    Trigger模式下，车辆会先保存在spawned_objects中，进入触发区域后才变为active traffic_vehicles；
    Basic模式则直接使用traffic_vehicles。这里优先读取spawned_objects，以兼容官方两种模式。
    """
    traffic_manager = getattr(env.engine, "traffic_manager", None)
    if traffic_manager is None:
        return []

    spawned_objects = list(getattr(traffic_manager, "spawned_objects", {}).values())
    if spawned_objects:
        return spawned_objects
    return list(getattr(traffic_manager, "traffic_vehicles", []))


def summarize_reset(env, scene: str, episode_idx: int, info: Dict[str, Any], traffic_limit: int) -> None:
    """
    在每次reset后打印环境状态摘要。
    
    显示自车和交通车辆的初始位置、速度等信息，便于调试和验证随机化效果。
    
    Args:
        env: 环境实例
        scene: 场景名称
        episode_idx: 当前episode索引
        info: reset返回的info字典
        traffic_limit: 最多打印多少辆交通车辆
    """
    # 获取自车对象
    ego = next(iter(env.agents.values()))
    # 获取所有交通车辆列表
    traffic = collect_initialized_traffic(env)
    
    # 打印分隔线和基本信息
    print("\n===== reset {} / scene={} =====".format(episode_idx, scene))
    
    # 打印自车初始状态
    print(
        "自车: 车道={}, 纵向位置={:.2f}m, 横向位置={:.2f}m, 速度={}, 目的地={}".format(
            ego.config.get("spawn_lane_index"),
            float(ego.config.get("spawn_longitude", 0.0)),
            float(ego.config.get("spawn_lateral", 0.0)),
            format_spawn_velocity(ego),
            ego.config.get("destination"),
        )
    )
    
    # 打印偏离道路相关信息
    print(
        "信息: 累计成本={}, 偏离模式={}, warning次数={}".format(
            info.get("total_cost"),
            info.get("out_of_road_mode"),
            info.get("out_of_road_warning_count"),
        )
    )
    
    # 打印交通车辆总数
    print("交通车辆总数={}".format(len(traffic)))
    
    # 打印前traffic_limit辆交通车辆的详细信息
    for idx, vehicle in enumerate(traffic[:traffic_limit]):
        print(
            "  交通车[{idx}]: 车道={lane}, 纵向={long:.2f}m, 横向={lat:.2f}m, 速度={speed}".format(
                idx=idx,
                lane=vehicle.config.get("spawn_lane_index"),
                long=float(vehicle.config.get("spawn_longitude", 0.0)),
                lat=float(vehicle.config.get("spawn_lateral", 0.0)),
                speed=format_spawn_velocity(vehicle),
            )
        )


def build_render_text(
    scene: str,
    ego_controller: str,
    episode_idx: int,
    episode_steps: int,
    env,
    reward,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    构建渲染窗口中显示的文本信息。
    
    Args:
        scene: 场景名称
        ego_controller: 自车控制器类型
        episode_idx: 当前episode索引
        episode_steps: 当前episode的步数
        env: 环境实例
        reward: 当前步奖励
        info: step返回的info字典
        
    Returns:
        包含渲染信息的字典
    """
    ego = next(iter(env.agents.values()))
    traffic_count = len(collect_initialized_traffic(env))
    
    return {
        "scene": scene,                                    # 场景名称
        "controller": ego_controller,                      # 控制器类型
        "episode": episode_idx,                            # Episode索引
        "step": episode_steps,                             # 当前步数
        "reward": round(float(reward), 3),                 # 当前步奖励（保留3位小数）
        "cost": round(float(info.get("cost", 0.0)), 3),   # 当前步成本
        "traffic_n": traffic_count,                        # 交通车辆数量
        "ego_lane": str(ego.config.get("spawn_lane_index")), # 自车车道
        "ego_long": round(float(ego.config.get("spawn_longitude", 0.0)), 2), # 自车纵向位置
        "ego_dest": str(ego.config.get("destination")),    # 自车目的地
    }


# ==================== 主函数 ====================

def main() -> int:
    """
    主函数，协调整个运行流程。
    
    流程概述：
    1. 解析命令行参数
    2. 构建环境配置
    3. 创建环境实例
    4. （可选）加载PPO模型
    5. 进入主循环：观测 -> 动作 -> 步进 -> 渲染
    6. 处理episode结束和重置
    7. 清理资源
    
    Returns:
        退出码（0表示正常退出）
    """
    # 1. 解析命令行参数
    args = parse_args()
    
    # 2. 加载环境API
    get_single_scene_training_env, get_single_scene_validation_env = load_env_api()
    
    # 3. 构建额外配置
    extra_config = build_extra_config(args)
    
    # 4. 确定手控状态
    manual_control = extra_config["manual_control"]
    
    # 5. 验证PPO配置
    if args.ego_controller == "ppo" and not args.ppo_path:
        raise ValueError("当 --ego-controller=ppo 时必须提供 --ppo-path")
    
    # 6. 确定动作模式
    # 优先级：显式指定的action_mode > 根据控制器类型推断
    action_mode = args.action_mode or ("idle" if manual_control else "random")
    if args.ego_controller == "idm":
        action_mode = "idle"  # IDM控制器自己生成动作，外部只需传递[0,0]
    if args.ego_controller == "ppo":
        action_mode = "ppo"   # PPO控制器自己生成动作
    
    # 7. 选择环境构建函数（训练集或验证集）
    env_builder = get_single_scene_training_env if args.split == "train" else get_single_scene_validation_env
    
    # 8. 打印启动信息
    print(
        "启动场景='{}', 分割='{}', 渲染={}, 控制器='{}', 手控={}, 动作模式='{}'".format(
            args.scene,
            args.split,
            args.render,
            args.ego_controller,
            manual_control,
            action_mode,
        )
    )
    
    # 9. 创建环境实例
    env = env_builder(args.scene, extra_config=extra_config)
    
    # 10. （可选）加载PPO控制器
    ppo_controller = None
    if args.ego_controller == "ppo":
        try:
            import yaml  # YAML配置文件解析
        except ModuleNotFoundError as exc:
            print("缺少依赖: {}".format(exc.name), file=sys.stderr)
            raise SystemExit(1) from exc
        
        # 解析PPO模型文件路径
        artifacts = resolve_ppo_artifacts(args.ppo_path, args.ppo_best)
        
        # 加载训练配置
        with open(artifacts["config_path"], "r", encoding="utf-8") as file_obj:
            ppo_config = yaml.safe_load(file_obj) or {}
        
        # 创建PPO控制器
        ppo_controller = PPOEgoController(env, ppo_config, artifacts["model_path"], args.ppo_device)
        print("已从 '{}' 加载PPO检查点".format(artifacts["model_path"]))
    
    # 11. 初始化计数器
    completed_episodes = 0  # 已完成的episode数量
    total_steps = 0         # 总步数
    episode_steps = 0       # 当前episode的步数
    
    try:
        # 12. 首次重置环境
        obs, info = env.reset()
        summarize_reset(env, args.scene, completed_episodes, info, args.print_traffic_limit)
        
        # 13. 如果启用渲染，显示帮助信息
        if args.render:
            env.engine.toggle_help_message()
        
        # 14. 主循环
        while True:
            # 14.1 生成动作
            if action_mode == "ppo":
                # 使用PPO策略生成动作
                action = ppo_controller.act(obs)
            else:
                # 使用简单动作模式
                action = get_action(env, action_mode)
            
            # 14.2 执行环境步进
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            episode_steps += 1
            
            # 14.3 渲染（如果启用）
            if args.render and args.render_mode == "topdown":
                env.render(
                    mode="topdown",                          # 俯视鸟瞰图
                    target_agent_heading_up=True,            # 自车始终朝上
                    text=build_render_text(                  # 叠加文本信息
                        args.scene,
                        args.ego_controller,
                        completed_episodes,
                        episode_steps,
                        env,
                        reward,
                        info,
                    ),
                )
            
            # 14.4 检查episode是否结束
            if terminated or truncated:
                # 打印结束信息
                print(
                    "结束: terminated={}, truncated={}, 步奖励={}, 累计奖励={}, episode长度={}".format(
                        terminated,
                        truncated,
                        round(float(reward), 3),
                        round(float(info.get("episode_reward", 0.0)), 3),
                        info.get("episode_length"),
                    )
                )
                
                completed_episodes += 1
                
                # 检查是否达到episode数量上限
                if args.episodes > 0 and completed_episodes >= args.episodes:
                    break
                
                # 检查是否达到总步数上限
                if args.max_steps > 0 and total_steps >= args.max_steps:
                    break
                
                # 重置环境，开始新的episode
                obs, info = env.reset()
                episode_steps = 0
                summarize_reset(env, args.scene, completed_episodes, info, args.print_traffic_limit)
            
            # 14.5 再次检查总步数上限（防止在episode中间超出）
            if args.max_steps > 0 and total_steps >= args.max_steps:
                break
    
    except KeyboardInterrupt:
        # 捕获Ctrl+C中断信号
        print("\n用户中断，正在关闭环境。")
    
    finally:
        # 无论正常退出还是异常退出，都确保关闭环境
        env.close()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
