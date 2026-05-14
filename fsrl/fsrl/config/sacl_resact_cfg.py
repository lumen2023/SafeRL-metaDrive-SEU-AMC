"""
SAC-Lagrangian + ResAct 配置

ResAct (残差动作): final_action = tanh(residual + prev_action)
让策略输出动作修正量而非绝对值,实现更平滑的驾驶控制
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from fsrl.fsrl.config.sacl_cfg import (
    Bullet1MCfg,
    Bullet5MCfg,
    Bullet10MCfg,
    Mujoco2MCfg,
    Mujoco10MCfg,
    Mujoco20MCfg,
    MujocoBaseCfg,
    TrainCfg as SACLTrainCfg,
)


@dataclass
class TrainCfg(SACLTrainCfg):
    """SAC-Lag + ResAct 训练配置"""
    
    task: str = "SafeMetaDrive"  # 任务名称
    
    # N-Step Learning: 多步回报的步数
    # n_step=1: 单步TD学习,更新快、稳定(推荐)
    # n_step=2~5: 考虑更多未来信息,但方差增大
    n_step: int = 5
    
    # ===== ResAct 残差动作配置 =====
    resact_enabled: bool = True  # 是否启用ResAct
    
    # 转向残差缩放系数 (0.05~0.3)
    # 值越小动作越平滑,但响应变慢; 0.15为中等保守值
    resact_steer_delta_scale: float = 0.8
    
    # 油门残差缩放系数 (0.05~0.2)
    # 通常比转向系数小,避免加减速顿挫; 0.10较保守
    resact_throttle_delta_scale: float = 1
    
    # Episode初始动作值 (steering, throttle)
    # (0.0, 0.0)表示静止起步,最安全
    resact_initial_action: Tuple[float, float] = (0.0, 0.0)
    
    # WandB实验名称后缀，保持简短，具体ResAct尺度由训练脚本补短标签
    suffix: Optional[str] = "-steer0.8"
