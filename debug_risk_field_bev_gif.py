"""生成风险场BEV视角GIF动画 - 使用IDM策略控制主车运行MetaDrive环境

功能说明：
    该脚本在MetaDrive环境中运行IDM（智能驾驶员模型）控制的车辆，同时实时计算并可视化
    风险场的鸟瞰图（BEV），最终生成GIF动画用于调试和分析风险场分布。

使用方法：
    python debug_risk_field_bev_gif.py --output debug/risk_dense.gif --frames 20 --resolution 0.75 --dpi 100

注意：
    该脚本会逐网格采样风险场，适合精确检查势场形状，但不适合长GIF。
    长GIF和快速排查请优先使用 debug_risk_field_topdown_overlay_gif.py。
"""

import argparse
import os

import imageio.v2 as imageio
import matplotlib
import numpy as np

# 设置matplotlib后端为非交互式模式，避免显示窗口
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from safe_metadrive_adapter.local_import import prefer_local_metadrive

prefer_local_metadrive()

from env import get_training_env
from risk_field_bev import plot_risk_field_bev

try:
    from metadrive.policy.idm_policy import IDMPolicy
except ImportError:
    from metadrive.metadrive.policy.idm_policy import IDMPolicy


DENSE_BEV_HELP = """\
推荐路径：
  快速长GIF:
    python debug_risk_field_topdown_overlay_gif.py --output debug/risk_topdown_overlay.gif --frames 120 --fps 10

  精确风险场少量帧:
    python debug_risk_field_bev_gif.py --output debug/risk_dense.gif --frames 20 --fps 10 --component total --resolution 0.75 --vmax 1.2 --dpi 100

性能说明：
  本脚本逐网格调用RiskFieldCalculator.calculate_at_position，分辨率越小越慢。
  如果只是排查MetaDrive真实地图上的风险位置，优先用官方topdown叠加脚本。
"""


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有配置参数的命名空间对象
    """
    parser = argparse.ArgumentParser(
        description="运行IDM控制的主车并生成精确但较慢的密集风险场BEV GIF动画",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DENSE_BEV_HELP,
    )
    
    # ========== 输出配置 ==========
    parser.add_argument(
        "--output", 
        default="debug/risk_field_idm.gif", 
        help="GIF输出文件路径"
    )
    parser.add_argument(
        "--frame-dir",
        default=None,
        help="保存PNG帧图片的目录路径。默认为'<输出文件名_stem>_frames'"
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="仅生成GIF，不保存中间PNG帧图片（节省磁盘空间）"
    )
    
    # ========== 动画参数 ==========
    parser.add_argument(
        "--frames", 
        type=int, 
        default=120, 
        help="GIF总帧数"
    )
    parser.add_argument(
        "--fps", 
        type=float, 
        default=10.0, 
        help="GIF播放帧率（帧/秒）"
    )
    parser.add_argument(
        "--step-per-frame", 
        type=int, 
        default=1, 
        help="每生成一帧GIF需要执行的环境步数（控制动画速度）"
    )
    
    # ========== 风险场可视化配置 ==========
    parser.add_argument(
        "--component", 
        default="total", 
        help="风险场组件类型: total/road/boundary/lane/offroad/vehicle/object/headway/ttc"
    )
    parser.add_argument(
        "--resolution", 
        type=float, 
        default=0.5, 
        help="风险热力图网格分辨率（米/像素）"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="PNG/GIF帧保存DPI。数值越高清晰度越高，但matplotlib保存越慢"
    )
    parser.add_argument(
        "--cmap",
        default="risk_legacy",
        help="颜色映射方案：risk_legacy接近原版势场图，risk_red为纯红风险图"
    )
    parser.add_argument(
        "--interpolation",
        default="bilinear",
        help="热力图插值方式，例如 bilinear/nearest/bicubic"
    )
    parser.add_argument(
        "--vmax", 
        type=float, 
        default=1.2, 
        help="颜色映射最大值（红色上限）。设为<=0则每帧自动调整"
    )
    parser.add_argument(
        "--risk-min-visible", 
        type=float, 
        default=1e-4, 
        help="低于此值的风险将被设为透明（过滤噪声）"
    )
    
    # ========== BEV采样范围配置 ==========
    parser.add_argument(
        "--front-min", 
        type=float, 
        default=-20.0,
        help="前方采样范围最小值（米，负值表示车后）"
    )
    parser.add_argument(
        "--front-max", 
        type=float, 
        default=70.0,
        help="前方采样范围最大值（米）"
    )
    parser.add_argument(
        "--lateral-min", 
        type=float, 
        default=-20.0,
        help="横向采样范围最小值（米，负值表示左侧）"
    )
    parser.add_argument(
        "--lateral-max", 
        type=float, 
        default=20.0,
        help="横向采样范围最大值（米）"
    )
    
    # ========== 环境配置 ==========
    parser.add_argument(
        "--seed", 
        type=int, 
        default=100, 
        help="训练环境的起始随机种子（start_seed）"
    )
    parser.add_argument(
        "--traffic-density", 
        type=float, 
        default=0.2,
        help="交通密度（0.0-1.0，控制周围车辆数量）"
    )
    parser.add_argument(
        "--accident-prob", 
        type=float, 
        default=0.0,
        help="事故概率（触发其他车辆异常行为的概率）"
    )
    
    # ========== 可视化选项 ==========
    parser.add_argument(
        "--world-frame", 
        action="store_true", 
        help="使用世界坐标系绘图（默认使用主车航向朝上的局部坐标系）"
    )
    parser.add_argument(
        "--draw-lane-centers", 
        action="store_true", 
        help="绘制车道中心线（仅用于几何调试，非真实道路标线）"
    )
    parser.add_argument(
        "--draw-lane-surfaces",
        action="store_true",
        help="填充车道表面多边形（默认关闭，只描边）"
    )
    parser.add_argument(
        "--hide-lane-surfaces", 
        action="store_true", 
        help="兼容旧参数。车道表面默认已经不填充"
    )
    
    # ========== IDM策略配置 ==========
    parser.add_argument(
        "--disable-idm-lane-change", 
        action="store_true", 
        help="禁用IDM换道逻辑（车辆保持在当前车道）"
    )
    parser.add_argument(
        "--disable-idm-deceleration", 
        action="store_true", 
        help="禁用IDM减速逻辑（车辆不会因前车而减速）"
    )
    
    # ========== 运行控制 ==========
    parser.add_argument(
        "--reset-on-done", 
        action="store_true", 
        help="当episode结束时自动重置环境并继续生成帧（否则停止）"
    )
    
    return parser.parse_args()


def main():
    """主函数：运行IDM控制的环境并生成风险场BEV GIF动画"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # 初始化帧图片保存目录
    frame_dir = None
    if not args.no_save_frames:
        frame_dir = args.frame_dir or f"{os.path.splitext(args.output)[0]}_frames"
        os.makedirs(frame_dir, exist_ok=True)
    
    # 确定颜色映射的最大值（None表示每帧自动调整）
    fixed_vmax = None if args.vmax <= 0 else args.vmax

    # ========== 创建训练环境 ==========
    env = get_training_env({
        "num_scenarios": 1,              # 单场景模式
        "start_seed": args.seed,         # 随机种子
        "use_render": False,             # 不使用3D渲染（提升性能）
        "manual_control": False,         # 非人工控制模式
        "agent_policy": IDMPolicy,       # 使用IDM策略控制主车
        "traffic_density": args.traffic_density,      # 交通密度
        "accident_prob": args.accident_prob,          # 事故概率
        "enable_idm_lane_change": not args.disable_idm_lane_change,     # 是否允许换道
        "disable_idm_deceleration": args.disable_idm_deceleration,      # 是否禁用减速
    })

    # 创建GIF写入器
    writer = imageio.get_writer(
        args.output, 
        mode="I",                        # 图像模式
        duration=1.0 / max(args.fps, 1e-6),  # 每帧持续时间（秒）
        loop=0                           # 无限循环播放
    )
    
    try:
        # 重置环境，开始新的episode
        env.reset()
        
        # ========== 逐帧生成GIF ==========
        for frame_index in range(max(args.frames, 1)):
            # 获取主车对象
            vehicle = next(iter(env.agents.values()))
            
            # 确定当前帧的保存路径
            frame_path = None
            if frame_dir is not None:
                frame_path = os.path.join(frame_dir, f"frame_{frame_index:06d}.png")
            
            # ========== 绘制风险场BEV图 ==========
            fig, _, _ = plot_risk_field_bev(
                env,
                vehicle,
                save_path=frame_path,                    # PNG帧保存路径（可选）
                component=args.component,                # 风险场组件类型
                front_range=(args.front_min, args.front_max),           # 前方采样范围
                lateral_range=(args.lateral_min, args.lateral_max),     # 横向采样范围
                resolution=args.resolution,              # 网格分辨率
                dpi=args.dpi,                            # 保存DPI，控制清晰度/速度
                ego_frame=not args.world_frame,          # 是否使用主车局部坐标系
                draw_lane_centers=args.draw_lane_centers,               # 是否绘制车道中心线
                draw_lane_surfaces=args.draw_lane_surfaces and not args.hide_lane_surfaces,  # 默认只描边
                cmap=args.cmap,                            # 颜色映射方案
                vmax=fixed_vmax,                         # 颜色映射最大值
                risk_min_visible=args.risk_min_visible,  # 最小可见风险阈值
                interpolation=args.interpolation,         # 平滑插值，降低像素感
                show=False,                              # 不显示matplotlib窗口
            )
            
            # 将帧图片添加到GIF
            if frame_path is not None:
                # 从PNG文件读取
                writer.append_data(_read_rgb_image(frame_path))
            else:
                # 直接从matplotlib图形转换
                writer.append_data(_figure_to_rgb_array(fig))
            
            # 关闭图形以释放内存
            plt.close(fig)

            # ========== 执行环境步进 ==========
            done = False
            for _ in range(max(args.step_per_frame, 1)):
                # 执行空动作（IDM策略会自动计算动作）
                _, _, terminated, truncated, _ = env.step([0, 0])
                done = bool(terminated or truncated)
                if done:
                    break

            # 打印进度信息
            if frame_index % 10 == 0:
                print(f"已写入第 {frame_index + 1}/{args.frames} 帧")
            
            # 处理episode结束
            if done:
                if args.reset_on_done:
                    # 重置环境并继续
                    env.reset()
                else:
                    # 停止生成
                    print(f"Episode在第 {frame_index + 1} 帧结束")
                    break
        
        # 输出完成信息
        if frame_dir is not None:
            print(f"PNG帧图片已保存到 {frame_dir}")
        print(f"风险场IDM GIF已保存到 {args.output}")
        
    finally:
        # 确保资源正确释放
        writer.close()
        env.close()


def _figure_to_rgb_array(fig):
    """将matplotlib图形转换为RGB数组
    
    Args:
        fig: matplotlib Figure对象
        
    Returns:
        np.ndarray: RGB图像数组 (H, W, 3)，dtype=uint8
    """
    fig.canvas.draw()                                    # 渲染图形到画布
    rgba = np.asarray(fig.canvas.buffer_rgba())         # 获取RGBA缓冲区
    return rgba[:, :, :3].copy()                        # 提取RGB通道并复制


def _read_rgb_image(path):
    """从文件读取RGB图像
    
    Args:
        path: 图片文件路径
        
    Returns:
        np.ndarray: RGB图像数组 (H, W, 3)，dtype=uint8
    """
    image = imageio.imread(path)                        # 读取图片
    if image.ndim == 2:                                 # 如果是灰度图
        return np.stack([image, image, image], axis=-1) # 转换为RGB
    return image[:, :, :3]                              # 提取RGB通道


if __name__ == "__main__":
    main()
