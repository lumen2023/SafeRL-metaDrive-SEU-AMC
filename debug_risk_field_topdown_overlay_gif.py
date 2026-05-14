"""基于MetaDrive官方topdown帧的快速GIF调试视图

本脚本通过复用MetaDrive官方的topdown渲染结果，仅叠加轻量级风险元素，
比逐网格采样的matplotlib BEV方法快得多，适合生成长GIF和快速排查问题。

核心特性：
    - 速度快：直接复用MetaDrive渲染的地图和车辆
    - 支持三种车辆风险组件：静态/动态/总和
    - 可配置的风险可视化参数（透明度、颜色映射等）
    - 可选的道路边界/车道线风险叠加

使用示例：
    # 基础用法：生成总风险场GIF
    python debug_risk_field_topdown_overlay_gif.py --output debug/risk_topdown_overlay.gif --frames 120 --fps 10
    
    # 调整缩放比例（像素/米）
    python debug_risk_field_topdown_overlay_gif.py --output debug/risk_topdown_overlay.gif --frames 120 --fps 10 --scaling 6.0
    
    # 仅绘制车辆风险（隐藏道路边界）
    python debug_risk_field_topdown_overlay_gif.py --output debug/risk_vehicle_only.gif --frames 120 --fps 10 --vehicle-only
    
    # 仅绘制动态速度风险场
    python debug_risk_field_topdown_overlay_gif.py --output debug/risk_vehicle_dynamic.gif --frames 120 --fps 10 --vehicle-only --vehicle-risk-component dynamic
"""

import argparse
import math
import os
from typing import Any, Iterable, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from safe_metadrive_adapter.local_import import prefer_local_metadrive

prefer_local_metadrive()

from risk_field import RiskFieldCalculator


RISK_FIELD_OVERRIDE_ARGS = {
    "risk_field_lane_edge_sigma": "risk_field_lane_edge_sigma",
    "risk_field_lane_core_sigma_scale": "risk_field_lane_core_sigma_scale",
    "risk_field_lane_shoulder_sigma_scale": "risk_field_lane_shoulder_sigma_scale",
    "risk_field_lane_shoulder_weight": "risk_field_lane_shoulder_weight",
    "risk_field_boundary_sigma": "risk_field_boundary_sigma",
    "risk_field_broken_line_sigma": "risk_field_broken_line_sigma",
    "risk_field_lane_weight": "risk_field_lane_weight",
    "risk_field_boundary_weight": "risk_field_boundary_weight",
    "risk_field_broken_line_factor": "risk_field_broken_line_factor",
    "risk_field_solid_line_factor": "risk_field_solid_line_factor",
    "risk_field_boundary_line_factor": "risk_field_boundary_line_factor",
    "risk_field_oncoming_line_factor": "risk_field_oncoming_line_factor",
}


def build_risk_overlay_args(config: dict) -> argparse.Namespace:
    """把配置字典整理成风险叠加 helper 需要的轻量参数对象。"""
    return argparse.Namespace(
        width=int(config.get("risk_overlay_width", 800)),
        height=int(config.get("risk_overlay_height", 800)),
        film_size=int(config.get("risk_overlay_film_size", 2000)),
        scaling=float(config.get("risk_overlay_scaling", 8.0)),
        vehicle_only=bool(config.get("risk_overlay_vehicle_only", False)),
        no_road_edge_risk=bool(config.get("risk_overlay_no_road", False)),
        no_vehicle_risk=bool(config.get("risk_overlay_no_vehicle", False)),
        edge_width=int(config.get("risk_overlay_edge_width", 3)),
        edge_alpha=int(config.get("risk_overlay_edge_alpha", 120)),
        road_risk_component=str(config.get("risk_overlay_road_component", "total")),
        road_risk_vmax=float(config.get("risk_overlay_road_vmax", 0.0)),
        road_risk_min_visible=float(config.get("risk_overlay_road_min_visible", 0.02)),
        road_alpha=int(config.get("risk_overlay_road_alpha", 135)),
        road_min_alpha=int(config.get("risk_overlay_road_min_alpha", 28)),
        road_risk_layers=int(config.get("risk_overlay_road_layers", 10)),
        road_risk_sample_step=float(config.get("risk_overlay_road_sample_step", 1.0)),
        road_risk_use_weights=bool(config.get("risk_overlay_road_use_weights", True)),
        vehicle_alpha=int(config.get("risk_overlay_vehicle_alpha", 120)),
        vehicle_layers=int(config.get("risk_overlay_vehicle_layers", 9)),
        vehicle_risk_component=str(config.get("risk_overlay_vehicle_component", "total")),
        vehicle_risk_vmax=float(config.get("risk_overlay_vehicle_vmax", 0.0)),
        vehicle_risk_min_visible=float(config.get("risk_overlay_vehicle_min_visible", 0.02)),
        vehicle_risk_use_weights=bool(config.get("risk_overlay_vehicle_use_weights", True)),
        risk_distance=float(config.get("risk_overlay_risk_distance", 80.0)),
        no_risk_panel=bool(config.get("risk_overlay_no_panel", False)),
        no_colorbar=bool(config.get("risk_overlay_no_colorbar", False)),
        no_config_panel=bool(config.get("risk_overlay_no_config_panel", False)),
        side_by_side=bool(config.get("risk_overlay_side_by_side", False)),
        show_ego_shape=bool(config.get("risk_overlay_show_ego_shape", True)),
        show_surrounding_vehicle_shapes=bool(config.get("risk_overlay_show_surrounding_vehicle_shapes", False)),
        show_static_object_shapes=bool(config.get("risk_overlay_show_static_object_shapes", False)),
    )


TOPDOWN_HELP = """\
推荐用途：
  长GIF、快速排查、确认风险线/周车是否跟随MetaDrive真实地图运动。

对比：
  debug_risk_field_bev_gif.py 会逐网格采样真实风险场，数学上更精确但慢。
  本脚本复用MetaDrive官方topdown帧，只叠加轻量风险元素，因此适合生成长GIF。

示例：
  python debug_risk_field_topdown_overlay_gif.py --output debug/risk_topdown_overlay.gif --frames 120 --fps 10
  python debug_risk_field_topdown_overlay_gif.py --output debug/risk_topdown_overlay.gif --frames 120 --fps 10 --scaling 6.0
  python debug_risk_field_topdown_overlay_gif.py --output debug/risk_road_overlay.gif --frames 120 --fps 10 --no-vehicle-risk
  python debug_risk_field_topdown_overlay_gif.py --output debug/risk_road_wide.gif --frames 120 --fps 10 --no-vehicle-risk --risk-field-lane-edge-sigma 1.5 --risk-field-boundary-sigma 1.5
  python debug_risk_field_topdown_overlay_gif.py --output debug/risk_line_types.gif --frames 120 --fps 10 --no-vehicle-risk --risk-field-broken-line-factor 0.05 --risk-field-solid-line-factor 0.65 --risk-field-oncoming-line-factor 1.35
  python debug_risk_field_topdown_overlay_gif.py --output debug/risk_vehicle_only.gif --frames 120 --fps 10 --vehicle-only
  python debug_risk_field_topdown_overlay_gif.py --output debug/risk_vehicle_dynamic.gif --frames 120 --fps 10 --vehicle-only --vehicle-risk-component dynamic
"""


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="Fast official topdown GIF with risk overlays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=TOPDOWN_HELP,
    )
    
    # ========== 输出配置 ==========
    parser.add_argument("--output", default="debug/risk_topdown_overlay.gif", help="GIF保存路径")
    parser.add_argument("--frame-dir", default=None, help="PNG帧保存目录，默认为'<output_stem>_frames'")
    parser.add_argument("--no-save-frames", action="store_true", help="仅生成GIF，不保存中间PNG帧")
    
    # ========== 动画参数 ==========
    parser.add_argument("--frames", type=int, default=120, help="GIF总帧数")
    parser.add_argument("--fps", type=float, default=10.0, help="GIF播放帧率")
    parser.add_argument("--step-per-frame", type=int, default=1, help="每帧之间的环境步进次数")
    
    # ========== 环境配置 ==========
    parser.add_argument("--seed", type=int, default=100, help="训练环境随机种子")
    parser.add_argument("--traffic-density", type=float, default=0.2, help="交通密度（0-1）")
    parser.add_argument("--accident-prob", type=float, default=0.0, help="事故概率")
    
    # ========== 渲染配置 ==========
    parser.add_argument("--width", type=int, default=800, help="Topdown帧宽度（像素）")
    parser.add_argument("--height", type=int, default=800, help="Topdown帧高度（像素）")
    parser.add_argument("--film-size", type=int, default=2000, help="MetaDrive topdown地图胶片尺寸")
    parser.add_argument("--scaling", type=float, default=8.0, help="Topdown缩放比例（像素/米）")
    
    # ========== 风险可视化选项 ==========
    parser.add_argument("--vehicle-only", action="store_true", help="仅绘制车辆风险，隐藏道路/车道边缘风险")
    parser.add_argument("--no-road-edge-risk", action="store_true", help="不绘制道路/车道边缘风险叠加层")
    parser.add_argument("--no-vehicle-risk", action="store_true", help="不绘制车辆风险斑块")
    
    # ========== 道路边缘样式 ==========
    parser.add_argument("--edge-width", type=int, default=3, help="车道边缘叠加层线宽（像素）")
    parser.add_argument("--edge-alpha", type=int, default=120, help="车道边缘叠加层透明度（0-255）")
    parser.add_argument(
        "--road-risk-component",
        choices=("lane", "broken", "solid", "boundary", "oncoming", "total"),
        default="total",
        help="道路风险组件类型：lane=非边界车道线, broken=虚线, solid=实线, boundary=道路边界, oncoming=黄色/对向分隔线, total=全部"
    )
    parser.add_argument("--road-risk-vmax", type=float, default=0.0, help="道路风险值映射到最红色的阈值。<=0时按当前线型组件自动选择")
    parser.add_argument("--road-risk-min-visible", type=float, default=0.02, help="低于此值的道路风险将被隐藏")
    parser.add_argument("--road-alpha", type=int, default=135, help="道路/车道线风险最大透明度（0-255）")
    parser.add_argument("--road-risk-layers", type=int, default=10, help="道路风险带分层数，越大越平滑")
    parser.add_argument("--road-risk-sample-step", type=float, default=1.0, help="道路边缘采样步长（米）")
    parser.add_argument(
        "--road-risk-use-weights",
        action="store_true",
        help="将risk_field_*_weight也乘进道路风险可视化；默认显示未加权原始风险形状"
    )
    
    # ========== 车辆风险样式 ==========
    parser.add_argument("--vehicle-alpha", type=int, default=120, help="车辆风险最大透明度（0-255）")
    parser.add_argument("--vehicle-layers", type=int, default=9, help="[已弃用] 兼容性选项，现采用最大值栅格化")
    parser.add_argument(
        "--vehicle-risk-component",
        choices=("static", "dynamic", "total"),
        default="total",
        help="车辆风险组件类型：static=静态占据, dynamic=动态速度风险, total=总和"
    )
    parser.add_argument(
        "--vehicle-risk-vmax",
        type=float,
        default=0.0,
        help="车辆风险值映射到最红色的阈值。<=0时使用组件默认值（total=2.0, static/dynamic=1.0）"
    )
    parser.add_argument("--vehicle-risk-min-visible", type=float, default=0.02, help="低于此值的车辆风险将被隐藏")
    parser.add_argument("--risk-distance", type=float, default=80.0, help="车辆风险叠加的最大距离（米）")
    parser.add_argument("--show-ego-shape", action="store_true", help="在风险场上绘制主车 footprint。")
    parser.add_argument("--no-risk-panel", action="store_true", help="隐藏右上角当前风险数值面板")
    parser.add_argument("--no-colorbar", action="store_true", help="隐藏右上角风险色条")
    parser.add_argument("--risk-field-lane-edge-sigma", type=float, default=None, help="覆盖车道线风险sigma，用于快速调参")
    parser.add_argument("--risk-field-lane-core-sigma-scale", type=float, default=None, help="覆盖车道线窄核sigma缩放，仅强化贴线中心惩罚")
    parser.add_argument("--risk-field-lane-shoulder-sigma-scale", type=float, default=None, help="覆盖车道线宽肩sigma缩放，用于扩大低强度预警范围")
    parser.add_argument("--risk-field-lane-shoulder-weight", type=float, default=None, help="覆盖车道线宽肩权重（0~1），避免宽范围惩罚过重")
    parser.add_argument("--risk-field-boundary-sigma", type=float, default=None, help="覆盖道路边界风险sigma，用于快速调参")
    parser.add_argument("--risk-field-broken-line-sigma", type=float, default=None, help="覆盖虚线单独sigma，用于缩窄虚线风险带")
    parser.add_argument("--risk-field-lane-weight", type=float, default=None, help="覆盖车道线风险权重，用于快速调参")
    parser.add_argument("--risk-field-boundary-weight", type=float, default=None, help="覆盖道路边界风险权重，用于快速调参")
    parser.add_argument("--risk-field-broken-line-factor", type=float, default=None, help="覆盖虚线风险系数，用于快速调参")
    parser.add_argument("--risk-field-solid-line-factor", type=float, default=None, help="覆盖普通实线风险系数，用于快速调参")
    parser.add_argument("--risk-field-boundary-line-factor", type=float, default=None, help="覆盖道路边界/护栏线风险系数，用于快速调参")
    parser.add_argument("--risk-field-oncoming-line-factor", type=float, default=None, help="覆盖黄色/对向分隔线风险系数，用于快速调参")
    
    # ========== IDM策略配置 ==========
    parser.add_argument("--disable-idm-lane-change", action="store_true", help="禁用IDM换道逻辑")
    parser.add_argument("--disable-idm-deceleration", action="store_true", help="禁用IDM减速逻辑")
    parser.add_argument("--reset-on-done", action="store_true", help="Episode结束后自动重置并继续")
    
    return parser.parse_args()


def _apply_risk_field_overrides(config: dict, args: argparse.Namespace) -> None:
    """把命令行风险场调参项写入MetaDrive配置。"""
    for arg_name, config_key in RISK_FIELD_OVERRIDE_ARGS.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config[config_key] = float(value)


def _load_env_dependencies():
    """Load MetaDrive dependencies only for the standalone GIF script."""
    from env import get_training_env

    try:
        from metadrive.policy.idm_policy import IDMPolicy
    except ImportError:
        from metadrive.metadrive.policy.idm_policy import IDMPolicy

    return get_training_env, IDMPolicy


def main():
    """主函数：生成带风险场叠加的Topdown GIF
    
    流程：
    1. 初始化MetaDrive环境和风险场计算器
    2. 逐帧渲染：获取topdown图像 → 叠加风险场 → 保存帧/GIF
    3. 清理资源
    """
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    get_training_env, IDMPolicy = _load_env_dependencies()
    
    # 创建帧保存目录
    frame_dir = None
    if not args.no_save_frames:
        frame_dir = args.frame_dir or f"{os.path.splitext(args.output)[0]}_frames"
        os.makedirs(frame_dir, exist_ok=True)

    # ========== 初始化训练环境 ==========
    env_config = {
        "num_scenarios": 10,
        "start_seed": args.seed,
        "use_render": False,
        "manual_control": False,
        "agent_policy": IDMPolicy,
        "traffic_density": args.traffic_density,
        "accident_prob": args.accident_prob,
        "enable_idm_lane_change": not args.disable_idm_lane_change,
        "disable_idm_deceleration": args.disable_idm_deceleration,
    }
    _apply_risk_field_overrides(env_config, args)
    env = get_training_env(env_config)
    
    # 初始化风险场计算器
    calculator = RiskFieldCalculator(getattr(env, "config", {}))
    
    # 创建GIF写入器
    writer = imageio.get_writer(args.output, mode="I", duration=1.0 / max(args.fps, 1e-6), loop=0)

    try:
        env.reset()
        
        # ========== 逐帧生成GIF ==========
        for frame_index in range(max(args.frames, 1)):
            ego = next(iter(env.agents.values()))
            
            # 1. 渲染MetaDrive官方topdown图像
            base = env.render(
                mode="topdown",
                window=False,
                screen_size=(args.width, args.height),
                film_size=(args.film_size, args.film_size),
                scaling=args.scaling,
                target_agent_heading_up=True,  # 主车朝上
                screen_record=False,
                num_stack=1,
                history_smooth=0,
                draw_center_line=False,  # 不画车道中心线
                draw_contour=True,       # 绘制车辆轮廓
            )
            frame = _as_rgb(base)
            
            # 2. 叠加风险场
            frame = _overlay_risk(frame, env, ego, calculator, args)

            # 3. 保存帧（可选）
            frame_path = None
            if frame_dir is not None:
                frame_path = os.path.join(frame_dir, f"frame_{frame_index:06d}.png")
                Image.fromarray(frame).save(frame_path)
            
            # 4. 写入GIF
            writer.append_data(frame)

            # 5. 环境步进
            done = False
            for _ in range(max(args.step_per_frame, 1)):
                _, _, terminated, truncated, _ = env.step([0, 0])
                done = bool(terminated or truncated)
                if done:
                    break

            # 6. 进度提示
            if frame_index % 10 == 0:
                print(f"wrote frame {frame_index + 1}/{args.frames}")
            if done:
                if args.reset_on_done:
                    env.reset()
                else:
                    print(f"episode ended at frame {frame_index + 1}")
                    break

        if frame_dir is not None:
            print(f"Saved PNG frames to {frame_dir}")
        print(f"Saved official-topdown risk overlay GIF to {args.output}")
    finally:
        writer.close()
        env.close()


def _as_rgb(image: Any) -> np.ndarray:
    """将图像转换为RGB格式
    
    Args:
        image: 输入图像（可以是灰度图、RGBA等）
        
    Returns:
        RGB格式的numpy数组 (H, W, 3) uint8
    """
    array = np.asarray(image)
    if array.ndim == 2:
        # 灰度图转RGB
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] > 3:
        # RGBA转RGB（丢弃alpha通道）
        array = array[:, :, :3]
    return np.ascontiguousarray(array.astype(np.uint8))


def _fmt_info(info: dict | None, key: str, default: float = 0.0) -> str:
    if not info or key not in info:
        return "n/a"
    try:
        return f"{float(info.get(key, default)):.3f}"
    except (TypeError, ValueError):
        return str(info.get(key))


def _draw_panel(base: Image.Image, xy: tuple[int, int], title: str, lines: list[str], width: int) -> Image.Image:
    x0, y0 = xy
    line_h = 13
    panel_h = 18 + len(lines) * line_h
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle(
        (x0, y0, x0 + width, y0 + panel_h),
        radius=7,
        fill=(255, 255, 255, 222),
        outline=(15, 23, 42, 155),
    )
    draw.text((x0 + 8, y0 + 6), title, fill=(15, 23, 42, 255))
    for idx, line in enumerate(lines):
        draw.text((x0 + 8, y0 + 20 + idx * line_h), line, fill=(15, 23, 42, 255))
    return Image.alpha_composite(base, overlay)


def draw_eval_step_cost_panel(
    base: Image.Image,
    info: dict | None,
    episode_cost: float,
    episode_length: int,
) -> Image.Image:
    lines = [
        f"step cost: {_fmt_info(info, 'cost')}",
        f"episode cost: {episode_cost:.3f}",
        f"step/len: {max(episode_length, 0)}",
        f"event: {_fmt_info(info, 'event_cost')}",
        f"risk raw: {_fmt_info(info, 'risk_field_cost')}",
        f"risk eq: {_fmt_info(info, 'risk_field_event_equivalent_cost')}",
        f"road/bound/lane: {_fmt_info(info, 'risk_field_road_cost')}/{_fmt_info(info, 'risk_field_boundary_cost')}/{_fmt_info(info, 'risk_field_lane_cost')}",
        f"off/veh/obj: {_fmt_info(info, 'risk_field_offroad_cost')}/{_fmt_info(info, 'risk_field_vehicle_cost')}/{_fmt_info(info, 'risk_field_object_cost')}",
        f"headway/ttc: {_fmt_info(info, 'risk_field_headway_cost')}/{_fmt_info(info, 'risk_field_ttc_cost')}",
        f"out/crash: {int(bool(info and info.get('out_of_road', False)))}/{int(bool(info and (info.get('crash_vehicle', False) or info.get('crash_object', False))))}",
    ]
    return _draw_panel(base, (10, 10), "eval cost chain", lines, width=310)


def _draw_view_badge(base: Image.Image, title: str, *, anchor: tuple[int, int] = (10, 10)) -> Image.Image:
    """在视图左上角打一个短标签，方便区分左右子图。"""
    x0, y0 = anchor
    padding_x = 10
    padding_y = 6
    text_width = max(90, 8 * len(title))
    badge_w = text_width + padding_x * 2
    badge_h = 24
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle(
        (x0, y0, x0 + badge_w, y0 + badge_h),
        radius=8,
        fill=(15, 23, 42, 210),
        outline=(255, 255, 255, 70),
    )
    draw.text((x0 + padding_x, y0 + padding_y), title, fill=(255, 255, 255, 255))
    return Image.alpha_composite(base, overlay)


def _config_float(config: dict, key: str, default: float) -> float:
    value = RiskFieldCalculator._safe_float(config.get(key, default))
    return value if np.isfinite(value) else default


def draw_eval_config_panel(
    base: Image.Image,
    env: Any,
    calculator: RiskFieldCalculator,
    overlay_args: argparse.Namespace,
) -> Image.Image:
    config = getattr(env, "config", {})
    road_vmax = _resolve_road_risk_vmax(overlay_args, calculator)
    vehicle_vmax = _resolve_vehicle_risk_vmax(overlay_args, calculator)
    lines = [
        "source: current env.py validation config",
        f"combine/mapping: {config.get('risk_field_cost_combine', 'max')}/{config.get('risk_field_cost_mapping', 'linear_scale')}",
        f"scale/raw_clip: {_config_float(config, 'risk_field_cost_scale', 1.0):.2f}/{calculator._cfg('risk_field_raw_clip'):.1f}",
        f"event weight/max_dist: {_config_float(config, 'risk_field_event_cost_weight', 1.0):.2f}/{calculator._cfg('risk_field_max_distance'):.1f}",
        f"w road: b{calculator._cfg('risk_field_boundary_weight'):.2f} l{calculator._cfg('risk_field_lane_weight'):.2f} off{calculator._cfg('risk_field_offroad_weight'):.2f}",
        f"w obj: veh{calculator._cfg('risk_field_vehicle_weight'):.2f} obj{calculator._cfg('risk_field_object_weight'):.2f} hw{calculator._cfg('risk_field_headway_weight'):.2f} ttc{calculator._cfg('risk_field_ttc_weight'):.2f}",
        f"sigma road: br{calculator._cfg('risk_field_broken_line_sigma'):.2f} lane{calculator._cfg('risk_field_lane_edge_sigma'):.2f} bound{calculator._cfg('risk_field_boundary_sigma'):.2f}",
        f"lane shape: gaussian beta{calculator._cfg('risk_field_lane_beta'):.2f}",
        f"sigma off: {calculator._cfg('risk_field_offroad_sigma'):.2f}",
        f"sigma veh: long{calculator._cfg('risk_field_vehicle_longitudinal_sigma'):.1f} lat{calculator._cfg('risk_field_vehicle_lateral_sigma'):.1f} beta{calculator._cfg('risk_field_vehicle_beta'):.1f}",
        f"line factor: br{calculator._cfg('risk_field_broken_line_factor'):.2f} solid{calculator._cfg('risk_field_solid_line_factor'):.2f} bd{calculator._cfg('risk_field_boundary_line_factor'):.2f} y{calculator._cfg('risk_field_oncoming_line_factor'):.2f}",
        f"color vmax: road{road_vmax:.2f} veh{vehicle_vmax:.2f}",
        f"overlay weighted: road{int(bool(overlay_args.road_risk_use_weights))} veh{int(bool(overlay_args.vehicle_risk_use_weights))}",
    ]
    panel_h = 18 + len(lines) * 13
    x0 = 10
    y0 = max(base.size[1] - panel_h - 10, 10)
    return _draw_panel(base, (x0, y0), "env risk config", lines, width=390)


def render_topdown_frame(env: Any, render: bool, overlay_args: argparse.Namespace) -> np.ndarray:
    """统一封装原始 topdown BEV 渲染，供 eval 和单图快照共用。"""
    base = env.render(
        mode="topdown",
        window=render,
        screen_size=(overlay_args.width, overlay_args.height),
        film_size=(overlay_args.film_size, overlay_args.film_size),
        scaling=overlay_args.scaling,
        target_agent_heading_up=True,
        screen_record=False,
        num_stack=1,
        history_smooth=0,
        draw_center_line=False,
        draw_contour=True,
    )
    return _as_rgb(base)


def render_risk_only_frame(
    env: Any,
    render: bool,
    calculator: RiskFieldCalculator,
    overlay_args: argparse.Namespace,
    step_info: dict | None = None,
    episode_cost: float = 0.0,
    episode_length: int = 0,
    show_badge: bool = True,
) -> np.ndarray:
    """只渲染风险场，不混入底图 BEV，用于纯风险视图或左右对照左图。"""
    agents = getattr(env, "agents", {})
    if not agents:
        return np.zeros((overlay_args.height, overlay_args.width, 3), dtype=np.uint8)

    ego = next(iter(agents.values()))
    blank = np.zeros((overlay_args.height, overlay_args.width, 3), dtype=np.uint8)
    blank[:, :] = np.array([12, 16, 24], dtype=np.uint8)
    frame = _overlay_risk(blank, env, ego, calculator, overlay_args)
    base = Image.fromarray(frame).convert("RGBA")
    if show_badge:
        base = _draw_view_badge(base, "risk field only")
    if not overlay_args.no_risk_panel:
        base = draw_eval_step_cost_panel(base, step_info, episode_cost, episode_length)
    if not overlay_args.no_config_panel:
        base = draw_eval_config_panel(base, env, calculator, overlay_args)
    return np.asarray(base.convert("RGB"))


def render_risk_side_by_side_frame(
    env: Any,
    render: bool,
    calculator: RiskFieldCalculator,
    overlay_args: argparse.Namespace,
    step_info: dict | None = None,
    episode_cost: float = 0.0,
    episode_length: int = 0,
) -> np.ndarray:
    """左侧风险场、右侧原始 BEV 的并排视图。"""
    left = render_risk_only_frame(
        env,
        render,
        calculator,
        overlay_args,
        step_info=step_info,
        episode_cost=episode_cost,
        episode_length=episode_length,
    )
    right = render_topdown_frame(env, render, overlay_args)
    right_image = _draw_view_badge(Image.fromarray(right).convert("RGBA"), "raw bev")
    right = np.asarray(right_image.convert("RGB"))

    gap = 8
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + gap + right.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = np.array([8, 10, 16], dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap : left.shape[1] + gap + right.shape[1]] = right
    return canvas


def render_risk_overlay_frame(
    env: Any,
    render: bool,
    calculator: RiskFieldCalculator,
    overlay_args: argparse.Namespace,
    step_info: dict | None = None,
    episode_cost: float = 0.0,
    episode_length: int = 0,
    show_badge: bool = True,
) -> np.ndarray:
    """在原始 topdown BEV 上叠加当前风险场。"""
    frame = render_topdown_frame(env, render, overlay_args)
    agents = getattr(env, "agents", {})
    if not agents:
        return frame
    ego = next(iter(agents.values()))
    frame = _overlay_risk(frame, env, ego, calculator, overlay_args)
    base = Image.fromarray(frame).convert("RGBA")
    if show_badge:
        base = _draw_view_badge(base, "risk overlay")
    if not overlay_args.no_risk_panel:
        base = draw_eval_step_cost_panel(base, step_info, episode_cost, episode_length)
    if not overlay_args.no_config_panel:
        base = draw_eval_config_panel(base, env, calculator, overlay_args)
    return np.asarray(base.convert("RGB"))


def _overlay_risk(frame: np.ndarray, env: Any, ego: Any, calculator: RiskFieldCalculator, args: argparse.Namespace) -> np.ndarray:
    """在topdown帧上叠加风险场
    
    Args:
        frame: 基础topdown图像 (RGB)
        env: MetaDrive环境
        ego: 主车对象
        calculator: 风险场计算器
        args: 命令行参数
        
    Returns:
        叠加风险场后的RGB图像
    """
    base = Image.fromarray(frame).convert("RGBA")
    width, height = base.size
    
    # 获取缩放比例和坐标变换函数
    scaling = _topdown_scaling(env, args.scaling)
    transform = _make_ego_to_pixel_transform(ego, width, height, scaling)

    road_enabled = not args.vehicle_only and not args.no_road_edge_risk
    vehicle_enabled = not args.no_vehicle_risk

    # ========== 绘制道路/车道边缘风险（参数驱动热力带） ==========
    if road_enabled:
        base = _draw_road_risk(base, env, calculator, transform, scaling, args)
    
    # ========== 绘制静态障碍物风险（彩色热力图） ==========
    if vehicle_enabled:
        base = _draw_object_risk(base, env, ego, calculator, transform, scaling, args)

    # ========== 绘制车辆风险（彩色热力图） ==========
    if vehicle_enabled:
        base = _draw_vehicle_risk(base, env, ego, calculator, transform, scaling, args)

    if getattr(args, "show_static_object_shapes", False):
        base = _draw_static_object_footprints(base, env, ego, calculator, transform, scaling, args)

    if getattr(args, "show_surrounding_vehicle_shapes", False):
        base = _draw_surrounding_vehicle_footprints(base, env, ego, calculator, transform, scaling, args)

    if getattr(args, "show_ego_shape", False):
        base = _draw_ego_footprint(base, ego, transform, scaling)

    if not args.no_colorbar:
        base = _draw_colorbar(base, calculator, args, road_enabled, vehicle_enabled)
    if not args.no_risk_panel:
        base = _draw_risk_panel(base, env, ego, calculator, args)

    return np.asarray(base.convert("RGB"))


def _topdown_scaling(env: Any, fallback: float) -> float:
    """获取topdown渲染器的实际缩放比例
    
    Args:
        env: MetaDrive环境
        fallback: 备用缩放值
        
    Returns:
        缩放比例（像素/米）
    """
    renderer = getattr(env, "top_down_renderer", None)
    scaling = RiskFieldCalculator._safe_float(getattr(renderer, "scaling", fallback))
    if math.isfinite(scaling) and scaling > RiskFieldCalculator.EPS:
        return scaling
    return max(float(fallback), RiskFieldCalculator.EPS)


def _make_ego_to_pixel_transform(ego: Any, width: int, height: int, scaling: float):
    """创建从世界坐标到像素坐标的变换函数
    
    坐标系说明：
    - 世界坐标：以主车为中心，forward为y轴正方向，left为x轴正方向
    - 像素坐标：图像左上角为原点，向右为x正，向下为y正
    
    Args:
        ego: 主车对象
        width: 图像宽度（像素）
        height: 图像高度（像素）
        scaling: 缩放比例（像素/米）
        
    Returns:
        变换函数 transform(point) -> (pixel_x, pixel_y)
    """
    ego_pos = RiskFieldCalculator._xy(getattr(ego, "position", (0.0, 0.0)))
    forward = _unit_heading(ego)  # 主车前向单位向量
    left = np.array([-forward[1], forward[0]], dtype=float)  # 主车左侧单位向量

    def transform(point: Any) -> Tuple[float, float]:
        """将世界坐标点转换为像素坐标
        
        Args:
            point: 世界坐标 [x, y]
            
        Returns:
            (pixel_x, pixel_y) 像素坐标
        """
        delta = RiskFieldCalculator._xy(point) - ego_pos
        # 投影到主车局部坐标系
        forward_m = float(np.dot(delta, forward))   # 纵向距离（米）
        lateral_left_m = float(np.dot(delta, left))  # 横向距离（米，向左为正）
        
        # 转换到像素坐标（主车在图像中心）
        pixel_x = width / 2.0 - lateral_left_m * scaling
        pixel_y = height / 2.0 - forward_m * scaling
        return pixel_x, pixel_y

    return transform


def _draw_road_risk(
    base: Image.Image,
    env: Any,
    calculator: RiskFieldCalculator,
    transform: Any,
    scaling: float,
    args: argparse.Namespace,
) -> Image.Image:
    """绘制道路/车道线高斯风险带
    
    这里使用risk_field.py中的sigma参数，把车道线和道路边界画成有宽度的风险带。
    多条线重叠时按最大风险合成，避免重叠区域被假性叠红。
    """
    width, height = base.size
    road_vmax = _resolve_road_risk_vmax(args, calculator)
    risk_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(risk_image)
    commands = []

    for lane in _iter_lanes(env):
        for side in (0, 1):
            profile = calculator.lane_line_risk_profile(lane, side)
            kind = profile["kind"]
            if kind == "none" or not _road_component_enabled(kind, args.road_risk_component):
                continue

            edge_points = _sample_lane_edge(lane, side, step=args.road_risk_sample_step)
            if edge_points.size == 0:
                continue
            pixels = [transform(point) for point in edge_points if np.all(np.isfinite(point))]
            pixels = [(int(round(x)), int(round(y))) for x, y in pixels]
            if not _has_visible_pixel(pixels, width, height, margin=20):
                continue

            sigma = max(float(profile["sigma"]), RiskFieldCalculator.EPS)
            weight = max(float(profile["factor"]), 0.0)
            if args.road_risk_use_weights:
                weight *= _road_global_weight(calculator, kind)
            if weight <= 0.0:
                continue
            weighted_profile = dict(profile)
            weighted_profile["factor"] = weight
            max_distance = _lane_line_visible_distance(
                calculator,
                weighted_profile,
                args.road_risk_min_visible,
                fallback_sigma=sigma,
            )
            if max_distance <= 0.0:
                continue

            for distance in np.linspace(max_distance, 0.0, max(int(args.road_risk_layers), 2)):
                risk_value = float(calculator.lane_line_risk_components(float(distance), weighted_profile)["total"])
                normalized = min(max(risk_value / road_vmax, 0.0), 1.0)
                if normalized <= 0.0:
                    continue
                line_width = max(int(args.edge_width), int(math.ceil(2.0 * distance * scaling)))
                commands.append((normalized, line_width, pixels))

    for normalized, line_width, pixels in sorted(commands, key=lambda item: item[0]):
        draw.line(pixels, fill=int(round(normalized * 255)), width=max(line_width, 1))

    risk_map = np.asarray(risk_image, dtype=np.float32) / 255.0 * road_vmax
    return _blend_risk_map_values(
        base,
        risk_map,
        min_visible=args.road_risk_min_visible,
        vmax=road_vmax,
        max_alpha=args.road_alpha,
        min_alpha=getattr(args, "road_min_alpha", 0),
    )


def _draw_ego_footprint(base: Image.Image, ego: Any, transform: Any, scaling: float) -> Image.Image:
    """Draw the ego vehicle footprint over the risk field."""
    return _draw_vehicle_footprint(
        base,
        ego,
        transform,
        scaling,
        fill=(16, 185, 129, 170),
        outline=(255, 255, 255, 245),
        heading=(255, 255, 255, 245),
    )


def _draw_surrounding_vehicle_footprints(
    base: Image.Image,
    env: Any,
    ego: Any,
    calculator: RiskFieldCalculator,
    transform: Any,
    scaling: float,
    args: argparse.Namespace,
) -> Image.Image:
    """Draw original surrounding-vehicle footprints as an observation aid."""
    ego_pos = RiskFieldCalculator._xy(getattr(ego, "position", (0.0, 0.0)))
    output = base
    for other in calculator._iter_surrounding_vehicles(env, ego):
        other_pos = RiskFieldCalculator._xy(getattr(other, "position", (0.0, 0.0)))
        if float(np.linalg.norm(other_pos - ego_pos)) > args.risk_distance:
            continue

        cx, cy = transform(other_pos)
        margin = 90
        if not (-margin <= cx <= base.size[0] + margin and -margin <= cy <= base.size[1] + margin):
            continue

        color = _vehicle_display_rgb(other, default=(251, 146, 60))
        output = _draw_vehicle_footprint(
            output,
            other,
            transform,
            scaling,
            fill=(*color, 72),
            outline=(255, 255, 255, 235),
            heading=(*color, 255),
        )
    return output


def _draw_static_object_footprints(
    base: Image.Image,
    env: Any,
    ego: Any,
    calculator: RiskFieldCalculator,
    transform: Any,
    scaling: float,
    args: argparse.Namespace,
) -> Image.Image:
    """Draw static traffic-object footprints as an observation aid."""
    ego_pos = RiskFieldCalculator._xy(getattr(ego, "position", (0.0, 0.0)))
    output = base
    for obj in calculator._iter_static_objects(env):
        obj_pos = RiskFieldCalculator._xy(getattr(obj, "position", (0.0, 0.0)))
        if float(np.linalg.norm(obj_pos - ego_pos)) > args.risk_distance:
            continue

        cx, cy = transform(obj_pos)
        margin = 90
        if not (-margin <= cx <= base.size[0] + margin and -margin <= cy <= base.size[1] + margin):
            continue

        color = _vehicle_display_rgb(obj, default=(244, 63, 94))
        output = _draw_vehicle_footprint(
            output,
            obj,
            transform,
            scaling,
            fill=(*color, 96),
            outline=(255, 255, 255, 235),
            heading=(*color, 255),
            length_fallback=0.9,
            width_fallback=0.9,
            heading_marker=False,
        )
    return output


def _draw_vehicle_footprint(
    base: Image.Image,
    vehicle: Any,
    transform: Any,
    scaling: float,
    *,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    heading: tuple[int, int, int, int],
    length_fallback: float = 4.5,
    width_fallback: float = 2.0,
    heading_marker: bool = True,
) -> Image.Image:
    """Draw an oriented vehicle rectangle and a short heading marker."""
    vehicle_pos = RiskFieldCalculator._xy(getattr(vehicle, "position", (0.0, 0.0)))
    forward = _unit_heading(vehicle)
    left = np.array([-forward[1], forward[0]], dtype=float)
    length = _dimension_value(vehicle, "top_down_length", length_fallback)
    width = _dimension_value(vehicle, "top_down_width", width_fallback)

    corners = [
        vehicle_pos + forward * length / 2.0 + left * width / 2.0,
        vehicle_pos + forward * length / 2.0 - left * width / 2.0,
        vehicle_pos - forward * length / 2.0 - left * width / 2.0,
        vehicle_pos - forward * length / 2.0 + left * width / 2.0,
    ]
    pixels = [transform(point) for point in corners]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in pixels):
        return base

    center = transform(vehicle_pos)
    nose = transform(vehicle_pos + forward * length * 0.42)
    if not all(math.isfinite(v) for v in (*center, *nose)):
        return base

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    outline_width = max(1, int(round(0.18 * max(float(scaling), 1.0))))
    heading_width = max(1, int(round(0.12 * max(float(scaling), 1.0))))
    draw.polygon(pixels, fill=fill, outline=outline)
    if heading_marker:
        draw.line([center, nose], fill=heading, width=heading_width)
    draw.line(pixels + [pixels[0]], fill=outline, width=outline_width)
    if heading_marker:
        marker_radius = max(1, int(round(0.18 * max(float(scaling), 1.0))))
        draw.ellipse(
            (
                nose[0] - marker_radius,
                nose[1] - marker_radius,
                nose[0] + marker_radius,
                nose[1] + marker_radius,
            ),
            fill=heading,
        )
    return Image.alpha_composite(base, overlay)


def _road_component_enabled(kind: str, component: str) -> bool:
    if component == "total":
        return True
    if component == "lane":
        return kind in {"broken", "solid", "oncoming"}
    return component == kind


def _road_global_weight(calculator: RiskFieldCalculator, kind: str) -> float:
    key = "risk_field_boundary_weight" if kind == "boundary" else "risk_field_lane_weight"
    return max(calculator._cfg(key), 0.0)


def _lane_line_visible_distance(
    calculator: RiskFieldCalculator,
    profile: dict[str, Any],
    min_visible: float,
    *,
    fallback_sigma: float,
) -> float:
    """数值估计车道线单高斯风险带的可见半径。"""
    if max(float(profile.get("factor", 0.0)), 0.0) <= max(float(min_visible), 0.0):
        return 0.0

    sigma = max(fallback_sigma, RiskFieldCalculator.EPS)
    max_distance = sigma * 5.0
    samples = np.linspace(0.0, max_distance, 160)
    risk_values = np.asarray(calculator.lane_line_risk_components(samples, profile)["total"], dtype=float)
    visible = np.flatnonzero(risk_values >= max(float(min_visible), 0.0))
    if visible.size == 0:
        return 0.0
    return float(samples[int(visible[-1])])


def _resolve_road_risk_vmax(args: argparse.Namespace, calculator: RiskFieldCalculator) -> float:
    requested = RiskFieldCalculator._safe_float(args.road_risk_vmax)
    if math.isfinite(requested) and requested > 0.0:
        return requested

    component_to_kinds = {
        "broken": ("broken",),
        "solid": ("solid",),
        "boundary": ("boundary",),
        "oncoming": ("oncoming",),
        "lane": ("broken", "solid", "oncoming"),
        "total": ("broken", "solid", "boundary", "oncoming"),
    }
    factor_keys = {
        "broken": "risk_field_broken_line_factor",
        "solid": "risk_field_solid_line_factor",
        "boundary": "risk_field_boundary_line_factor",
        "oncoming": "risk_field_oncoming_line_factor",
    }
    vmax = 0.0
    for kind in component_to_kinds.get(args.road_risk_component, component_to_kinds["total"]):
        value = max(calculator._cfg(factor_keys[kind]), 0.0)
        if args.road_risk_use_weights:
            value *= _road_global_weight(calculator, kind)
        vmax = max(vmax, value)
    return max(vmax, RiskFieldCalculator.EPS)


def _draw_vehicle_risk(
    base: Image.Image,
    env: Any,
    ego: Any,
    calculator: RiskFieldCalculator,
    transform: Any,
    scaling: float,
    args: argparse.Namespace,
) -> Image.Image:
    """绘制车辆风险热力图
    
    核心流程：
    1. 遍历周围车辆
    2. 对每辆车计算风险场栅格
    3. 取所有车辆风险的最大值（max pooling）
    4. 混合到基础图像
    
    Args:
        base: 基础图像（RGBA）
        env: MetaDrive环境
        ego: 主车对象
        calculator: 风险场计算器
        transform: 坐标变换函数
        scaling: 缩放比例
        args: 命令行参数
        
    Returns:
        叠加车辆风险后的图像
    """
    width, height = base.size
    ego_pos = RiskFieldCalculator._xy(getattr(ego, "position", (0.0, 0.0)))
    
    # 获取主车所在车道和参考航向
    ego_lane = calculator._current_lane(env, ego)
    ref_heading = calculator._reference_heading(ego, ego_lane)
    ego_forward_speed = calculator._forward_speed(ego, ref_heading)
    
    # 初始化风险栅格（存储每个像素的最大风险值）
    risk_map = np.zeros((height, width), dtype=np.float32)

    # ========== 遍历周围车辆 ==========
    for other in calculator._iter_surrounding_vehicles(env, ego):
        other_pos = RiskFieldCalculator._xy(getattr(other, "position", (0.0, 0.0)))
        
        # 距离过滤
        if float(np.linalg.norm(other_pos - ego_pos)) > args.risk_distance:
            continue
        
        # 转换到像素坐标
        cx, cy = transform(other_pos)
        if not (-80 <= cx <= width + 80 and -80 <= cy <= height + 80):
            continue
        
        # 累积车辆风险到栅格
        _accumulate_vehicle_risk(
            risk_map,
            ego,
            other,
            cx,
            cy,
            calculator,
            scaling,
            max(args.vehicle_risk_min_visible, 1e-6),
            args.vehicle_risk_component,  # static/dynamic/total
            ego_forward_speed,
            calculator._forward_speed(other, ref_heading),
            calculator._dimension(other, "top_down_length", 4.5),
        )
    
    if getattr(args, "vehicle_risk_use_weights", False):
        risk_map *= max(calculator._cfg("risk_field_vehicle_weight"), 0.0)

    # 混合风险栅格到图像
    return _blend_risk_map(base, risk_map, args, calculator)


def _draw_object_risk(
    base: Image.Image,
    env: Any,
    ego: Any,
    calculator: RiskFieldCalculator,
    transform: Any,
    scaling: float,
    args: argparse.Namespace,
) -> Image.Image:
    """绘制静态障碍物风险热力图。

    设计上把静态障碍物视为和车辆类似的“参与体风险”，因此复用同一套色带、
    透明度和最小可见阈值，避免新引入另一套视觉语义。
    """
    width, height = base.size
    ego_pos = RiskFieldCalculator._xy(getattr(ego, "position", (0.0, 0.0)))
    ego_lane = calculator._current_lane(env, ego)
    ref_heading = calculator._reference_heading(ego, ego_lane)

    risk_map = np.zeros((height, width), dtype=np.float32)

    for obj in calculator._iter_static_objects(env):
        obj_pos = RiskFieldCalculator._xy(getattr(obj, "position", (0.0, 0.0)))
        if float(np.linalg.norm(obj_pos - ego_pos)) > args.risk_distance:
            continue

        cx, cy = transform(obj_pos)
        if not (-80 <= cx <= width + 80 and -80 <= cy <= height + 80):
            continue

        _accumulate_object_risk(
            risk_map,
            obj,
            cx,
            cy,
            calculator,
            scaling,
            max(args.vehicle_risk_min_visible, 1e-6),
            ref_heading,
        )

    if getattr(args, "vehicle_risk_use_weights", False):
        risk_map *= max(calculator._cfg("risk_field_object_weight"), 0.0)

    return _blend_risk_map_values(
        base,
        risk_map,
        min_visible=args.vehicle_risk_min_visible,
        vmax=_resolve_object_risk_vmax(args, calculator),
        max_alpha=args.vehicle_alpha,
    )


def _accumulate_vehicle_risk(
    risk_map: np.ndarray,
    ego: Any,
    other: Any,
    cx: float,
    cy: float,
    calculator: RiskFieldCalculator,
    scaling: float,
    min_visible: float,
    component: str,
    ego_forward_speed: float,
    other_forward_speed: float,
    other_length: float,
) -> None:
    """累积单个车辆的风险场到栅格
    
    核心算法：
    1. 根据风险组件类型计算渲染半径
    2. 在局部窗口内计算每个像素的风险值
    3. 使用max pooling更新风险栅格
    
    Args:
        risk_map: 风险栅格 (H, W)，会被原地修改
        ego: 主车对象
        other: 其他车辆对象
        cx: 车辆中心像素x坐标
        cy: 车辆中心像素y坐标
        calculator: 风险场计算器
        scaling: 缩放比例
        min_visible: 最小可见风险阈值
        component: 风险组件类型 ("static"/"dynamic"/"total")
        ego_forward_speed: 主车前向速度
        other_forward_speed: 其他车辆前向速度
        other_length: 其他车辆长度
    """
    min_visible = min(max(float(min_visible), 1e-6), 0.999)
    height, width = risk_map.shape
    
    # ========== 获取风险场参数 ==========
    sigma_long = max(calculator._cfg("risk_field_vehicle_longitudinal_sigma"), RiskFieldCalculator.EPS)
    sigma_lat = max(calculator._cfg("risk_field_vehicle_lateral_sigma"), RiskFieldCalculator.EPS)
    beta = max(calculator._cfg("risk_field_vehicle_beta"), RiskFieldCalculator.EPS)
    
    # 静态风险半径因子
    static_radius_factor = max((-math.log(min_visible)) ** (1.0 / (2.0 * beta)), 1.0)
    
    # 动态风险sigma（与速度差成正比）
    speed_delta = abs(float(other_forward_speed) - float(ego_forward_speed))
    dynamic_sigma = max(
        calculator._cfg("risk_field_vehicle_dynamic_sigma_scale") * speed_delta,
        calculator._cfg("risk_field_vehicle_min_dynamic_sigma"),
    )
    dynamic_radius_factor = max(math.sqrt(-math.log(min_visible)), 1.0)
    
    # ========== 计算渲染半径 ==========
    radius_m = 0.0
    if component in ("static", "total"):
        radius_m = max(radius_m, sigma_long * static_radius_factor, sigma_lat * static_radius_factor)
    if component in ("dynamic", "total"):
        radius_m = max(radius_m, dynamic_sigma * dynamic_radius_factor, sigma_lat * dynamic_radius_factor)
    
    radius_px = int(math.ceil(radius_m * scaling)) + 3
    
    # ========== 确定渲染窗口 ==========
    x_min = max(int(math.floor(cx - radius_px)), 0)
    x_max = min(int(math.ceil(cx + radius_px)) + 1, width)
    y_min = max(int(math.floor(cy - radius_px)), 0)
    y_max = min(int(math.ceil(cy + radius_px)) + 1, height)
    if x_min >= x_max or y_min >= y_max:
        return

    # ========== 坐标变换：像素 → 障碍物局部坐标系 ==========
    ego_forward = _unit_heading(ego)
    ego_left = np.array([-ego_forward[1], ego_forward[0]], dtype=float)
    other_forward = _unit_heading(other)
    other_left = np.array([-other_forward[1], other_forward[0]], dtype=float)
    
    # 将障碍物朝向向量投影到主车局部坐标系
    other_forward_in_ego = np.array(
        [np.dot(other_forward, ego_forward), np.dot(other_forward, ego_left)],
        dtype=float,
    )
    other_left_in_ego = np.array(
        [np.dot(other_left, ego_forward), np.dot(other_left, ego_left)],
        dtype=float,
    )

    # 生成窗口内的像素网格
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    
    # 像素坐标 → 主车局部坐标系（米）
    delta_forward = -(yy.astype(float) - cy) / scaling  # 纵向（前向为正）
    delta_left = -(xx.astype(float) - cx) / scaling      # 横向（向左为正）
    
    # 主车局部坐标 → 障碍物局部坐标
    longitudinal = delta_forward * other_forward_in_ego[0] + delta_left * other_forward_in_ego[1]
    lateral = delta_forward * other_left_in_ego[0] + delta_left * other_left_in_ego[1]
    
    # ========== 调用风险场计算器获取风险值 ==========
    components = calculator.vehicle_potential_components(
        longitudinal,
        lateral,
        ego_forward_speed,
        other_forward_speed,
        other_length,
    )
    
    # 选择指定的风险组件
    vehicle_risk = np.asarray(components[component], dtype=np.float32)
    
    # Max pooling：保留最大风险值
    risk_map[y_min:y_max, x_min:x_max] = np.maximum(risk_map[y_min:y_max, x_min:x_max], vehicle_risk)


def _accumulate_object_risk(
    risk_map: np.ndarray,
    obj: Any,
    cx: float,
    cy: float,
    calculator: RiskFieldCalculator,
    scaling: float,
    min_visible: float,
    fallback_heading: np.ndarray,
) -> None:
    """累积单个静态障碍物的风险场到栅格。"""
    min_visible = min(max(float(min_visible), 1e-6), 0.999)
    height, width = risk_map.shape

    sigma_long = max(calculator._cfg("risk_field_object_longitudinal_sigma"), RiskFieldCalculator.EPS)
    sigma_lat = max(calculator._cfg("risk_field_object_lateral_sigma"), RiskFieldCalculator.EPS)
    beta = max(calculator._cfg("risk_field_object_beta"), RiskFieldCalculator.EPS)
    radius_factor = max((-math.log(min_visible)) ** (1.0 / (2.0 * beta)), 1.0)
    radius_m = max(sigma_long * radius_factor, sigma_lat * radius_factor)
    radius_px = int(math.ceil(radius_m * scaling)) + 3

    x_min = max(int(math.floor(cx - radius_px)), 0)
    x_max = min(int(math.ceil(cx + radius_px)) + 1, width)
    y_min = max(int(math.floor(cy - radius_px)), 0)
    y_max = min(int(math.ceil(cy + radius_px)) + 1, height)
    if x_min >= x_max or y_min >= y_max:
        return

    obj_forward = calculator._object_heading(obj, fallback_heading)
    obj_left = np.array([-obj_forward[1], obj_forward[0]], dtype=float)
    ego_forward = fallback_heading / max(np.linalg.norm(fallback_heading), RiskFieldCalculator.EPS)
    ego_left = np.array([-ego_forward[1], ego_forward[0]], dtype=float)

    obj_forward_in_ego = np.array(
        [np.dot(obj_forward, ego_forward), np.dot(obj_forward, ego_left)],
        dtype=float,
    )
    obj_left_in_ego = np.array(
        [np.dot(obj_left, ego_forward), np.dot(obj_left, ego_left)],
        dtype=float,
    )

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    delta_forward = -(yy.astype(float) - cy) / scaling
    delta_left = -(xx.astype(float) - cx) / scaling

    longitudinal = delta_forward * obj_forward_in_ego[0] + delta_left * obj_forward_in_ego[1]
    lateral = delta_forward * obj_left_in_ego[0] + delta_left * obj_left_in_ego[1]

    object_risk = np.exp(
        np.maximum(
            -(
                ((longitudinal ** 2) / (sigma_long ** 2)) ** beta
                + ((lateral ** 2) / (sigma_lat ** 2)) ** beta
            ),
            -80.0,
        )
    ).astype(np.float32)
    risk_map[y_min:y_max, x_min:x_max] = np.maximum(risk_map[y_min:y_max, x_min:x_max], object_risk)


def _blend_risk_map(
    base: Image.Image,
    risk_map: np.ndarray,
    args: argparse.Namespace,
    calculator: RiskFieldCalculator = None,
) -> Image.Image:
    """将风险栅格混合到基础图像
    
    混合策略：
    1. 归一化风险值到 [0, 1]
    2. 使用颜色映射表转换为RGB
    3. 根据风险值计算透明度（非线性增强）
    4. Alpha合成
    
    Args:
        base: 基础图像（RGBA）
        risk_map: 风险栅格 (H, W)
        args: 命令行参数
        
    Returns:
        混合后的图像（RGBA）
    """
    return _blend_risk_map_values(
        base,
        risk_map,
        min_visible=args.vehicle_risk_min_visible,
        vmax=_resolve_vehicle_risk_vmax(args, calculator),
        max_alpha=args.vehicle_alpha,
    )


def _blend_risk_map_values(
    base: Image.Image,
    risk_map: np.ndarray,
    *,
    min_visible: float,
    vmax: float,
    max_alpha: int,
    min_alpha: int = 0,
) -> Image.Image:
    """将任意风险栅格按统一色带混合到基础图像。"""
    min_visible = max(float(min_visible), 0.0)
    visible = risk_map > min_visible
    if not np.any(visible):
        return base

    vmax = max(float(vmax), RiskFieldCalculator.EPS)
    normalized = np.clip(risk_map / vmax, 0.0, 1.0)
    
    # 风险值 → RGB颜色
    rgb = _risk_to_rgb(normalized)
    
    # 计算透明度（非线性增强：alpha = max_alpha * risk^0.75）
    alpha = np.zeros_like(risk_map, dtype=np.uint8)
    max_alpha = max(0, min(int(max_alpha), 255))
    min_alpha = max(0, min(int(min_alpha), max_alpha))
    visible_alpha = np.clip(max_alpha * (normalized[visible] ** 0.75), 0, 255).astype(np.uint8)
    if min_alpha > 0:
        visible_alpha = np.maximum(visible_alpha, min_alpha).astype(np.uint8)
    alpha[visible] = visible_alpha

    # 创建RGBA叠加层
    overlay = np.zeros((risk_map.shape[0], risk_map.shape[1], 4), dtype=np.uint8)
    overlay[:, :, :3] = rgb
    overlay[:, :, 3] = alpha
    
    # Alpha合成
    return Image.alpha_composite(base, Image.fromarray(overlay, "RGBA"))


def _resolve_vehicle_risk_vmax(args: argparse.Namespace, calculator: RiskFieldCalculator = None) -> float:
    """解析车辆风险vmax参数
    
    Args:
        args: 命令行参数
        
    Returns:
        vmax值
    """
    requested = RiskFieldCalculator._safe_float(args.vehicle_risk_vmax)
    if math.isfinite(requested) and requested > 0.0:
        return requested
    # 默认值：total=2.0（包含动态部分），static/dynamic=1.0
    vmax = 2.0 if args.vehicle_risk_component == "total" else 1.0
    if calculator is not None and getattr(args, "vehicle_risk_use_weights", False):
        vmax *= max(calculator._cfg("risk_field_vehicle_weight"), 0.0)
    return max(vmax, RiskFieldCalculator.EPS)


def _resolve_object_risk_vmax(args: argparse.Namespace, calculator: RiskFieldCalculator = None) -> float:
    """解析静态障碍物风险的色条上限。"""
    vmax = 1.0
    if calculator is not None and getattr(args, "vehicle_risk_use_weights", False):
        vmax *= max(calculator._cfg("risk_field_object_weight"), 0.0)
    return max(vmax, RiskFieldCalculator.EPS)


def _risk_palette() -> np.ndarray:
    """获取风险颜色映射表
    
    颜色渐变：蓝色（低风险）→ 青色 → 绿色 → 黄色 → 橙色 → 红色（高风险）
    
    Returns:
        颜色表 (6, 3) RGB值
    """
    palette = [
        (31, 64, 217),    # 蓝色
        (0, 199, 242),    # 青色
        (77, 230, 71),    # 绿色
        (255, 224, 31),   # 黄色
        (255, 115, 25),   # 橙色
        (204, 0, 0),      # 红色
    ]
    return np.asarray(palette, dtype=float)


def _risk_to_rgb(risk: np.ndarray) -> np.ndarray:
    """将归一化风险值映射到RGB颜色
    
    使用线性插值在颜色表之间平滑过渡。
    
    Args:
        risk: 归一化风险值 (H, W) [0, 1]
        
    Returns:
        RGB图像 (H, W, 3) uint8
    """
    palette = _risk_palette()
    scaled = np.clip(risk, 0.0, 1.0) * (len(palette) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.clip(lower + 1, 0, len(palette) - 1)
    frac = scaled - lower
    
    # 线性插值
    rgb = palette[lower] * (1.0 - frac[..., None]) + palette[upper] * frac[..., None]
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def _draw_colorbar(
    base: Image.Image,
    calculator: RiskFieldCalculator,
    args: argparse.Namespace,
    road_enabled: bool,
    vehicle_enabled: bool,
) -> Image.Image:
    """在右上角绘制当前叠加层的风险色条。"""
    if not (road_enabled or vehicle_enabled):
        return base

    width, _ = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    x0 = max(width - 238, 8)
    y0 = 10
    panel_w = 220
    panel_h = 42
    draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=7, fill=(255, 255, 255, 220), outline=(15, 23, 42, 150))

    if road_enabled and vehicle_enabled:
        vmax = max(
            _resolve_road_risk_vmax(args, calculator),
            _resolve_vehicle_risk_vmax(args, calculator),
            _resolve_object_risk_vmax(args, calculator),
        )
        label = "risk color"
    elif road_enabled:
        vmax = _resolve_road_risk_vmax(args, calculator)
        label = f"road:{args.road_risk_component}"
    else:
        vmax = max(_resolve_vehicle_risk_vmax(args, calculator), _resolve_object_risk_vmax(args, calculator))
        label = f"actor:{args.vehicle_risk_component}+object"

    bar_w = 128
    bar_h = 10
    bar_x = x0 + 80
    bar_y = y0 + 8
    grad = np.linspace(0.0, 1.0, bar_w, dtype=float)[None, :]
    colors = _risk_to_rgb(np.repeat(grad, bar_h, axis=0))
    alpha = np.full((bar_h, bar_w, 1), 230, dtype=np.uint8)
    bar = Image.fromarray(np.concatenate([colors, alpha], axis=2), "RGBA")
    overlay.alpha_composite(bar, (bar_x, bar_y))
    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline=(15, 23, 42, 180))
    draw.text((x0 + 8, y0 + 6), label, fill=(15, 23, 42, 255))
    draw.text((bar_x, bar_y + 14), "0", fill=(15, 23, 42, 255))
    draw.text((bar_x + bar_w - 36, bar_y + 14), f"{vmax:.1f}", fill=(15, 23, 42, 255))
    return Image.alpha_composite(base, overlay)


def _draw_risk_panel(
    base: Image.Image,
    env: Any,
    ego: Any,
    calculator: RiskFieldCalculator,
    args: argparse.Namespace,
) -> Image.Image:
    """在右上角绘制当前ego位置的风险分解数值。"""
    risk_cost, info = calculator.calculate(env, ego)
    width, _ = base.size
    panel_w = 268
    x0 = max(width - panel_w - 18, 8)
    y0 = 58
    line_h = 13
    lines = [
        f"risk: {risk_cost:.3f}",
        f"road: {info.get('risk_field_road_cost', 0.0):.3f}",
        f"boundary: {info.get('risk_field_boundary_cost', 0.0):.3f}",
        f"lane: {info.get('risk_field_lane_cost', 0.0):.3f}",
        f"offroad: {info.get('risk_field_offroad_cost', 0.0):.3f}",
        f"vehicle: {info.get('risk_field_vehicle_cost', 0.0):.3f}",
        f"object: {info.get('risk_field_object_cost', 0.0):.3f}",
        f"headway: {info.get('risk_field_headway_cost', 0.0):.3f}",
        f"ttc: {info.get('risk_field_ttc_cost', 0.0):.3f}",
        f"n_vehicle: {int(info.get('risk_field_surrounding_vehicle_count', 0))}",
        f"n_object: {int(info.get('risk_field_surrounding_object_count', 0))}",
        f"on_road: {int(bool(info.get('risk_field_on_road', True)))}",
        f"broken_sigma: {calculator._cfg('risk_field_broken_line_sigma'):.2f}",
        f"lane_sigma: {calculator._cfg('risk_field_lane_edge_sigma'):.2f}",
        f"boundary_sigma: {calculator._cfg('risk_field_boundary_sigma'):.2f}",
        f"side0(-): {info.get('risk_field_side0_line_kind', 'none')} x{info.get('risk_field_side0_line_factor', 0.0):.2f}",
        f"side1(+): {info.get('risk_field_side1_line_kind', 'none')} x{info.get('risk_field_side1_line_factor', 0.0):.2f}",
        f"broken/solid: {calculator._cfg('risk_field_broken_line_factor'):.2f}/{calculator._cfg('risk_field_solid_line_factor'):.2f}",
        f"boundary/yellow: {calculator._cfg('risk_field_boundary_line_factor'):.2f}/{calculator._cfg('risk_field_oncoming_line_factor'):.2f}",
    ]
    panel_h = 16 + len(lines) * line_h
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=7, fill=(255, 255, 255, 220), outline=(15, 23, 42, 150))
    draw.text((x0 + 8, y0 + 6), "current risk", fill=(15, 23, 42, 255))
    for idx, line in enumerate(lines):
        draw.text((x0 + 8, y0 + 20 + idx * line_h), line, fill=(15, 23, 42, 255))
    return Image.alpha_composite(base, overlay)


def _iter_lanes(env: Any):
    """迭代环境中所有车道
    
    Args:
        env: MetaDrive环境
        
    Yields:
        车道对象
    """
    road_network = getattr(getattr(env, "current_map", None), "road_network", None)
    if road_network is None:
        return []
    try:
        return list(road_network.get_all_lanes())
    except Exception:
        return []


def _sample_lane_edges(lane: Any, step: float):
    """采样车道两侧边缘点，返回顺序与MetaDrive line_types一致。
    
    Args:
        lane: 车道对象
        step: 采样步长（米）
        
    Returns:
        [side0_edge_points, side1_edge_points] 两个点集数组。
        MetaDrive原生约定：side=0对应lateral=-width/2，side=1对应lateral=+width/2。
    """
    length = RiskFieldCalculator._safe_float(getattr(lane, "length", math.nan))
    if not math.isfinite(length) or length <= 0:
        return []
    
    count = max(2, int(math.ceil(length / max(step, 0.1))) + 1)
    longitudinal_values = np.linspace(0.0, length, count)
    
    side0_edge = []
    side1_edge = []
    for longitudinal in longitudinal_values:
        width = _lane_width(lane, longitudinal)
        if not math.isfinite(width):
            continue
        side0_edge.append(_lane_position(lane, longitudinal, -width / 2.0))
        side1_edge.append(_lane_position(lane, longitudinal, width / 2.0))
    
    return [np.asarray(side0_edge), np.asarray(side1_edge)]


def _sample_lane_edge(lane: Any, side: int, step: float) -> np.ndarray:
    """采样单侧车道边缘点，side=0为lateral=-width/2，side=1为lateral=+width/2。"""
    length = RiskFieldCalculator._safe_float(getattr(lane, "length", math.nan))
    if not math.isfinite(length) or length <= 0:
        return np.empty((0, 2))

    count = max(2, int(math.ceil(length / max(step, 0.1))) + 1)
    longitudinal_values = np.linspace(0.0, length, count)
    points = []
    sign = -1.0 if side == 0 else 1.0
    for longitudinal in longitudinal_values:
        width = _lane_width(lane, longitudinal)
        if not math.isfinite(width):
            continue
        points.append(_lane_position(lane, longitudinal, sign * width / 2.0))
    return np.asarray(points)


def _lane_width(lane: Any, longitudinal: float) -> float:
    """获取指定纵向位置的车道宽度
    
    Args:
        lane: 车道对象
        longitudinal: 纵向坐标（米）
        
    Returns:
        车道宽度（米）
    """
    try:
        return RiskFieldCalculator._safe_float(lane.width_at(longitudinal))
    except Exception:
        return RiskFieldCalculator._safe_float(getattr(lane, "width", math.nan))


def _lane_position(lane: Any, longitudinal: float, lateral: float) -> np.ndarray:
    """获取车道上指定Frenet坐标的世界坐标
    
    Args:
        lane: 车道对象
        longitudinal: 纵向坐标（米）
        lateral: 横向偏移（米，向左为正）
        
    Returns:
        世界坐标 [x, y]
    """
    try:
        return RiskFieldCalculator._xy(lane.position(longitudinal, lateral))
    except Exception:
        return np.array([math.nan, math.nan], dtype=float)


def _has_visible_pixel(pixels: Iterable[Tuple[int, int]], width: int, height: int, margin: int) -> bool:
    """检查像素列表中是否有可见像素（在图像范围内）
    
    Args:
        pixels: 像素坐标列表
        width: 图像宽度
        height: 图像高度
        margin: 容差边距（像素）
        
    Returns:
        是否有可见像素
    """
    for x, y in pixels:
        if -margin <= x <= width + margin and -margin <= y <= height + margin:
            return True
    return False


def _unit_heading(vehicle: Any) -> np.ndarray:
    """获取车辆的单位航向向量
    
    Args:
        vehicle: 车辆对象
        
    Returns:
        单位航向向量 [cos(theta), sin(theta)]
    """
    heading = getattr(vehicle, "heading", None)
    if heading is not None:
        heading = RiskFieldCalculator._xy(heading)
        norm = float(np.linalg.norm(heading))
        if norm > RiskFieldCalculator.EPS:
            return heading / norm
    
    theta = RiskFieldCalculator._safe_float(getattr(vehicle, "heading_theta", 0.0))
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


def _dimension_value(obj: Any, attr: str, default: float) -> float:
    value = getattr(obj, attr, default)
    if callable(value):
        value = value()
    value = RiskFieldCalculator._safe_float(value)
    return value if math.isfinite(value) and value > RiskFieldCalculator.EPS else default


def _vehicle_display_rgb(vehicle: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    color = getattr(vehicle, "top_down_color", None)
    if color is None:
        color = getattr(vehicle, "color", None)
    try:
        values = np.asarray(color, dtype=float).reshape(-1)[:3]
    except Exception:
        return default
    if values.size < 3 or not np.all(np.isfinite(values)):
        return default
    if float(np.nanmax(values)) <= 1.0:
        values = values * 255.0
    values = np.clip(values, 0, 255).astype(np.uint8)
    return int(values[0]), int(values[1]), int(values[2])


if __name__ == "__main__":
    main()
