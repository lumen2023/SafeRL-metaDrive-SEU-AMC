"""Generate a one-frame BEV risk-field debug plot.

Example:
    python debug_risk_field_bev.py --output debug/risk_field_bev.png --no-show --dpi 120
"""

import argparse

from env import get_training_env
from risk_field_bev import plot_risk_field_bev


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MetaDrive risk-field BEV debug view.")
    parser.add_argument("--output", default="debug/risk_field_bev.png", help="Path to save the BEV image.")
    parser.add_argument("--component", default="total", help="total/road/boundary/lane/offroad/vehicle/object/headway/ttc.")
    parser.add_argument("--cmap", default="risk_legacy", help="Matplotlib cmap name, risk_legacy, or risk_red.")
    parser.add_argument("--vmax", type=float, default=1.2, help="Fixed colorbar max. Use <=0 for auto per frame.")
    parser.add_argument("--risk-min-visible", type=float, default=1e-4, help="Risk below this value is transparent.")
    parser.add_argument("--resolution", type=float, default=0.5, help="Risk heatmap grid resolution in meters.")
    parser.add_argument("--dpi", type=int, default=120, help="Saved image DPI. Higher is sharper but slower.")
    parser.add_argument("--interpolation", default="bilinear", help="Heatmap interpolation, e.g. bilinear/nearest/bicubic.")
    parser.add_argument("--front-min", type=float, default=-20.0)
    parser.add_argument("--front-max", type=float, default=70.0)
    parser.add_argument("--lateral-min", type=float, default=-20.0)
    parser.add_argument("--lateral-max", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=0, help="Advance the env before plotting.")
    parser.add_argument("--seed", type=int, default=100, help="Training env start_seed.")
    parser.add_argument("--traffic-density", type=float, default=0.2)
    parser.add_argument("--accident-prob", type=float, default=0.0)
    parser.add_argument("--world-frame", action="store_true", help="Plot in world frame instead of ego-heading-up frame.")
    parser.add_argument("--draw-lane-centers", action="store_true", help="Draw lane centerlines for geometry debugging.")
    parser.add_argument("--draw-lane-surfaces", action="store_true", help="Fill lane surface polygons for geometry debugging.")
    parser.add_argument("--hide-lane-surfaces", action="store_true", help="Compatibility flag. Lane surfaces are hidden by default.")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a matplotlib window.")
    return parser.parse_args()


def main():
    args = parse_args()
    env = get_training_env({
        "num_scenarios": 1,
        "start_seed": args.seed,
        "use_render": False,
        "traffic_density": args.traffic_density,
        "accident_prob": args.accident_prob,
    })

    try:
        env.reset()
        for _ in range(max(args.steps, 0)):
            _, _, terminated, truncated, _ = env.step([0, 0])
            if terminated or truncated:
                env.reset()

        vehicle = next(iter(env.agents.values()))
        plot_risk_field_bev(
            env,
            vehicle,
            save_path=args.output,
            component=args.component,
            front_range=(args.front_min, args.front_max),
            lateral_range=(args.lateral_min, args.lateral_max),
            resolution=args.resolution,
            dpi=args.dpi,
            ego_frame=not args.world_frame,
            draw_lane_centers=args.draw_lane_centers,
            draw_lane_surfaces=args.draw_lane_surfaces and not args.hide_lane_surfaces,
            cmap=args.cmap,
            vmax=args.vmax,
            risk_min_visible=args.risk_min_visible,
            interpolation=args.interpolation,
            show=not args.no_show,
        )
        print(f"Saved risk-field BEV to {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
