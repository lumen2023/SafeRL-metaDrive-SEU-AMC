#!/usr/bin/env python3
"""
批量导出多个 MetaDrive seed 的地图与 route 可视化，并自动整理成论文图版。

输出目录结构示例：
    debug/metadrive_route_batches/train_seed100-108_n9_20260505-160000/
      images/
      summaries/
      boards/
      manifests/
      README.md

示例：
    python batch_visualize_metadrive_routes.py --split train --seed-start 100 --num-seeds 9
    python batch_visualize_metadrive_routes.py --split train --seeds 100-105,120
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from visualize_metadrive_route import (
    generate_route_visualization,
    get_default_start_seed,
    parse_map_arg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量导出 MetaDrive route 可视化并生成论文图版。")
    parser.add_argument("--split", choices=("train", "val"), default="train", help="使用训练集还是验证集预设。")
    parser.add_argument(
        "--seeds",
        default=None,
        help="显式指定 seed 列表，支持例如 100,101,105-108。",
    )
    parser.add_argument("--seed-start", type=int, default=None, help="批量起始 seed。")
    parser.add_argument("--num-seeds", type=int, default=9, help="连续导出的 seed 数量。默认 9。")
    parser.add_argument("--map", dest="map_arg", default=None, help="可选覆盖 map 参数，例如 3、5、TXO。")
    parser.add_argument("--batch-name", default=None, help="批次目录名；默认自动生成。")
    parser.add_argument(
        "--root-dir",
        default="debug/metadrive_route_batches",
        help="批量输出根目录。",
    )
    parser.add_argument("--film-size", type=int, default=2400, help="单张底图画布大小（像素）。")
    parser.add_argument("--scaling", type=float, default=None, help="每米对应像素；默认自动缩放。")
    parser.add_argument(
        "--semantic-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用语义色 top-down 底图。",
    )
    parser.add_argument(
        "--draw-center-line",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否绘制所有 lane 中心线。",
    )
    parser.add_argument(
        "--annotate-segments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否在单图中标注 route segment 编号。",
    )
    parser.add_argument("--route-sample-interval", type=float, default=2.0, help="route 采样间隔（米）。")
    parser.add_argument("--turn-threshold-deg", type=float, default=15.0, help="路口转向分类阈值（度）。")
    parser.add_argument(
        "--curvature-threshold-deg",
        type=float,
        default=10.0,
        help="单 segment 判定为曲线路段的阈值（度）。",
    )
    parser.add_argument("--traffic-density", type=float, default=0.0, help="可视化时使用的交通密度。默认 0。")
    parser.add_argument("--accident-prob", type=float, default=0.0, help="可视化时使用的事故概率。默认 0。")
    parser.add_argument("--dpi", type=int, default=200, help="保存单图 PNG 的 DPI。")
    parser.add_argument("--overview-cols", type=int, default=4, help="总览拼图的列数。")
    parser.add_argument("--paper-cols", type=int, default=3, help="论文图版的列数。")
    parser.add_argument("--overview-tile-width", type=int, default=360, help="总览拼图中每张图的缩略宽度。")
    parser.add_argument("--paper-tile-width", type=int, default=460, help="论文图版中每张图的缩略宽度。")
    parser.add_argument("--paper-topk", type=int, default=9, help="论文图版挑选的 top-k 样本数。")
    parser.add_argument(
        "--paper-sort-by",
        choices=("complexity_score", "route_length_m", "curved_segment_count", "segment_count", "seed"),
        default="complexity_score",
        help="论文图版的排序字段。",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="某个 seed 失败时是否继续处理剩余 seed。",
    )
    return parser.parse_args()


def parse_seed_spec(spec: str) -> List[int]:
    tokens = [token.strip() for token in str(spec).split(",") if token.strip()]
    if not tokens:
        raise ValueError("Seed 列表为空。")
    seeds: List[int] = []
    for token in tokens:
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start_seed = int(start_text.strip())
            end_seed = int(end_text.strip())
            if end_seed < start_seed:
                raise ValueError(f"非法 seed 区间: {token}")
            seeds.extend(range(start_seed, end_seed + 1))
        else:
            seeds.append(int(token))
    deduped: List[int] = []
    seen = set()
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        deduped.append(seed)
    return deduped


def resolve_seeds(split: str, explicit_seeds: Optional[str], seed_start: Optional[int], num_seeds: int) -> List[int]:
    if explicit_seeds:
        return parse_seed_spec(explicit_seeds)
    start_seed = get_default_start_seed(split) if seed_start is None else int(seed_start)
    if num_seeds <= 0:
        raise ValueError("num_seeds 必须为正整数。")
    return list(range(start_seed, start_seed + int(num_seeds)))


def sanitize_slug(text: Any) -> str:
    raw = str(text)
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", raw).strip("-")
    return slug or "default"


def ensure_unique_directory(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def build_default_batch_name(split: str, seeds: Sequence[int], map_arg: Optional[Any]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed_min = min(seeds)
    seed_max = max(seeds)
    map_tag = "default" if map_arg is None else sanitize_slug(map_arg)
    return f"{split}_seed{seed_min}-{seed_max}_n{len(seeds)}_map{map_tag}_{timestamp}"


def batch_directories(root_dir: Path, batch_name: str) -> Dict[str, Path]:
    batch_root = ensure_unique_directory(root_dir / batch_name)
    return {
        "batch_root": batch_root,
        "images": batch_root / "images",
        "summaries": batch_root / "summaries",
        "boards": batch_root / "boards",
        "manifests": batch_root / "manifests",
    }


def seed_file_stem(seed: int) -> str:
    return f"seed_{int(seed):05d}"


def build_batch_record(payload: Dict[str, Any], image_path: Path, summary_path: Path, batch_root: Optional[Path] = None) -> Dict[str, Any]:
    summary = payload["summary"]
    map_summary = summary["map"]
    route_summary = summary["route"]
    complexity = route_summary.get("complexity_metrics", {})

    def rel(path: Path) -> str:
        if batch_root is None:
            return str(path)
        return str(path.relative_to(batch_root))

    return {
        "seed": int(payload["seed"]),
        "split": payload["split"],
        "image_path": rel(image_path),
        "summary_path": rel(summary_path),
        "map_parameter": map_summary.get("map_parameter"),
        "map_class_name": map_summary.get("class_name"),
        "map_num_blocks": int(map_summary.get("num_blocks", 0)),
        "map_num_lanes": int(map_summary.get("num_lanes", 0)),
        "map_block_id_sequence": "".join(map_summary.get("map_block_id_sequence", [])),
        "route_segment_count": int(route_summary.get("segment_count", 0)),
        "route_length_m": float(route_summary.get("length_m", 0.0)),
        "route_centerline_length_m": float(route_summary.get("centerline_length_m", 0.0)),
        "curved_segment_count": int(route_summary.get("curved_segment_count", 0)),
        "route_block_id_sequence": "".join(route_summary.get("route_block_id_sequence_compressed", [])),
        "left_turn_count": int(complexity.get("left_turn_count", 0)),
        "right_turn_count": int(complexity.get("right_turn_count", 0)),
        "straight_turn_count": int(complexity.get("straight_turn_count", 0)),
        "non_straight_turn_count": int(complexity.get("non_straight_turn_count", 0)),
        "route_block_transition_count": int(complexity.get("route_block_transition_count", 0)),
        "complexity_score": float(complexity.get("composite_complexity_score", 0.0)),
    }


def write_summary_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _load_pillow():
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    return Image, ImageDraw, ImageFont, ImageOps


def load_font(font_size: int, *, bold: bool = False, serif: bool = False):
    _, _, ImageFont, _ = _load_pillow()
    serif_candidates = (
        "Times New Roman.ttf",
        "Times New Roman Bold.ttf" if bold else "Times New Roman.ttf",
        "LiberationSerif-Bold.ttf" if bold else "LiberationSerif-Regular.ttf",
        "NimbusRoman-Bold.otf" if bold else "NimbusRoman-Regular.otf",
        "NotoSerifCJK-Bold.ttc" if bold else "NotoSerifCJK-Regular.ttc",
    )
    sans_candidates = (
        "Arial Bold.ttf" if bold else "Arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc",
    )
    candidates = serif_candidates if serif else sans_candidates
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def format_tile_caption(record: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"seed {record['seed']} | map {record['map_block_id_sequence']}",
            f"len {record['route_length_m']:.1f}m | seg {record['route_segment_count']} | curve {record['curved_segment_count']}",
            f"score {record['complexity_score']:.2f} | L/R {record['left_turn_count']}/{record['right_turn_count']}",
        ]
    )


def format_paper_tile_caption(record: Dict[str, Any]) -> str:
    route_comp = "-".join(list(str(record["route_block_id_sequence"])))
    return "\n".join(
        [
            f"Length {record['route_length_m']:.1f} m | Curves {record['curved_segment_count']} | Blocks {record['map_num_blocks']}",
            f"Route {route_comp}",
        ]
    )


def create_contact_sheet(
    *,
    title: str,
    records: Sequence[Dict[str, Any]],
    batch_root: Path,
    output_path: Path,
    cols: int,
    tile_width: int,
    caption_formatter: Callable[[Dict[str, Any]], str] = format_tile_caption,
    rows: Optional[int] = None,
    show_title: bool = True,
    style: str = "default",
    subfigure_labels: bool = False,
    # 字体大小配置参数（可选，默认使用 IEEE 风格的推荐值）
    title_font_size: Optional[int] = None,
    caption_font_size: Optional[int] = None,
    label_font_size: Optional[int] = None,
) -> Optional[Path]:
    """
    创建联系表（contact sheet），将多张图片拼接成一张大图
    
    Args:
        title: 图版标题
        records: 记录列表，每个记录包含图片路径和元数据
        batch_root: 批次根目录
        output_path: 输出文件路径
        cols: 列数
        tile_width: 每张子图的宽度（像素）
        caption_formatter: caption 格式化函数
        rows: 行数（可选，默认自动计算）
        show_title: 是否显示标题
        style: 样式风格（"ieee" 或 "default"）
        subfigure_labels: 是否显示子图标签 (a), (b), ...
        title_font_size: 标题字体大小（磅），默认 IEEE 风格 22，普通风格 26
        caption_font_size: 说明文字字体大小（磅），默认 IEEE 风格 15，普通风格 16
        label_font_size: 子图标签字体大小（磅），默认 IEEE 风格 18，普通风格 16
        
    Returns:
        输出文件路径，如果无记录则返回 None
    """
    if not records:
        return None

    Image, ImageDraw, _, ImageOps = _load_pillow()
    ieee_style = style == "ieee"
    
    # 设置字体大小：优先使用传入的参数，否则使用风格默认值
    actual_title_font_size = title_font_size if title_font_size is not None else (22 if ieee_style else 26)
    actual_caption_font_size = caption_font_size if caption_font_size is not None else (15 if ieee_style else 16)
    actual_label_font_size = label_font_size if label_font_size is not None else (18 if ieee_style else 16)
    
    # 加载字体
    title_font = load_font(actual_title_font_size, bold=True, serif=ieee_style)
    text_font = load_font(actual_caption_font_size, bold=False, serif=ieee_style)
    label_font = load_font(actual_label_font_size, bold=True, serif=ieee_style)

    image_height = tile_width
    text_height = 76 if ieee_style else 92
    padding = 16 if ieee_style else 18
    gap = 14 if ieee_style else 18
    title_height = 0 if not show_title else (34 if ieee_style else 54)
    tile_height = image_height + text_height
    rows = int(math.ceil(len(records) / cols)) if rows is None else int(rows)

    canvas_width = padding * 2 + cols * tile_width + max(cols - 1, 0) * gap
    canvas_height = padding * 2 + title_height + rows * tile_height + max(rows - 1, 0) * gap

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255) if ieee_style else (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    if show_title and title:
        draw.text((padding, padding), title, fill=(20, 20, 20), font=title_font)

    for index, record in enumerate(records):
        row = index // cols
        col = index % cols
        x0 = padding + col * (tile_width + gap)
        y0 = padding + title_height + row * (tile_height + gap)
        image_box = (x0, y0, x0 + tile_width, y0 + image_height)
        text_box = (x0, y0 + image_height, x0 + tile_width, y0 + tile_height)

        if ieee_style:
            draw.rectangle(image_box, fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        else:
            draw.rounded_rectangle(image_box, radius=12, fill=(255, 255, 255), outline=(220, 220, 220), width=2)
            draw.rounded_rectangle(text_box, radius=12, fill=(255, 255, 255), outline=(220, 220, 220), width=2)

        image_path = batch_root / record["image_path"]
        with Image.open(image_path).convert("RGB") as image:
            fitted = ImageOps.contain(image, (tile_width - 10 if ieee_style else tile_width - 12, image_height - 10 if ieee_style else image_height - 12))
            paste_x = x0 + (tile_width - fitted.width) // 2
            paste_y = y0 + (image_height - fitted.height) // 2
            canvas.paste(fitted, (paste_x, paste_y))

        if subfigure_labels:
            label = f"({chr(ord('a') + index)})"
            label_x = x0 + 10
            label_y = y0 + 8
            if ieee_style:
                draw.text((label_x, label_y), label, fill=(0, 0, 0), font=label_font)
            else:
                draw.text((label_x, label_y), label, fill=(0, 0, 0), font=label_font)

        caption = caption_formatter(record)
        draw.multiline_text(
            (x0 + 6 if ieee_style else x0 + 10, y0 + image_height + 8),
            caption,
            fill=(0, 0, 0) if ieee_style else (30, 30, 30),
            font=text_font,
            spacing=4,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def sort_records(records: Sequence[Dict[str, Any]], sort_by: str, descending: bool) -> List[Dict[str, Any]]:
    return sorted(records, key=lambda record: record.get(sort_by, 0), reverse=descending)


def write_batch_readme(path: Path, manifest: Dict[str, Any]) -> None:
    records = manifest.get("records", [])
    failures = manifest.get("failures", [])
    lines = [
        "# MetaDrive Route Batch",
        "",
        f"- Batch name: `{manifest['batch_name']}`",
        f"- Split: `{manifest['split']}`",
        f"- Requested seeds: `{manifest['requested_seeds']}`",
        f"- Success count: `{len(records)}`",
        f"- Failure count: `{len(failures)}`",
        f"- Images dir: `{manifest['directories']['images']}`",
        f"- Summaries dir: `{manifest['directories']['summaries']}`",
        f"- Boards dir: `{manifest['directories']['boards']}`",
        f"- Manifest JSON: `{manifest['manifest_path']}`",
        f"- Summary CSV: `{manifest['summary_csv_path']}`",
        "",
        "## Top Complexity Seeds",
        "",
    ]

    top_records = sort_records(records, "complexity_score", descending=True)[: min(5, len(records))]
    if not top_records:
        lines.append("No successful records.")
    else:
        for record in top_records:
            lines.append(
                "- seed `{seed}` | score `{score:.2f}` | len `{length:.1f}m` | curve `{curve}` | block `{block}`".format(
                    seed=record["seed"],
                    score=record["complexity_score"],
                    length=record["route_length_m"],
                    curve=record["curved_segment_count"],
                    block=record["route_block_id_sequence"],
                )
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    map_arg = parse_map_arg(args.map_arg)
    seeds = resolve_seeds(args.split, args.seeds, args.seed_start, args.num_seeds)

    batch_name = args.batch_name or build_default_batch_name(args.split, seeds, map_arg)
    directories = batch_directories(Path(args.root_dir).expanduser().resolve(), batch_name)
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    print(f"[route-batch] batch_root={directories['batch_root']}")
    print(f"[route-batch] seeds={seeds}")

    for index, seed in enumerate(seeds, start=1):
        stem = seed_file_stem(seed)
        image_path = directories["images"] / f"{stem}.png"
        summary_path = directories["summaries"] / f"{stem}.json"
        print(f"[route-batch] ({index}/{len(seeds)}) seed={seed}")
        try:
            result = generate_route_visualization(
                split=args.split,
                seed=seed,
                map_arg=map_arg,
                output_path=image_path,
                summary_path=summary_path,
                film_size=args.film_size,
                scaling=args.scaling,
                semantic_map=bool(args.semantic_map),
                draw_center_line=bool(args.draw_center_line),
                annotate_segments=bool(args.annotate_segments),
                route_sample_interval=args.route_sample_interval,
                turn_threshold_deg=args.turn_threshold_deg,
                curvature_threshold_deg=args.curvature_threshold_deg,
                traffic_density=float(args.traffic_density),
                accident_prob=float(args.accident_prob),
                dpi=int(args.dpi),
                print_summary_log=False,
            )
            records.append(
                build_batch_record(
                    result["payload"],
                    result["output_path"],
                    result["summary_path"],
                    batch_root=directories["batch_root"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            failure = {"seed": int(seed), "error": "{}: {}".format(type(exc).__name__, exc)}
            failures.append(failure)
            print(f"[route-batch] seed={seed} failed: {failure['error']}")
            if not args.continue_on_error:
                raise

    overview_board_path = create_contact_sheet(
        title=f"MetaDrive Route Overview | split={args.split} | seeds={len(records)}",
        records=sort_records(records, "seed", descending=False),
        batch_root=directories["batch_root"],
        output_path=directories["boards"] / "overview_board_seed_order.png",
        cols=max(1, int(args.overview_cols)),
        tile_width=int(args.overview_tile_width),
    )

    paper_candidates = sort_records(
        records,
        "complexity_score" if args.paper_sort_by == "complexity_score" else args.paper_sort_by,
        descending=False if args.paper_sort_by == "seed" else True,
    )[: max(1, int(args.paper_topk))]
    paper_board_path = create_contact_sheet(
        title=f"MetaDrive Route Paper Board | sort={args.paper_sort_by} | topk={len(paper_candidates)}",
        records=paper_candidates,
        batch_root=directories["batch_root"],
        output_path=directories["boards"] / f"paper_board_{args.paper_sort_by}_top{len(paper_candidates)}.png",
        cols=max(1, int(args.paper_cols)),
        tile_width=int(args.paper_tile_width),
    )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "batch_name": directories["batch_root"].name,
        "split": args.split,
        "requested_seeds": seeds,
        "map_arg": map_arg,
        "directories": {key: str(path.relative_to(directories["batch_root"])) for key, path in directories.items() if key != "batch_root"},
        "visualization": {
            "film_size": int(args.film_size),
            "scaling": None if args.scaling is None else float(args.scaling),
            "semantic_map": bool(args.semantic_map),
            "draw_center_line": bool(args.draw_center_line),
            "annotate_segments": bool(args.annotate_segments),
            "route_sample_interval_m": float(args.route_sample_interval),
            "turn_threshold_deg": float(args.turn_threshold_deg),
            "curvature_threshold_deg": float(args.curvature_threshold_deg),
            "traffic_density": float(args.traffic_density),
            "accident_prob": float(args.accident_prob),
            "dpi": int(args.dpi),
        },
        "boards": {
            "overview_board_seed_order": None if overview_board_path is None else str(overview_board_path.relative_to(directories["batch_root"])),
            "paper_board": None if paper_board_path is None else str(paper_board_path.relative_to(directories["batch_root"])),
        },
        "records": records,
        "failures": failures,
    }

    manifest_path = directories["manifests"] / "manifest.json"
    summary_csv_path = directories["manifests"] / "summary.csv"
    manifest["manifest_path"] = str(manifest_path.relative_to(directories["batch_root"]))
    manifest["summary_csv_path"] = str(summary_csv_path.relative_to(directories["batch_root"]))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    write_summary_csv(summary_csv_path, records)
    write_batch_readme(directories["batch_root"] / "README.md", manifest)

    print(f"[route-batch] success={len(records)} failure={len(failures)}")
    print(f"[route-batch] manifest={manifest_path}")
    print(f"[route-batch] csv={summary_csv_path}")
    if overview_board_path is not None:
        print(f"[route-batch] overview_board={overview_board_path}")
    if paper_board_path is not None:
        print(f"[route-batch] paper_board={paper_board_path}")


if __name__ == "__main__":
    main()
