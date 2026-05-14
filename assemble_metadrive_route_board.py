#!/usr/bin/env python3
"""
从现有导出的 MetaDrive route 可视化结果中，按指定 seed 网格拼出论文图版。

适合在图片与 summary 已经生成之后，重复调整论文版式与 caption，而不必重新渲染地图。

功能特性：
- 画质控制：通过 --dpi 参数调整输出分辨率（推荐：屏幕查看 150-200，论文印刷 300-600）
- PDF 生成：通过 --output-format 选择输出格式（png/pdf/both）
- 字体控制：可自定义标题、说明文字、子图标签的字体大小

使用示例：
    # 基础用法：生成 PNG + PDF，DPI 300
    python assemble_metadrive_route_board.py \
      --seed-grid "100,124,152,127;108,158,143,134" \
      --board-name paper_4x2_selected \
      --dpi 300 \
      --output-format both

    # 自定义字体大小
    python assemble_metadrive_route_board.py \
      --seed-grid "100,124;108,158" \
      --title-font-size 24 \
      --caption-font-size 16 \
      --label-font-size 18 \
      --dpi 600 \
      --output-format pdf
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from batch_visualize_metadrive_routes import (
    build_batch_record,
    create_contact_sheet,
    format_paper_tile_caption,
    sanitize_slug,
    write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="从现有 MetaDrive route 结果中拼接定制论文图版。")
    
    # ===== 基础配置 =====
    parser.add_argument(
        "--seed-grid",
        required=True,
        help="按行指定 seed 网格，例如 '100,124,152,127;108,158,143,134'。",
    )
    parser.add_argument(
        "--search-root",
        default="debug/metadrive_route_batches",
        help="已有 batch 结果的根目录。",
    )
    parser.add_argument(
        "--output-root",
        default="debug/metadrive_route_custom_boards",
        help="定制图版输出根目录。",
    )
    parser.add_argument(
        "--board-name",
        default=None,
        help="输出目录名；默认根据 seed 网格自动生成。",
    )
    
    # ===== 标题与样式配置 =====
    parser.add_argument("--title", default="", help="图版标题。IEEE 风格通常建议留空。")
    parser.add_argument(
        "--show-title",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否在图内显示标题。IEEE 风格默认关闭。",
    )
    parser.add_argument(
        "--caption-mode",
        choices=("paper",),
        default="paper",
        help="caption 风格。默认不展示 seed，只展示论文指标。",
    )
    
    # ===== 画质与输出格式配置 =====
    parser.add_argument(
        "--tile-width", 
        type=int, 
        default=520, 
        help="每张子图缩略宽度（像素）。增大可提高清晰度但会增加文件大小。"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=300, 
        help="输出图片的 DPI（每英寸点数）。推荐值：屏幕查看用 150-200，论文印刷用 300-600。"
    )
    parser.add_argument(
        "--output-format",
        choices=("png", "pdf", "both"),
        default="both",
        help="输出格式：png（仅PNG）、pdf（仅PDF）、both（同时生成PNG和PDF）。"
    )
    
    # ===== 字体大小配置 =====
    parser.add_argument(
        "--title-font-size",
        type=int,
        default=22,
        help="标题字体大小（磅）。IEEE 风格推荐 20-24。"
    )
    parser.add_argument(
        "--caption-font-size",
        type=int,
        default=15,
        help="说明文字字体大小（磅）。IEEE 风格推荐 14-16。"
    )
    parser.add_argument(
        "--label-font-size",
        type=int,
        default=18,
        help="子图标签字体大小（磅），如 (a)、(b)。IEEE 风格推荐 16-20。"
    )
    
    return parser.parse_args()


def parse_seed_grid(spec: str) -> Tuple[List[int], int, int]:
    """
    解析 seed 网格字符串
    
    Args:
        spec: 种子网格字符串，分号分隔行，逗号分隔列
        
    Returns:
        (所有种子列表, 行数, 列数)
        
    Example:
        "100,124;108,158" -> ([100, 124, 108, 158], 2, 2)
    """
    rows = []
    for row_text in str(spec).split(";"):
        row_text = row_text.strip()
        if not row_text:
            continue
        row = [int(token.strip()) for token in row_text.split(",") if token.strip()]
        if not row:
            continue
        rows.append(row)
    
    if not rows:
        raise ValueError("seed-grid 为空。")
    
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError("seed-grid 每一行的列数必须一致。")
    
    seeds = [seed for row in rows for seed in row]
    return seeds, len(rows), width


def find_latest_summary(seed: int, search_root: Path) -> Path:
    """查找指定 seed 的最新 summary JSON 文件"""
    pattern = f"*/summaries/seed_{seed:05d}.json"
    candidates = list(search_root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"未找到 seed={seed} 的 summary 文件。")
    # 按修改时间排序，返回最新的
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def load_record_from_summary(summary_path: Path, search_root: Path) -> Dict:
    """从 summary JSON 文件加载记录信息"""
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    batch_root = summary_path.parent.parent
    image_path = batch_root / "images" / summary_path.name.replace(".json", ".png")
    if not image_path.exists():
        raise FileNotFoundError(f"缺少对应图片: {image_path}")
    
    record = build_batch_record(payload, image_path=image_path, summary_path=summary_path, batch_root=batch_root)
    record["source_batch_root"] = str(batch_root)
    record["source_summary_path"] = str(summary_path)
    record["source_image_path"] = str(image_path)
    record["search_root"] = str(search_root)
    return record


def build_default_board_name(seeds: Sequence[int]) -> str:
    """根据种子列表和时间戳生成默认的图版名称"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed_text = "-".join(str(seed) for seed in seeds)
    return f"paper_board_{sanitize_slug(seed_text)}_{timestamp}"


def save_as_pdf(canvas, output_path: Path, dpi: int) -> None:
    """
    将 PIL Image 对象保存为 PDF 格式
    
    Args:
        canvas: PIL Image 对象
        output_path: PDF 输出路径
        dpi: 分辨率（dots per inch）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 保存为 PDF，设置分辨率和质量参数
    canvas.save(
        output_path,
        format="PDF",
        resolution=dpi,
        quality=95  # JPEG 压缩质量（如果包含图像）
    )


def main() -> None:
    """主函数：组装自定义论文图版"""
    args = parse_args()
    
    # 解析种子网格
    seeds, rows, cols = parse_seed_grid(args.seed_grid)
    search_root = Path(args.search_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    board_name = args.board_name or build_default_board_name(seeds)
    board_root = output_root / board_name
    board_root.mkdir(parents=True, exist_ok=True)

    # 加载每个 seed 的记录
    records = []
    for seed in seeds:
        summary_path = find_latest_summary(seed, search_root)
        records.append(load_record_from_summary(summary_path, search_root))

    # 标准化记录路径（转换为相对路径）
    normalized_records = []
    for record in records:
        source_batch_root = Path(record["source_batch_root"])
        normalized = dict(record)
        normalized["image_path"] = str(Path(record["source_image_path"]).relative_to(source_batch_root))
        normalized["summary_path"] = str(Path(record["source_summary_path"]).relative_to(source_batch_root))
        normalized["board_batch_root"] = str(source_batch_root)
        normalized_records.append(normalized)

    # 准备用于 contact sheet 生成的记录（使用绝对路径）
    board_records = []
    for record in normalized_records:
        source_image = Path(record["source_image_path"]).resolve()
        source_summary = Path(record["source_summary_path"]).resolve()
        board_record = dict(record)
        board_record["image_path"] = str(source_image.relative_to(board_root)) if str(source_image).startswith(str(board_root)) else str(source_image)
        board_record["summary_path"] = str(source_summary.relative_to(board_root)) if str(source_summary).startswith(str(board_root)) else str(source_summary)
        board_records.append(board_record)

    # 生成接触表（contact sheet）
    # 注意：create_contact_sheet 内部使用硬编码的字体大小，需要修改该函数以支持自定义字体
    board_path = create_contact_sheet(
        title=args.title,
        records=board_records,
        batch_root=Path("/"),
        output_path=board_root / "paper_board.png",
        cols=cols,
        rows=rows,
        tile_width=int(args.tile_width),
        caption_formatter=format_paper_tile_caption,
        show_title=bool(args.show_title),
        style="ieee",
        subfigure_labels=True,
        # 传递字体大小参数（需要在 create_contact_sheet 中添加这些参数支持）
        title_font_size=args.title_font_size,
        caption_font_size=args.caption_font_size,
        label_font_size=args.label_font_size,
    )

    # 如果需要生成 PDF，则转换格式
    if args.output_format in ("pdf", "both") and board_path is not None:
        from PIL import Image
        # 打开 PNG 文件并保存为 PDF
        with Image.open(board_path) as img:
            pdf_path = board_root / "paper_board.pdf"
            save_as_pdf(img, pdf_path, args.dpi)
            print(f"[custom-board] pdf={pdf_path}")

    # 生成元数据文件
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed_grid": args.seed_grid,
        "rows": rows,
        "cols": cols,
        "title": args.title,
        "caption_mode": args.caption_mode,
        "search_root": str(search_root),
        "board_path": None if board_path is None else str(board_path),
        "output_format": args.output_format,
        "dpi": args.dpi,
        "font_sizes": {
            "title": args.title_font_size,
            "caption": args.caption_font_size,
            "label": args.label_font_size,
        },
        "records": normalized_records,
    }

    manifest_path = board_root / "manifest.json"
    csv_path = board_root / "selected_records.csv"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    write_summary_csv(csv_path, normalized_records)

    # 生成 README 说明文件
    readme_lines = [
        "# Custom MetaDrive Route Board",
        "",
        f"- Seed grid: `{args.seed_grid}`",
        f"- Layout: `{cols} x {rows}`",
        f"- Title: `{args.title}`",
        f"- Output format: `{args.output_format}`",
        f"- DPI: `{args.dpi}`",
        f"- Font sizes: title={args.title_font_size}, caption={args.caption_font_size}, label={args.label_font_size}",
        f"- Board image: `paper_board.png`",
        f"- Board PDF: `paper_board.pdf`" if args.output_format in ("pdf", "both") else "",
        f"- Manifest: `manifest.json`",
        f"- Selected records CSV: `selected_records.csv`",
        "",
        "## Selected Seeds",
        "",
    ]
    for record in normalized_records:
        readme_lines.append(
            "- seed `{seed}` | len `{length:.1f}m` | curve `{curve}` | blocks `{blocks}` | route `{route}`".format(
                seed=record["seed"],
                length=record["route_length_m"],
                curve=record["curved_segment_count"],
                blocks=record["map_num_blocks"],
                route=record["route_block_id_sequence"],
            )
        )
    (board_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"[custom-board] output_dir={board_root}")
    if board_path is not None:
        print(f"[custom-board] board={board_path}")
    print(f"[custom-board] manifest={manifest_path}")
    print(f"[custom-board] csv={csv_path}")


if __name__ == "__main__":
    main()
