from pathlib import Path

from batch_visualize_metadrive_routes import (
    build_batch_record,
    build_default_batch_name,
    ensure_unique_directory,
    format_paper_tile_caption,
    parse_seed_spec,
    resolve_seeds,
)
from assemble_metadrive_route_board import parse_seed_grid


def test_parse_seed_spec_supports_ranges_and_deduplication():
    assert parse_seed_spec("100,101,103-105,101") == [100, 101, 103, 104, 105]


def test_resolve_seeds_uses_explicit_list_first():
    assert resolve_seeds("train", "200-202", None, 9) == [200, 201, 202]


def test_build_default_batch_name_contains_core_metadata():
    name = build_default_batch_name("train", [100, 101, 102], 3)
    assert "train" in name
    assert "100-102" in name
    assert "n3" in name
    assert "map3" in name


def test_ensure_unique_directory_appends_suffix(tmp_path: Path):
    base = tmp_path / "batch"
    base.mkdir()
    unique = ensure_unique_directory(base)
    assert unique != base
    assert unique.name.startswith("batch_")


def test_build_batch_record_flattens_payload_for_csv():
    payload = {
        "seed": 100,
        "split": "train",
        "summary": {
            "map": {
                "map_parameter": 3,
                "class_name": "PGMap",
                "num_blocks": 4,
                "num_lanes": 96,
                "map_block_id_sequence": ["I", "S", "C", "O"],
            },
            "route": {
                "segment_count": 13,
                "length_m": 521.9,
                "centerline_length_m": 520.2,
                "curved_segment_count": 8,
                "route_block_id_sequence_compressed": [">", "S", "C", "O"],
                "complexity_metrics": {
                    "left_turn_count": 1,
                    "right_turn_count": 2,
                    "straight_turn_count": 9,
                    "non_straight_turn_count": 3,
                    "route_block_transition_count": 3,
                    "composite_complexity_score": 17.5,
                },
            },
        },
    }
    record = build_batch_record(
        payload,
        image_path=Path("/tmp/images/seed_00100.png"),
        summary_path=Path("/tmp/summaries/seed_00100.json"),
        batch_root=Path("/tmp"),
    )
    assert record["seed"] == 100
    assert record["map_block_id_sequence"] == "ISCO"
    assert record["route_block_id_sequence"] == ">SCO"
    assert record["complexity_score"] == 17.5


def test_parse_seed_grid_preserves_row_major_order():
    seeds, rows, cols = parse_seed_grid("100,124,152,127;108,158,143,134")
    assert seeds == [100, 124, 152, 127, 108, 158, 143, 134]
    assert rows == 2
    assert cols == 4


def test_format_paper_tile_caption_excludes_seed():
    caption = format_paper_tile_caption(
        {
            "route_length_m": 522.0,
            "curved_segment_count": 8,
            "map_num_blocks": 4,
            "route_block_id_sequence": ">SCO",
        }
    )
    assert "seed" not in caption.lower()
    assert "Length" in caption
    assert "Curves" in caption
    assert "Blocks" in caption
    assert "Route" in caption
