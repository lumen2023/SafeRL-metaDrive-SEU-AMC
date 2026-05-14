from visualize_metadrive_route import (
    classify_heading_change,
    compress_consecutive,
    compute_complexity_metrics,
    polyline_endpoint_heading_deg,
    polyline_length,
    wrap_to_180,
)


def test_wrap_to_180_normalizes_large_angles():
    assert wrap_to_180(270.0) == -90.0
    assert wrap_to_180(-270.0) == 90.0
    assert wrap_to_180(45.0) == 45.0


def test_compress_consecutive_keeps_order_and_deduplicates_runs():
    assert compress_consecutive(["I", "I", "C", "C", "T", "T", "C"]) == ["I", "C", "T", "C"]


def test_polyline_length_sums_segment_lengths():
    points = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]
    assert polyline_length(points) == 10.0


def test_polyline_endpoint_heading_deg_reads_start_and_end_tangent():
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert polyline_endpoint_heading_deg(points, at_end=False) == 0.0
    assert polyline_endpoint_heading_deg(points, at_end=True) == 90.0


def test_classify_heading_change_uses_threshold():
    assert classify_heading_change(30.0, threshold_deg=15.0) == "left"
    assert classify_heading_change(-30.0, threshold_deg=15.0) == "right"
    assert classify_heading_change(5.0, threshold_deg=15.0) == "straight"


def test_compute_complexity_metrics_combines_turns_curves_and_blocks():
    metrics = compute_complexity_metrics(
        {"num_blocks": 4},
        {
            "junction_turn_count": {"left": 1, "right": 2, "straight": 4},
            "route_block_type_sequence_compressed": ["UnknownBlock", "Straight", "Curve", "Roundabout"],
            "curved_segment_count": 5,
            "cumulative_abs_segment_curvature_deg": 270.0,
            "mean_abs_junction_turn_change_deg": 12.0,
            "segment_count": 10,
        },
    )
    assert metrics["non_straight_turn_count"] == 3
    assert metrics["route_block_transition_count"] == 3
    assert metrics["composite_complexity_score"] > 0.0
