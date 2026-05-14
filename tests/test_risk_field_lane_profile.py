import numpy as np
import pytest

from risk_field import RiskFieldCalculator


def _profile() -> dict:
    return {
        "kind": "solid",
        "factor": 1.0,
        "sigma": 0.35,
    }


def test_lane_line_risk_is_monotonic_non_increasing():
    calculator = RiskFieldCalculator({})
    distances = np.linspace(0.0, 2.0, 128)
    total = np.asarray(calculator.lane_line_risk_components(distances, _profile())["total"], dtype=float)

    assert np.all(total[:-1] >= total[1:] - 1e-9)


def test_lane_line_risk_is_single_gaussian_from_edge():
    calculator = RiskFieldCalculator({})

    at_edge = calculator.lane_line_risk_components(0.0, _profile())
    one_sigma = calculator.lane_line_risk_components(0.35, _profile())
    far = calculator.lane_line_risk_components(1.75, _profile())

    assert at_edge["total"] == pytest.approx(1.0)
    assert one_sigma["total"] == pytest.approx(np.exp(-0.5))
    assert far["total"] < 1e-5
    assert at_edge["core"] == pytest.approx(at_edge["total"])
    assert at_edge["shoulder"] == 0.0


def test_lane_line_risk_uses_absolute_distance_outside_lane():
    calculator = RiskFieldCalculator({})

    positive = calculator.lane_line_risk_components(0.35, _profile())
    negative = calculator.lane_line_risk_components(-0.35, _profile())

    assert negative["total"] == pytest.approx(positive["total"])
