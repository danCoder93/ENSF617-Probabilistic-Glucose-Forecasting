from __future__ import annotations

# These tests protect the primitive metric formulas used by the evaluation
# package before those metrics are composed into higher-level summaries.

"""
AI-assisted implementation note:
These tests protect the evaluation package's primitive metric formulas.

The intent is to keep this module narrow and exact:
- validate the deterministic point-metric helpers
- validate the probabilistic interval/pinball helpers
- keep the assertions numeric and small so regressions are easy to diagnose
"""

import pytest

torch = pytest.importorskip("torch")

from evaluation.metrics import (
    empirical_interval_coverage,
    mean_absolute_error,
    mean_bias,
    mean_prediction_interval_width,
    pinball_loss,
    pinball_loss_by_quantile,
    root_mean_squared_error,
)


def test_metric_primitives_compute_expected_values() -> None:
    # Point metrics should remain exact and easy to sanity-check with tiny
    # hand-computable tensors.
    target = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    point_prediction = torch.tensor([[2.0, 1.0]], dtype=torch.float32)

    assert mean_absolute_error(point_prediction, target).item() == pytest.approx(1.5)
    assert root_mean_squared_error(point_prediction, target).item() == pytest.approx(
        (2.5) ** 0.5
    )
    assert mean_bias(point_prediction, target).item() == pytest.approx(-0.5)


def test_probabilistic_metric_primitives_compute_expected_values() -> None:
    # The probabilistic helpers share one small synthetic example so interval
    # width, coverage, and pinball behavior can be validated together.
    quantiles = (0.1, 0.5, 0.9)
    predictions = torch.tensor(
        [[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]],
        dtype=torch.float32,
    )
    target = torch.tensor([[1.0, 3.0]], dtype=torch.float32)

    overall_pinball = pinball_loss(predictions, target, quantiles)
    per_quantile = pinball_loss_by_quantile(predictions, target, quantiles)
    interval_width = mean_prediction_interval_width(predictions)
    coverage = empirical_interval_coverage(predictions, target)

    assert overall_pinball.item() == pytest.approx((0.1 + 0.0 + 0.1) / 3.0)
    assert per_quantile[0.1].item() == pytest.approx(0.1)
    assert per_quantile[0.5].item() == pytest.approx(0.0)
    assert per_quantile[0.9].item() == pytest.approx(0.1)
    assert interval_width is not None
    assert interval_width.item() == pytest.approx(2.0)
    assert coverage is not None
    assert coverage.item() == pytest.approx(1.0)
