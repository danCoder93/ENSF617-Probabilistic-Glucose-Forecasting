from __future__ import annotations

"""
AI-assisted implementation note:
These tests protect the evaluation package's end-to-end evaluator contract.

They intentionally focus on:
- scalar summary outputs
- grouped metric surfaces
- batch-aligned prediction/target evaluation behavior
"""

import pytest

torch = pytest.importorskip("torch")

from evaluation import evaluate_batch, evaluate_prediction_batches


def test_evaluate_batch_returns_grouped_metrics() -> None:
    quantiles = (0.1, 0.5, 0.9)
    predictions = torch.tensor(
        [
            [[90.0, 100.0, 110.0], [140.0, 150.0, 160.0]],
            [[190.0, 200.0, 210.0], [50.0, 60.0, 70.0]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[100.0, 152.0], [205.0, 65.0]],
        dtype=torch.float32,
    )
    metadata = {"subject_id": ["subject_a", "subject_b"]}

    result = evaluate_batch(
        predictions=predictions,
        target=target,
        quantiles=quantiles,
        metadata=metadata,
    )

    assert result.summary.count == 4
    assert result.summary.mae == pytest.approx(3.0)
    assert result.summary.rmse == pytest.approx((13.5) ** 0.5)
    assert result.summary.mean_interval_width == pytest.approx(20.0)
    assert result.summary.empirical_interval_coverage == pytest.approx(1.0)
    assert len(result.by_horizon) == 2
    assert len(result.by_subject) == 2
    assert {row.group_value for row in result.by_glucose_range} == {
        "70_to_180",
        "gt_180",
        "lt_70",
    }


def test_evaluate_prediction_batches_matches_batch_evaluation() -> None:
    quantiles = (0.1, 0.5, 0.9)
    prediction_batches = [
        torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32),
        torch.tensor([[[4.0, 5.0, 6.0]]], dtype=torch.float32),
    ]
    batches = [
        {"target": torch.tensor([[2.0]], dtype=torch.float32), "metadata": {"subject_id": ["a"]}},
        {"target": torch.tensor([[5.0]], dtype=torch.float32), "metadata": {"subject_id": ["b"]}},
    ]

    result = evaluate_prediction_batches(
        predictions=prediction_batches,
        batches=batches,
        quantiles=quantiles,
    )

    assert result.summary.count == 2
    assert result.summary.mae == pytest.approx(0.0)
    assert result.summary.rmse == pytest.approx(0.0)
    assert result.summary.pinball_loss_by_quantile["q10"] == pytest.approx(0.1)
    assert result.summary.pinball_loss_by_quantile["q50"] == pytest.approx(0.0)
    assert result.summary.pinball_loss_by_quantile["q90"] == pytest.approx(0.1)
