import numpy as np
import pytest

from lnp_benchmark.evaluation import bootstrap_metrics, metrics


def test_metrics_known_values():
    scores = metrics(np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert scores["rmse"] == 1
    assert scores["mae"] == 1
    assert scores["r2"] == 0
    assert scores["spearman"] is None


def test_cluster_bootstrap_keeps_families_intact():
    # Two families, with two equal-error rows each. Possible RMSEs are 1, sqrt(5), 3.
    y = np.array([0.0, 0.0, 0.0, 0.0])
    pred = np.array([1.0, 1.0, 3.0, 3.0])
    intervals, draws = bootstrap_metrics(y, pred, np.array(["a", "a", "b", "b"]), 100)
    assert set(np.round(draws["rmse"], 6)) <= {1.0, round(np.sqrt(5), 6), 3.0}
    assert intervals["rmse"] == pytest.approx([1.0, 3.0])
    _, again = bootstrap_metrics(y, pred, np.array(["a", "a", "b", "b"]), 100)
    np.testing.assert_array_equal(draws["rmse"], again["rmse"])


def test_constant_float_predictions_have_no_rank_correlation():
    # Floating-point std can be nonzero even for a mathematically constant vector.
    assert metrics(np.arange(110.0), np.full(110, 3.14))["spearman"] is None


def test_nonfinite_predictions_are_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        metrics(np.array([1.0, 2.0]), np.array([float("nan"), 2.0]))
