"""Audit full-run artifacts against public structures and per-row held-out predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lnp_benchmark.data import SHA256, digest, download, load_data
from lnp_benchmark.evaluation import bootstrap_metrics, metrics
from lnp_benchmark.splits import scaffold
from update_readme import result_block


def main():
    download("data/AGILE.csv")
    source = load_data("data/AGILE.csv")
    root = Path("results/reference")
    training = json.loads((root / "training.json").read_text())
    evaluation = json.loads((root / "evaluation.json").read_text())
    predictions = pd.read_csv(root / "predictions.csv")
    assignment = pd.read_csv(root / "assignments.csv")
    cv = pd.read_csv(root / "cv_results.csv")
    assert not training["quick"], "A smoke run must never be the reference result"
    assert training["dataset_sha256"] == SHA256
    actual_groups = np.array([scaffold(s) for s in source.canonical_smiles])
    bootstrap_draws = {}
    for kind, info in training["splits"].items():
        part = assignment[assignment.split == kind]
        assert len(part) == len(source) and part.row_id.nunique() == len(source)
        assert set(part.row_id) == set(source.row_id)
        np.testing.assert_array_equal(part.scaffold, actual_groups[part.row_id])
        train = part[part.partition == "train"]
        test = part[part.partition == "test"]
        assert len(train) == info["train_rows"] and len(test) == info["test_rows"]
        if kind == "scaffold":
            assert not set(train.scaffold) & set(test.scaffold)
            for fold in sorted(train.cv_validation_fold.unique()):
                assert not set(train[train.cv_validation_fold == fold].scaffold) & set(
                    train[train.cv_validation_fold != fold].scaffold
                )
        for name, fit in info["models"].items():
            p = predictions[(predictions.split == kind) & (predictions.model == name)]
            assert set(p.row_id) == set(test.row_id) and p.row_id.is_unique
            np.testing.assert_allclose(p.target, source.Target.iloc[p.row_id])
            np.testing.assert_array_equal(p.scaffold, actual_groups[p.row_id])
            scores = metrics(p.target.to_numpy(), p.prediction.to_numpy())
            saved = next(
                s for s in evaluation["scores"] if s["split"] == kind and s["model"] == name
            )
            for key, value in scores.items():
                if value is not None:
                    assert np.isclose(saved[key], value, atol=1e-10), (kind, name, key)
            intervals, draws = bootstrap_metrics(
                p.target.to_numpy(),
                p.prediction.to_numpy(),
                p.scaffold.to_numpy(),
                evaluation["bootstrap_repeats"],
                training["seed"] + (kind == "scaffold"),
            )
            bootstrap_draws[kind, name] = draws["rmse"]
            for metric, bounds in intervals.items():
                np.testing.assert_allclose(saved["ci95"][metric], bounds, atol=1e-10)
            tuning = cv[(cv.split == kind) & (cv.model == name)]
            assert np.isclose(tuning.mean_cv_rmse.min(), fit["cv_rmse"])
    for gap in evaluation["gaps"]:
        scores = {s["split"]: s["rmse"] for s in evaluation["scores"] if s["model"] == gap["model"]}
        assert np.isclose(gap["scaffold_minus_random_rmse"], scores["scaffold"] - scores["random"])
        deltas = bootstrap_draws["scaffold", gap["model"]] - bootstrap_draws["random", gap["model"]]
        np.testing.assert_allclose(gap["ci95"], np.quantile(deltas, [0.025, 0.975]), atol=1e-10)
    for path, expected in json.loads((root / "source_manifest.json").read_text()).items():
        assert digest(path) == expected, f"Source changed since reference run: {path}"
    for filename in ["generalization.png", "feature_importance.png", "scaffold_errors.png"]:
        assert (root / filename).stat().st_size > 1000
    block = (
        Path("README.md")
        .read_text()
        .split("<!-- benchmark:start -->\n")[1]
        .split("\n<!-- benchmark:end -->")[0]
    )
    assert block == result_block(root), "Run uv run python scripts/update_readme.py"
    print(
        f"Verified {len(source)} public structures, {len(predictions)} held-out predictions, "
        f"{len(cv)} CV configurations, exact split boundaries and generated README values."
    )


if __name__ == "__main__":
    main()
