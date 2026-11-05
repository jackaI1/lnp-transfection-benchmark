"""Held-out predictions and scaffold-cluster bootstrap uncertainty."""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from threadpoolctl import threadpool_limits

from .models import load_features
from .data import digest


def metrics(y, pred):
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Nonfinite target or prediction")
    return {
        "rmse": float(root_mean_squared_error(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None,
        "spearman": float(spearmanr(y, pred).statistic)
        if np.ptp(y) > 0 and np.ptp(pred) > 0
        else None,
    }


def bootstrap_metrics(y, pred, groups, repeats=2000, seed=42):
    """Resample scaffold families with replacement, preserving all rows in each draw."""
    if repeats < 2:
        raise ValueError("Need at least two bootstrap replicates")
    rng = np.random.default_rng(seed)
    families = [np.flatnonzero(groups == group) for group in sorted(set(groups))]
    samples = {key: [] for key in ("rmse", "mae", "r2")}
    for _ in range(repeats):
        idx = np.concatenate([families[i] for i in rng.integers(len(families), size=len(families))])
        error = y[idx] - pred[idx]
        samples["rmse"].append(float(np.sqrt(np.mean(error**2))))
        samples["mae"].append(float(np.mean(np.abs(error))))
        variance = np.sum((y[idx] - np.mean(y[idx])) ** 2)
        samples["r2"].append(1 - float(np.sum(error**2) / variance) if variance else np.nan)
    draws = {key: np.asarray(value) for key, value in samples.items()}
    intervals = {
        key: np.quantile(value[np.isfinite(value)], [0.025, 0.975]).tolist()
        if np.isfinite(value).any()
        else [None, None]
        for key, value in draws.items()
    }
    return intervals, draws


def permutation_groups(model, X, y, names, repeats=10, seed=42):
    """Permute the Morgan block together and each physicochemical descriptor separately."""
    rng = np.random.default_rng(seed)
    names = list(names)
    fingerprint = [i for i, name in enumerate(names) if name.startswith("Morgan_")]
    blocks = {"Morgan block (all counts)": fingerprint}
    blocks.update({name: [i] for i, name in enumerate(names) if i not in fingerprint})
    base = root_mean_squared_error(y, model.predict(X))
    records = []
    for name, columns in blocks.items():
        differences = []
        for _ in range(repeats):
            permuted = X.copy()
            permuted[:, columns] = X[rng.permutation(len(X))][:, columns]
            differences.append(float(root_mean_squared_error(y, model.predict(permuted)) - base))
        records.append(
            {
                "feature": name,
                "rmse_increase_mean": np.mean(differences),
                "permutation_sd": np.std(differences),
                "repeats": repeats,
            }
        )
    return records


def evaluate(feature_path, output, repeats=2000):
    data = load_features(feature_path)
    output = Path(output)
    meta = json.loads((output / "training.json").read_text())
    if digest(feature_path) != meta["features_sha256"]:
        raise ValueError("Feature artifact changed since training; rerun train")
    if str(data["source_sha256"]) != meta["dataset_sha256"]:
        raise ValueError("Features do not match training dataset")
    if json.loads(str(data["config"])) != meta["feature_config"]:
        raise ValueError("Feature configuration does not match training")
    assignments = pd.read_csv(output / "assignments.csv", keep_default_na=False)
    results = {
        "bootstrap_repeats": repeats,
        "ci_method": "95% scaffold-cluster percentile bootstrap",
        "uncertainty_scope": "Conditional on this fixed train/test split and fitted models; "
        "not training-seed, split-selection, assay or external-study uncertainty.",
        "scores": [],
        "gaps": [],
    }
    predictions, errors, importance = [], [], []
    draws = {}
    with warnings.catch_warnings(record=True) as seen, threadpool_limits(limits=1):
        warnings.simplefilter("once")
        for kind, info in meta["splits"].items():
            ids = assignments.loc[
                (assignments.split == kind) & (assignments.partition == "test"), "row_id"
            ].to_numpy(dtype=int)
            # Dataset row IDs are positional and validated at feature creation.
            X, y, groups = data["X"][ids], data["y"][ids], data["groups"][ids]
            for name in info["models"]:
                model = joblib.load(output / f"{kind}_{name}.joblib")
                pred = model.predict(X)
                intervals, replicates = bootstrap_metrics(
                    y, pred, groups, repeats, meta["seed"] + (kind == "scaffold")
                )
                draws[kind, name] = replicates
                results["scores"].append(
                    {
                        "split": kind,
                        "model": name,
                        "n_test": len(y),
                        "scaffolds": len(set(groups)),
                        **metrics(y, pred),
                        "ci95": intervals,
                    }
                )
                for row, truth, estimate, group in zip(ids, y, pred, groups):
                    predictions.append(
                        {
                            "split": kind,
                            "model": name,
                            "row_id": int(row),
                            "scaffold": group,
                            "target": truth,
                            "prediction": estimate,
                            "residual": estimate - truth,
                        }
                    )
                for group in sorted(set(groups)):
                    idx = np.flatnonzero(groups == group)
                    errors.append(
                        {
                            "split": kind,
                            "model": name,
                            "scaffold": group,
                            "n": len(idx),
                            "target_mean": float(y[idx].mean()),
                            "prediction_mean": float(pred[idx].mean()),
                            "bias": float((pred[idx] - y[idx]).mean()),
                            **metrics(y[idx], pred[idx]),
                        }
                    )
                if name == info["winner_by_cv"]:
                    for record in permutation_groups(model, X, y, data["names"], seed=meta["seed"]):
                        importance.append({"split": kind, "model": name, **record})
        for name in meta["splits"]["random"]["models"]:
            a, b = [
                next(s for s in results["scores"] if s["split"] == kind and s["model"] == name)
                for kind in ("random", "scaffold")
            ]
            deltas = draws["scaffold", name]["rmse"] - draws["random", name]["rmse"]
            results["gaps"].append(
                {
                    "model": name,
                    "scaffold_minus_random_rmse": b["rmse"] - a["rmse"],
                    "ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
                }
            )
    results["evaluation_warnings"] = sorted(
        {w.category.__name__ + ": " + str(w.message) for w in seen}
    )
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(errors).to_csv(output / "scaffold_errors.csv", index=False)
    pd.DataFrame(importance).to_csv(output / "feature_importance.csv", index=False)
    (output / "evaluation.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results
