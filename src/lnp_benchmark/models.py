"""Training-only hyperparameter searches, with fold-local scaling and target scaling."""

import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from rdkit import rdBase
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .splits import inner_folds, split_indices
from .data import digest


def model_specs(seed, quick=False):
    specs = {
        "mean": (DummyRegressor(), {}),
        "random_forest": (
            RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=1),
            {"max_features": [0.3, 1.0], "min_samples_leaf": [1, 3]},
        ),
        "gradient_boosting": (
            HistGradientBoostingRegressor(random_state=seed, max_iter=150, early_stopping=False),
            {"max_leaf_nodes": [7, 15], "l2_regularization": [1.0, 10.0]},
        ),
        "mlp": (
            TransformedTargetRegressor(
                regressor=Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "net",
                            MLPRegressor(
                                hidden_layer_sizes=(64,),
                                max_iter=400,
                                random_state=seed,
                                early_stopping=False,
                                learning_rate_init=0.001,
                            ),
                        ),
                    ]
                ),
                transformer=StandardScaler(),
            ),
            {
                "regressor__net__alpha": [0.1, 1.0],
                "regressor__net__hidden_layer_sizes": [(32,), (64,)],
            },
        ),
    }
    if quick:
        specs = {
            k: (m, {p: values[:1] for p, values in grid.items()}) for k, (m, grid) in specs.items()
        }
        specs["random_forest"][0].set_params(n_estimators=12)
        specs["gradient_boosting"][0].set_params(max_iter=12)
        specs["mlp"][0].set_params(regressor__net__max_iter=12)
    return specs


def load_features(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def train(feature_path, output, seed=42, quick=False):
    data = load_features(feature_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": seed,
        "quick": quick,
        "cv_folds": 3,
        "test_fraction_requested": 0.2,
        "dataset_sha256": str(data["source_sha256"]),
        "features_sha256": digest(feature_path),
        "feature_config": json.loads(str(data["config"])),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "splits": {},
    }
    assignments, cv_records = [], []
    with threadpool_limits(limits=1):
        for kind in ("random", "scaffold"):
            tr, te = split_indices(data["groups"], kind, seed)
            folds = inner_folds(data["groups"][tr], kind, seed)
            fold_by_index = {
                int(tr[idx]): fold for fold, (_, val) in enumerate(folds) for idx in val
            }
            for i in range(len(data["y"])):
                assignments.append(
                    {
                        "split": kind,
                        "row_id": int(data["row_id"][i]),
                        "partition": "train" if i in fold_by_index else "test",
                        "cv_validation_fold": fold_by_index.get(i, -1),
                        "scaffold": data["groups"][i],
                    }
                )
            summary = {
                "train_rows": len(tr),
                "test_rows": len(te),
                "train_scaffolds": len(set(data["groups"][tr])),
                "test_scaffolds": len(set(data["groups"][te])),
                "shared_scaffolds": len(set(data["groups"][tr]) & set(data["groups"][te])),
                "models": {},
            }
            if kind == "scaffold":
                assert summary["shared_scaffolds"] == 0
            for name, (estimator, grid) in model_specs(seed, quick).items():
                print(f"Training {kind}/{name}: {len(tr)} rows, 3 CV folds", flush=True)
                search = GridSearchCV(
                    estimator,
                    grid,
                    scoring="neg_root_mean_squared_error",
                    cv=folds,
                    n_jobs=1,
                    refit=True,
                    error_score="raise",
                )
                warning_counts = {}

                def count_warning(message, category, filename, lineno, file=None, line=None):
                    key = category.__name__ + ": " + str(message)
                    warning_counts[key] = warning_counts.get(key, 0) + 1

                # Count repeated numerical warnings without retaining thousands of objects.
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.showwarning = count_warning
                    search.fit(data["X"][tr], data["y"][tr])
                for j, params in enumerate(search.cv_results_["params"]):
                    record = {
                        "split": kind,
                        "model": name,
                        "params": json.dumps(params),
                        "mean_cv_rmse": -search.cv_results_["mean_test_score"][j],
                        "std_cv_rmse": search.cv_results_["std_test_score"][j],
                    }
                    for fold in range(3):
                        record[f"fold_{fold}_rmse"] = -search.cv_results_[
                            f"split{fold}_test_score"
                        ][j]
                    cv_records.append(record)
                summary["models"][name] = {
                    "params": search.best_params_,
                    "cv_rmse": -search.best_score_,
                    "warnings": warning_counts,
                }
                joblib.dump(search.best_estimator_, output / f"{kind}_{name}.joblib")
            summary["winner_by_cv"] = min(
                summary["models"], key=lambda name: summary["models"][name]["cv_rmse"]
            )
            metadata["splits"][kind] = summary
    pd.DataFrame(assignments).to_csv(output / "assignments.csv", index=False)
    pd.DataFrame(cv_records).to_csv(output / "cv_results.csv", index=False)
    (output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
