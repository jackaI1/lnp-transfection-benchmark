"""Export one cross-validation-selected benchmark model for inference."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from rdkit import rdBase

from .features import DESCRIPTORS, featurize

MAX_MODEL_BYTES = 128 * 1024 * 1024


def versions():
    return {"sklearn": sklearn.__version__, "numpy": np.__version__, "rdkit": rdBase.rdkitVersion}


def validate_bundle(bundle):
    if bundle.get("schema") != 1 or bundle.get("versions") != versions():
        raise ValueError("Unsupported artifact schema or runtime versions")
    config = bundle["feature_config"]
    if (
        config.get("morgan") != "hashed count"
        or config.get("chirality") is not True
        or config.get("descriptor_names") != list(DESCRIPTORS)
        or type(config.get("n_bits")) is not int
        or not 1 <= config["n_bits"] <= 8192
        or type(config.get("radius")) is not int
        or not 0 <= config["radius"] <= 4
    ):
        raise ValueError("Unsupported feature configuration")
    expected = config["n_bits"] + len(DESCRIPTORS)
    if bundle["model"].n_features_in_ != expected:
        raise ValueError("Model and feature configuration disagree")
    return bundle


def export_model(run, output, split="scaffold"):
    run, output = Path(run), Path(output)
    record = json.loads((run / "training.json").read_text())
    if record["versions"]["sklearn"] != sklearn.__version__:
        raise ValueError("Export with the training runtime")
    winner = record["splits"][split]["winner_by_cv"]
    bundle = validate_bundle(
        {
            "schema": 1,
            "model": joblib.load(run / f"{split}_{winner}.joblib"),
            "feature_config": record["feature_config"],
            "versions": {key: record["versions"][key] for key in versions()},
            "model_id": f"{split}/{winner}",
            "dataset_sha256": record["dataset_sha256"],
            "quick": record["quick"],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(checksum + "\n")
    return checksum


def load_verified(payload, checksum):
    # joblib uses pickle, so verify bytes BEFORE deserializing. Only export trusted models.
    if len(payload) > MAX_MODEL_BYTES:
        raise ValueError("Model exceeds size limit")
    if hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("Model checksum mismatch")
    return validate_bundle(joblib.load(io.BytesIO(payload)))


def predict(bundle, smiles):
    config = bundle["feature_config"]
    features, _ = featurize(smiles, n_bits=config["n_bits"], radius=config["radius"])
    values = np.asarray(bundle["model"].predict(features))
    if values.shape != (len(smiles),) or not np.isfinite(values).all():
        raise RuntimeError("Model returned invalid predictions")
    return values.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/run")
    parser.add_argument("--output", default="results/deploy/model.joblib")
    parser.add_argument("--split", choices=["random", "scaffold"], default="scaffold")
    args = parser.parse_args()
    print(export_model(args.run, args.output, args.split))


if __name__ == "__main__":
    main()
