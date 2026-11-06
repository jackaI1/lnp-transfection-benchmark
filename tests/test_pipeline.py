import json

import pandas as pd

from lnp_benchmark.cli import main
from lnp_benchmark.artifact import export_model, load_verified, predict


def test_all_cli_stages_on_small_chemical_fixture(tmp_path):
    # A test fixture only. The reportable run always uses the checksum-verified public data.
    rings = ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCCC1", "C1CCOC1", "C1CCNCC1"]
    smiles = [prefix + ring for prefix in ("C", "CC", "CCC") for ring in rings]
    path = tmp_path / "fixture.csv"
    pd.DataFrame({"SMILES": smiles, "Target": [i / 2 for i in range(len(smiles))]}).to_csv(
        path, index=False
    )
    out = str(tmp_path / "run")
    main(["featurize", "--data", str(path), "--output", out, "--bits", "64"])
    main(["train", "--output", out, "--quick"])
    main(["evaluate", "--output", out, "--bootstrap", "30"])
    main(["report", "--output", out])
    results = json.loads((tmp_path / "run/evaluation.json").read_text())
    assert len(results["scores"]) == 8
    meta = json.loads((tmp_path / "run/training.json").read_text())
    assert meta["splits"]["scaffold"]["shared_scaffolds"] == 0
    assert (tmp_path / "run/feature_importance.png").stat().st_size > 1000
    assert "Quick smoke run: True" in (tmp_path / "run/report.md").read_text()
    artifact = tmp_path / "deploy/model.joblib"
    checksum = export_model(out, artifact)
    bundle = load_verified(artifact.read_bytes(), checksum)
    assert bundle["model_id"] == "scaffold/" + meta["splits"]["scaffold"]["winner_by_cv"]
    assert len(predict(bundle, smiles[:2])) == 2
