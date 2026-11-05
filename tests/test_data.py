import pandas as pd
import pytest

from lnp_benchmark.data import download, load_data


def test_corrupt_cache_is_rejected_without_network(tmp_path):
    path = tmp_path / "AGILE.csv"
    path.write_text("unverified")
    with pytest.raises(ValueError, match="SHA256"):
        download(path)


@pytest.mark.parametrize(
    "smiles,targets,match",
    [
        (["CCO", "OCC", "CCN", "CCC"], [1, 2, 3, 4], "Duplicate"),
        (["CCO", "CCN", "CCC", "invalid"], [1, 2, 3, 4], "Invalid"),
        (["CCO", "CCN", "CCC", "CCCC"], [1, 2, 3, float("inf")], "finite"),
    ],
)
def test_invalid_dataset_fails_before_splitting(tmp_path, smiles, targets, match):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"SMILES": smiles, "Target": targets}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        load_data(path)
