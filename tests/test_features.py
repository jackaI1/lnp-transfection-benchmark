import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from lnp_benchmark.features import featurize


def test_shape_finite_counts_and_known_descriptors():
    X, names = featurize(["CCO", "c1ccccc1"], n_bits=128)
    assert X.shape == (2, 140)
    assert np.isfinite(X).all()
    assert (X[:, :128] >= 0).all()
    assert (X[:, :128] == np.floor(X[:, :128])).all()
    assert X[0, names.index("MolWt")] == pytest.approx(Descriptors.MolWt(Chem.MolFromSmiles("CCO")))
    assert X[1, names.index("RingCount")] == 1
    assert X[0, names.index("HBD")] == 1


def test_equivalent_smiles_and_repeated_calls_are_identical():
    X, names = featurize(["CCO", "OCC"], n_bits=128)
    Y, other = featurize(["CCO", "OCC"], n_bits=128)
    np.testing.assert_allclose(X[0], X[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(X, Y)
    assert names == other


def test_chirality_is_encoded():
    X, _ = featurize(["N[C@H](C)C(=O)O", "N[C@@H](C)C(=O)O"])
    assert not np.array_equal(X[0, :1024], X[1, :1024])


@pytest.mark.parametrize("smiles", [["bad smiles"], [""], []])
def test_invalid_or_empty_structures_fail(smiles):
    with pytest.raises(ValueError):
        featurize(smiles)
