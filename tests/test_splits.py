import numpy as np
import pytest

from lnp_benchmark.splits import inner_folds, scaffold, split_indices


def test_murcko_ignores_sidechains_and_groups_acyclics():
    assert scaffold("Cc1ccccc1") == scaffold("CCc1ccccc1") == "c1ccccc1"
    assert scaffold("CCCCN") == scaffold("CCO") == "ACYCLIC"
    assert scaffold("C1CCCCC1") != scaffold("c1ccccc1")


@pytest.mark.parametrize("kind", ["scaffold", "random"])
def test_split_covers_all_rows_deterministically(kind):
    groups = np.array(["a"] * 8 + ["b"] * 4 + ["c"] * 3 + ["d"] * 3 + ["e"] * 2)
    train, test = split_indices(groups, kind)
    again, other = split_indices(groups, kind)
    np.testing.assert_array_equal(train, again)
    np.testing.assert_array_equal(test, other)
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(range(20))
    if kind == "scaffold":
        assert not set(groups[train]) & set(groups[test])
        assert len(test) == 5  # Whole families make the requested fraction approximate.


def test_grouped_cv_has_no_scaffold_leakage():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    validation_rows = []
    for train, val in inner_folds(groups, "scaffold"):
        assert not set(groups[train]) & set(groups[val])
        validation_rows.extend(val)
    assert sorted(validation_rows) == list(range(len(groups)))


def test_largest_family_is_never_fragmented():
    groups = ["ACYCLIC"] * 9 + ["ring"]
    train, test = split_indices(groups, "scaffold")
    assert set(test) == set(range(9))
    assert list(train) == [9]


@pytest.mark.parametrize(
    "call",
    [
        lambda: split_indices(["ACYCLIC"] * 10, "scaffold"),
        lambda: split_indices(["a", "b"], "scaffold", test_fraction=1),
        lambda: split_indices(["a", "b"], "unknown"),
        lambda: inner_folds(["a", "a", "b"], "scaffold"),
        lambda: scaffold("invalid"),
    ],
)
def test_impossible_splits_fail_clearly(call):
    with pytest.raises(ValueError):
        call()
