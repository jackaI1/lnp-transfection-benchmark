"""Keep entire Bemis-Murcko families together, including the empty acyclic family."""

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold, KFold, train_test_split


def scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Invalid SMILES")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "ACYCLIC"


def split_indices(groups, kind, seed=42, test_fraction=0.2):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie between zero and one")
    indices = np.arange(len(groups))
    if kind == "random":
        train, test = train_test_split(indices, test_size=test_fraction, random_state=seed)
    elif kind == "scaffold":
        families = defaultdict(list)
        for i, group in enumerate(groups):
            families[group].append(i)
        if len(families) < 2:
            raise ValueError("Need at least two scaffolds; acyclic molecules share one scaffold")
        # Largest-first, with lexical tie breaking: target-free and independent of CSV order.
        ordered = sorted(families.items(), key=lambda item: (-len(item[1]), item[0]))
        capacity = int(len(groups) * (1 - test_fraction))
        train, test = [], []
        for _, members in ordered:
            (train if len(train) + len(members) <= capacity else test).extend(members)
        if not train or not test:
            raise ValueError("Scaffold sizes cannot support this split fraction")
    else:
        raise ValueError(f"Unknown split: {kind}")
    return np.sort(train), np.sort(test)


def inner_folds(groups, kind, seed=42, n_splits=3):
    if kind == "scaffold":
        if len(set(groups)) < n_splits:
            raise ValueError("Too few training scaffolds for grouped cross-validation")
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
    if kind != "random":
        raise ValueError(f"Unknown split: {kind}")
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(groups))
