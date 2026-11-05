"""Deterministic Morgan count fingerprints and a small interpretable descriptor panel."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

DESCRIPTORS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "HBD": Lipinski.NumHDonors,
    "HBA": Lipinski.NumHAcceptors,
    "RotatableBonds": Lipinski.NumRotatableBonds,
    "RingCount": Lipinski.RingCount,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": Lipinski.HeavyAtomCount,
    "FormalCharge": Chem.GetFormalCharge,
    "NHOHCount": Lipinski.NHOHCount,
    "NOCount": Lipinski.NOCount,
}


def featurize(smiles, n_bits=1024, radius=2):
    if n_bits < 1 or radius < 0:
        raise ValueError("Fingerprint size must be positive and radius nonnegative")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=True
    )
    rows = []
    for i, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES at row {i}: {value!r}")
        fp = generator.GetCountFingerprintAsNumPy(mol)
        desc = [func(mol) for func in DESCRIPTORS.values()]
        rows.append(np.concatenate([fp, desc]))
    if not rows:
        raise ValueError("No molecules to featurize")
    features = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite descriptor value")
    names = [f"Morgan_{i:04d}" for i in range(n_bits)] + list(DESCRIPTORS)
    return features, names
