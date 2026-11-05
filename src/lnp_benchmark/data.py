"""Pinned public data, checksum verification, and explicit chemistry validation."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from rdkit import Chem

COMMIT = "11240f29ef92323649ae60d177b21df77e2d428b"
URL = f"https://raw.githubusercontent.com/AsalMehradfar/LANTERN/{COMMIT}/data/AGILE.csv"
SHA256 = "d0ecfd94ec40a7f013af546113520175e70f78045af1798673c2680fe6f23c54"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path):
    frame = pd.read_csv(path)
    if not {"SMILES", "Target"} <= set(frame):
        raise ValueError("CSV needs SMILES and Target columns")
    if len(frame) < 4 or frame.SMILES.isna().any():
        raise ValueError("Need at least four rows and no missing structures")
    frame["Target"] = pd.to_numeric(frame.Target, errors="raise")
    if not np.isfinite(frame.Target).all():
        raise ValueError("Targets must be finite")
    canonical = []
    for i, smiles in enumerate(frame.SMILES):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES at row {i}")
        canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    if len(set(canonical)) != len(canonical):
        raise ValueError("Duplicate canonical structures: resolve labels before splitting")
    frame["canonical_smiles"] = canonical
    frame["row_id"] = np.arange(len(frame))
    return frame


def download(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Existing caches are verified too. An incomplete network read never replaces the cache.
    content = path.read_bytes() if path.exists() else urlopen(URL, timeout=60).read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("Dataset SHA256 mismatch; refusing an unverified dataset")
    path.write_bytes(content)
    frame = load_data(path)
    provenance = {
        "url": URL,
        "upstream_commit": COMMIT,
        "sha256": SHA256,
        "rows": len(frame),
        "target": "Target column as distributed, unchanged; HeLa transfection assay",
        "curation": "LANTERN curated version of the Xu et al. AGILE experiment",
    }
    path.with_name("provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
