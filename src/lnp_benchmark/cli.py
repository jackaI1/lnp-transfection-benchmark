"""Five independently runnable, reproducible pipeline stages."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import digest, download, load_data
from .evaluation import evaluate
from .features import featurize
from .models import train
from .report import report
from .splits import scaffold


def create_features(data_path, output, n_bits=1024):
    frame = load_data(data_path)
    X, names = featurize(frame.canonical_smiles, n_bits=n_bits)
    groups = np.array([scaffold(s) for s in frame.canonical_smiles])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "morgan": "hashed count",
        "radius": 2,
        "n_bits": n_bits,
        "chirality": True,
        "descriptor_names": names[n_bits:],
    }
    np.savez_compressed(
        output,
        X=X,
        y=frame.Target.to_numpy(),
        groups=groups,
        names=np.array(names),
        row_id=frame.row_id.to_numpy(),
        smiles=frame.canonical_smiles.to_numpy(dtype=str),
        source_sha256=np.array(digest(data_path)),
        config=np.array(json.dumps(config)),
    )
    return {
        "rows": len(frame),
        "features": X.shape[1],
        "scaffolds": len(set(groups)),
        "acyclic": int(sum(groups == "ACYCLIC")),
        "config": config,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("download", help="Fetch and checksum the exact public LANTERN CSV")
    get.add_argument("--data", default="data/AGILE.csv")
    feat = sub.add_parser("featurize", help="Validate SMILES and create Morgan + descriptor inputs")
    feat.add_argument("--data", default="data/AGILE.csv")
    feat.add_argument("--output", default="results/run")
    feat.add_argument("--bits", type=int, default=1024)
    fit = sub.add_parser("train", help="Tune on training CV only, save models and exact splits")
    fit.add_argument("--output", default="results/run")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--quick", action="store_true", help="Smoke test only, not reportable results")
    ev = sub.add_parser("evaluate", help="Predict held-out rows and bootstrap by scaffold")
    ev.add_argument("--output", default="results/run")
    ev.add_argument("--bootstrap", type=int, default=2000)
    rep = sub.add_parser("report", help="Generate figures and a Markdown results report")
    rep.add_argument("--output", default="results/run")
    args = parser.parse_args(argv)
    if args.command == "download":
        result = download(args.data)
    elif args.command == "featurize":
        result = create_features(args.data, Path(args.output) / "features.npz", args.bits)
    elif args.command == "train":
        result = train(Path(args.output) / "features.npz", args.output, args.seed, args.quick)
    elif args.command == "evaluate":
        result = evaluate(Path(args.output) / "features.npz", args.output, args.bootstrap)
    else:
        result = {"report": str(report(args.output))}
    # Bound console output; full scores and parameters live in small local artifacts.
    if args.command in ("train", "evaluate"):
        result = {"completed": args.command, "output": args.output}
    print(json.dumps(result, indent=2))
    if args.command == "report":
        output = Path(args.output)
        # Hash the source and dependency lock so the reported run is tied to exact code.
        manifest = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("src/lnp_benchmark").glob("*.py"))
        }
        if Path("uv.lock").exists():
            manifest["uv.lock"] = digest("uv.lock")
        (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
