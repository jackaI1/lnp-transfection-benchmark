"""Refresh the README's result block from the committed machine-readable report."""

import json

import pandas as pd
from pathlib import Path


def result_block(root):
    evaluation = json.loads((root / "evaluation.json").read_text())
    training = json.loads((root / "training.json").read_text())
    rows = sum(training["splits"]["random"][key] for key in ("train_rows", "test_rows"))
    scaffold = training["splits"]["scaffold"]
    lines = [
        f"I ran the full search on **{rows:,} public lipids**. Both holdouts contain "
        f"**{scaffold['test_rows']} lipids**, but the scaffold holdout has only "
        f"**{scaffold['test_scaffolds']} unseen ring families**. The random holdout shares "
        f"**{training['splits']['random']['shared_scaffolds']} families** with training; "
        f"the scaffold holdout shares **{scaffold['shared_scaffolds']}**.",
        "",
        "RMSE is prediction error in the source dataset's target units; lower is better. "
        "The brackets are 95% confidence intervals from resampling whole scaffold families.",
        "",
        "| Model | Random RMSE [95% CI] | Scaffold RMSE [95% CI] | Gap [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for name in training["splits"]["random"]["models"]:
        a, b = [
            next(s for s in evaluation["scores"] if s["split"] == kind and s["model"] == name)
            for kind in ("random", "scaffold")
        ]
        gap = next(g for g in evaluation["gaps"] if g["model"] == name)
        cells = [
            f"{s['rmse']:.3f} [{s['ci95']['rmse'][0]:.3f}, {s['ci95']['rmse'][1]:.3f}]"
            for s in (a, b)
        ]
        lines.append(
            f"| {name.replace('_', ' ')} | {cells[0]} | {cells[1]} | "
            f"{gap['scaffold_minus_random_rmse']:+.3f} "
            f"[{gap['ci95'][0]:.3f}, {gap['ci95'][1]:.3f}] |"
        )
    chosen = training["splits"]["random"]["winner_by_cv"]
    gap = next(g for g in evaluation["gaps"] if g["model"] == chosen)
    lines += [
        "",
        f"Training cross-validation chose **{chosen.replace('_', ' ')}** for the random "
        f"split and **{scaffold['winner_by_cv'].replace('_', ' ')}** for the scaffold split. "
        f"For the random split's chosen family, the scaffold-minus-random error gap is "
        f"**{gap['scaffold_minus_random_rmse']:+.3f}**. Positive means worse predictions "
        "on the scaffold holdout.",
        "",
        "These are fixed-split results, not the paper's scores. The intervals condition on "
        "the fitted models, and the gap compares different test populations with an unpaired "
        "bootstrap. With only two scaffold test families, uncertainty about completely new "
        "lipid chemistry is much larger than these intervals can measure. I cannot use this "
        "experiment to claim performance in animals, patients, or another cell type.",
    ]
    errors = pd.read_csv(root / "scaffold_errors.csv")
    chosen_errors = errors[
        (errors.split == "scaffold") & (errors.model == scaffold["winner_by_cv"])
    ]
    worst = chosen_errors.sort_values("rmse", ascending=False).iloc[0]
    direction = "underpredicts" if worst.bias < 0 else "overpredicts"
    lines += [
        "",
        f"The scaffold model's largest family error is on `{worst.scaffold}` "
        f"({int(worst.n)} lipids, RMSE {worst.rmse:.3f}). It {direction} this family by "
        f"{abs(worst.bias):.3f} target units on average. That systematic bias is hidden "
        "by a single overall score.",
    ]
    return "\n".join(lines)


def main():
    path = Path("README.md")
    start, end = "<!-- benchmark:start -->", "<!-- benchmark:end -->"
    original = path.read_text()
    before, rest = original.split(start)
    _, after = rest.split(end)
    path.write_text(
        before + start + "\n" + result_block(Path("results/reference")) + "\n" + end + after
    )


if __name__ == "__main__":
    main()
