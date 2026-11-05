# What this experiment measures

The input is an ionizable lipid's molecular structure. The output is the numeric HeLa transfection `Target` supplied by LANTERN. A lipid nanoparticle carries RNA into cells; an ionizable lipid changes charge with its surroundings and helps that delivery. Transfection is the delivery readout in this cell experiment. This is a structure-only model of that assay, not a clinical prediction or a model of every LNP formulation.

## Data provenance and scope

`lnp-benchmark download` fetches `data/AGILE.csv` from LANTERN commit `11240f29ef92323649ae60d177b21df77e2d428b` and checks SHA256 `d0ecfd94ec40a7f013af546113520175e70f78045af1798673c2680fe6f23c54`. The file has 1,100 rows. Despite the filename, this is LANTERN's curated table, not the uncorrected AGILE table. I do not substitute synthetic measurements, reconstruct labels, rescale the published target column, or silently drop rows. Negative targets are retained. Scores are in the source `Target` units, not percentage points or raw fluorescence units.

Validation fails on invalid/empty structures, nonfinite targets, or duplicate canonical isomeric SMILES. Canonicalization preserves stereochemistry. The downloader creates `data/provenance.json`. Public measurements are downloaded rather than bundled. Upstream licensing and attribution are in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

This is a reproduction of the **task and evaluation idea**, with an independently implemented, smaller baseline suite. I do not reproduce the paper's exact hyperparameter budget, supplied splits, exhaustive descriptor panel, pretrained graph embeddings, or reported performance. It is not an independent validation of the authors' data corrections. The original experimental context is HeLa, a laboratory cell line; there is no animal or human validation here.

## Inputs

RDKit constructs radius-2, chirality-aware Morgan **count** fingerprints with 1,024 hashed positions, plus 12 explicitly named descriptors. Morgan counts summarize atom neighborhoods, with possible hash collisions. The descriptors measure molecular mass, estimated oil/water preference (LogP), polar surface area (TPSA), hydrogen bonding, rotatable bonds, rings, carbon hybridization, heavy atoms, formal charge and heteroatom groups. The full names and implementations are in `features.py`. No target labels enter featurization.

The MLP standardizes its input and output inside each training fold. That keeps validation values out of the scaling calculation. Tree models receive the same unscaled columns. No entire-dataset normalization, feature selection, or label-derived filtering occurs before splitting.

## Splits and tuning

The random split is an 80/20 row split with seed 42. Canonically identical structures cannot occur on both sides because they are rejected before splitting. Related molecules can still share a scaffold across this split, which is the point of the comparison.

For the scaffold split, RDKit extracts the Bemis-Murcko ring-and-linker skeleton, ignoring scaffold chirality. All acyclic lipids share the single `ACYCLIC` group. I sort groups largest first, break ties lexically, and put a whole group in training if it fits the 80% capacity; otherwise it goes in test. I never give each acyclic molecule its own artificial scaffold. In other datasets the requested fraction is only approximate, and impossible splits fail explicitly.

This dataset has six families, including the 440-member acyclic family. Scaffold training contains four families; test contains the morpholine-like `C1COCCN1` and pyrazole-like `c1cn[nH]c1` families, with 110 rows each. This is one particular extrapolation task, not an average over unseen lipid chemistry. Scaffold identity also reflects only the ring skeleton, not every meaningful head-group or tail difference.

Each split has its own three-fold training-only cross-validation. Random training uses shuffled KFold; scaffold training uses GroupKFold, keeping scaffold families out of the validation folds in which they were trained. With so few families the folds can differ greatly in size. Models are refit on all outer training rows after selection. The test data never chooses model families, hyperparameters, scaling or stopping time. `winner_by_cv` records the model selected by mean validation RMSE; test results for all models are retained, including the mean-prediction baseline.

The full grid is in `models.py`: random forest tunes feature fraction and minimum leaf size; histogram gradient boosting tunes leaf count and L2 regularization; a small one-hidden-layer MLP tunes hidden width and L2 regularization. Each trainable family searches four configurations. MLP early stopping is disabled so it cannot introduce a hidden random validation split into scaffold training. Fit warnings are counted in `training.json` and shown in the report. `--quick` cuts the search and iteration budget for CI and is explicitly marked as a smoke test.

## Metrics and uncertainty

RMSE (root mean squared error) emphasizes large errors. MAE is average absolute error. R² compares predictions with the mean of the **test** targets and can be negative. Spearman measures ranking agreement. A separate mean regressor learns its constant from training targets only. Comparing it with the learned models shows whether molecular features help on this holdout.

The evaluator resamples entire test scaffold families with replacement 2,000 times and reports percentile 95% intervals for RMSE, MAE and R². It uses cluster rather than independent-row bootstrap because molecules in a family are related. Bootstrap samples retain all rows in the sampled family, including repeated copies when a family is drawn twice. Scores are row-weighted, not family-weighted.

The gap is scaffold RMSE minus random RMSE for the **same model family** after its separate training-only tuning. Positive means worse scaffold performance. Its interval subtracts independently generated bootstrap RMSE draws for the two holdouts. The test sets can share molecules, so this is a descriptive unpaired uncertainty calculation, not a paired significance test. Differences include changed test populations and changed training data as well as scaffold novelty; they are not an isolated causal effect of removing shared scaffolds.

These intervals condition on the fixed splits and fitted models. They exclude training-seed variation, alternative scaffold assignments, measurement uncertainty and transfer to other studies. With only two scaffold test families the cluster bootstrap has very limited support and cannot justify broad claims about new chemistry. I leave this limitation visible rather than presenting tight row-bootstrap intervals. No split seed is selected by test performance.

## Error analysis and importance

`scaffold_errors.csv` lists every test family for every model, with count, mean target, mean prediction, signed bias, RMSE, MAE and R². `scaffold_plot_key.csv` links the plotted S labels to full scaffold strings. A positive bias means overprediction. Errors for small families are noisy; the random split includes smaller groups than the scaffold holdout.

For the model chosen by training CV in each split, permutation importance measures how much held-out RMSE increases after shuffling an input. All Morgan columns are shuffled together as a block, preserving the internal fingerprint pattern; each descriptor is shuffled individually. Ten repeats produce mean effects and repeat standard deviations. A large block has more chances to carry information than one descriptor. Correlated descriptors can substitute for one another, and importance is an association, not a chemical mechanism or proof of causation. Test-set importance is post-hoc analysis and does not feed back into tuning.

## Reproduction and artifacts

Use the commands in the README. `uv.lock` pins the full dependency graph for Python 3.11/3.12; direct scientific dependencies are also pinned in `pyproject.toml`. Training uses one numerical thread and one job per fit to keep laptop resource use bounded. Exact floating-point results can vary across operating systems and numerical libraries. The checked-in reference run records the Python, RDKit, scikit-learn and NumPy versions, seed, feature configuration, all grid scores, exact outer partitions and inner fold assignments. The generated source manifest hashes package code and the dependency lock.

`features.npz` and fitted `.joblib` models stay local; rerunning reconstructs them. Only load model artifacts created by this pipeline in a trusted directory. Small predictions, split assignments, tuning results, figures and JSON summaries are committed under `results/reference/` so results can be audited without retraining. `scripts/check_reference.py` recomputes saved metrics from per-row predictions and validates partitions against the real public dataset. CI also runs all five CLI stages on that public dataset with a reduced training budget. Its smoke figures are uploaded separately and never overwrite the full reference results.

On this Mac, the numerical backend emitted repeated `matmul` runtime warnings while producing finite MLP predictions. The training record counts them without storing large diagnostic buffers, and the evaluation JSON lists unique warnings. [NumPy has documented similar macOS reports](https://github.com/numpy/numpy/issues/28687). I retain these warnings and check all targets and predictions for finite values; Linux CI provides a separate execution check. A successful run is not proof of a numerical-backend bug fix.
