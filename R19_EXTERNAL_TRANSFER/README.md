# R19: Held-out aerofoil geometry and local-rule audit (frozen source-to-result snapshot)

**Manuscript working title:** *Local-Statement Robustness in Takagi–Sugeno Airfoil Self-Noise Regression: A Finite-Input Audit*.

This package adds a **separate R19 Southampton external-transfer extension** to the earlier NASA NACA0012 R5–R9 reproducibility repository. It does not overwrite or recompute the locked R18 NASA conclusions. The R19 results are **source-normalized spectral-shape** predictions, **not absolute noise predictions in dB** and not calibration-free deployment on a novel geometry. All 7 geometries and 38 spectra come from one Southampton archive, rather than seven independent campaigns.

## Fixed numerical endpoints

The Southampton archive contains **343 frequency rows / 38 spectra / 7 geometries**. All spectra of the held-out geometry are excluded from model fitting and 4-fold *spectrum-grouped* hyperparameter selection. Training data alone determine scaling, clustering and parameter selection.

| Metric on dimensionless normalized response | TSK | Extra Trees |
|---|---:|---:|
| Equal-geometry RMSE | 0.0593961028381699 | 0.04783880803883461 |
| Equal-spectrum RMSE | 0.05636793120717745 | 0.04322684449353849 |
| Row RMSE | 0.055491916066947145 | 0.04153805373018892 |

The corrected run uses all **12** source-declared Extra Trees configurations per held-out geometry, instead of the **3** saved in the inconsistent preliminary candidate CSV files; 45 TSK candidates are considered per outer held-out geometry. Independent replay was completed. R19 is a scientific extension; it does not supersede R18 NASA results.

## Run from source

Python versions and pinned scientific-library versions are recorded in `R19_Executable_Reproduction/requirements_R19.txt`. From the extracted package root:

```bash
python -m pip install -r R19_Executable_Reproduction/requirements_R19.txt
cd R19_Executable_Reproduction
bash run_full_reproduction.sh
```

The pipeline re-extracts the canonical data from the Southampton `DataFiles.zip`, checks byte identity against the frozen 343-row CSV, fits the corrected seven LOGO models, reconstructs saved states, runs independent numerical checks, computes activation/coverage diagnostics, feature-role sensitivity (both fixed and retuned), and local frequency-slope certificates. A full rerun performs fitting and may require substantial CPU time. The top-level archive also holds the **previously executed** source, predictions, training/test partitions, candidate tables and NPZ states for inspection without refitting.

**Do not run** the archived `57_H3_Airfoil_R19_LOGO_Candidate/run_logo_external.py` as a standalone script: its `main()` contains historical machine-specific paths; it is included solely for the originally registered fitting functions imported by the portable `run_logo_external_clean.py` wrapper and to document why the preliminary outputs were rejected.

## Integrity and quick audit

The compact `VERIFY_RELEASE.py` in the GitHub directory checks ZIP CRC, every packaged SHA-256 payload, row and grouping counts, all 7 × 12 tree candidate tables, all 7 × 45 TSK candidate tables, and independently recomputed row/spectrum/geometry RMSE from out-of-fold predictions using only the Python standard library. It **does not** claim to rerun model fitting.

```bash
python R19_EXTERNAL_TRANSFER/VERIFY_RELEASE.py
```

Detailed `INDEPENDENT_REPLAY_QA.json` and `SPLIT_AUDIT.json` are retained inside the ZIP.

## Scientific boundaries

* Extra Trees performs better than TSK on both aggregate held-out normalized target scores.
* Normalized fuzzy firing weights can be high when the absolute best Gaussian activation is extremely small: 30/343 rows fell below a fold-specific train-only fifth-percentile cutoff; 10/30 had maximum normalized weights at least 0.9. This is **not** a calibrated error-prediction rule.
* The Southampton trip-feature antecedent omission is sensitive to retuning and aggregation. It is **not** a replicated NASA velocity effect.
* 18/18 frozen R19 LOGO training-model local frequency slopes were conditionally protected within the specified finite-input and 1%-ridge-objective budgets. Protection does not certify correctness of the aerodynamic mechanism.
* R19 has not established causality, absolute-SPL prediction, or universal geometry generalization.
* This snapshot is a research-software release candidate, **not a published journal manuscript, journal acceptance, Zenodo DOI, or blanket code license**.

## Source and attribution

Southampton data: Lau and Kim, [DOI:10.5258/SOTON/398169](https://doi.org/10.5258/SOTON/398169); companion article Lau et al., [DOI:10.1002/we.2003](https://doi.org/10.1002/we.2003). The data archive carries the attribution statements noted by the source research. Keep the source attribution and avoid treating published source target normalization as independently known operational calibration.

The earlier NASA dataset and the current R5–R9 results remain in the repository root. This R19 package contains no private identities, credentials or unpublished author-review manuscript drafts. No third-party license is granted to software by this README; confirm an explicit software license with the author before reuse beyond GitHub's platform terms.
