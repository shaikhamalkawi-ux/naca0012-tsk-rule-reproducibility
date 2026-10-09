# NACA 0012 — TSK rule reproducibility

Research software and numerical evidence for **Predictive Accuracy and Local-Rule Reproducibility in Takagi–Sugeno Models of Airfoil Self-Noise** (manuscript under author/venue preparation).

> **PUBLIC RESEARCH-SOFTWARE SNAPSHOT (2026-10-09).** Uploaded at the account holder’s express request. This snapshot is not a journal submission, does not constitute a verified complete archival rerun, and has no Zenodo DOI or software reuse license at this stage. The data originate from the NASA/UCI Airfoil Self-Noise benchmark; no independent physical validation or generalization to other airfoil geometries is claimed.

## Purpose

The study compares held-out acoustic predictions with the reproducibility of cluster-initialized, first-order Takagi–Sugeno (TSK) regression rules under complete operating-configuration resampling. **It is not a newly trained ANFIS architecture, a new TSK optimization algorithm, FFLS, or a physics-informed neural network.**

- 1,503 frequency observations; 106 configurations identified by attack angle, chord length, and free-stream velocity.
- Four-input F4 and five-input F5 analyses.
- Outer 10-fold grouped CV and inner 5-fold grouped model selection.
- Baselines: linear/ridge, RBF-SVR, Extra Trees and XGBoost.
- 28,000 total bootstrap attempts in the executed parent study; 27,998 succeeded and two nonconverged FCM fits were retained as failures.
- Relearning stability and common-activation-support sensitivity are **post hoc** analyses; no independence claims are attached to individual bootstrap pairs.

Reference group-weighted RMSE values (dB, lower is better):

| Method | F4 | F5 |
| --- | ---: | ---: |
| Extra Trees | 2.687 | 2.381 |
| XGBoost | 2.705 | 2.648 |
| One-SE TSK-Ridge | 3.950 | 4.166 |

The TSK system is **not** the lowest-error predictor in this benchmark. Rule reproducibility, rather than predictive superiority, is the research question.

## Source layout

- `57_H3_Airfoil_R5/`: original executable training and resampling code and the numerically checked derived input CSV (filenames preserved for path compatibility).
- `R6_EXTENSION/analysis/`: after-the-fact similarity versus local-function analysis.
- `R9_EXTENSION/code/`: common-activation-support sensitivity and checks.
- `results/`: selected original R5, R6 and R9 result tables and pair-level numeric outputs. This is an **analysis snapshot, not the entire model/replicate archive**.
- `audits/`: recorded execution environment and checks from the archived computational run.
- `tests/`: portable source-data, math and result-table consistency checks for the release candidate.
- `docs/`: provenance, rights, conditional-inference limitations and execution instructions.

## Reproduction

Python 3.13.5 and package versions from `requirements.txt` reproduce the recorded environment. For a quick check after installation:

```bash
python tests/check_release.py
```

For the full resource-intensive pipeline, see [REPRODUCE.md](REPRODUCE.md). To rerun the post hoc R6/R9 extensions, the parent R5 bootstrap tensor/NPZ artifacts must first be regenerated or retrieved from a separately authorized and versioned research archive; the compact GitHub release deliberately does **not** carry those large tensors. Do not describe the compact release as a byte-for-byte full-archive reproduction.

## Dataset citation and license

Brooks, T., Pope, D., & Marcolini, M. (1989). *Airfoil Self-Noise* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW2C. The UCI dataset is **CC BY 4.0**. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for attribution and the explicit source verification limitation.

## Publication status and software rights

No Zenodo DOI has been issued for this candidate. No software license has been selected by the authors, and no right to redistribute publisher PDFs is claimed. See [RELEASE_APPROVAL.md](RELEASE_APPROVAL.md) for publication status and remaining release requirements. These materials are not submitted to a journal through GitHub merely by being uploaded.
