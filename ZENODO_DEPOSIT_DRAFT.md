# Zenodo deposit record — DRAFT, NOT APPROVED OR PUBLISHED

This document is preparation metadata only. No Zenodo deposit has been made, no DOI has been reserved or assigned, and no public software-license grant has been approved.

## Proposed record metadata

- **Proposed title:** NACA 0012 Takagi–Sugeno Rule Reproducibility: Research Code and Result Tables for Airfoil Self-Noise
- **Resource type:** Software (research software and selected numeric result tables)
- **Proposed version:** v1.0.0 — not tagged or released; contingent on authors' rights and archive approval
- **Creators:** Complete author names, order, affiliations, ORCIDs and depositor authority **to be confirmed by the authors**. Do not infer creators from the GitHub username or commit metadata.
- **Publication year:** 2026 (only if deposited in 2026)
- **Keywords:** airfoil self-noise; NACA 0012; Takagi–Sugeno fuzzy regression; local-rule reproducibility; grouped cross-validation; bootstrap stability; common-activation support; research software
- **Related dataset:** Brooks, T., Pope, D., & Marcolini, M. (1989), Airfoil Self-Noise, UCI Machine Learning Repository, https://doi.org/10.24432/C5VW2C
- **Related manuscript:** Predictive Accuracy and Local-Rule Reproducibility in Takagi–Sugeno Models of Airfoil Self-Noise (journal submission/persistent identifier pending)
- **Code URL:** https://github.com/shaikhamalkawi-ux/naca0012-tsk-rule-reproducibility

## Proposed abstract / description

This research-software snapshot accompanies a study distinguishing held-out prediction error from stability of local consequent functions in first-order Takagi–Sugeno models of the UCI/NASA NACA 0012 airfoil self-noise benchmark. The dataset comprises 1,503 observations from 106 operating configurations. The repository contains source for grouped nested model comparison and local-rule resampling, baseline models, and reproducibility audits. It also includes complete, SHA-256-recorded R5 out-of-fold predictions, R6 matched rule-pair results, and R9 common-activation-support pair results. A lightweight independent checkout test checks source integrity, 17 deterministic mathematical controls, 20 out-of-fold benchmark metrics, and R6/R9 table coverage. **This snapshot does not include the large parent/bootstrap NPZ training-state archives and does not itself establish complete experimental retraining, independent external validation, or measured physical generalization.**

## Rights and release gates

1. The underlying UCI dataset is CC BY 4.0; preserve attribution and disclose the researcher-generated derived identifiers.
2. The research code is **not yet licensed**. Confirm institutional/coauthor ownership, consent, and an appropriate software license before creating a formal version tag or Zenodo deposit. Do not assign CC BY 4.0 to the entire repository by assumption.
3. Reconfirm publication-safe contents: no private parent/bootstrap archive, publisher PDFs, reviewer correspondence, secrets, or unapproved coauthor material.
4. Validate the exact intended tag against the CI record, Git tree and manifest.
5. Specify repository access statement, whether data/code are included or linked, and whether publisher permits preprint/author manuscript deposits.
6. Connect GitHub to Zenodo and review the preview **only after explicit authorization**. Publishing Zenodo requires a separate affirmative instruction.

## Journal-facing data and code statement — provisional

“The NACA 0012 Airfoil Self-Noise benchmark was obtained from the UCI Machine Learning Repository (DOI: 10.24432/C5VW2C; CC BY 4.0). Source code, selected derived data, complete R5/R6/R9 numeric result tables, and integrity checks are available at https://github.com/shaikhamalkawi-ux/naca0012-tsk-rule-reproducibility. The full training-state/bootstrap archive is not part of this public snapshot. A permanent DOI and applicable software license will be supplied once the archival deposit and rights review are complete.”

This statement must be rechecked after license selection and the final Zenodo record are approved.
