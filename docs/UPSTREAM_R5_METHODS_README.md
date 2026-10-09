# 57 H3 — NASA Airfoil Self-Noise: R5 Independent Reconstruction

**Version:** R5, 8 October 2026. **Scientific status:** complete, reproducible *within-benchmark methodological research draft*; not journal-submitted, and not physically/external-airfoil validated.

## Relationship to prior work

R5 was independently rebuilt from the preserved method specification and semantically verified NASA/UCI numerical data after the previously cited R4 calculation directory became unavailable. **R5 is not a restoration, replication or certified replay of R4.** Earlier Stage A outputs are retained as historical records but **are not the numeric source for this manuscript**. Differences between Stage A and R5 are not presented as a causal methodological contrast because the full Stage A executable state was unavailable. `audit/experiment_spec_v2.json` is the original unmodified plan. `audit/experiment_spec_R5_EFFECTIVE.json` describes what was executed.

## Data provenance and admission limits

The numerical table `data/airfoil_F5_SEMANTICALLY_VERIFIED.csv` has 1,503 frequency-specific observations, 106 unique operating configurations defined by `(angle_deg, chord_m, velocity_m_s)`, and five source inputs plus target. Its values were checked semantically against a public copy of the UCI dataset and the UCI repository's count, columns and relevant aggregate values. **The original official UCI archive bytes were not acquired and no exact original-release hash match is claimed.** Dataset credit: Brooks, Pope and Marcolini, NASA Airfoil Self-Noise, UCI dataset 291, DOI https://doi.org/10.24432/C5VW2C, CC BY 4.0. Preserve the citation and license attribution in redistributed data.

## Implemented experimental design

- F4: frequency, angle, chord and velocity; F5 adds observed suction-side displacement thickness.
- Primary transforms: `log10(frequency)`, plus `log10(displacement)` in F5; training-only scaling; an original-numeric-representation sensitivity is kept separate.
- Primary validation: outer ten-fold **GroupKFold** across 106 operating configurations, inner five-fold group CV for tuning; no rows from a held-out configuration enter its training fold.
- TSK: input-only FCM (fuzzifier m=2), R in {2,4,6,8,10,12}, Gaussian widths h in {0.75,1,1.5}, five restarts seeded 11/23/47/71/101; compact one-SE and minimum-error selection. Paired ordinary least squares and ridge consequents. Primary FCM max iterations 2,000 and tolerance 1e-6, an openly documented amendment from v2 max 500 after a convergence problem; **not preregistered**.
- Tuned controls: linear/ridge regression, RBF-SVR, Extra Trees and XGBoost; row-wise splits reported only as a *different prediction question*, not as primary evidence of generalization.
- Stability: F4/F5 × ten outer folds × compact/fixed-R6 structures × (500 fixed-rule + 200 relearned-rule) configuration-bootstrap draws. LS and Ridge both evaluated per successful draw, with dummy-enabled weighted-Jaccard/Hungarian rule alignment.
- Conditional paired intervals: 2,000 resamplings of fixed outer-OOF per-configuration losses, **not retraining-based population confidence intervals**. No independent-fold p-values.

## Principal results (equal-configuration held-out RMSE, dB)

| Model | F4 | F5 |
|---|---:|---:|
| Extra Trees | 2.687370 | 2.381163 |
| XGBoost | 2.705073 | 2.648407 |
| Compact TSK-Ridge | 3.949870 | 4.166475 |

Bootstrap **28,000 attempted**, **27,998 successful**, two failed five-start FCM convergence (recorded in `results/BOOTSTRAP_FAILED_REPLICATES.csv`, never replaced).

The paper asks about prediction reliability versus rule stability. It does NOT introduce ANFIS training, a new identification theorem, a fuzzy-valued data model, or a universal physical interpretation of TSK coefficients. Results apply only to this dataset, candidate structures and training conditions.

## Verification and file map

- `manuscript/main.pdf`, `manuscript/supplement.pdf`: finished journal-neutral scientific drafts; LaTeX source and figures available alongside.
- `results/TABLE_PRIMARY_BENCHMARK.csv`, `results/TABLE_BOOTSTRAP_SUMMARY.csv`, `results/ALL_PRIMARY_OOF.csv`: computed and independently checked data behind the manuscript.
- `results/tsk/`, `results/baselines/`: complete candidate-level inner-CV logs, full outer predictions, selected fold models and metadata; includes row-wise and raw-scale sensitivity runs.
- `results/bootstrap/`: all 40 tasks with per-replicate outcome registers, predictions, rule matching, and failure logs.
- `code/`: full fitting, statistics, figures, and manuscript generation scripts.
- `audit/TEST_REPORT.json`: 17/17 mathematical and coverage checks.
- `audit/INDEPENDENT_REPLAY_QA.json`: 13/13 independent replay checks; max observed prediction difference 7.544e-6 for float32 XGBoost replay.
- `audit/FINAL_QA_R5.json`: final count, PDF and manifest QA metadata.
- `RUN_FROM_SCRATCH.sh` and `REBUILD_REPORTS_FROM_STORED_OUTPUTS.sh`: required dependencies and execution order; complete retraining is compute intensive and may take hours.

## Release and venue stage

Before submission: author-confirmed names/order/affiliations/email, conflict and funding declarations, CRediT, data-redistribution permissions/attribution and institutional approvals; choose a specific venue/template; deposit archive and obtain DOI if required. The current draft is **not yet journal-submission-compliant** without these author-side declarations. Original UCI archive byte-level provenance remains an explicitly disclosed limitation; nothing here is represented as third-party independent physical validation.
