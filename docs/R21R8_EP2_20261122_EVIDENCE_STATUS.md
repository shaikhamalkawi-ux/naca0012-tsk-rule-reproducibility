# R21R8-EP2 — Scientific evidence status (22 November 2026)

**Study:** Local-Statement Robustness in Takagi–Sugeno Airfoil Self-Noise Regression: A Finite-Input Audit.  
**Status:** Author-review manuscript / IEEE Access resubmission candidate, **not accepted, resubmitted, or published**. This public note is an update to scientific provenance; it does not replace the earlier released code/data snapshot, publish a controlled archive, or grant a software license.

## Result and bounded claim

The pointwise finite-training-input constraint tests changes in predictions of a *frozen-antecedent TSK-ridge* system; the second limit restricts increase in the **original penalized ridge objective**. The units are not interchangeable.

| Pointwise training-prediction budget | Ridge objective increase | Protected signs | Verified strict sign reversals | Slopes |
| --- | --- | ---: | ---: | ---: |
| **0.1 dB — original, unchanged** | 1% | 403 | 107 | 510 |
| **1.0 dB — post-review contextual sensitivity** | 1% | 135 | 375 | 510 |

At 1.0 dB, the controlled independent audit checked 375 full coefficient-displacement witnesses and 135 protection certificates. Among protected signs, 110 are supported by an objective-only lower bound; 25 need joint constraints. This remains a **conditional numerical test**, not a physical law, uncertainty distribution, or population rate.

The motivation for examining 1.0 dB is [NASA Reference Publication 1218 (1989), Appendix A, printed p. 104](https://ntrs.nasa.gov/citations/19890016302): processed spectra derived independently from two microphone pairs generally agreed within 1 dB after shear-layer and directivity corrections. **That statement is not a calibrated ±1 dB uncertainty for each UCI measurement.** The primary 0.1-dB result was not replaced.

## Replay and source audit

- Frozen R5 code and recorded seeds were used to **newly rerun** all **40/40 bootstrap tasks**, comprising **28,000 attempts**, **27,998 successful refits**, and the original two FCM convergence failures at `F5_fold00_compact/relearn/141` and `F5_fold02_reference/relearn/115`.
- An independent source-to-replayed-output audit passed the saved numerical criteria (`atol=1e-7`, `rtol=1e-9`): maximal absolute prediction difference **1.2574×10⁻¹⁰**, coefficient difference **1.4698×10⁻⁸**, similarity difference **1.8486×10⁻¹⁴**.
- **27,998 newly recomputed internal fit states** were retained in a separate **author-controlled 40-archive collection** (773,361,055 bytes). They are **not historical October 2026 state-file bytes** and are **not contained in this GitHub snapshot**.
- A *current* official UCI raw `airfoil_self_noise.dat` of 1,503×6 values was checked against the inherited R5 derived CSV: all **9,018/9,018** numeric cells agreed within `1e-12`. Its current source SHA-256 is `74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3`. **The original October acquisition-byte hash remains unknown.**

## Held-out prediction and other sources

The selected F5 one-SE TSK-ridge produces equal-configuration RMSE **4.166 dB**, while Extra Trees produces **2.381 dB**. TSK is not claimed to be the most accurate regressor. Separate Southampton LOGO results are on a **dimensionless, experimentally normalized** response, with equal-geometry RMSE **0.059396** (TSK) and **0.047839** (Extra Trees), not dB and not absolute-noise predictions for previously unseen geometries.

## Availability, rights and Zenodo

This repository is a **historical, limited public snapshot**. This status page contains findings, not new controlled model states, the current manuscript/reviewer correspondence, third-party publisher PDFs, or unapproved file-level release rights. No repository-wide `LICENSE` has been approved; upstream UCI CC BY 4.0 does not grant a license for original author-written software.

A pre-existing **unpublished Zenodo draft** was prepared against an older research revision. It must be revised to reflect the actual deposited file contents, authors and rights before public publication. **No published project/article DOI is claimed.** Depositing an approved software/evidence archive before article acceptance may be possible, but DOI registration, file release, rights and the author list must be checked independently.

**Scope boundary:** This entry records the independently audited author-review evidence. To reproduce all 28,000 refits afresh, the archived R5 inputs, fixed seeds, software environment and the separately controlled replay kit are required; `tests/check_release.py` verifies the public historical snapshot and is not an end-to-end fresh retraining test.
