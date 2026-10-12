# R19 external-transfer source-to-result publication note

**Status:** Verified research reproducibility snapshot; journal manuscript under author review. No Zenodo DOI or GitHub Release is asserted by this file.

**Added:** source-authenticated Southampton data (343 rows, 38 spectra, 7 geometry labels), nested leave-one-geometry-out TSK and Extra Trees experiments, corrected twelve-candidate tree selection, saved candidate tables/NPZ training states, explicit geometry/spectrum grouping checks, absolute-vs-relative Gaussian firing audit, feature-role fixed and retuned sensitivities, conditional slope protection report and independent numerical replay.

**Unchanged:** all pre-existing repository files, including NASA R5–R9 snapshots and the separately archived R18 manuscript baseline. No previously published result is replaced. The 2026-10-11 preliminary R19 tree predictions are superseded *within this extension* by the corrected fit, not silently overwritten in historical evidence.

**Protected publication boundary:** source-normalized target relies on experimental reference maxima. The evaluation is not an end-to-end absolute dB noise predictor for an unmeasured geometry. Extra Trees is more accurate than TSK in aggregate; weak absolute Gaussian activation can coexist with high normalized weights, but it does not itself certify errors. Seven geometry folds do not represent independent laboratories. Stability conclusions are model-, support-, and budget-conditional.

**Reproducibility:** run `python3 R19_EXTERNAL_TRANSFER/VERIFY_RELEASE.py` for a fast no-dependency source/record audit; unzip `R19_Southampton_ExternalTransfer_SourceToResult_Public.zip` and follow `R19_README.md` to execute the full model-fitting pipeline with the exact recorded environment. The fast audit checks fixed predictions; it is not independent model retraining.

**Licensing:** source datasets retain upstream attribution, and software licensing is not yet author-approved. Public GitHub visibility does not itself license reuse.
