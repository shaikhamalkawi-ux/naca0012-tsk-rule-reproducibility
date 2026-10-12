# R19 provenance, licenses, and claim limits

* Original external data: A. S. H. Lau and J. W. Kim, University of Southampton dataset, DOI 10.5258/SOTON/398169. Related wind-energy study: DOI 10.1002/we.2003. Preserve source attribution. The normalized responses are experimental-source representations, not absolute dB levels.
* This archive retains original `DataFiles.zip` and deterministic parser; its source SHA-256 must match the recorded manifest. Model-prediction source columns are excluded by the parser.
* NASA NACA0012 upstream data: UCI Airfoil Self-Noise, DOI 10.24432/C5VW2C, CC BY 4.0; primary results are in the prior repository snapshot and unchanged by this upload.
* Software: no explicit license decision was received; do not assert MIT, Apache, GPL, CC0 or another permission grant. Public repository accessibility by itself is not an open-source license.
* Duplicated rows within spectra and overlapping geometry-training partitions are dependent. Seven geometries from one experimental archive do not provide seven independent laboratories.
* In LOGO, the response is source-normalized using experimental maxima that are not necessarily available when deploying to a truly new geometry; validation is on the reported normalized target only.
* All numerical stability/protection claims are conditional on the specified fitting basis, finite inputs, and budgets. They are not physical laws or deployment guarantees.
* R19 TitleFocus IEEE Access manuscript remains under author review; no journal acceptance or repository DOI is claimed.
