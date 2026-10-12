#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python R17_Southampton_External_Reproducibility/code/extract_southampton.py
python - <<'PYEND'
from pathlib import Path
import hashlib
p=Path('R17_Southampton_External_Reproducibility/data')
assert hashlib.sha256((p/'SOUTHAMPTON_CANONICAL_REEXTRACTED.csv').read_bytes()).hexdigest()==hashlib.sha256((p/'SOUTHAMPTON_CANONICAL.csv').read_bytes()).hexdigest(), 'Source re-extract differs from locked canonical CSV'
print('Source extraction byte-replay PASS')
PYEND
python run_logo_external_clean.py
python independent_verify_clean.py
python coverage_audit_clean.py
python feature_roles_clean.py
python feature_role_retune.py
python logo_statement_audit_clean.py
# manuscript compile is OPTIONAL: from R19 research root run 'python build_r19_manuscript.py', then pdflatex twice in R19_Manuscript_Source_ReviewCandidate.
echo 'R19 source -> models -> states -> feature/coverage/statement audit PASS'
