# Reproduction workflow

## Hardware / environment

Original Python 3.13.5; see `requirements.txt` for recorded numeric package versions. Install into a clean Python 3.13 virtual environment. Full nested CV plus bootstraps is CPU intensive (may require hours). No fresh aeroacoustic measurements are required.

```bash
python -m venv .venv
# Activate .venv using your platform's standard command
python -m pip install -r requirements.txt
python tests/check_release.py
```

## Full numerical retraining (not required for quick checks)

From repo root, after installing requirements:

```bash
cd 57_H3_Airfoil_R5
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for feature in F4 F5; do
  python code/run_tsk.py --features "$feature" --mode group
  python code/run_tsk.py --features "$feature" --mode group --raw
  python code/run_tsk.py --features "$feature" --mode row
  for model in linear_ridge svr extra_trees xgboost; do
    python code/run_baselines.py --features "$feature" --model "$model" --mode group
    python code/run_baselines.py --features "$feature" --model "$model" --mode row
  done
done
python code/run_bootstrap.py --workers 4
python code/verify.py
python code/independent_qa.py
python code/aggregate.py
python code/run_sensitivities.py
python code/split_comparison.py
```

This creates the original expected `results/tsk`, `results/baselines` and `results/bootstrap` directories, including parent and bootstrap `.npz` files. Do not replace failed draws; both failures are documented in the archival results. No guarantee of cross-platform byte-identical output is made for numerical libraries.

## R6 and R9 extensions

**Only after full parent R5 regeneration or a separately approved complete parent-results snapshot**:

```bash
# from repository root
python R6_EXTENSION/analysis/run_r6_extension.py
python R9_EXTENSION/code/shared_support.py --root 57_H3_Airfoil_R5 --out R9_EXTENSION --workers 4
python R9_EXTENSION/code/independent_checks.py --root 57_H3_Airfoil_R5 --out R9_EXTENSION --r6 R6_EXTENSION
```

The archived R6 and R9 code retain original folder assumptions; staged R6 outputs already present under `results/r6` are evidence snapshots, not the writable output directory used by re-runs. Inspect source and tests before rerunning and retain failures.

## Evidence boundaries

`results/` in this slim repository includes benchmark and pair-level tables, not the full original 2,000+ file model/replicate tree. For exact frozen-archive reproduction, retain the full private archive (with restricted/copyrighted PDF content **excluded** from any public deposit) and deposit an authorized science-only archive on Zenodo with a DOI. Conditional group-bootstrap intervals are not universal model/population confidence intervals.
