#!/usr/bin/env python3
"""Quick public snapshot checks; no model fitting or manuscript build."""
from pathlib import Path
import sys
import subprocess
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/"scripts/restore_large_tables.py")],check=True,cwd=root)
sys.path.insert(0,str(root/'57_H3_Airfoil_R5'/'code'))
from verify import verify_math

checks=verify_math()
assert len(checks)==17, f"Expected 17 mathematical controls, found {len(checks)}"
assert all(checks.values()), checks

p=root/'results/r5/ALL_PRIMARY_OOF.csv'
t=root/'results/r5/TABLE_PRIMARY_BENCHMARK.csv'
a=pd.read_csv(p);ref=pd.read_csv(t)
assert a['row_id'].nunique()==1503, 'Expected 1,503 observation IDs'
assert len(ref)==20, 'Expected 20 primary result rows'
assert len(a)==30060, 'Expected one held-out prediction per model + feature combination'
for (feature,model),g in a.groupby(['feature_set','model']):
    assert len(g)==1503 and g.row_id.nunique()==1503 and g.group_id.nunique()==106
    score=np.sqrt(g.assign(err2=(g.y-g.pred)**2).groupby('group_id').err2.mean().mean())
    want=ref.query('feature_set==@feature and model==@model').equal_config_RMSE
    assert len(want)==1 and abs(score-float(want.iloc[0]))<1e-8, (feature,model,score,want.tolist())

b=pd.read_csv(root/'results/r6/R6_MATCHED_PAIRS_0p5.csv')
c=pd.read_csv(root/'results/r9/SHARED_SUPPORT_PAIRS.csv')
assert len(b)==50376 and len(c)==151128,(len(b),len(c))
assert len(c.query('cutoff == 0.1'))==50376
assert (c.common_rows<=c.parent_rows).all()
assert (c.common_groups<=c.parent_groups).all()
print('RELEASE_CHECKS: PASS — 17 math tests, 20 primary OOF metrics, R6/R9 paired-row coverage')
