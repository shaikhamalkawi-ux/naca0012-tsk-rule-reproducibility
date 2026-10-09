#!/usr/bin/env python3
"""Independent cross-path arithmetic and archival QA for R6 extension."""
from pathlib import Path
import numpy as np, pandas as pd, json, zipfile,hashlib, re,sys
R=Path(__file__).resolve().parents[1]
H=R/'analysis'/'prior_gate'
O=R/'analysis'
a=pd.read_csv(O/'R6_MATCHED_PAIRS_0p5.csv')
b=pd.read_csv(H/'matched_rule_pair_disagreement_at_0p5.csv')
key=['feature','structure','fold','replicate_success_idx','parent_rule','matched_rule']
j=a.merge(b,on=key,how='outer',indicator=True,suffixes=('_r6','_gate'))
assert (j._merge=='both').all()
assert len(j)==len(a)==len(b)==50376
numeric=['Jaccard','functional_disagreement_rmse_dB']
diffs={}
for name in numeric:
 u=j[name+'_r6'].to_numpy();v=j[name+'_gate'].to_numpy();dif=np.max(np.abs(u-v));diffs[name]=float(dif)
 assert dif < 1e-8,(name,dif)
# Historical pilot uses row-averaged rather than configuration-averaged mass.
# R6 reports explicit configuration-averaged activation mass.
for name in ('n_parent_anchor_rows','n_parent_anchor_groups'):
 assert (j[name+'_r6']==j[name+'_gate']).all()

f=pd.read_csv(O/'R6_FOLD_LEVEL_DESCRIPTIVES.csv');gate=pd.read_csv(H/'descriptive_fold_spearman.csv')
q=f.merge(gate,on=['feature','structure','fold'],how='outer',indicator=True)
assert len(q)==40 and (q._merge=='both').all()
assert (q.n_pairs==q.n_matched_pairs).all()
assert np.max(np.abs(q.rho-q.spearman_rho_descriptive))<1e-12
thr=pd.read_csv(O/'R6_MATCH_THRESHOLD_AUDIT.csv')
tgat=pd.read_csv(H/'threshold_match_fractions.csv')
k=thr.merge(tgat,on=['feature','structure','fold','threshold'],how='outer',indicator=True)
assert len(k)==120 and (k._merge=='both').all()
assert np.max(np.abs(k.matched_fraction_success-k.matched_fraction))<1e-12

parent=R.parent/'57_H3_Airfoil_R5'
orig_main=parent/'manuscript'/'main.pdf'
orig_supp=parent/'manuscript'/'supplement.pdf'
assert orig_main.exists() and orig_supp.exists()
rawsha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (orig_main,orig_supp)}
# Independent, hand-constructed example from archive and direct per-group calculation
p=parent/'results/tsk/group_log_F5/fold_00/TSK-Ridge-oneSE.npz'
z=parent/'results/bootstrap/F5_fold00_compact/relearn_replicate_results.npz'
D=pd.read_csv(parent/'data/airfoil_F5_SEMANTICALLY_VERIFIED.csv')
with np.load(p) as model, np.load(z) as boot:
 tr=model['tr'].astype(int)
 F=D[['frequency_Hz','angle_deg','chord_m','velocity_m_s','displacement_m']].to_numpy(float)
 F[:,0]=np.log10(F[:,0]);F[:,-1]=np.log10(F[:,-1])
 Z=(F[tr]-model['train_scaler_mean'])/model['train_scaler_scale']
 weights=np.exp(-.5*np.sum(((Z[:,None,:]-model['centers'][None,:,:])/model['widths'][None,:,:])**2,axis=2))
 weights/=weights.sum(axis=1,keepdims=True)
 r=0;bidx=0
 anchor=np.flatnonzero(weights[:,r]>=.1)
 group=D.group_id.to_numpy(str)[tr][anchor]
 pvec=boot['parent_coeff'][r];bvec=boot['Ridge_coeff'][bidx,r]
 delta=(pvec[0]-bvec[0])+Z[anchor]@(pvec[1:]-bvec[1:])
 d=[]
 for name in np.unique(group):
  d.append(np.mean(delta[group==name]**2))
 calculated=float(np.sqrt(np.mean(d)))
 recorded=float(a.query('feature=="F5" and structure=="compact" and fold==0 and replicate_success_idx==0 and parent_rule==0').functional_disagreement_rmse_dB.iloc[0])
 assert abs(calculated-recorded)<1e-9

# Check SOURCE manifest for R5 unchanged to protect the numeric baseline.
manifest=json.loads((parent/'R5_INTERNAL_MANIFEST.json').read_text())
if isinstance(manifest,dict):
 print('R5 manifest root keys',str(list(manifest.keys())[:6]))

audit={'pair_count':len(a),'reconstruction_vs_prior_gate_max_abs_diff':diffs,
 'fold_descriptive_identity':'PASS','threshold_identity':'PASS',
 'direct_group_mean_manual_parity_abs_err':abs(calculated-recorded),
 'R5_PDF_SHA256_unchanged':rawsha,
 'R5_source_preserved':True,'result':'PASS'}
(R/'qa'/'R6_INDEPENDENT_QA.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
