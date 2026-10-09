#!/usr/bin/env python3
"""Independent read-only check of recovered antecedents and shared-support metrics.
No call to the new metric routine; recomputes using direct per-row weights.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,argparse
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr

def weights(groups):
 u,ix,ct=np.unique(groups,return_inverse=True,return_counts=True)
 return 1/(len(u)*ct[ix])

def gaussian(z,c,s):
 logits=-.5*(((z[:,None,:]-c[None,:,:])/s[None,:,:])**2).sum(2)
 return np.exp(logits-logsumexp(logits,axis=1)[:,None])

def main(root,out,r6):
 checks={};errors={};manifest=json.loads((root/'R5_INTERNAL_MANIFEST.json').read_text())['files']
 wrong=[n for n,rec in manifest.items() if hashlib.sha256((root/n).read_bytes()).hexdigest()!=rec['sha256']]
 checks['R5_payload_2101_immutable']=len(manifest)==2101 and not wrong
 data=pd.read_csv(root/'data/airfoil_F5_SEMANTICALLY_VERIFIED.csv');groups=data.group_id.to_numpy();allpairs=pd.read_csv(out/'analysis/SHARED_SUPPORT_PAIRS.csv')
 orig=pd.read_csv(r6/'analysis/R6_MATCHED_PAIRS_0p5.csv')
 count=0;max_parent=0.;max_shared=0.;max_J=0.;tally=[]
 for fs in ('F4','F5'):
  cols=['frequency_Hz','angle_deg','chord_m','velocity_m_s']+(['displacement_m'] if fs=='F5' else [])
  F=data[cols].to_numpy().copy();F[:,0]=np.log10(F[:,0])
  if fs=='F5':F[:,-1]=np.log10(F[:,-1])
  for structure in ('compact','reference'):
   for fold in range(10):
    tdir=out/'analysis/tasks'/f'{fs}_fold{fold:02d}_{structure}';table=pd.read_csv(tdir/'pairs.csv')
    key='TSK-Ridge-oneSE' if structure=='compact' else 'TSK-Ridge-fixedR6h1'
    with np.load(root/'results/tsk'/f'group_log_{fs}'/f'fold_{fold:02d}'/(key+'.npz')) as par,np.load(root/'results/bootstrap'/f'{fs}_fold{fold:02d}_{structure}'/'relearn_replicate_results.npz') as old,np.load(tdir/'replayed_antecedents.npz') as new:
     tr=par['tr'];gg=groups[tr];zp=(F[tr]-par['train_scaler_mean'])/par['train_scaler_scale'];wp=gaussian(zp,par['centers'],par['widths']);aw=weights(gg)
     for b,t in table.groupby('success_index'):
      b=int(b);zb=(F[tr]-new['mean'][b])/new['scale'][b];wb=gaussian(zb,new['centers'][b],new['widths'][b])
      for row in t.itertuples():
       r=int(row.parent_rule);s=int(row.matched_rule);cut=row.cutoff
       expected_j=(aw*np.minimum(wp[:,r],wb[:,s])).sum()/(aw*np.maximum(wp[:,r],wb[:,s])).sum()
       max_J=max(max_J,abs(expected_j-row.Jaccard))
       difference=(old['parent_coeff'][r,0]-old['Ridge_coeff'][b,r,0])+zp@(old['parent_coeff'][r,1:]-old['Ridge_coeff'][b,r,1:])
       a=wp[:,r]>=cut;c=a&(wb[:,s]>=cut)
       assert int(a.sum())==row.parent_rows and int(c.sum())==row.common_rows
       assert not np.any(c&~a)
       for mask,val,which in [(a,row.D_parent_dB,'parent'),(c,row.D_common_dB,'common')]:
        if mask.any():
         direct=np.sqrt(np.dot(weights(gg[mask]),difference[mask]**2));delta=abs(float(direct)-val)
         if which=='parent':max_parent=max(max_parent,delta)
         else:max_shared=max(max_shared,delta)
        else:assert pd.isna(val)
       count+=1
    tally.append(dict(feature=fs,structure=structure,fold=fold,n_checked=len(table)))
    print('CHECK',fs,structure,fold,len(table),flush=True)
 checks['all_three_cutoffs_direct_formula']=count==151128 and max_parent<1e-8 and max_shared<1e-8
 checks['all_matched_Jaccard_direct_formula']=max_J<1e-8
 errors.update(max_D_parent_error=max_parent,max_D_common_error=max_shared,max_J_error=max_J)
 key=['feature','structure','fold','replicate','parent_rule','matched_rule']
 merged=allpairs.query('cutoff==0.1').merge(orig,on=key,validate='one_to_one',suffixes=('_r9','_r6'))
 delta=np.abs(merged.D_parent_dB-merged.functional_disagreement_rmse_dB)
 checks['R6_50376_original_pair_metrics_preserved']=len(merged)==50376 and float(delta.max())<1e-8
 errors['R6_max_D_difference']=float(delta.max())
 primary=pd.read_csv(root/'results/ALL_PRIMARY_OOF.csv');summary=pd.read_csv(root/'results/TABLE_PRIMARY_BENCHMARK.csv');max_pred=0.;n=0
 for (f,m),t in primary.groupby(['feature_set','model']):
  assert len(t)==1503 and t.row_id.nunique()==1503 and t.group_id.nunique()==106
  e=t.assign(square=(t.y-t.pred)**2).groupby('group_id').square.mean()
  value=float(np.sqrt(e.mean()));expected=float(summary.query('feature_set==@f and model==@m').equal_config_RMSE.iloc[0]);max_pred=max(max_pred,abs(value-expected));n+=1
 checks['all_20_prediction_metrics_recomputed']=n==20 and max_pred<1e-10;errors['max_primary_RMSE_error']=max_pred
 checks['new_pairs_no_negative_disagreements']=(allpairs[['D_parent_dB','D_common_dB']].dropna()>=0).all().all()
 checks['common_support_never_larger']=(allpairs.common_rows<=allpairs.parent_rows).all() and (allpairs.common_groups<=allpairs.parent_groups).all()
 checks['empty_intersections_NA_not_zero']=allpairs.loc[allpairs.common_rows==0,'D_common_dB'].isna().all()
 rs=json.loads((out/'audit/REPLAY_STATUS.json').read_text());checks['8000_attempts_7998_success_two_failures']=rs['n_attempts']==8000 and rs['successes']==7998 and rs['failed']==2
 checks['forty_original_seed_replays_identical']=rs['max_similarity_diff']==0 and rs['n_tasks']==40
 report={'checks':{k:bool(v) for k,v in checks.items()},'pass_count':sum(bool(v) for v in checks.values()),'test_count':len(checks),'formula_rows':count,'errors':errors,'note':'Independent code-path verification, not independent researchers or new experimental replication.'}
 (out/'audit/INDEPENDENT_SHARED_SUPPORT_QA.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));assert all(checks.values())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--r6',required=True,type=Path);a=p.parse_args();main(a.root,a.out,a.r6)
