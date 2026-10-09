#!/usr/bin/env python3
"""R6 read-only, deterministic extension of 57 H3 Airfoil R5.

Does not refit R5 models, perturb locked R5 results or reuse historical
posthoc summaries in calculation. All output starts from original R5 NPZs.
Dependent bootstrap matched pairs are reported descriptively only.
"""
from __future__ import annotations
import json, hashlib, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

BASE=Path(__file__).resolve().parents[2] / '57_H3_Airfoil_R5'
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code'))
from common import load_data,features,fire

def assign(sim,th):
    R,S=sim.shape
    gains=np.concatenate([sim-th,np.zeros((R,R),dtype=float)],axis=1)
    rows,cols=linear_sum_assignment(-gains)
    ans=np.full(R,-1,int)
    for r,s in zip(rows,cols):
        if s<S and sim[r,s]>=th: ans[r]=s
    return ans

def weights_by_config(keys):
    _,ids=np.unique(keys,return_inverse=True)
    size=np.bincount(ids)
    return 1./(len(size)*size[ids])

def rmse_equal_configs(dif,key):
    ids,ind=np.unique(key,return_inverse=True)
    mse=np.bincount(ind,weights=dif*dif)/np.bincount(ind)
    return float(np.sqrt(mse.mean()))

records=[];coverage=[];folds=[];thr_rows=[];identity=[]
D=load_data(); group_all=D.group_id.to_numpy(str)
for fs in ('F4','F5'):
  F=features(D,fs)
  for structure in ('compact','reference'):
   for fold in range(10):
    par_name='TSK-Ridge-oneSE' if structure=='compact' else 'TSK-Ridge-fixedR6h1'
    parfile=BASE/'results'/'tsk'/f'group_log_{fs}'/f'fold_{fold:02d}'/(par_name+'.npz')
    bootdir=BASE/'results'/'bootstrap'/f'{fs}_fold{fold:02d}_{structure}'
    with np.load(parfile) as parent, np.load(bootdir/'relearn_replicate_results.npz') as arc:
     tr=parent['tr'].astype(int)
     groups=group_all[tr]
     Z=(F[tr]-parent['train_scaler_mean'])/parent['train_scaler_scale']
     P_w, _=fire(Z,parent['centers'],parent['widths'])
     p=Z.shape[1];R=int(parent['selected_R'])
     assert len(P_w)==len(tr) and P_w.shape[1]==R
     assert arc['match'].shape[1]==R
     assert arc['sim'].shape[1:] ==(R,R)
     assert arc['Ridge_coeff'].shape[1:]==(R,p+1)
     assert np.allclose(arc['parent_coeff'], np.asarray(parent['coef']).reshape(R,p+1)+np.column_stack((np.full(R,float(parent['train_target_mean'])),np.zeros((R,p)))),atol=1e-9,rtol=0)
     register=pd.read_csv(bootdir/'relearn_replicate_register.csv')
     success=register.query("status=='success'")
     failed=register.query("status!='success'")
     assert len(success)==len(arc['match'])==len(arc['Ridge_coeff'])
     assert len(register)==200
     # Anchors fixed at the observed parent membership threshold, as in R5.
     anchor_ix=[np.flatnonzero(P_w[:,r]>=0.1) for r in range(R)]
     expected_anchor_sizes=arc['parent_support_sizes']
     assert np.array_equal([len(x) for x in anchor_ix],expected_anchor_sizes)
     groupw=weights_by_config(groups)
     group_u=np.unique(groups)
     gm=np.vstack([P_w[groups==g,:].mean(axis=0) for g in group_u])
     masses=gm.mean(axis=0)
     eff_group_support=gm.sum(axis=0)**2/np.maximum((gm**2).sum(axis=0),1e-300)
     s=arc['sim']; M=arc['match']; C=arc['Ridge_coeff']; target=arc['parent_coeff']
     for b in range(s.shape[0]):
       chk=assign(s[b],0.5)
       identity.append(bool(np.array_equal(chk,M[b])))
       if not identity[-1]:raise ValueError(f'Archived matching disagrees {fs} {structure} fold{fold} replicate-success-index{b}')
       for r,match_id in enumerate(M[b]):
        if match_id<0:continue
        anchor=anchor_ix[r]
        if len(anchor)==0:continue
        coef=C[b,r]
        if not np.isfinite(coef).all():raise ValueError('Nonfinite matched coefficient')
        z=Z[anchor]
        diff=(target[r,0]+z@target[r,1:])-(coef[0]+z@coef[1:])
        g=groups[anchor]
        dis=rmse_equal_configs(diff,g)
        # Independently equivalent method: assign a weight to each configuration,
        # not to the multiplicity of its frequency samples.
        ww=weights_by_config(g)
        direct=float(np.sqrt(np.sum(ww*diff**2)))
        if abs(dis-direct)>1e-9:raise AssertionError('Group-weight parity failure')
        records.append(dict(feature=fs,structure=structure,fold=fold,
            replicate_success_idx=b,replicate=int(success.iloc[b]['replicate']),
            parent_rule=r,matched_rule=int(match_id),Jaccard=float(s[b,r,match_id]),
            functional_disagreement_rmse_dB=dis,n_parent_anchor_rows=len(anchor),
            n_parent_anchor_groups=len(np.unique(g)),
            parent_mean_activation=float(masses[r]),parent_effective_group_support=float(eff_group_support[r]),
            low_rule_mass_within_fold=bool(masses[r]<np.median(masses)),
            max_parent_weight=float(P_w[:,r].max())))
     for th in (.3,.5,.7):
      matches=np.array([assign(v,th) for v in s])
      thr_rows.append(dict(feature=fs,structure=structure,fold=fold,threshold=th,
          n_draw_success=len(success),n_draw_fail=len(failed),n_parent_rules=R,
          matched_count=int(np.sum(matches>=0)),total_parent_rule_slots=200*R,
          matched_fraction_success=float(np.mean(matches>=0)),
          matched_fraction_all_attempts=float(np.sum(matches>=0)/(200*R)),
          unmatched_count=int(np.sum(matches<0)),
          matched_median_per_successful_draw=float(np.median(np.mean(matches>=0,axis=1)))))
     coverage.append(dict(feature=fs,structure=structure,fold=fold,R=R,
           n_parent_rows=len(tr),n_parent_groups=len(group_u),B_attempt=200,B_success=len(success),
           B_failure=len(failed),matched_pairs_0p5=int((M>=0).sum()),
           full_match_n_draws=int(np.sum(np.all(M>=0,axis=1)))))

pairs=pd.DataFrame(records)
thresholds=pd.DataFrame(thr_rows)
cv=pd.DataFrame(coverage)
assert len(identity)==sum(cv.B_success)
assert all(identity)
assert len(pairs)>40000
assert len(cv)==40 and cv.B_attempt.sum()==8000 and cv.B_failure.sum()==2
assert (pairs.functional_disagreement_rmse_dB>=0).all()
assert (pairs.Jaccard>=.5).all()
assert (pairs.n_parent_anchor_groups>0).all()
assert (thresholds.query('threshold==0.5').matched_count.sum()==len(pairs))
for col,table in [('R6_MATCHED_PAIRS_0p5.csv',pairs),('R6_MATCH_THRESHOLD_AUDIT.csv',thresholds),('R6_RELEARN_COVERAGE.csv',cv)]:
 table.to_csv(OUT/'analysis'/col,index=False,float_format='%.15g')

for (fs,structure,fold),group in pairs.groupby(['feature','structure','fold'],sort=True):
  rr=spearmanr(group.Jaccard,group.functional_disagreement_rmse_dB).statistic
  folds.append(dict(feature=fs,structure=structure,fold=int(fold),n_pairs=len(group),
                    rho=float(rr),median_J=float(group.Jaccard.median()),
                    median_D_dB=float(group.functional_disagreement_rmse_dB.median()),
                    p90_D_dB=float(group.functional_disagreement_rmse_dB.quantile(.9)),
                    high_similarity_pairs=int((group.Jaccard>=.8).sum()),
                    high_sim_Dgt10_count=int(((group.Jaccard>=.8)&(group.functional_disagreement_rmse_dB>10)).sum())))
foldtab=pd.DataFrame(folds)
foldtab.to_csv(OUT/'analysis'/'R6_FOLD_LEVEL_DESCRIPTIVES.csv',index=False,float_format='%.15g')
# High-level summaries, descriptive only: median across the ten outer training folds.
summary=[]
for (fs,structure),g in pairs.groupby(['feature','structure'],sort=True):
 f=foldtab.query('feature == @fs and structure == @structure')
 for th in (.3,.5,.7):
  tt=thresholds.query('feature==@fs and structure==@structure and threshold==@th')
  summary.append(dict(feature=fs,structure=structure,threshold=th,
      median_fold_match_fraction=float(tt.matched_fraction_success.median()),
      min_fold_match_fraction=float(tt.matched_fraction_success.min()),
      max_fold_match_fraction=float(tt.matched_fraction_success.max()),
      total_successful_matches=int(tt.matched_count.sum()),
      total_parent_rule_attempt_slots=int(tt.total_parent_rule_slots.sum()),
      n_fit_failures=int(tt.n_draw_fail.sum()),
      pairs_with_function_evaluations=len(g) if th==.5 else None,
      median_fold_spearman_rho=float(f.rho.median()) if th==.5 else None,
      median_fold_D_dB=float(f.median_D_dB.median()) if th==.5 else None,
      median_fold_p90_D_dB=float(f.p90_D_dB.median()) if th==.5 else None))
summary=pd.DataFrame(summary)
summary.to_csv(OUT/'analysis'/'R6_SUMMARY_BY_SPECIFICATION.csv',index=False,float_format='%.15g')

# Distinct diagnostics: high similarity does NOT automatically fix function.
for fs in ('F4','F5'):
 primary=pairs.query('feature==@fs and structure=="compact"').copy()
 primary['sim_bin']=pd.cut(primary.Jaccard,[.5,.6,.7,.8,.9,1.000000001],right=False,
                labels=['0.50-0.60','0.60-0.70','0.70-0.80','0.80-0.90','0.90-1.00'])
 primary['rule_mass_stratum']=np.where(primary.low_rule_mass_within_fold,'lower-than-fold-median','upper-half')
 grp=primary.groupby(['sim_bin','rule_mass_stratum'],observed=True).functional_disagreement_rmse_dB
 stats=grp.agg(n='count',median='median',p90=lambda x: np.quantile(x,.9),
               frac_Dgt10=lambda x:np.mean(x>10)).reset_index()
 stats.to_csv(OUT/'analysis'/f'R6_{fs}_JACCARD_SUPPORT_STRATA.csv',index=False,float_format='%.15g')

out={'scope':'POST-HOC cross-resample matched-rule analysis, no new model fitting',
 'n_pairs_0p5':len(pairs),'n_outer_folds':10,'n_feature_sets':2,'n_structures':2,
 'relearn_attempts_total':int(cv.B_attempt.sum()),'relearn_success_total':int(cv.B_success.sum()),
 'relearn_failure_total':int(cv.B_failure.sum()),'matched_pairs_check_all':bool(all(identity)),
 'all_data_source':'R5 archived parent models and bootstrap NPZ arrays (immutable input)',
 'functions_at_thresholds':'Functional differences computed only at J>=0.5: R5 archives 0.5-aligned coefficients',
 'inference':'Fold-level descriptive only; bootstrap/rule pairs are dependent',
 'comparison_previous_gate':'independent read-only reconstruction: compare csv files separately'}
(OUT/'qa'/'R6_EXTENSION_STATUS.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
print(summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
