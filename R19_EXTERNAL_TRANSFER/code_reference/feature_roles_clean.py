#!/usr/bin/env python3
"""External predictive role audit: fixed source-trained parent TSK R,h,lambda per LOGO fold.
No heldout geometry used for tuning, scaling, membership fitting, or coefficients.
Frequency role primary (measured and varying within every spectrum); trip secondary.
Velocity/chord omitted from direct NASA feature-transfer claims owing to confounding.
"""
from pathlib import Path
import json, numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import importlib.util
BASE=Path(__file__).parent;src=BASE/'57_H3_Airfoil_R19_LOGO_Candidate'/'run_logo_external.py'
spec=importlib.util.spec_from_file_location('r19_source',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=pd.read_csv(BASE/'R17_Southampton_External_Reproducibility'/'data'/'SOUTHAMPTON_CANONICAL.csv')
X,fn=m.features(D);y=D.response_norm.to_numpy(float);g=D.group_id.to_numpy(str);geom=D.geometry_id.to_numpy(str)
OUT=BASE/'R19_Clean_12grid'; folds=pd.read_csv(OUT/'LOGO_FOLDS.csv')
# Feature admissibility checked independently of role-test results
fa=[]
for name,ix in zip(fn,range(len(fn))):
 bygeom=D.assign(x=X[:,ix]).groupby('geometry_id').x.nunique()
 byspec=D.assign(x=X[:,ix]).groupby('group_id').x.nunique()
 fa.append({'feature':name,'observed':True,'unique_dataset_values':int(len(set(X[:,ix]))),'n_geometries_with_within_geometry_variation':int((bygeom>1).sum()),'n_spectra_with_within_spectrum_variation':int((byspec>1).sum()),'median_unique_values_within_geometry':float(bygeom.median()),'eligible_role_test':name in ('log10_frequency','trip_flag'),'role_test_scope':'PRIMARY' if name=='log10_frequency' else ('SECONDARY' if name=='trip_flag' else 'NOT TESTED')})
pd.DataFrame(fa).to_csv(OUT/'FEATURE_COMPARABILITY_AUDIT.csv',index=False)
results=[]; preds=[]
with threadpool_limits(limits=1):
 for _,f in folds.iterrows():
  hold=f.heldout_geometry; tr=np.flatnonzero(geom!=hold);te=np.flatnonzero(geom==hold)
  sc=StandardScaler().fit(X[tr]);Ztr=sc.transform(X[tr]);Zte=sc.transform(X[te])
  R=int(f.TSK_R); h=float(f.TSK_h); lam=float(f.TSK_lambda)
  for feat in (0,2):
   for variant in ('FULL','OMIT_ANTECEDENT','OMIT_CONSEQUENT','OMIT_BOTH'):
    A=[i for i in range(X.shape[1]) if not(variant in ('OMIT_ANTECEDENT','OMIT_BOTH') and i==feat)]
    C=[i for i in range(X.shape[1]) if not(variant in ('OMIT_CONSEQUENT','OMIT_BOTH') and i==feat)]
    fc=m.fcm(Ztr[:,A],R); assert fc is not None
    Wtr=m.fire(Ztr[:,A],fc['C'],fc['S0']*h);Wte=m.fire(Zte[:,A],fc['C'],fc['S0']*h)
    Ptr=m.design(Ztr[:,C],Wtr);Pte=m.design(Zte[:,C],Wte)
    b=m.ridge(Ptr,y[tr],lam); pt=y[tr].mean()+Pte@b
    val=m.group_rmse(y[te],pt,g[te]);valr=float(np.sqrt(np.mean((y[te]-pt)**2)))
    if variant=='FULL':
     z=np.load(OUT/'states'/f'{hold}.npz',allow_pickle=False);expected=z['offset']+z['P_test']@z['beta'];match=float(np.max(np.abs(pt-expected)))
     assert match < 1e-9,(hold,feat,match)
    results.append({'heldout_geometry':hold,'feature':fn[feat],'variant':variant,'same_parent_R':R,'same_parent_h':h,'same_parent_lambda':lam,'n_train':len(tr),'n_test':len(te),'spectra':len(set(g[te])),'row_RMSE':valr,'equal_spectrum_RMSE':val,'n_antecedent_inputs':len(A),'n_consequent_inputs':len(C)})
    for i,idx in enumerate(te):preds.append({'row_id':int(D.row_id.iloc[idx]),'heldout_geometry':hold,'group_id':g[idx],'feature':fn[feat],'variant':variant,'actual':y[idx],'prediction':pt[i]})
    print(hold,fn[feat],variant,round(val,6),flush=True)
allfold=pd.DataFrame(results);allfold.to_csv(OUT/'FEATURE_ROLE_GEOMETRY_RESULTS.csv',index=False,float_format='%.17g')
p=pd.DataFrame(preds);p.to_csv(OUT/'FEATURE_ROLE_PREDICTIONS.csv',index=False,float_format='%.17g')
summary=[]
for (feature,variant),pp in p.groupby(['feature','variant']):
 summary.append({'feature':feature,'variant':variant,'equal_geometry_RMSE':m.geom_equal_rmse(pp.actual.to_numpy(),pp.prediction.to_numpy(),pp.heldout_geometry.to_numpy()), 'equal_spectrum_RMSE':m.group_rmse(pp.actual.to_numpy(),pp.prediction.to_numpy(),pp.group_id.to_numpy()),'row_RMSE':float(np.sqrt(np.mean((pp.actual-pp.prediction)**2))), 'heldout_geometries':pp.heldout_geometry.nunique()})
summary=pd.DataFrame(summary);summary.to_csv(OUT/'FEATURE_ROLE_SUMMARY.csv',index=False,float_format='%.17g')
(OUT/'FEATURE_ROLE_PROTOCOL.json').write_text(json.dumps({'primary':'frequency role omission; 7 geometry-heldout experiments','secondary':'trip flag role omission; 7 geometry-heldout experiments','comparison':'same R h lambda selected on full outer training data, recluster changed antecedents, preserve all other input roles','heldout_model_selection':'none','causal_inference':'prohibited','direct_transfer_of_NASA_velocity_effect':'not admissible because Southampton velocity is mostly geometry-constant and confounded','target':'source-reported normalized response, not absolute dB','R18_status':'unchanged'},indent=2))
print('SUMMARY\n',summary.to_string(index=False));print('FEATURE_AVAILABILITY\n',pd.DataFrame(fa).to_string(index=False))
