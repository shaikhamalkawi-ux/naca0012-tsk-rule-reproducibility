#!/usr/bin/env python3
"""R19 corrective fresh rerun of archived 12-candidate ET source; independent from archived outputs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import sys, pathlib, json, platform, hashlib
import numpy as np,pandas as pd,scipy,sklearn
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
import importlib.util
BASE=pathlib.Path(__file__).parent
SRC=BASE/'57_H3_Airfoil_R19_LOGO_Candidate'/'run_logo_external.py'
spec=importlib.util.spec_from_file_location('r19_archived_source',SRC)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
DATA=BASE/'R17_Southampton_External_Reproducibility'/'data'/'SOUTHAMPTON_CANONICAL.csv'
OUT=BASE/'R19_Clean_12grid'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=pd.read_csv(DATA);X,fn=m.features(d); y=d.response_norm.to_numpy(float);g=d.group_id.to_numpy(str);geom=d.geometry_id.to_numpy(str)
 assert (len(d),len(set(g)),len(set(geom)))==(343,38,7) and not d.isna().any().any() and d.row_id.is_unique
 OUT.mkdir(exist_ok=True)
 for sub in ('candidates','states','splits'): (OUT/sub).mkdir(exist_ok=True)
 folds=[];preds=[];audit=[]
 with threadpool_limits(limits=1):
  for hold in sorted(set(geom)):
   tr=np.flatnonzero(geom!=hold);te=np.flatnonzero(geom==hold)
   assert not set(g[tr])&set(g[te]) and not set(d.row_id.iloc[tr])&set(d.row_id.iloc[te]) and not set(geom[tr])&set(geom[te])
   cv=[]
   for ix,(a,b) in enumerate(GroupKFold(4).split(X[tr],y[tr],g[tr])):
    assert not set(g[tr][a])&set(g[tr][b]);cv.append({'inner_fold':ix,'train_row_ids':d.row_id.iloc[tr[a]].tolist(),'validation_row_ids':d.row_id.iloc[tr[b]].tolist()})
   (OUT/'splits'/f'{hold}.json').write_text(json.dumps({'heldout':hold,'outer_train_rows':d.row_id.iloc[tr].tolist(),'outer_test_rows':d.row_id.iloc[te].tolist(),'inner_folds':cv},indent=2))
   tsel,ct=m.select_tsk(X[tr],y[tr],g[tr]); assert len(ct)==45;ct.to_csv(OUT/'candidates'/f'TSK_holdout_{hold}.csv',index=False,float_format='%.17g')
   model=m.tsk_fit(X[tr],y[tr],int(tsel.R),float(tsel.h),float(tsel['lambda']))
   assert model is not None
   pt,W,Pt,lm,Z=m.tsk_predict(model,X[te],True)
   esel,ce=m.select_et(X[tr],y[tr],g[tr]);assert len(ce)==12
   ce.to_csv(OUT/'candidates'/f'ET_holdout_{hold}.csv',index=False,float_format='%.17g')
   pa={'min_samples_leaf':int(esel.min_samples_leaf),'max_features':float(esel.max_features),'max_depth':None if pd.isna(esel.max_depth) else int(esel.max_depth)}
   et=ExtraTreesRegressor(n_estimators=400,random_state=20261007,n_jobs=1,bootstrap=False,**pa).fit(X[tr],y[tr]);pe=et.predict(X[te])
   cdist=np.linalg.norm((Z[:,None,:]-model['C'][None,:,:])/model['S'][None,:,:],axis=2).min(axis=1)
   assert np.allclose(-.5*cdist**2,lm,atol=1e-10)
   zd=np.linalg.norm(Z,axis=1)
   f={'heldout_geometry':hold,'rows':len(te),'spectra':len(set(g[te])),'TSK_rmse':m.group_rmse(y[te],pt,g[te]),'ET_rmse':m.group_rmse(y[te],pe,g[te]),'TSK_R':int(tsel.R),'TSK_h':float(tsel.h),'TSK_lambda':float(tsel['lambda']),'TSK_inner_rmse':float(tsel.inner_rmse),'ET_min_samples_leaf':pa['min_samples_leaf'],'ET_max_features':pa['max_features'],'ET_max_depth':pa['max_depth'],'ET_inner_rmse':float(esel.inner_rmse),'median_log_best_firing':float(np.median(lm)),'min_log_best_firing':float(np.min(lm)),'median_standardized_radius':float(np.median(zd)),'max_standardized_radius':float(np.max(zd))}
   folds.append(f)
   for j,ix in enumerate(te):preds.append({'row_id':int(d.row_id.iloc[ix]),'heldout_geometry':hold,'group_id':g[ix],'actual':float(y[ix]),'TSK_pred':float(pt[j]),'ET_pred':float(pe[j]),'log_best_firing':float(lm[j]),'max_normalized_rule_weight':float(W[j].max()),'nearest_rule_standardized_distance':float(cdist[j]),'standardized_radius':float(zd[j])})
   np.savez_compressed(OUT/'states'/f'{hold}.npz',P_train=model['P'],P_test=Pt,beta=model['beta'],lam=np.array(model['lam']),offset=np.array(model['offset']),y_train=y[tr],y_test=y[te],train_rows=d.row_id.iloc[tr].to_numpy(),test_rows=d.row_id.iloc[te].to_numpy(),train_geom=geom[tr],test_geom=geom[te],train_group=g[tr],test_group=g[te],feature_names=np.array(fn),epsilon=np.array((.1/6.896361277488791)*np.std(y[tr],ddof=0)),mean=model['sc'].mean_,scale=model['sc'].scale_,C=model['C'],S=model['S'],weights_test=W,logbest_test=lm,ET_param_json=np.array(json.dumps(pa)))
   audit.append({'heldout_geometry':hold,'n_train_rows':len(tr),'n_test_rows':len(te),'n_ET_candidates':len(ce),'n_TSK_candidates':len(ct),'fitted_scaler_train_rows':len(tr),'outer_test_geometry_exclusive':True,'outer_test_spectrum_exclusive':True})
   print(json.dumps(f),flush=True)
 p=pd.DataFrame(preds).sort_values('row_id');p.to_csv(OUT/'LOGO_PREDICTIONS.csv',index=False,float_format='%.17g')
 pd.DataFrame(folds).to_csv(OUT/'LOGO_FOLDS.csv',index=False,float_format='%.17g')
 s={'n_rows':len(d),'n_geometries':len(set(geom)),'n_spectra':len(set(g)), 'TSK_row_RMSE':float(np.sqrt(np.mean((p.actual-p.TSK_pred)**2))), 'ET_row_RMSE':float(np.sqrt(np.mean((p.actual-p.ET_pred)**2))), 'TSK_equal_geometry_RMSE':m.geom_equal_rmse(p.actual.to_numpy(),p.TSK_pred.to_numpy(),p.heldout_geometry.to_numpy()), 'ET_equal_geometry_RMSE':m.geom_equal_rmse(p.actual.to_numpy(),p.ET_pred.to_numpy(),p.heldout_geometry.to_numpy()), 'TSK_equal_spectrum_RMSE':m.group_rmse(p.actual.to_numpy(),p.TSK_pred.to_numpy(),p.group_id.to_numpy()),'ET_equal_spectrum_RMSE':m.group_rmse(p.actual.to_numpy(),p.ET_pred.to_numpy(),p.group_id.to_numpy())}
 (OUT/'SUMMARY.json').write_text(json.dumps(s,indent=2));(OUT/'SPLIT_AUDIT.json').write_text(json.dumps(audit,indent=2))
 (OUT/'ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'original_run_script_sha256':sha(SRC),'canonical_data_sha256':sha(DATA),'ET_search_grid':'12 source-declared candidates', 'TSK_seeds':m.SEEDS,'ET_seed':20261007},indent=2,default=str))
 print('FINAL',json.dumps(s,indent=2),flush=True)
if __name__=='__main__':main()
