from pathlib import Path
import itertools, json,hashlib,sys
import numpy as np,pandas as pd,scipy,sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
BASE=Path(__file__).parent;NEW=BASE/'R19_Clean_12grid';OLD=BASE/'57_H3_Airfoil_R19_LOGO_Candidate'
D=pd.read_csv(BASE/'R17_Southampton_External_Reproducibility'/'data'/'SOUTHAMPTON_CANONICAL.csv').set_index('row_id')
X=np.c_[np.log10(D.frequency_Hz.to_numpy(float)),D.angle_deg.to_numpy(float),D.trip_flag.to_numpy(float),D.chord_m.to_numpy(float),D.velocity_m_s.to_numpy(float)]
Y=D.response_norm.to_numpy(float);geom=D.geometry_id.to_numpy(str);group=D.group_id.to_numpy(str)
p=pd.read_csv(NEW/'LOGO_PREDICTIONS.csv').set_index('row_id');op=pd.read_csv(OLD/'LOGO_PREDICTIONS.csv').set_index('row_id')
assert p.index.is_unique and len(p)==343 and set(p.index)==set(D.index)
assert p.heldout_geometry.eq(D.loc[p.index,'geometry_id'].to_numpy()).all()
assert np.max(abs(p.actual-D.loc[p.index,'response_norm']))<1e-14
check=[];max_tsk=0.;max_et=0.;max_gram=0.;max_old_tsk=0.;max_old_et=0.;max_cv=0.
for hold in sorted(set(geom)):
 tr=np.flatnonzero(geom!=hold);te=np.flatnonzero(geom==hold); row_train=D.index.to_numpy()[tr]; row_test=D.index.to_numpy()[te]
 assert not set(group[tr])&set(group[te]);assert not set(geom[tr])&set(geom[te]);assert not set(row_train)&set(row_test)
 z=np.load(NEW/'states'/f'{hold}.npz',allow_pickle=False)
 assert set(z['train_rows'])==set(row_train) and set(z['test_rows'])==set(row_test)
 xtr=(X[tr]-z['mean'])/z['scale']; xte=(X[te]-z['mean'])/z['scale']
 assert np.max(abs(xtr.mean(axis=0)))<1e-12
 Ltr=-.5*np.sum(((xtr[:,None,:]-z['C'][None,:,:])/z['S'][None,:,:])**2,axis=2)
 Lte=-.5*np.sum(((xte[:,None,:]-z['C'][None,:,:])/z['S'][None,:,:])**2,axis=2)
 def weights(log):
  a=log-np.max(log,axis=1,keepdims=True);w=np.exp(a);return w/w.sum(axis=1,keepdims=True)
 def design(x,log):return np.einsum('nr,nj->nrj',weights(log),np.c_[np.ones(len(x)),x]).reshape(len(x),-1)
 Ptr=design(xtr,Ltr);Pte=design(xte,Lte)
 max_gram=max(max_gram,float(np.max(abs(Ptr-z['P_train']))),float(np.max(abs(Pte-z['P_test']))))
 off=float(z['offset']);beta=z['beta'];rr=off+Pte@beta
 local=p.loc[row_test];err=np.max(abs(local.TSK_pred.to_numpy()-rr));max_tsk=max(max_tsk,float(err))
 # independent KKT gradient and reg objective check
 lam=float(z['lam']);grad=Ptr.T@(Ptr@beta-(Y[tr]-off))/len(tr)+lam*beta
 assert np.max(abs(grad))<5e-9
 assert np.max(abs(Pte-z['P_test']))<1e-11
 new_grid=pd.read_csv(NEW/'candidates'/f'ET_holdout_{hold}.csv')
 assert len(new_grid)==12 and set(new_grid.min_samples_leaf)=={1,3,5} and set(new_grid.max_features.round(3))=={.7,1.0}
 assert set(new_grid.max_depth.dropna().astype(int))=={12} and new_grid.max_depth.isna().sum()==6
 best=new_grid.sort_values(['inner_rmse','grid_index']).iloc[0]
 pa={'min_samples_leaf':int(best.min_samples_leaf),'max_features':float(best.max_features),'max_depth':None if pd.isna(best.max_depth) else int(best.max_depth)}
 et=ExtraTreesRegressor(n_estimators=400,random_state=20261007,n_jobs=1,bootstrap=False,**pa).fit(X[tr],Y[tr])
 ee=et.predict(X[te]);et_err=float(np.max(abs(local.ET_pred.to_numpy()-ee)));max_et=max(max_et,et_err)
 # compare old TSK and ET saved predictions without equating the two search procedures
 max_old_tsk=max(max_old_tsk,float(np.max(abs(local.TSK_pred.to_numpy()-op.loc[row_test,'TSK_pred'].to_numpy()))))
 max_old_et=max(max_old_et,float(np.max(abs(local.ET_pred.to_numpy()-op.loc[row_test,'ET_pred'].to_numpy()))))
 tskc=pd.read_csv(NEW/'candidates'/f'TSK_holdout_{hold}.csv');tsko=pd.read_csv(OLD/'candidates'/f'TSK_holdout_{hold}.csv')
 assert tskc.shape==tsko.shape==(45,6)
 assert np.max(abs(tskc.inner_rmse-tsko.inner_rmse))<2e-12
 assert len(pd.read_csv(OLD/'candidates'/f'ET_holdout_{hold}.csv'))==3
 with open(NEW/'splits'/f'{hold}.json') as f:split=json.load(f)
 assert set(split['outer_train_rows'])==set(row_train) and set(split['outer_test_rows'])==set(row_test)
 for inner in split['inner_folds']:
  ra=inner['train_row_ids'];rb=inner['validation_row_ids'];assert not set(ra)&set(rb)
  assert not set(D.loc[ra,'group_id'])&set(D.loc[rb,'group_id'])
  assert not set(ra)&set(row_test) and not set(rb)&set(row_test)
 check.append({'geometry':hold,'rows':len(te),'spectra':len(set(group[te])),'fcm_design_max_abs':float(np.max(abs(Pte-z['P_test']))),'TSK_replay_max_abs':float(err),'ET_retrained_max_abs':et_err,'ET_best_params':pa,'old_new_ET_prediction_max_abs':float(np.max(abs(local.ET_pred.to_numpy()-op.loc[row_test,'ET_pred'].to_numpy()))),'train_test_geometry_disjoint':True,'inner_spectrum_disjoint':True})
assert max_tsk<1e-12 and max_et<1e-12 and max_gram<1e-11 and max_old_tsk<1e-12
# Independently recompute all three aggregation endpoints, without calling the original methods.
q=p.copy();q['TSK_err2']=(q.actual-q.TSK_pred)**2;q['ET_err2']=(q.actual-q.ET_pred)**2
scores={}
for name in ('TSK','ET'):
 e=q[f'{name}_err2'];scores[name+'_row_RMSE']=float(np.sqrt(e.mean()));scores[name+'_equal_spectrum_RMSE']=float(np.sqrt(q.assign(e2=e).groupby('group_id').e2.mean().mean()));scores[name+'_equal_geometry_RMSE']=float(np.sqrt(q.assign(e2=e).groupby('heldout_geometry').e2.mean().mean()))
expected=json.loads((NEW/'SUMMARY.json').read_text());assert all(abs(scores[k]-expected[k])<5e-14 for k in scores)
QA={'scientific_scope':'R19 source-declared 12-candidate Extra Trees; R18 locked unchanged','data_rows':343,'spectra':38,'geometries':7,'old_stored_ET_grid_per_fold':3,'source_script_ET_grid_per_fold':12,'corrected_clean_grid_per_fold':12,'old_stored_TSK_grid_per_fold':45,'source_script_TSK_grid_per_fold':45,'max_TSK_saved_vs_independent_replay':max_tsk,'max_ET_saved_vs_selected_retraining':max_et,'max_TSK_P_matrices_vs_independent_design':max_gram,'max_original_candidate_vs_corrected_TSK_prediction':max_old_tsk,'max_original_candidate_vs_corrected_ET_prediction':max_old_et,'scores':scores,'folds':check,'pass':True}
(NEW/'INDEPENDENT_REPLAY_QA.json').write_text(json.dumps(QA,indent=2))
print(json.dumps({k:v for k,v in QA.items() if k not in ('folds','scores')},indent=2));print('computed',scores)
