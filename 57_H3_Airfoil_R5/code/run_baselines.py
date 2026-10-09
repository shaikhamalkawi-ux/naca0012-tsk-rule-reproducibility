"""Nested group-aware baseline search, same 10x5 group splits as TSK."""
import os,argparse,json,time,itertools
import numpy as np,pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
from common import *

def param_grid(model):
 if model=='linear_ridge':return [{'alpha':v} for v in (0.,0.001,0.01,0.1,1.,10.,100.)]
 if model=='svr':return [{'C':C,'gamma':gamma,'epsilon':eps} for C,gamma,eps in itertools.product((1.,10.,100.),('scale',0.1,1.),(0.1,0.5))]
 if model=='extra_trees':return [{'min_samples_leaf':l,'max_features':mf,'max_depth':depth} for l,mf,depth in itertools.product((1,3,5),(.7,1.0),(12,None))]
 if model=='xgboost':return [{'n_estimators':n,'max_depth':depth,'learning_rate':rate,'reg_lambda':lam} for n,depth,rate,lam in itertools.product((200,500),(2,4),(.03,.1),(1.,10.))]
 raise ValueError(model)

def fit_predict(model,parameters,Xtr,ytr,Xte):
 if model in ('linear_ridge','svr'):
  scaler=StandardScaler().fit(Xtr);A=scaler.transform(Xtr);B=scaler.transform(Xte)
  if model=='linear_ridge':
   if parameters['alpha']==0: obj=LinearRegression()
   else:obj=Ridge(alpha=parameters['alpha'],solver='svd')
  else:obj=SVR(kernel='rbf',**parameters)
 elif model=='extra_trees':
  A=Xtr;B=Xte;obj=ExtraTreesRegressor(n_estimators=400,random_state=SEED,n_jobs=1,bootstrap=False,**parameters)
 else:
  A=Xtr;B=Xte;obj=XGBRegressor(objective='reg:squarederror',tree_method='hist',n_jobs=1,
                       random_state=SEED,subsample=1.,colsample_bytree=1.,verbosity=0,**parameters)
 obj.fit(A,ytr)
 return obj.predict(B)

def run(feature_set,model,mode='group',raw=False):
 d=load_data();X=features(d,feature_set,raw);y=d.scaled_SPL_dB.to_numpy(float);g=d.group_id.to_numpy(str)
 tag=f'{mode}_{"raw" if raw else "log"}_{feature_set}'; root=ROOT/'results'/'baselines'/tag/model;root.mkdir(parents=True,exist_ok=True)
 outer=fold_pairs(X,y,g,mode); params=param_grid(model)
 for fold,(tr,te) in enumerate(outer):
  ot=root/f'fold_{fold:02d}';ot.mkdir(exist_ok=True)
  if (ot/'COMPLETE.json').exists():print('SKIP',tag,model,fold,flush=True);continue
  started=time.time();inner=five_splits(X[tr],y[tr],g[tr],mode)
  records=[];chosen=None
  for k,p in enumerate(params):
   loss=[]
   for ii,(itr,ival) in enumerate(inner):
    ta=tr[itr];va=tr[ival]
    if mode=='group': assert not set(g[ta])&set(g[va])
    pred=fit_predict(model,p,X[ta],y[ta],X[va]); score=candidate_score(y[va],pred,g[va],mode)
    records.append({'model':model,'feature_set':feature_set,'outer_fold':fold,'inner_fold':ii,'candidate_index':k,
      'parameters':json.dumps(p,sort_keys=True),'inner_fold_score':score,'n_inner_val_groups':len(set(g[va]))})
    loss.append(score)
   scores=np.array(loss)
   counts=np.array([len(set(g[tr[iv]])) for _,iv in inner])
   pooled=float(np.sqrt(np.average(scores**2,weights=counts))) if mode=='group' else float(np.sqrt(np.average(scores**2,weights=[len(iv) for _,iv in inner])))
   if chosen is None or pooled<chosen['score']-1e-12:
    chosen={'candidate_index':k,'parameters':p,'score':pooled,'five_fold_scores':loss}
  pd.DataFrame(records).to_csv(ot/'inner_candidate_folds.csv',index=False)
  pp=fit_predict(model,chosen['parameters'],X[tr],y[tr],X[te]);assert np.isfinite(pp).all()
  pd.DataFrame({'feature_set':feature_set,'mode':mode,'representation':'raw' if raw else 'log',
    'outer_fold':fold,'model':model,'row_id':d.row_id.iloc[te].to_numpy(int), 'group_id':g[te],
    'y':y[te],'pred':pp}).to_csv(ot/'outer_predictions.csv',index=False)
  save_json(ot/'selection.json',chosen)
  save_json(ot/'COMPLETE.json',{'tag':tag,'model':model,'fold':fold,'n_train':len(tr),'n_test':len(te),
    'n_train_groups':len(set(g[tr])),'n_test_groups':len(set(g[te])),'elapsed_sec':time.time()-started,'n_candidates':len(params)})
  print('BASE COMPLETE',tag,model,fold,'score',chosen['score'],'sec',round(time.time()-started,2),flush=True)
 pp=pd.concat([pd.read_csv(root/f'fold_{fold:02d}'/'outer_predictions.csv') for fold in range(10)],ignore_index=True)
 if len(pp)!=1503 or pp.row_id.nunique()!=1503:raise RuntimeError('incomplete outer test coverage')
 pp.to_csv(root/'OUTER_PREDICTIONS.csv',index=False)
 pd.concat([pd.read_csv(root/f'fold_{fold:02d}'/'inner_candidate_folds.csv') for fold in range(10)],ignore_index=True).to_csv(root/'ALL_INNER_CANDIDATES.csv',index=False)
 save_json(root/'COMPLETE.json',{'feature_set':feature_set,'model':model,'covered_rows':len(pp),'outer_folds':10,
     'grid_size':len(params),'n_inner_fits':10*5*len(params)})
 print('DONE BASELINE',tag,model,flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--features',choices=['F4','F5'],default='F5');
 parser.add_argument('--model',choices=['linear_ridge','svr','extra_trees','xgboost'],required=True);
 parser.add_argument('--mode',default='group');parser.add_argument('--raw',action='store_true');a=parser.parse_args()
 with threadpool_limits(1):run(a.features,a.model,a.mode,a.raw)
