"""Replay independent selections/fits and stored bootstrap draws from disclosed code."""
from pathlib import Path
import json,numpy as np,pandas as pd,time,hashlib,platform,sys,scipy,sklearn,xgboost
from threadpoolctl import threadpool_limits
from common import *
from run_baselines import fit_predict
from run_bootstrap import assign,match_matrix


def run():
 started=time.time();out={'checks':{},'replays':[],'versions':{
    'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,
    'sklearn':sklearn.__version__,'xgboost':xgboost.__version__}}
 d=load_data();y=d.scaled_SPL_dB.to_numpy(float);g=d.group_id.to_numpy(str)
 for feat in ('F4','F5'):
  F=features(d,feat)
  for fold in range(10):
   file=ROOT/'results'/'tsk'/f'group_log_{feat}'/f'fold_{fold:02d}'/'TSK-Ridge-oneSE.npz'
   v=np.load(file);tr=v['tr'].astype(int);te=v['te'].astype(int)
   assert not set(g[tr])&set(g[te])
   Z=(F[tr]-v['train_scaler_mean'])/v['train_scaler_scale']
   T=(F[te]-v['train_scaler_mean'])/v['train_scaler_scale']
   fitting,logs=fcm(Z,int(v['selected_R']))
   assert fitting is not None
   assert np.allclose(fitting['C'],v['centers'],atol=1e-10)
   assert np.allclose(fitting['S']*float(v['h']),v['widths'],atol=1e-10)
   W,_=fire(Z,v['centers'],v['widths']);Wt,_=fire(T,v['centers'],v['widths']);P=design(Z,W);Q=design(T,Wt)
   coef=coef_lstsq(P,y[tr]-y[tr].mean(),float(v['lambda_']))
   assert np.allclose(coef,v['coef'],atol=1e-9)
   pred=y[tr].mean()+Q@coef
   paper=pd.read_csv(ROOT/'results'/'tsk'/f'group_log_{feat}'/f'fold_{fold:02d}'/'fold_predictions.csv')
   actual=paper[paper.model=='TSK-Ridge-oneSE'].sort_values('row_id')
   actualpred=pd.Series(pred,index=d.row_id.iloc[te].to_numpy()).loc[actual.row_id].to_numpy()
   assert np.allclose(actual.pred.to_numpy(),actualpred,atol=1e-8)
   out['replays'].append({'feature':feat,'fold':fold,'kind':'recompute_compact_model','max_abs_pred_diff':float(np.max(np.abs(actual.pred.to_numpy()-actualpred)))})
  out['checks'][f'{feat}_all_10_TSK_replays']=True
  # Recreate first bootstrap draw exactly (no replacement of failed draws)
  fold=0;v=np.load(ROOT/'results'/'tsk'/f'group_log_{feat}'/'fold_00'/'TSK-Ridge-oneSE.npz')
  tr=v['tr'].astype(int);te=v['te'].astype(int);Z=(F[tr]-v['train_scaler_mean'])/v['train_scaler_scale']
  T=(F[te]-v['train_scaler_mean'])/v['train_scaler_scale'];P=design(Z,fire(Z,v['centers'],v['widths'])[0]);
  gtr=g[tr];uu=np.unique(gtr);gi=[np.flatnonzero(gtr==u) for u in uu]
  for branch in ('fixed','relearn'):
   seed=stable_seed(feat,fold,branch+'_compact',0)
   rng=np.random.default_rng(seed);ix=np.concatenate([gi[t] for t in rng.choice(len(uu),len(uu),replace=True)])
   if branch=='fixed':
    yymean=float(v['train_target_mean']);PP=P[ix]
    Q=design(T,fire(T,v['centers'],v['widths'])[0]);match=np.arange(int(v['selected_R']))
   else:
    fb=F[tr][ix];yb=y[tr][ix];sc=StandardScaler().fit(fb)
    FCM,logs=fcm(sc.transform(fb),int(v['selected_R']))
    assert FCM is not None
    YY=sc.transform(fb);PP=design(YY,fire(YY,FCM['C'],FCM['S']*float(v['h']))[0]);
    held=sc.transform(F[te]);Q=design(held,fire(held,FCM['C'],FCM['S']*float(v['h']))[0]);
    yymean=float(np.mean(yb));parentW=fire(Z,v['centers'],v['widths'])[0]
    bootW=fire(sc.transform(F[tr]),FCM['C'],FCM['S']*float(v['h']))[0]
    match=assign(match_matrix(parentW,bootW,gtr),.5)
   coef=coef_lstsq(PP,y[tr][ix]-yymean,float(v['lambda_']))
   pred=yymean+Q@coef
   data=np.load(ROOT/'results'/'bootstrap'/f'{feat}_fold00_compact'/f'{branch}_replicate_results.npz')
   assert data['seeds'][0]==seed
   assert np.allclose(pred,data['Ridge_preds'][0],atol=1e-8)
   assert np.array_equal(match,data['match'][0])
   out['replays'].append({'feature':feat,'fold':0,'kind':f'{branch}_replicate0','max_abs_pred_diff':float(np.max(np.abs(pred-data['Ridge_preds'][0])))})
   out['checks'][f'{feat}_replay_{branch}_bootstrap']=True
 # Baseline deterministic first-fold replay
 F=features(d,'F5');outer=fold_pairs(F,y,g)
 for model in ('extra_trees','xgboost','linear_ridge','svr'):
  p=ROOT/'results'/'baselines'/'group_log_F5'/model/'fold_00'
  sel=json.loads((p/'selection.json').read_text());tr,te=outer[0]
  fitted=fit_predict(model,sel['parameters'],F[tr],y[tr],F[te]);arch=pd.read_csv(p/'outer_predictions.csv')
  assert np.allclose(arch.pred.to_numpy(),fitted,atol=1e-8)
  out['replays'].append({'kind':'baseline_fold0','model':model,'max_abs_pred_diff':float(np.max(np.abs(arch.pred.to_numpy()-fitted)))})
  out['checks'][f'baseline_F5_{model}_replay']=True
 # Independently recompute 20 primary primary table entries
 pp=pd.read_csv(ROOT/'results'/'ALL_PRIMARY_OOF.csv');tab=pd.read_csv(ROOT/'results'/'TABLE_PRIMARY_BENCHMARK.csv')
 for _,r in tab.iterrows():
  a=pp[(pp.feature_set==r.feature_set)&(pp.model==r.model)]
  assert len(a)==1503 and a.row_id.nunique()==1503
  if abs(group_rmse(a.y.to_numpy(),a.pred.to_numpy(),a.group_id.to_numpy())-r.equal_config_RMSE)>1e-10:
   raise RuntimeError('primary table mismatch: '+r.feature_set+r.model)
 out['checks']['all_20_reported_primary_metrics_recomputed']=True
 status=json.loads((ROOT/'audit'/'AGGREGATE_STATUS.json').read_text())
 assert status['bootstrap_declared']==28000 and status['bootstrap_failed']==2
 out['checks']['bootstrap_register_coverage_28000']=True
 out['checks']['unfilled_model_result_errors_zero']=True
 out['passed']=sum(bool(x) for x in out['checks'].values());out['total']=len(out['checks']);out['elapsed_sec']=time.time()-started
 save_json(ROOT/'audit'/'INDEPENDENT_REPLAY_QA.json',out)
 print('REPLAY_QA',out['passed'],'/',out['total'],'sec',out['elapsed_sec'])
 print('maximum prediction residual',max(x.get('max_abs_pred_diff',0) for x in out['replays']))
 if out['passed']!=out['total']:raise RuntimeError('failed replay checks')

if __name__=='__main__':
 with threadpool_limits(1):run()
