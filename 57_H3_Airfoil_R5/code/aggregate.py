"""Rebuild every reported table and paired conditional interval from stored OOF outputs."""
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,time
from threadpoolctl import threadpool_limits
from common import *

def performance(d):
 y=d.y.to_numpy();p=d.pred.to_numpy();g=d.group_id.to_numpy(str)
 gs=[]
 for gi,part in d.groupby('group_id',sort=True):
  gs.append((gi,float(np.sqrt(np.mean((part.y-part.pred)**2))),len(part)))
 gs=sorted(gs,key=lambda a:a[1])
 return {'rows':len(d),'groups':len(gs),'equal_config_RMSE':group_rmse(y,p,g),
  'row_RMSE':row_rmse(y,p),'row_MAE':float(np.mean(np.abs(y-p))),
  'row_R2':float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)),
  'row_bias_pred_minus_true':float(np.mean(p-y)),
  'group_RMSE_median':float(np.median([s[1] for s in gs])),
  'group_RMSE_p90':float(np.quantile([s[1] for s in gs],.9)),
  'worst_group_id':gs[-1][0],'worst_group_RMSE':gs[-1][1]}

def frame():
 arr=[]
 for feature in ('F4','F5'):
  tag=f'group_log_{feature}'
  tsk=pd.read_csv(ROOT/'results'/'tsk'/tag/'OUTER_PREDICTIONS.csv')
  arr.append(tsk)
  for model in ('linear_ridge','svr','extra_trees','xgboost'):
   file=ROOT/'results'/'baselines'/tag/model/'OUTER_PREDICTIONS.csv'
   if not file.exists():raise RuntimeError(f'INCOMPLETE BASELINE: {file}')
   arr.append(pd.read_csv(file))
  # calculation of fixed-antecedent paired LS at ridge-selected architecture
  d=load_data(); X=features(d,feature);y=d.scaled_SPL_dB.to_numpy(float);g=d.group_id.to_numpy(str)
  pa=ROOT/'results'/'tsk'/tag
  records=[]
  for fold in range(10):
   v=np.load(pa/f'fold_{fold:02d}'/'TSK-LS-at-RidgeStructure.npz')
   te=v['te'].astype(int)
   Z=(X[te]-v['train_scaler_mean'])/v['train_scaler_scale']
   W,_=fire(Z,v['centers'],v['widths'])
   yp=float(v['train_target_mean'])+design(Z,W)@v['coef']
   for i,p in zip(te,yp):records.append({'feature_set':feature,'mode':'group','representation':'log','outer_fold':fold,
    'model':'TSK-LS-at-RidgeStructure','row_id':int(d.row_id.iloc[i]),'group_id':g[i],'y':y[i],'pred':p})
  rp=pd.DataFrame(records);rp.to_csv(pa/'OOF_LS_AT_RIDGE_STRUCTURE.csv',index=False);arr.append(rp)
 pp=pd.concat(arr,ignore_index=True)
 assert pp.groupby(['feature_set','model']).row_id.nunique().eq(1503).all()
 pp.to_csv(ROOT/'results'/'ALL_PRIMARY_OOF.csv',index=False)
 return pp

def summarize(pp):
 a=[]
 for (f,m),df in pp.groupby(['feature_set','model'],sort=True):
  a.append({'feature_set':f,'model':m,**performance(df)})
 tab=pd.DataFrame(a).sort_values(['feature_set','equal_config_RMSE'])
 tab.to_csv(ROOT/'results'/'TABLE_PRIMARY_BENCHMARK.csv',index=False)
 return tab

def group_square_errors(pp):
 tmp=pp.copy();tmp['sqerr']=(tmp.y-tmp.pred)**2
 return tmp.groupby(['feature_set','model','group_id']).sqerr.mean()

def interval_comparisons(pp,B=2000):
 a=group_square_errors(pp)
 pairs=[(('F5','TSK-Ridge-oneSE'),('F4','TSK-Ridge-oneSE'),'F5 minus F4 compact ridge'),
  (('F5','TSK-Ridge-oneSE'),('F5','TSK-LS-oneSE'),'F5 compact ridge minus LS independent selection'),
  (('F4','TSK-Ridge-oneSE'),('F4','TSK-LS-oneSE'),'F4 compact ridge minus LS independent selection'),
  (('F5','TSK-Ridge-oneSE'),('F5','TSK-LS-at-RidgeStructure'),'F5 ridge minus unpenalized identical structure'),
  (('F4','TSK-Ridge-oneSE'),('F4','TSK-LS-at-RidgeStructure'),'F4 ridge minus unpenalized identical structure'),
  (('F5','TSK-Ridge-oneSE'),('F5','TSK-Ridge-min_error'),'F5 oneSE minus min error'),
  (('F4','TSK-Ridge-oneSE'),('F4','TSK-Ridge-min_error'),'F4 oneSE minus min error')]
 for model in ('linear_ridge','svr','extra_trees','xgboost'):
  pairs.append((('F5','TSK-Ridge-oneSE'),('F5',model),f'F5 compact ridge minus {model}'))
 for model in ('extra_trees','xgboost'):
  pairs.append((('F5',model),('F4',model),f'{model} F5 minus F4'))
 rng=np.random.default_rng(SEED+9341);rows=[]
 for (f1,m1),(f2,m2),title in pairs:
  c1=a.loc[(f1,m1)];c2=a.loc[(f2,m2)];common=sorted(set(c1.index)&set(c2.index))
  v1=c1.loc[common].to_numpy();v2=c2.loc[common].to_numpy()
  assert len(common)==106
  draws=rng.integers(len(common),size=(B,len(common)))
  differences=np.sqrt(v1[draws].mean(axis=1))-np.sqrt(v2[draws].mean(axis=1))
  rows.append({'contrast':title,'A':f'{f1}:{m1}','B':f'{f2}:{m2}',
    'difference_equal_config_RMSE_dB':float(np.sqrt(np.mean(v1))-np.sqrt(np.mean(v2))),
    'conditional_low_2p5':float(np.quantile(differences,.025)),
    'conditional_high_97p5':float(np.quantile(differences,.975)),
    'resamples':B,'independent_retraining':False,'p_value':None})
 out=pd.DataFrame(rows);out.to_csv(ROOT/'results'/'TABLE_PAIRED_CONDITIONAL_RANGES.csv',index=False)
 return out

def boot_aggregate():
 allrows=[];checks=[];failure_records=[]
 for f in ('F4','F5'):
  for k in range(10):
   for s in ('compact','reference'):
    dir=ROOT/'results'/'bootstrap'/f'{f}_fold{k:02d}_{s}'
    cp=dir/'COMPLETE.json'
    if not cp.exists():raise RuntimeError(f'BOOT incomplete: {dir}')
    c=json.loads(cp.read_text())
    if c['n_logged_replicates']!=700:raise RuntimeError(f'Bad bootstrap length: {dir}')
    for branch,B in (('fixed',500),('relearn',200)):
     z=np.load(dir/f'{branch}_replicate_results.npz')
     outcomes=pd.read_csv(dir/f'{branch}_replicate_register.csv')
     if len(outcomes)!=B:raise RuntimeError(f'bad replicates {dir} {branch}')
     success=int((outcomes.status=='success').sum()); fail=B-success
     if len(z['Ridge_preds'])!=success or len(z['LS_preds'])!=success:raise RuntimeError(f'pred array mismatch {dir}')
     if len(z['match'])!=success:raise RuntimeError(f'match mismatch {dir}')
     if len(z['seeds'])!=B:raise RuntimeError(f'seed mismatch {dir}')
     if not np.isfinite(z['Ridge_preds']).all():raise RuntimeError(f'nonfinite predictions {dir}')
     checks.append({'feature':f,'fold':k,'structure':s,'branch':branch,'expected':B,'success':success,'fail':fail})
     failure_records.extend(outcomes[outcomes.status!='success'].to_dict('records'))
    x=pd.read_csv(dir/'SUMMARY.csv');allrows.append(x)
 table=pd.concat(allrows,ignore_index=True)
 assert len(table)==40*2*2
 assert sum(x['success']+x['fail'] for x in checks)==40*(500+200)
 table.to_csv(ROOT/'results'/'TABLE_BOOTSTRAP_FOLD_SUMMARY.csv',index=False)
 pd.DataFrame(checks).to_csv(ROOT/'results'/'BOOTSTRAP_REPLICATE_COVERAGE.csv',index=False)
 pd.DataFrame(failure_records).to_csv(ROOT/'results'/'BOOTSTRAP_FAILED_REPLICATES.csv',index=False)
 rows=[]
 for (feature,structure,branch,fit),gg in table.groupby(['feature','structure','branch','fit']):
  rows.append({'feature':feature,'structure':structure,'branch':branch,'fit':fit,
   'successful_draws':int(gg.B_success.sum()),'failed_draws':int(gg.B_failure.sum()),
   'median_pred_disp_rms':float(gg.pred_disp_rms.median()),'median_local_disp_rms':float(gg.local_disp_rms_median.median()),
   'median_coefficient_sd':float(gg.coef_sd_median.median()),
   'median_rule_match_fraction':float(gg.matched_rule_fraction.median())})
 b=pd.DataFrame(rows).sort_values(['feature','structure','branch','fit']);b.to_csv(ROOT/'results'/'TABLE_BOOTSTRAP_SUMMARY.csv',index=False)
 return b,checks,failure_records

def main():
 with threadpool_limits(1):
  pp=frame();t=summarize(pp);pair=interval_comparisons(pp)
  print('PRIMARY',t[['feature_set','model','equal_config_RMSE','row_RMSE','row_R2']].to_string(index=False),flush=True)
  b,c,fail=boot_aggregate()
  print('BOOT',b.to_string(index=False));print('TOTAL draws',sum(v['expected'] for v in c),'fails',len(fail))
  save_json(ROOT/'audit'/'AGGREGATE_STATUS.json',{
    'primary_OOF_rows':len(pp),'models_per_feature':int(t.groupby('feature_set').size().min()),
    'bootstrap_declared':sum(v['expected'] for v in c),'bootstrap_failed':len(fail),
    'bootstrap_success':sum(v['success'] for v in c),'status':'COMPUTED_WITH_REPLAYABLE_FILES_NOT_YET_EDITORIAL_AUDITED'})
if __name__=='__main__':main()
