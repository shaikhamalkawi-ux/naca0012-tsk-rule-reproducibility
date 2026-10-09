"""Prespecified diagnostic perturbations; never select primary model by these scores."""
import numpy as np,pandas as pd,scipy
from threadpoolctl import threadpool_limits
from common import *

def do_width():
 df=load_data(); y=df.scaled_SPL_dB.to_numpy();g=df.group_id.to_numpy(str)
 rows=[]
 for feat in ('F4','F5'):
  F=features(df,feat)
  for fold in range(10):
   item=ROOT/'results'/'tsk'/f'group_log_{feat}'/f'fold_{fold:02d}'/'TSK-Ridge-oneSE.npz'
   v=np.load(item);tr=v['tr'].astype(int);te=v['te'].astype(int)
   Z=(F[tr]-v['train_scaler_mean'])/v['train_scaler_scale'];T=(F[te]-v['train_scaler_mean'])/v['train_scaler_scale']
   C=v['centers'];R=int(v['selected_R']);h=float(v['h']);lam=float(v['lambda_']);
   # recompute FCM training memberships on exact training partition and five fixed restarts
   fit,logs=fcm(Z,R)
   if fit is None:raise RuntimeError('Selected fold FCM nonconvergence')
   if not np.allclose(fit['C'],C,atol=1e-8):raise RuntimeError('Non-deterministic FCM centres under same data: '+feat+str(fold))
   um=fit['U']**2
   widths=np.sqrt(np.maximum((um.T@(Z*Z))/um.sum(0)[:,None]-C*C,0))
   for floor in (0.025,.05,.1):
    S=np.maximum(widths,floor)*h;W,_=fire(Z,C,S);WT,_=fire(T,C,S)
    P=design(Z,W);Q=design(T,WT)
    beta=coef_lstsq(P,y[tr]-y[tr].mean(),lam)
    pred=y[tr].mean()+Q@beta
    rows.append({'feature':feat,'fold':fold,'width_floor':floor,'R':R,'h':h,'lambda':lam,
         'group_RMSE_dB':group_rmse(y[te],pred,g[te]),'row_RMSE_dB':row_rmse(y[te],pred),
         'n_changed_widths':int(np.sum((widths<floor)&(widths>=.05)) if floor>.05 else np.sum(widths<.05))})
 out=pd.DataFrame(rows);out.to_csv(ROOT/'results'/'DIAG_WIDTH_FLOOR.csv',index=False)
 print('FLOOR SENS',out.groupby(['feature','width_floor']).group_RMSE_dB.median().to_string(),flush=True)
 return out

def all_transform():
 rows=[]
 for f in ('F4','F5'):
  for mode in ('group','row'):
   for form in ('log','raw') if mode=='group' else ('log',):
    file=ROOT/'results'/'tsk'/f'{mode}_{form}_{f}'/'OUTER_PREDICTIONS.csv'
    if not file.exists():continue
    a=pd.read_csv(file)
    for model,part in a.groupby('model'):
     rows.append({'feature':f,'mode':mode,'form':form,'model':model,
       'row_RMSE':row_rmse(part.y,part.pred),
       'equal_configuration_RMSE':group_rmse(part.y.to_numpy(),part.pred.to_numpy(),part.group_id.to_numpy())})
 pd.DataFrame(rows).to_csv(ROOT/'results'/'TABLE_TRANSFORM_AND_ROW_SENSITIVITY.csv',index=False)

if __name__=='__main__':
 with threadpool_limits(1):do_width();all_transform()
