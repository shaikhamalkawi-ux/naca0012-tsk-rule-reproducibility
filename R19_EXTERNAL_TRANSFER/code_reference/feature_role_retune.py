#!/usr/bin/env python3
"""Robustness sensitivity: one-SE inner spectral tuning per changed-role candidate.
No test-geometry information enters CV; frozen-parent role test remains primary.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util, json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
BASE=Path(__file__).parent;spec=importlib.util.spec_from_file_location('r19',BASE/'57_H3_Airfoil_R19_LOGO_Candidate'/'run_logo_external.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=pd.read_csv(BASE/'R17_Southampton_External_Reproducibility'/'data'/'SOUTHAMPTON_CANONICAL.csv')
X,names=m.features(D);y=D.response_norm.to_numpy(float);g=D.group_id.to_numpy(str);geom=D.geometry_id.to_numpy(str);OUT=BASE/'R19_Clean_12grid'

def select(X,y,g,A,C):
  splits=list(GroupKFold(4).split(X,y,g));r=[]
  for R in m.RSET:
    errors={(h,l):{} for h in m.HSET for l in m.LSET};fs={(h,l):[] for h in m.HSET for l in m.LSET};valid=True
    for tr,va in splits:
      scaler=StandardScaler().fit(X[tr]);Ztr=scaler.transform(X[tr]);Zva=scaler.transform(X[va])
      fc=m.fcm(Ztr[:,A],R)
      if fc is None:valid=False;break
      for h in m.HSET:
        ptr=m.design(Ztr[:,C],m.fire(Ztr[:,A],fc['C'],fc['S0']*h));pva=m.design(Zva[:,C],m.fire(Zva[:,A],fc['C'],fc['S0']*h));betas,ym=m.gram_betas(ptr,y[tr]);pred=pva@betas+ym
        for jj,lam in enumerate(m.LSET):
          fs[(h,lam)].append(m.group_rmse(y[va],pred[:,jj],g[va]))
          for gp in set(g[va]):
            s=g[va]==gp; errors[(h,lam)][gp]=float(np.mean((y[va][s]-pred[s,jj])**2))
    n_grp=len(set(g));nfold=len(splits)
    for h in m.HSET:
      for lam in m.LSET:
        complete=valid and len(errors[(h,lam)])==n_grp
        rmse=float(np.sqrt(np.mean(list(errors[(h,lam)].values())))) if complete else np.inf
        se=float(np.std(fs[(h,lam)],ddof=1)/np.sqrt(nfold)) if complete else np.nan
        r.append({'R':R,'h':h,'lambda':lam,'inner_rmse':rmse,'inner_se':se,'valid':complete})
  c=pd.DataFrame(r);rsel=[]
  for R in m.RSET:
    z=c[(c.R==R)&c.valid].sort_values(['inner_rmse','h','lambda'])
    if len(z):rsel.append(z.iloc[0].to_dict())
  z=pd.DataFrame(rsel);best=z.sort_values(['inner_rmse','R']).iloc[0]
  cand=z[z.inner_rmse<=best.inner_rmse+best.inner_se+1e-12].copy();cand['hdist']=(cand.h-1).abs()
  sel=cand.sort_values(['R','inner_rmse','lambda','hdist','h'],ascending=[True,True,False,True,True]).iloc[0]
  return sel,c

def fit_predict(Xtr,ytr,Xte,A,C,sel):
  sc=StandardScaler().fit(Xtr);zt=sc.transform(Xtr);ze=sc.transform(Xte)
  fc=m.fcm(zt[:,A],int(sel.R));assert fc is not None
  wtr=m.fire(zt[:,A],fc['C'],fc['S0']*float(sel.h));wte=m.fire(ze[:,A],fc['C'],fc['S0']*float(sel.h))
  b=m.ridge(m.design(zt[:,C],wtr),ytr,float(sel['lambda']))
  return ytr.mean()+m.design(ze[:,C],wte)@b

opts=[('FULL',None),('log10_frequency_OMIT_ANTECEDENT',('A',0)),('log10_frequency_OMIT_CONSEQUENT',('C',0)),('trip_flag_OMIT_ANTECEDENT',('A',2)),('trip_flag_OMIT_CONSEQUENT',('C',2))]
rows=[];preds=[]
with threadpool_limits(limits=1):
 for hold in sorted(set(geom)):
  tr=np.flatnonzero(geom!=hold);te=np.flatnonzero(geom==hold)
  for opt,drop in opts:
   A=list(range(5));C=list(range(5))
   if drop is not None:
    if drop[0]=='A': A.remove(drop[1])
    else: C.remove(drop[1])
   sel,c=select(X[tr],y[tr],g[tr],A,C)
   pred=fit_predict(X[tr],y[tr],X[te],A,C,sel)
   rows.append({'heldout_geometry':hold,'option':opt,'n_train_rows':len(tr),'n_test_rows':len(te),'equal_spectrum_RMSE':m.group_rmse(y[te],pred,g[te]),'row_RMSE':float(np.sqrt(np.mean((y[te]-pred)**2))),'R':int(sel.R),'h':float(sel.h),'lambda':float(sel['lambda']),'inner_rmse':float(sel.inner_rmse),'inner_se':float(sel.inner_se)})
   for i,j in enumerate(te):preds.append({'row_id':int(D.row_id.iloc[j]),'heldout_geometry':hold,'group_id':g[j],'option':opt,'actual':float(y[j]),'prediction':float(pred[i])})
   print('RETUNE',hold,opt,'RMSE',m.group_rmse(y[te],pred,g[te]),flush=True)
model=pd.DataFrame(rows);model.to_csv(OUT/'FEATURE_ROLE_RETUNED_FOLDS.csv',index=False,float_format='%.17g')
p=pd.DataFrame(preds);p.to_csv(OUT/'FEATURE_ROLE_RETUNED_PREDICTIONS.csv',index=False,float_format='%.17g')
s=[]
for name,pp in p.groupby('option'):
 s.append({'option':name,'equal_geometry_RMSE':m.geom_equal_rmse(pp.actual.to_numpy(),pp.prediction.to_numpy(),pp.heldout_geometry.to_numpy()),'equal_spectrum_RMSE':m.group_rmse(pp.actual.to_numpy(),pp.prediction.to_numpy(),pp.group_id.to_numpy()),'row_RMSE':float(np.sqrt(np.mean((pp.actual-pp.prediction)**2)))})
res=pd.DataFrame(s);res.to_csv(OUT/'FEATURE_ROLE_RETUNED_SUMMARY.csv',index=False,float_format='%.17g');print('SUMMARY\n',res.to_string(index=False))
