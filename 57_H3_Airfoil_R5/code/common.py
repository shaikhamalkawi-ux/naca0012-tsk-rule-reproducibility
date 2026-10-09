"""57 H3 NASA Airfoil R5 transparent independent reconstruction.

Protocol basis: v2 7 Oct 2026. Amendment: FCM maximum iterations 2000
(instead of 500), uniform and logged; not an outcome-blind preregistration.
Preplanned numerical cutoffs, seeds, models, feature sets and grouped CV retained.
"""
from __future__ import annotations
import json, hashlib, sys, platform, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import lstsq
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'airfoil_F5_SEMANTICALLY_VERIFIED.csv'
SEED=20261007
RESTARTS=(11,23,47,71,101)
RS=(2,4,6,8,10,12)
HS=(0.75,1.0,1.5)
LAMBDAS=(1e-6,1e-4,1e-2,1.,100.)
MAX_ITER=2000
TOL=1e-6
FLOOR=0.05


def load_data():
 d=pd.read_csv(DATA)
 assert len(d)==1503 and d.group_id.nunique()==106
 assert (d.row_id.to_numpy()==np.arange(1,1504)).all()
 assert not d.isnull().any().any()
 keys=d[['angle_deg','chord_m','velocity_m_s']].drop_duplicates()
 assert len(keys)==106
 for g, group in d.groupby('group_id'):
  assert len(group[['angle_deg','chord_m','velocity_m_s']].drop_duplicates())==1
  assert group.frequency_Hz.nunique()==len(group)
  assert group.displacement_m.nunique()==1
 assert (d[['frequency_Hz','displacement_m']]>0).all().all()
 return d

def features(d,feature_set='F5',raw=False):
 columns=['frequency_Hz','angle_deg','chord_m','velocity_m_s']
 if feature_set=='F5':columns+=['displacement_m']
 X=d[columns].to_numpy(dtype=float).copy()
 if not raw:
  X[:,0]=np.log10(X[:,0])
  if feature_set=='F5': X[:,-1]=np.log10(X[:,-1])
 return X

def fold_pairs(X,y,g,mode='group',n_outer=10):
 if mode=='group': return list(GroupKFold(n_outer).split(X,y,g))
 return list(KFold(n_outer,shuffle=True,random_state=SEED).split(X,y))

def five_splits(X,y,g,mode='group'):
 if mode=='group': return list(GroupKFold(5).split(X,y,g))
 return list(KFold(5,shuffle=True,random_state=SEED).split(X,y))

def fcm(Z,R,seeds=RESTARTS,max_iter=MAX_ITER,tol=TOL):
 """FCM (m=2) with positive initialized memberships and declared five restarts.

Convergence: max absolute difference of consecutive membership matrices. For
each converged run recomputes center and final objective at final memberships.
"""
 n,p=Z.shape; z2=(Z*Z).sum(axis=1,keepdims=True)
 best=None; all_runs=[]
 for seed in seeds:
  rng=np.random.default_rng(int(seed)); U=rng.random((n,R))+1e-12; U/=U.sum(axis=1,keepdims=True)
  converged=False; delta=float('inf')
  for it in range(1,max_iter+1):
   UM=U*U; C=(UM.T@Z)/UM.sum(axis=0)[:,None]
   D=np.maximum(z2+(C*C).sum(axis=1)[None,:]-2*Z@C.T,1e-25)
   inv=1/D; newU=inv/inv.sum(axis=1,keepdims=True)
   delta=float(np.max(np.abs(newU-U))); U=newU
   if delta<tol: converged=True;break
  UM=U*U; C=(UM.T@Z)/UM.sum(axis=0)[:,None]
  D=np.maximum(z2+(C*C).sum(axis=1)[None,:]-2*Z@C.T,0)
  objective=float(np.sum(UM*D))
  rec={'seed':int(seed),'converged':bool(converged),'iters':it,'final_change':delta,'objective':objective}
  all_runs.append(rec)
  if converged and (best is None or (objective,seed)<(best['objective'],best['seed'])):
   weighted_variance=((UM.T @ (Z*Z))/UM.sum(0)[:,None])-C*C
   S=np.sqrt(np.maximum(weighted_variance,0))
   S=np.maximum(S,FLOOR)
   best={'C':C,'S':S,'U':U,'objective':objective,'seed':int(seed),'iters':it}
 if best is None: return None, all_runs
 return best,all_runs

def fire(Z,C,S):
 v=(Z[:,None,:]-C[None,:,:])/S[None,:,:]
 log_a=-.5*np.sum(v*v,axis=2)
 W=np.exp(log_a-logsumexp(log_a,axis=1,keepdims=True))
 assert np.isfinite(W).all()
 return W,np.max(log_a,axis=1)

def design(Z,W):
 T=np.column_stack((np.ones(Z.shape[0]),Z))
 return np.einsum('nr,nj->nrj',W,T).reshape(Z.shape[0],-1)

def coef_svd(A,b,lambda_=0.0):
 # Stable SVD spectral solution; no inverse/normal matrix formed.
 # A is P/sqrt(n); b is yc/sqrt(n), so lambda is normalized MSE penalty.
 U,s,Vh=scipy.linalg.svd(A,full_matrices=False,check_finite=False)
 proj=U.T@b
 if lambda_==0:
  tol=max(A.shape)*np.finfo(float).eps*s[0]
  factors=np.where(s>tol,1/np.maximum(s,np.finfo(float).tiny),0)
 else:
  factors=s/(s*s+lambda_)
 return Vh.T@(factors*proj)

def coef_lstsq(P,yc,lam=0., verify=False):
 n,q=P.shape
 if lam<=0:
  result=scipy.linalg.lstsq(P,yc,lapack_driver='gelsd',check_finite=False)
  return result[0]
 A=np.vstack([P/np.sqrt(n),np.sqrt(lam)*np.eye(q)])
 b=np.r_[yc/np.sqrt(n),np.zeros(q)]
 return scipy.linalg.lstsq(A,b,lapack_driver='gelsd',check_finite=False)[0]

def group_rmse(y,yp,groups):
 D=(y-yp)**2; _,inverse=np.unique(groups,return_inverse=True)
 return float(np.sqrt(np.mean(np.bincount(inverse,weights=D)/np.bincount(inverse))))

def row_rmse(y,yp,groups=None):
 return float(np.sqrt(np.mean((y-yp)**2)))

def candidate_score(y,p,g,mode='group'):
 return group_rmse(y,p,g) if mode=='group' else row_rmse(y,p)

def diagnostics(P,W,g):
 n,q=P.shape
 S=np.linalg.svd(P/np.sqrt(n),compute_uv=False)
 eqnorm=np.linalg.norm(P/np.sqrt(n),axis=0)
 eqP=(P/np.sqrt(n))/np.maximum(eqnorm,1e-300)
 S2=np.linalg.svd(eqP,compute_uv=False)
 primarytol=max(n,q)*np.finfo(float).eps*S[0]
 r=int((S>primarytol).sum()); rankrel={str(v):int((S/S[0]>v).sum()) for v in (1e-12,1e-10,1e-8,1e-6)}
 _,inv=np.unique(g,return_inverse=True); gc=np.bincount(inv); gr=np.zeros((len(gc),W.shape[1]))
 for r2 in range(W.shape[1]):gr[:,r2]=np.bincount(inv,weights=W[:,r2])/gc
 mass=gr.mean(0);eff=gr.sum(0)**2/np.maximum(np.sum(gr*gr,axis=0),1e-300)
 return {'n':n,'q':q,'rank':r,'nullity':q-r,'condition':float(S[0]/S[-1]) if S[-1]>0 else float('inf'),
 'smin':float(S[-1]),'smax':float(S[0]),'column_equilibrated_cond':float(S2[0]/S2[-1]),
 'rank_by_relative_tolerance':rankrel,'svals':S.tolist(),
 'min_group_effective_support':float(eff.min()),'median_group_effective_support':float(np.median(eff)),
 'min_group_mass':float(mass.min()),'median_group_mass':float(np.median(mass)),
 'rule_group_mass':mass.tolist(),'rule_group_eff':eff.tolist(),
 'rule_active_groups_ge_0p1':np.sum(gr>=.1,axis=0).tolist(),
 'n_low_mass_rules':int(np.sum(mass<.01)),
 'n_low_activation_rules':int(np.sum(np.sum(gr>=.1,axis=0)==0))}

def best_score_idx(scores,tol=1e-12):
 best=min(scores)
 return next(i for i,s in enumerate(scores) if s<=best+tol)

def sel_tsk(rows,kind,policy='oneSE'):
 rr=[r for r in rows if r['kind']==kind and np.isfinite(r['score'])]
 if not rr:return None
 bi=best_score_idx([r['score'] for r in rr]); best=rr[bi]
 if policy=='min_error':return dict(best,policy='min_error',best_score=best['score'])
 threshold=best['score']+best['se']
 good=[r for r in rr if r['score']<=threshold+1e-12]
 # smallest R, smaller score, larger lambda, h nearest one, smaller h
 chosen=min(good,key=lambda r:(r['R'],r['score'],-r['lambda'],abs(r['h']-1),r['h']))
 return dict(chosen,policy='oneSE',best_score=best['score'],threshold=threshold)

def stable_seed(feature,fold,branch,rep):
 v=f'{SEED}|{feature}|{fold}|{branch}|{rep}'.encode()
 return int.from_bytes(hashlib.sha256(v).digest()[:4],'little')

def save_json(path,data):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8')

def json_conv(obj):
 if isinstance(obj,np.generic):return obj.item()
 if isinstance(obj,np.ndarray):return obj.tolist()
 return str(obj)
