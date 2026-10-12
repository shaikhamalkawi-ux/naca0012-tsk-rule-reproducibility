from pathlib import Path
import sys, numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent/'R17_Southampton_External_Reproducibility'/'code'))
from joint_core import minimum_cost
from scipy.optimize import linprog
ETA=.01; GAMMA=.001; TOL=5e-7
ROOT=Path(__file__).resolve().parent/'R19_Clean_12grid'; OUT=ROOT/'statement'; OUT.mkdir(exist_ok=True)

def objective_parts(z):
 P=z['P_train']; beta=z['beta']; lam=float(z['lam']); y=z['y_train']; off=float(z['offset']); yc=y-off; res=yc-P@beta
 J=float(np.mean(res**2)+lam*np.dot(beta,beta)); H=P.T@P/len(P)+lam*np.eye(P.shape[1]); st=float(np.max(np.abs(P.T@(P@beta-yc)/len(P)+lam*beta))); return H,J,st

def scaled_pointwise(P,j,eps):
 norms=np.linalg.norm(P,axis=0); S=1/np.maximum(norms,1e-300); Q=P*S[None,:]; c=np.zeros(P.shape[1]); c[j]=S[j]; B=np.vstack([Q,-Q]); ub=np.full(2*len(P),eps)
 res=linprog(-c,A_ub=B,b_ub=ub,bounds=[(None,None)]*P.shape[1],method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
 if not res.success:return {'status':'failed'}
 u=np.asarray(res.x); d=S*u; raw=-np.asarray(res.ineqlin.marginals); mu=np.maximum(raw,0.); return {'status':'optimal','d':d,'primal':float(d[j]),'dual':float(eps*mu.sum())}

def solve(P,H,J,j,bj,eps,gamma):
 target=-(1+gamma)*bj; lp=scaled_pointwise(P,j,eps)
 if lp['status']!='optimal': return {'class':'UNRES'}
 if lp['dual']+5e-9<abs(target): return {'class':'PROTECTED_POINT','target':target}
 if lp['primal']+5e-9<abs(target): return {'class':'UNRES','target':target}
 init=np.asarray(lp['d'])*(target/lp['primal']); q=minimum_cost(P,H,J,j,target,eps,initial=init)
 if q.get('status')!='optimal': return {'class':'UNRES','target':target}
 if q['max_training']<=eps+TOL and q['target_error']<=1e-6 and q['cost']<=ETA+2e-7: return {'class':'WITNESS','target':target,'d':q['d'],'cost':q['cost'],'dual':q['dual_lower'],'max_train':q['max_training']}
 if q['dual_lower']>ETA+1e-7 or (q['cost']>ETA and q['dual_lower']>ETA-5e-7): return {'class':'PROTECTED_COST','target':target,'cost':q['cost'],'dual':q['dual_lower']}
 return {'class':'UNRES','target':target,'cost':q['cost'],'dual':q['dual_lower']}

rows=[]
for sp in sorted((ROOT/'states').glob('*.npz')):
 z=np.load(sp,allow_pickle=False); P=z['P_train']; Pt=z['P_test']; beta=z['beta']; eps=float(z['epsilon']); R=(len(beta)//6); H,J,station=objective_parts(z); hold=sp.stem
 for r in range(R):
  j=r*6+1; bj=float(beta[j]); zero=solve(P,H,J,j,bj,eps,0.0); strict=solve(P,H,J,j,bj,eps,GAMMA)
  if zero['class'].startswith('PROTECTED'): cls='PROTECTED_FROM_ZERO'
  elif strict['class']=='WITNESS': cls='STRICT_REVERSAL_WITNESS'
  else: cls='UNRESOLVED_BOUNDARY'
  rec={'heldout_geometry':hold,'rule':r,'R':R,'coefficient':bj,'epsilon':eps,'eta':ETA,'classification':cls,'zero_class':zero['class'],'strict_class':strict['class'],'J0':J,'stationarity_inf':station}
  if strict.get('class')=='WITNESS':
   d=np.asarray(strict['d']); rec.update(strict_cost=float(strict['cost']),strict_max_train=float(strict['max_train']),heldout_max_change=float(np.max(np.abs(Pt@d))),heldout_rms_change=float(np.sqrt(np.mean((Pt@d)**2))),heldout_within_epsilon=bool(np.max(np.abs(Pt@d))<=eps+1e-12))
  rows.append(rec)
df=pd.DataFrame(rows); df.to_csv(OUT/'LOGO_STATEMENT_AUDIT.csv',index=False)
s=[]
for h,g in list(df.groupby('heldout_geometry'))+[('ALL',df)]:
 s.append({'heldout_geometry':h,'statements':len(g),'protected':int((g.classification=='PROTECTED_FROM_ZERO').sum()),'reversible':int((g.classification=='STRICT_REVERSAL_WITNESS').sum()),'unresolved':int((g.classification=='UNRESOLVED_BOUNDARY').sum()),'reversible_transport_within_epsilon':int(g.get('heldout_within_epsilon',pd.Series(False,index=g.index)).fillna(False).sum())})
pd.DataFrame(s).to_csv(OUT/'LOGO_STATEMENT_SUMMARY.csv',index=False); print(pd.DataFrame(s).to_string(index=False))
