import os, json, math
for v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[v]='1'
from pathlib import Path
import numpy as np, pandas as pd, scipy.linalg
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

SEEDS=(11,23,47,71,101); RSET=(2,4,6); HSET=(.75,1.,1.5); LSET=(1e-6,1e-4,1e-2,1.,100.); TOL=1e-6; MAX_ITER=2000; FLOOR=.05

def fcm(Z,R):
    n,p=Z.shape; z2=(Z*Z).sum(1,keepdims=True); best=None
    for seed in SEEDS:
        rng=np.random.default_rng(seed); U=rng.random((n,R))+1e-12; U/=U.sum(1,keepdims=True); conv=False
        for it in range(1,MAX_ITER+1):
            UM=U*U; C=(UM.T@Z)/UM.sum(0)[:,None]; D=np.maximum(z2+(C*C).sum(1)[None,:]-2*Z@C.T,1e-25); inv=1/D; new=inv/inv.sum(1,keepdims=True)
            delta=float(np.max(abs(new-U))); U=new
            if delta<TOL: conv=True; break
        UM=U*U; C=(UM.T@Z)/UM.sum(0)[:,None]; D=np.maximum(z2+(C*C).sum(1)[None,:]-2*Z@C.T,0); obj=float(np.sum(UM*D))
        if conv and (best is None or (obj,seed)<(best['obj'],best['seed'])):
            vari=(UM.T@(Z*Z))/UM.sum(0)[:,None]-C*C
            best={'C':C,'S0':np.maximum(np.sqrt(np.maximum(vari,0)),FLOOR),'obj':obj,'seed':seed,'iters':it}
    return best

def fire(Z,C,S, return_logmax=False):
    q=(Z[:,None,:]-C[None,:,:])/S[None,:,:]; la=-.5*np.sum(q*q,axis=2); W=np.exp(la-logsumexp(la,axis=1,keepdims=True))
    return (W,la.max(1)) if return_logmax else W

def design(Z,W):
    T=np.c_[np.ones(len(Z)),Z]; return np.einsum('nr,nj->nrj',W,T).reshape(len(Z),-1)

def gram_betas(P,y):
    n=P.shape[0]; ym=float(y.mean()); yc=y-ym; G=P.T@P/n; rhs=P.T@yc/n; ev,V=scipy.linalg.eigh(G,check_finite=False); a=V.T@rhs
    return np.column_stack([V@(a/(ev+lam)) for lam in LSET]),ym

def ridge(P,y,lam):
    n,q=P.shape; A=np.vstack([P/np.sqrt(n),np.sqrt(lam)*np.eye(q)]); v=np.r_[(y-y.mean())/np.sqrt(n),np.zeros(q)]
    return scipy.linalg.lstsq(A,v,lapack_driver='gelsd',check_finite=False)[0]

def group_rmse(y,p,g):
    e=(y-p)**2; u,iv=np.unique(g,return_inverse=True); return float(np.sqrt(np.mean(np.bincount(iv,weights=e)/np.bincount(iv))))

def geom_equal_rmse(y,p,geom):
    e=(y-p)**2; u,iv=np.unique(geom,return_inverse=True); return float(np.sqrt(np.mean(np.bincount(iv,weights=e)/np.bincount(iv))))

def features(d):
    return np.c_[np.log10(d.frequency_Hz.to_numpy(float)),d.angle_deg.to_numpy(float),d.trip_flag.to_numpy(float),d.chord_m.to_numpy(float),d.velocity_m_s.to_numpy(float)], ['log10_frequency','angle_deg','trip_flag','chord_m','velocity_m_s']

def tsk_fit(X,y,R,h,lam):
    sc=StandardScaler().fit(X); Z=sc.transform(X); fc=fcm(Z,R)
    if fc is None: return None
    S=fc['S0']*h; W=fire(Z,fc['C'],S); P=design(Z,W); b=ridge(P,y,lam)
    return {'sc':sc,'C':fc['C'],'S':S,'P':P,'beta':b,'offset':float(y.mean()),'R':R,'h':h,'lam':lam,'seed':fc['seed']}

def tsk_predict(m,X,return_details=False):
    Z=m['sc'].transform(X); W,lm=fire(Z,m['C'],m['S'],True); P=design(Z,W); pred=m['offset']+P@m['beta']
    return (pred,W,P,lm,Z) if return_details else pred

def select_tsk(X,y,g, inner_splits=4):
    ngrp=len(np.unique(g)); ns=min(inner_splits,ngrp)
    splits=list(GroupKFold(ns).split(X,y,g)); rec=[]
    for R in RSET:
        losses={(h,l):{} for h in HSET for l in LSET}; fscores={(h,l):[] for h in HSET for l in LSET}; valid=True
        for itr,iva in splits:
            sc=StandardScaler().fit(X[itr]); Ztr=sc.transform(X[itr]); Zva=sc.transform(X[iva]); fc=fcm(Ztr,R)
            if fc is None: valid=False; break
            for h in HSET:
                Ptr=design(Ztr,fire(Ztr,fc['C'],fc['S0']*h)); Pva=design(Zva,fire(Zva,fc['C'],fc['S0']*h)); B,ym=gram_betas(Ptr,y[itr]); yp=Pva@B+ym
                for jj,lam in enumerate(LSET):
                    val=group_rmse(y[iva],yp[:,jj],g[iva]); fscores[(h,lam)].append(val)
                    for gg in np.unique(g[iva]):
                        s=g[iva]==gg; losses[(h,lam)][gg]=float(np.mean((y[iva][s]-yp[s,jj])**2))
        ug=np.unique(g)
        for h in HSET:
            for lam in LSET:
                comp=valid and len(losses[(h,lam)])==len(ug); score=float(np.sqrt(np.mean(list(losses[(h,lam)].values())))) if comp else np.inf
                se=float(np.std(fscores[(h,lam)],ddof=1)/np.sqrt(ns)) if comp and len(fscores[(h,lam)])>1 else np.nan
                rec.append({'R':R,'h':h,'lambda':lam,'inner_rmse':score,'inner_se':se,'valid':bool(comp)})
    c=pd.DataFrame(rec)
    # best per R, then one-SE compact across R
    byR=[]
    for R in RSET:
        z=c[(c.R==R)&c.valid].sort_values(['inner_rmse','h','lambda'])
        if len(z): byR.append(z.iloc[0].to_dict())
    ch=pd.DataFrame(byR); best=ch.sort_values(['inner_rmse','R']).iloc[0]; thr=float(best.inner_rmse+best.inner_se)
    sel=ch[ch.inner_rmse<=thr+1e-12].copy(); sel['hdist']=abs(sel.h-1.0)
    sel=sel.sort_values(['R','inner_rmse','lambda','hdist','h'],ascending=[True,True,False,True,True]).iloc[0]
    return sel,c

def select_et(X,y,g):
    grid=list(ParameterGrid({'min_samples_leaf':[1,3,5],'max_features':[0.7,1.0],'max_depth':[12,None]}))
    ns=min(4,len(np.unique(g))); splits=list(GroupKFold(ns).split(X,y,g)); rows=[]
    for k,pa in enumerate(grid):
        per={}
        for tr,va in splits:
            m=ExtraTreesRegressor(n_estimators=400,random_state=20261007,n_jobs=1,bootstrap=False,**pa)
            m.fit(X[tr],y[tr]); pp=m.predict(X[va])
            for gg in np.unique(g[va]):
                s=g[va]==gg; per[gg]=float(np.mean((y[va][s]-pp[s])**2))
        rows.append({**pa,'inner_rmse':float(np.sqrt(np.mean(list(per.values())))),'grid_index':k})
    rr=pd.DataFrame(rows).sort_values(['inner_rmse','grid_index']); return rr.iloc[0],rr

def main():
    src=Path('/mnt/data/r17ext/57_H3_Airfoil_R17_SouthamptonExternal_20261011/data/SOUTHAMPTON_CANONICAL.csv')
    d=pd.read_csv(src); X,fn=features(d); y=d.response_norm.to_numpy(float); g=d.group_id.to_numpy(str); geom=d.geometry_id.to_numpy(str)
    out=Path('/mnt/data/57_H3_Airfoil_R19_LOGO_Candidate'); out.mkdir(exist_ok=True); (out/'candidates').mkdir(exist_ok=True); (out/'states').mkdir(exist_ok=True)
    folds=[]; preds=[]
    with threadpool_limits(limits=1):
      for hold in sorted(d.geometry_id.unique()):
        te=np.flatnonzero(geom==hold); tr=np.flatnonzero(geom!=hold)
        sel,ct=select_tsk(X[tr],y[tr],g[tr]); ct.to_csv(out/'candidates'/f'TSK_holdout_{hold}.csv',index=False)
        m=tsk_fit(X[tr],y[tr],int(sel.R),float(sel.h),float(sel['lambda']))
        pt,Wt,Pt,lm,Zt=tsk_predict(m,X[te],True); rm=group_rmse(y[te],pt,g[te])
        etsel,ce=select_et(X[tr],y[tr],g[tr]); ce.to_csv(out/'candidates'/f'ET_holdout_{hold}.csv',index=False)
        pa={k:(None if pd.isna(etsel[k]) else etsel[k]) for k in ['min_samples_leaf','max_features','max_depth']}
        if pa['max_depth'] is not None: pa['max_depth']=int(pa['max_depth'])
        pa['min_samples_leaf']=int(pa['min_samples_leaf']); pa['max_features']=float(pa['max_features'])
        et=ExtraTreesRegressor(n_estimators=400,random_state=20261007,n_jobs=1,bootstrap=False,**pa); et.fit(X[tr],y[tr]); pe=et.predict(X[te]); rme=group_rmse(y[te],pe,g[te])
        # OOD diagnostics: standardized coordinate distance and antecedent absolute log firing
        zdist=np.linalg.norm(Zt,axis=1)
        folds.append({'heldout_geometry':hold,'rows':len(te),'spectra':len(np.unique(g[te])),'TSK_rmse':rm,'ET_rmse':rme,'TSK_R':int(sel.R),'TSK_h':float(sel.h),'TSK_lambda':float(sel['lambda']),'TSK_inner_rmse':float(sel.inner_rmse),'ET_inner_rmse':float(etsel.inner_rmse),'median_log_best_firing':float(np.median(lm)),'min_log_best_firing':float(np.min(lm)),'median_standardized_radius':float(np.median(zdist)),'max_standardized_radius':float(np.max(zdist))})
        for ii,idx in enumerate(te): preds.append({'row_id':int(d.row_id.iloc[idx]),'heldout_geometry':hold,'group_id':g[idx],'actual':y[idx],'TSK_pred':pt[ii],'ET_pred':pe[ii],'log_best_firing':lm[ii],'standardized_radius':zdist[ii]})
        np.savez_compressed(out/'states'/f'{hold}.npz',P_train=m['P'],P_test=Pt,beta=m['beta'],lam=np.array(m['lam']),offset=np.array(m['offset']),y_train=y[tr],y_test=y[te],train_rows=d.row_id.iloc[tr].to_numpy(),test_rows=d.row_id.iloc[te].to_numpy(),train_geom=geom[tr],test_geom=geom[te],train_group=g[tr],test_group=g[te],feature_names=np.array(fn),epsilon=np.array((.1/6.896361277488791)*np.std(y[tr],ddof=0)))
        print(hold,rm,rme, 'R',int(sel.R),'h',float(sel.h),'lam',float(sel['lambda']),flush=True)
    fd=pd.DataFrame(folds); pd.DataFrame(preds).to_csv(out/'LOGO_PREDICTIONS.csv',index=False); fd.to_csv(out/'LOGO_FOLDS.csv',index=False)
    p=pd.DataFrame(preds)
    summary={'n_rows':len(d),'n_geometries':d.geometry_id.nunique(),'n_spectra':d.group_id.nunique(),'TSK_equal_geometry_RMSE':geom_equal_rmse(p.actual.to_numpy(),p.TSK_pred.to_numpy(),p.heldout_geometry.to_numpy()),'ET_equal_geometry_RMSE':geom_equal_rmse(p.actual.to_numpy(),p.ET_pred.to_numpy(),p.heldout_geometry.to_numpy()),'TSK_equal_spectrum_RMSE':group_rmse(p.actual.to_numpy(),p.TSK_pred.to_numpy(),p.group_id.to_numpy()),'ET_equal_spectrum_RMSE':group_rmse(p.actual.to_numpy(),p.ET_pred.to_numpy(),p.group_id.to_numpy())}
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
