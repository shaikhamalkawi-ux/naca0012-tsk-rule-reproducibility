"""Configuration bootstrap audit, with archived replicate predictions and matching.

Each of 40 tasks is independent and resumable. B=500 fixed antecedents and B=200
relearned antecedents (five FCM restarts) per task. Both LS and ridge are paired
in every draw; fit failures/unmatched rules retained as missing, never replaced.
"""
import os,time,json,sys,argparse,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,scipy
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from common import *

def locate(feature,fold,structure):
 root=ROOT/'results'/'tsk'/f'group_log_{feature}'/f'fold_{fold:02d}'
 key='TSK-Ridge-oneSE' if structure=='compact' else 'TSK-Ridge-fixedR6h1'
 return root/(key+'.npz'), root/key

def same_coords_beta(beta,scaler_b_mean,scaler_b_scale,target_mean,scaler_p_mean,scaler_p_scale):
 p=len(scaler_p_mean);a=beta.reshape(-1,p+1)
 slopes=a[:,1:]*(scaler_p_scale/scaler_b_scale)[None,:]
 full_intercept=target_mean+a[:,0]+np.sum(a[:,1:]*(scaler_p_mean-scaler_b_mean)[None,:]/scaler_b_scale[None,:],axis=1)
 return np.column_stack((full_intercept,slopes))

def local_preds(Z,theta):
 return theta[0]+Z@theta[1:]

def match_matrix(Wparent,Wnew,groups):
 _,inverse=np.unique(groups,return_inverse=True);count=np.bincount(inverse)
 weights=1/count[inverse] /len(count)
 R,S=Wparent.shape[1],Wnew.shape[1]
 sim=np.empty((R,S))
 for r in range(R):
  w=Wparent[:,r]
  for s in range(S):
   v=Wnew[:,s]
   den=np.sum(weights*np.maximum(w,v))
   sim[r,s]=float(np.sum(weights*np.minimum(w,v))/den) if den>0 else 0.
 return sim

def assign(sim,threshold=.5):
 R,S=sim.shape
 gain=np.concatenate([sim-threshold,np.zeros((R,R))],axis=1)
 i,j=linear_sum_assignment(-gain)
 out=np.full(R,-1,int)
 for a,b in zip(i,j):
  if b<S and sim[a,b]>=threshold:out[a]=b
 return out

def dispersion(preds,groups):
 # pred matrix: B x n_test; evaluate group-weighted pointwise bootstrap variance
 if len(preds)<2: return None
 ptsd=np.std(np.stack(preds),axis=0,ddof=1)
 _,ix=np.unique(groups,return_inverse=True)
 gv=np.bincount(ix,weights=ptsd**2)/np.bincount(ix)
 return dict(pred_disp_rms=float(np.sqrt(np.mean(gv))),point_sd_median=float(np.median(ptsd)),
             point_sd_p90=float(np.quantile(ptsd,.9)))

def one_task(feature,fold,structure):
 with threadpool_limits(limits=1):
  started=time.time();parfile,_=locate(feature,fold,structure)
  if not parfile.exists():return f'MISSING_PARENT {feature} {fold} {structure}'
  out=ROOT/'results'/'bootstrap'/f'{feature}_fold{fold:02d}_{structure}';out.mkdir(parents=True,exist_ok=True)
  if (out/'COMPLETE.json').exists():return f'EXISTS {feature} {fold} {structure}'
  par=np.load(parfile); d=load_data();F=features(d,feature);y=d.scaled_SPL_dB.to_numpy(float);groups=d.group_id.to_numpy(str)
  tr=par['tr'].astype(int);te=par['te'].astype(int)
  Zp=(F[tr]-par['train_scaler_mean'])/par['train_scaler_scale'];Zt=(F[te]-par['train_scaler_mean'])/par['train_scaler_scale']
  g=groups[tr];gt=groups[te];yy=y[tr];parent_ymean=float(par['train_target_mean'])
  C=par['centers'];S=par['widths'];W,_=fire(Zp,C,S);P=design(Zp,W)
  Rt=int(par['selected_R']);lam=float(par['lambda_']);h=float(par['h'])
  parent_beta=par['coef'];parent_full=same_coords_beta(parent_beta,par['train_scaler_mean'],par['train_scaler_scale'],parent_ymean,
    par['train_scaler_mean'],par['train_scaler_scale'])
  Wte,_=fire(Zt,C,S);Q=design(Zt,Wte)
  _,inv=np.unique(g,return_inverse=True);ug=np.unique(g);grpidx=[np.flatnonzero(g==u) for u in ug]
  unique_counts=len(ug)
  parent_anchors=[np.flatnonzero(W[:,r]>=.1) for r in range(Rt)]
  p=len(par['train_scaler_mean'])
  outcomes=[];stores={};summary=[]
  for branch,B in [('fixed',500),('relearn',200)]:
   TSKpred={'LS':[],'Ridge':[]}; coefficient={'LS':[],'Ridge':[]}; matching=[];all_seeds=[];failures=[];sims=[]
   localsums={v:[np.zeros(len(k)) for k in parent_anchors] for v in ('LS','Ridge')}
   localsq={v:[np.zeros(len(k)) for k in parent_anchors] for v in ('LS','Ridge')}
   localcnt={v:np.zeros(Rt,int) for v in ('LS','Ridge')}
   for b in range(B):
    seed=stable_seed(feature,fold,branch+'_'+structure,b); rng=np.random.default_rng(seed)
    groupdraw=rng.choice(unique_counts,size=unique_counts,replace=True)
    ix=np.concatenate([grpidx[j] for j in groupdraw]);all_seeds.append(seed)
    rec={'feature':feature,'fold':fold,'structure':structure,'branch':branch,'replicate':b,'seed':seed,'drawn_groups':len(groupdraw)}
    try:
     if branch=='fixed':
      ZZ=Zp[ix];PP=P[ix];yb=yy[ix];Ymean=parent_ymean; CC=C;SS=S
      scaler_b_mean=par['train_scaler_mean'];scaler_b_scale=par['train_scaler_scale'];
      QQ=Q;WWp=W
      M=np.arange(Rt,dtype=int);SIM=np.eye(Rt)
     else:
      fb=F[tr][ix];yb=yy[ix];sc=StandardScaler().fit(fb)
      scaler_b_mean=sc.mean_;scaler_b_scale=sc.scale_;ZZ=sc.transform(fb)
      fcmfit,runlog=fcm(ZZ,Rt)
      if fcmfit is None:raise ArithmeticError('FIVE_FCM_RESTARTS_DID_NOT_CONVERGE')
      CC=fcmfit['C'];SS=fcmfit['S']*h;Ymean=float(np.mean(yb));
      WW,_=fire(ZZ,CC,SS);PP=design(ZZ,WW)
      Qt,_=fire((F[te]-scaler_b_mean)/scaler_b_scale,CC,SS)
      QQ=design((F[te]-scaler_b_mean)/scaler_b_scale,Qt)
      anchorZ=(F[tr]-scaler_b_mean)/scaler_b_scale
      Wnew,_=fire(anchorZ,CC,SS)
      SIM=match_matrix(W,Wnew,g);M=assign(SIM,.5)
      rec['fcm_iterations']=int(fcmfit['iters']);rec['FCM_seed']=int(fcmfit['seed'])
     rec['status']='success';rec['n_matched']=int(np.sum(M>=0))
     rec['n_above_0p3']=int(np.sum(assign(SIM,.3)>=0));rec['n_above_0p7']=int(np.sum(assign(SIM,.7)>=0))
     y_center=yb-Ymean
     models={'LS':coef_lstsq(PP,y_center,0.),'Ridge':coef_lstsq(PP,y_center,lam)}
     for name,beta in models.items():
      yp=Ymean+QQ@beta
      if not np.isfinite(yp).all(): raise ValueError('nonfinite prediction')
      TSKpred[name].append(yp)
      # each corresponding rule's coefficients expressed in parent's standardized feature frame
      cur=same_coords_beta(beta,scaler_b_mean,scaler_b_scale,Ymean,par['train_scaler_mean'],par['train_scaler_scale'])
      aligned=np.full((Rt,p+1),np.nan)
      for r in range(Rt):
       k=M[r]
       if k<0:continue
       aligned[r]=cur[k]
       anchors=parent_anchors[r]
       if len(anchors)==0:continue
       values=local_preds(Zp[anchors],cur[k])
       localsums[name][r]+=values;localsq[name][r]+=values**2;localcnt[name][r]+=1
      coefficient[name].append(aligned)
     matching.append(M);sims.append(SIM)
    except Exception as e:
     rec['status']='FAILED';rec['error']=f'{type(e).__name__}: {e}';failures.append(rec)
    outcomes.append(rec)
   success=len(matching)
   np.savez_compressed(out/f'{branch}_replicate_results.npz',
    LS_preds=np.array(TSKpred['LS']),Ridge_preds=np.array(TSKpred['Ridge']),
    LS_coeff=np.array(coefficient['LS']),Ridge_coeff=np.array(coefficient['Ridge']),
    match=np.array(matching),sim=np.array(sims),seeds=np.array(all_seeds),
    parent_support_sizes=np.array([len(k) for k in parent_anchors]),
    parent_coeff=parent_full,parent_R=Rt,parent_h=h,parent_lambda=lam)
   for typ in ('LS','Ridge'):
    d0=dispersion(TSKpred[typ],gt)
    coeff=np.array(coefficient[typ]); parent=parent_full
    valid=np.isfinite(coeff)
    absdev=np.abs(coeff-parent[None,:,:]); sign=(np.sign(coeff)==np.sign(parent)[None,:,:])
    coefs=np.nanstd(coeff,axis=0,ddof=1) if len(coeff)>1 else np.full((Rt,p+1),np.nan)
    cf=coefs[np.isfinite(coefs)]
    local_rms=[]
    for r in range(Rt):
     nmatched=localcnt[typ][r]
     if nmatched>=2 and len(parent_anchors[r])>0:
      var=(localsq[typ][r]-(localsums[typ][r]**2)/nmatched)/(nmatched-1)
      local_rms.append(float(np.sqrt(np.mean(np.maximum(var,0)))))
    if d0 is None:d0={'pred_disp_rms':None,'point_sd_median':None,'point_sd_p90':None}
    m=np.array(matching)
    summary.append({'feature':feature,'fold':fold,'structure':structure,'branch':branch,'fit':typ,'B_expected':B,'B_success':success,'B_failure':B-success,
      'R':Rt,'h':h,'lambda':lam,'matched_rule_fraction':float(np.mean(m>=0)) if len(m) else None,
      'fraction_fully_matched':float(np.mean(np.all(m>=0,axis=1))) if len(m) else None,
      'coef_sd_median':float(np.median(cf)) if len(cf) else None,
      'coef_sd_p90':float(np.quantile(cf,.9)) if len(cf) else None,
      'local_disp_rms_median':float(np.median(local_rms)) if local_rms else None,
      'local_disp_rms_p90':float(np.quantile(local_rms,.9)) if local_rms else None,**d0})
   pd.DataFrame([r for r in outcomes if r['branch']==branch]).to_csv(out/f'{branch}_replicate_register.csv',index=False)
   print('BOOT',feature,fold,structure,branch,success,'/',B,'sec',round(time.time()-started,1),flush=True)
  pd.DataFrame(summary).to_csv(out/'SUMMARY.csv',index=False)
  save_json(out/'COMPLETE.json',{'feature':feature,'fold':fold,'structure':structure,'declared_B_fixed':500,'declared_B_relearn':200,
    'n_declared_replicates':700,'n_logged_replicates':len(outcomes),
    'n_success_fixed':sum(r['status']=='success' for r in outcomes if r['branch']=='fixed'),
    'n_success_relearn':sum(r['status']=='success' for r in outcomes if r['branch']=='relearn'),
    'elapsed_sec':time.time()-started,'algorithm':'FCM5 + weighted Jaccard dummy Hungarian'})
  return f'DONE {feature} {fold} {structure} sec={time.time()-started:.1f}'

def main(workers=4):
 tasks=[(f,fold,s) for f in ('F4','F5') for fold in range(10) for s in ('compact','reference')]
 print('tasks',len(tasks),'each 700 bootstrap draws, paired LS/Ridge',flush=True)
 with ProcessPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(one_task,*a) for a in tasks]
  for future in as_completed(futures):
   try:print(future.result(),flush=True)
   except Exception as e: print('TASK ERROR',repr(e),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.workers)
