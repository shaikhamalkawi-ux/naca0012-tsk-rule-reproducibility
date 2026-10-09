"""Faithful independent nested group-aware TSK run, with per-candidate evidence."""
import os,sys,time,json,argparse
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from common import *

def run(feature_set='F5',mode='group',raw=False):
 d=load_data(); X=features(d,feature_set,raw); y=d.scaled_SPL_dB.to_numpy(float); g=d.group_id.to_numpy(str)
 tag=f'{mode}_{"raw" if raw else "log"}_{feature_set}'
 root=ROOT/'results'/'tsk'/tag;root.mkdir(parents=True,exist_ok=True)
 rows=[]; preds=[]; selections=[]; fs=[]; errors=[]
 outer=fold_pairs(X,y,g,mode)
 for fold,(tr,te) in enumerate(outer):
  t0=time.time(); ot=root/f'fold_{fold:02d}';ot.mkdir(exist_ok=True)
  if (ot/'COMPLETE.json').exists():print('SKIP complete',tag,fold,flush=True);continue
  ts=[]; inner=five_splits(X[tr],y[tr],g[tr],mode)
  for ii,(itr,ival) in enumerate(inner):
   ttr=tr[itr];tva=tr[ival]
   assert len(set(ttr)&set(tva))==0
   if mode=='group':assert not set(g[ttr])&set(g[tva])
   scale=StandardScaler().fit(X[ttr]); A=scale.transform(X[ttr]);V=scale.transform(X[tva]);yc=y[ttr]-y[ttr].mean()
   for R in RS:
    f,info=fcm(A,R)
    if f is None:
     errors.append({'tag':tag,'outer':fold,'inner':ii,'R':R,'error':'NO_CONVERGED_RESTART','restart_log':info})
     continue
    C,S=f['C'],f['S']
    for h in HS:
     W,_=fire(A,C,h*S); WV,_=fire(V,C,h*S)
     P=design(A,W);Q=design(V,WV);As=P/np.sqrt(len(ttr));bs=yc/np.sqrt(len(ttr))
     us,ss,vh=scipy.linalg.svd(As,full_matrices=False,check_finite=False)
     aa=us.T@bs
     for kind,lam in [('LS',0.)]+[('Ridge',l) for l in LAMBDAS]:
      if lam==0:
       tol=max(As.shape)*np.finfo(float).eps*ss[0]; coef=vh.T@(np.where(ss>tol,1/ss,0)*aa)
      else:coef=vh.T@((ss/(ss*ss+lam))*aa)
      pred=y[ttr].mean()+Q@coef
      gr=candidate_score(y[tva],pred,g[tva],mode)
      ts.append({'kind':kind,'R':R,'h':h,'lambda':lam,'inner':ii,'rmse':gr,'restart_seed':f['seed'],'restart_iters':f['iters'],'outer':fold})
   print(f'TSK {tag} outer={fold} inner={ii} done',flush=True)
  if not ts:
   errors.append({'tag':tag,'outer':fold,'error':'NO_INNER_CANDIDATES'});continue
  inner_df=pd.DataFrame(ts); inner_df.to_csv(ot/'inner_candidate_folds.csv',index=False)
  agg=[]
  for (kind,R,h,lam),gg in inner_df.groupby(['kind','R','h','lambda'],sort=True):
   if len(gg)!=5:continue
   vals=gg.sort_values('inner').rmse.to_numpy(float)
   agg.append({'kind':kind,'R':int(R),'h':float(h),'lambda':float(lam),'score':float(np.sqrt(np.mean(vals**2))),
               'se':float(vals.std(ddof=1)/np.sqrt(5)),'fold_scores':vals.tolist(),
               'outer':fold})
  # protocol prescribes pooled equal-group score; GroupKFold group counts differ by at most one,
  # so recompute pooling weighted by actual inner-test group counts.
  if mode=='group':
   group_counts=[len(np.unique(g[tr[iv]])) for it,iv in inner]
   for r in agg:r['score']=float(np.sqrt(np.average(np.square(r['fold_scores']),weights=group_counts)))
  cand=pd.DataFrame([{k:v for k,v in r.items() if k!='fold_scores'} for r in agg]);cand.to_csv(ot/'inner_candidates.csv',index=False)
  sels=[]
  for kind in ['LS','Ridge']:
   for policy in ['oneSE','min_error']:
    s=sel_tsk(agg,kind,policy)
    if s is not None:sels.append(s)
  # fixed ref R6,h1 by best-inner lambdas
  ref=[a for a in agg if a['R']==6 and abs(a['h']-1)<1e-9 and a['kind']=='Ridge']
  if ref: sels.append(dict(ref[best_score_idx([r['score'] for r in ref])],policy='fixedR6h1',best_score=min(r['score'] for r in ref)))
  save_json(ot/'selection.json',sels)
  selections.extend([{k:v for k,v in s.items() if k!='fold_scores'} for s in sels])
  scale=StandardScaler().fit(X[tr]); A=scale.transform(X[tr]); B=scale.transform(X[te]);yc=y[tr]-y[tr].mean()
  learned={}
  for s in sels:
   key=(s['R'],s['h'])
   if key not in learned:
    f,info=fcm(A,s['R'])
    if f is None:errors.append({'tag':tag,'outer':fold,'error':'NONCONVERGED_OUTER','R':s['R'],'h':s['h']});continue
    learned[key]=(f,info)
   f,info=learned[key]; C,S=f['C'],f['S']*s['h'];W,logs=fire(A,C,S);WV,_=fire(B,C,S)
   P=design(A,W);Q=design(B,WV)
   diag=diagnostics(P,W,g[tr])
   # selected lambda and paired 0 LS at same antecedents, for comparison without R confounding
   fit_lam=s['lambda'] if s['kind']=='Ridge' else 0
   beta=coef_lstsq(P,yc,fit_lam)
   predicted=y[tr].mean()+Q@beta
   name=f"TSK-{s['kind']}-{s['policy']}"
   if s['policy']=='fixedR6h1': name='TSK-Ridge-fixedR6h1'
   for ix,val in zip(te,predicted):
    preds.append({'feature_set':feature_set,'mode':mode,'representation':'raw' if raw else 'log','outer_fold':fold,'model':name,
                  'row_id':int(d.row_id.iloc[ix]),'group_id':str(g[ix]),'y':float(y[ix]),'pred':float(val)})
   row={'feature_set':feature_set,'mode':mode,'representation':'raw' if raw else 'log','fold':fold,'name':name,'R':s['R'],'h':s['h'],'lambda':fit_lam,
        'RMSE_row':row_rmse(y[te],predicted),'RMSE_group':group_rmse(y[te],predicted,g[te]),
        'FCM_iter':f['iters'],'FCM_seed':f['seed'],**{k:v for k,v in diag.items() if not isinstance(v,(list,dict))}}
   rows.append(row)
   # preserve full reference fold structure to allow auditable B=500/B=200 subsequent work
   fpath=ot/f'{name}.npz'
   np.savez_compressed(fpath,tr=tr,te=te,train_scaler_mean=scale.mean_,train_scaler_scale=scale.scale_,
                       train_target_mean=y[tr].mean(),centers=C,widths=S,coef=beta,selected_R=s['R'],h=s['h'],lambda_=fit_lam,
                       W_train=W,logfiring_train=logs,restart_iters=f['iters'],restart_seed=f['seed'])
   save_json(ot/f'{name}_diagnostics.json',diag)
   # paired same antecedent LS diagnostic for ridge selected
   if s['kind']=='Ridge' and s['policy']=='oneSE':
    beta0=coef_lstsq(P,yc,0)
    np.savez_compressed(ot/'TSK-LS-at-RidgeStructure.npz',tr=tr,te=te,centers=C,widths=S,
                        train_scaler_mean=scale.mean_,train_scaler_scale=scale.scale_,
                        train_target_mean=y[tr].mean(),coef=beta0,selected_R=s['R'],h=s['h'],lambda_=0)
  pd.DataFrame(rows).query('fold==@fold').to_csv(ot/'fold_diagnostics.csv',index=False)
  pp=pd.DataFrame(preds); pp[pp.outer_fold==fold].to_csv(ot/'fold_predictions.csv',index=False)
  save_json(ot/'COMPLETE.json',{'tag':tag,'fold':fold,'train_groups':len(np.unique(g[tr])),'test_groups':len(np.unique(g[te])),
              'test_indices':te.tolist(),'train_indices':tr.tolist(),'candidate_count':len(cand),'elapsed_sec':time.time()-t0,
              'failed_candidates':sum(1 for e in errors if e.get('outer')==fold),'model_count':int(sum(s2['outer']==fold for s2 in sels))})
  print(f'COMPLETE TSK {tag} outer={fold} {time.time()-t0:.1f}s',flush=True)
 # aggregate from per fold files in fixed order
 pp=pd.concat([pd.read_csv(root/f'fold_{fold:02d}'/'fold_predictions.csv') for fold in range(10)],ignore_index=True)
 dd=pd.concat([pd.read_csv(root/f'fold_{fold:02d}'/'fold_diagnostics.csv') for fold in range(10)],ignore_index=True)
 pp.to_csv(root/'OUTER_PREDICTIONS.csv',index=False);dd.to_csv(root/'FOLD_DIAGNOSTICS.csv',index=False)
 pd.concat([pd.read_csv(root/f'fold_{fold:02d}'/'inner_candidates.csv') for fold in range(10)]).to_csv(root/'ALL_INNER_CANDIDATES.csv',index=False)
 save_json(root/'FAILURES.json',errors)
 oof=pp.groupby('model').agg(n=('row_id','count'),unique_rows=('row_id','nunique')).reset_index()
 for name,part in pp.groupby('model'):
  assert len(part)==len(d) and part.row_id.nunique()==len(d),f'Incomplete output {tag} {name}'
 oof.to_csv(root/'COVERAGE.csv',index=False)
 print('DONE TSK',tag,oof.to_string(index=False),flush=True)
 return root

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--features',default='F5');parser.add_argument('--mode',default='group');parser.add_argument('--raw',action='store_true');a=parser.parse_args()
 with threadpool_limits(1):run(a.features,a.mode,a.raw)
