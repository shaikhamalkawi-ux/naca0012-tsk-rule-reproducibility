#!/usr/bin/env python3
"""Read-only R9 extension: replay exact antecedents, verify, intersect support."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(k,'1')
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

def save(path,data):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8');t.replace(path)

def task(rootstr,outstr,feature,fold,structure):
 with threadpool_limits(limits=1):
    root=Path(rootstr);out=Path(outstr)/'analysis/tasks'/f'{feature}_fold{fold:02d}_{structure}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return json.loads((out/'COMPLETE.json').read_text())
    sys.path.insert(0,str(root/'code'))
    from common import load_data,features,fire,fcm,stable_seed,StandardScaler
    from run_bootstrap import match_matrix,assign
    start=time.time();d=load_data();F=features(d,feature);gg=d.group_id.to_numpy(str)
    key='TSK-Ridge-oneSE' if structure=='compact' else 'TSK-Ridge-fixedR6h1'
    parpath=root/'results/tsk'/f'group_log_{feature}'/f'fold_{fold:02d}'/(key+'.npz')
    bootdir=root/'results/bootstrap'/f'{feature}_fold{fold:02d}_{structure}';register=pd.read_csv(bootdir/'relearn_replicate_register.csv')
    with np.load(parpath) as par,np.load(bootdir/'relearn_replicate_results.npz') as arc:
      tr=par['tr'].astype(int);g=gg[tr];ug=np.unique(g);idx=[np.flatnonzero(g==u) for u in ug];_,inv=np.unique(g,return_inverse=True)
      Zp=(F[tr]-par['train_scaler_mean'])/par['train_scaler_scale'];Wp,_=fire(Zp,par['centers'],par['widths'])
      R=int(par['selected_R']);h=float(par['h']);parent=arc['parent_coeff'];pv=np.column_stack((np.ones(len(Zp)),Zp))@parent.T
      pairs=[];replay=[];models=[];bi=0
      for row in register.to_dict('records'):
        b=int(row['replicate']);seed=int(row['seed']);expected_ok=row['status']=='success'
        assert seed==stable_seed(feature,fold,'relearn_'+structure,b)
        rng=np.random.default_rng(seed);draw=rng.choice(len(ug),size=len(ug),replace=True);ix=np.concatenate([idx[j] for j in draw])
        sc=StandardScaler().fit(F[tr][ix]);fit,log=fcm(sc.transform(F[tr][ix]),R)
        rec=dict(feature=feature,fold=fold,structure=structure,replicate=b,seed=seed,expected_status=row['status'],status='success' if fit is not None else 'FAILED',fcmlog=log)
        if (fit is not None)!=expected_ok:raise RuntimeError(f'Changed convergence {feature}/{fold}/{structure}/{b}')
        if not expected_ok:replay.append(rec);continue
        C=fit['C'];S=fit['S']*h;Wb,_=fire(sc.transform(F[tr]),C,S);sim=match_matrix(Wp,Wb,g);match=assign(sim,.5)
        delta=float(np.max(np.abs(sim-arc['sim'][bi])))
        if delta>1e-8 or not np.array_equal(match,arc['match'][bi]):raise RuntimeError(f'Changed match {feature}/{fold}/{structure}/{b}: {delta}')
        if int(fit['seed'])!=int(row['FCM_seed']) or int(fit['iters'])!=int(row['fcm_iterations']):raise RuntimeError('Changed FCM seed/iterations')
        rec.update(success_index=bi,max_similarity_diff=delta,FCM_seed=int(fit['seed']),fcm_iterations=int(fit['iters']),matches=int((match>=0).sum()))
        replay.append(rec);models.append(dict(C=C,S=S,mean=sc.mean_,scale=sc.scale_,replicate=b,success_index=bi));coef=arc['Ridge_coeff'][bi]
        for r,s in enumerate(match):
          if s<0:continue
          if not np.isfinite(coef[r]).all():raise RuntimeError('Missing coefficients')
          dv=pv[:,r]-(coef[r,0]+Zp@coef[r,1:]);dsq=dv*dv
          def stats(mask):
            cnt=np.bincount(inv[mask],minlength=len(ug));sums=np.bincount(inv[mask],weights=dsq[mask],minlength=len(ug));ok=cnt>0
            return (float(np.sqrt(np.mean(sums[ok]/cnt[ok]))),int(ok.sum())) if ok.any() else (None,0)
          for cut in (.05,.1,.2):
            a=Wp[:,r]>=cut;c=a&(Wb[:,s]>=cut);na=int(a.sum());nc=int(c.sum());dp,gp=stats(a);dc,gc=stats(c)
            pairs.append(dict(feature=feature,structure=structure,fold=fold,replicate=b,success_index=bi,parent_rule=r,matched_rule=int(s),cutoff=cut,
              Jaccard=float(sim[r,s]),D_parent_dB=dp,D_common_dB=dc,parent_rows=na,common_rows=nc,parent_groups=gp,common_groups=gc,
              row_retention=nc/na if na else None,group_retention=gc/gp if gp else None,intersection_empty=(nc==0)))
        bi+=1
      assert bi==len(arc['match']) and len(replay)==200
    pd.DataFrame(pairs).to_csv(out/'pairs.csv',index=False,float_format='%.16g');save(out/'replay_register.json',{'attempts':replay})
    np.savez_compressed(out/'replayed_antecedents.npz',centers=np.array([m['C'] for m in models]),widths=np.array([m['S'] for m in models]),
       mean=np.array([m['mean'] for m in models]),scale=np.array([m['scale'] for m in models]),replicate=np.array([m['replicate'] for m in models]),success_index=np.array([m['success_index'] for m in models]))
    info=dict(feature=feature,fold=fold,structure=structure,attempts=200,successful=bi,failed=200-bi,pairs_at_0p1=sum(v['cutoff']==.1 for v in pairs),max_similarity_diff=max(v.get('max_similarity_diff',0) for v in replay),elapsed_seconds=time.time()-start)
    save(out/'COMPLETE.json',info);return info

def summarize(out):
    complete=sorted((out/'analysis/tasks').glob('*/COMPLETE.json'))
    if len(complete)!=40:raise RuntimeError(f'{len(complete)}/40 tasks complete')
    details=[json.loads(p.read_text()) for p in complete];df=pd.concat([pd.read_csv(p.parent/'pairs.csv') for p in complete],ignore_index=True)
    assert sum(x['attempts'] for x in details)==8000 and sum(x['successful'] for x in details)==7998
    assert len(df.query('cutoff==0.1'))==50376
    df.to_csv(out/'analysis/SHARED_SUPPORT_PAIRS.csv',index=False,float_format='%.16g');rows=[]
    for (f,s,fold,cut),a in df.groupby(['feature','structure','fold','cutoff']):
      v=a.dropna(subset=['D_common_dB']);hi=v.query('Jaccard>=.8')
      rows.append(dict(feature=f,structure=s,fold=int(fold),cutoff=cut,matched_pairs=len(a),nonempty_pairs=len(v),empty_pairs=len(a)-len(v),
        median_D_parent_dB=a.D_parent_dB.median(),median_D_parent_eligible_dB=v.D_parent_dB.median(),median_D_common_dB=v.D_common_dB.median(),p90_D_common_dB=v.D_common_dB.quantile(.9),
        median_paired_delta_dB=(v.D_common_dB-v.D_parent_dB).median(),median_row_retention=a.row_retention.median(),median_group_retention=a.group_retention.median(),
        rho_parent=float(spearmanr(a.Jaccard,a.D_parent_dB,nan_policy='omit').statistic),rho_common=float(spearmanr(v.Jaccard,v.D_common_dB).statistic),high_J_n=len(hi),high_J_Dgt10=int((hi.D_common_dB>10).sum())))
    ff=pd.DataFrame(rows);ff.to_csv(out/'analysis/SHARED_SUPPORT_FOLDS.csv',index=False,float_format='%.16g');rows=[]
    for (f,s,c),a in ff.groupby(['feature','structure','cutoff']):
      rows.append(dict(feature=f,structure=s,cutoff=c,folds=len(a),matched_pairs=int(a.matched_pairs.sum()),nonempty_pairs=int(a.nonempty_pairs.sum()),empty_pairs=int(a.empty_pairs.sum()),
        median_fold_D_parent_dB=a.median_D_parent_dB.median(),median_fold_D_common_dB=a.median_D_common_dB.median(),min_fold_D_common_dB=a.median_D_common_dB.min(),max_fold_D_common_dB=a.median_D_common_dB.max(),
        median_fold_paired_delta_dB=a.median_paired_delta_dB.median(),median_fold_row_retention=a.median_row_retention.median(),median_fold_group_retention=a.median_group_retention.median(),
        median_fold_rho_parent=a.rho_parent.median(),median_fold_rho_common=a.rho_common.median(),high_J_n=int(a.high_J_n.sum()),high_J_Dgt10=int(a.high_J_Dgt10.sum()),
        high_J_Dgt10_fraction=float(a.high_J_Dgt10.sum()/a.high_J_n.sum()) if a.high_J_n.sum() else None))
    pd.DataFrame(rows).to_csv(out/'analysis/SHARED_SUPPORT_SUMMARY.csv',index=False,float_format='%.16g')
    save(out/'audit/REPLAY_STATUS.json',{'tasks':details,'n_tasks':40,'n_attempts':8000,'successes':7998,'failed':2,'pairs_at_primary_cut':50376,'max_similarity_diff':max(x['max_similarity_diff'] for x in details)})

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--workers',type=int,default=4);p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
    for d in ['analysis','audit']:(a.out/d).mkdir(parents=True,exist_ok=True)
    if not a.summarize_only:
      with ProcessPoolExecutor(max_workers=a.workers) as pool:
        tasks=[pool.submit(task,str(a.root.resolve()),str(a.out.resolve()),f,i,s) for f in ('F4','F5') for i in range(10) for s in ('compact','reference')]
        for future in as_completed(tasks):print(json.dumps(future.result()),flush=True)
    summarize(a.out)
if __name__=='__main__':main()
