"""Independent deterministic checks and aggregate coverage checks for R5."""
import json,hashlib,os,sys,time
from pathlib import Path
import numpy as np,pandas as pd,scipy
from threadpoolctl import threadpool_limits
from common import *
from run_bootstrap import match_matrix,assign,same_coords_beta


def verify_math():
 checks={}
 d=load_data(); raw=d[['frequency_Hz','angle_deg','chord_m','velocity_m_s','displacement_m','scaled_SPL_dB']].to_numpy()
 reference_sum=np.array([4338230,10193.8,205.232,76443.7,16.743240228,187628.422])
 reference_sq=np.array([27449736300,121743.84,41.16636928,4252235.99,0.446256542732,23494253.330602])
 checks['mirror_six_column_sum']=bool(np.allclose(np.sum(raw,axis=0),reference_sum,rtol=1e-12,atol=1e-9))
 checks['mirror_six_column_squared_sum']=bool(np.allclose(np.sum(raw**2,axis=0),reference_sq,rtol=1e-11,atol=1e-8))
 assert np.allclose(raw[0],[800,0,.3048,71.3,.00266337,126.201])
 checks['mirror_first_last_rows']=bool(np.allclose(raw[-1],[6300,15.6,.1016,39.6,.0528487,104.204]))
 X=features(d,'F5');y=d.scaled_SPL_dB.to_numpy();g=d.group_id.to_numpy(str)
 outer=fold_pairs(X,y,g)
 checks['outer_group_disjoint']=all(not (set(g[a]) & set(g[b])) for a,b in outer)
 checks['outer_one_test_per_row']=bool(np.array_equal(np.sort(np.concatenate([b for a,b in outer])),np.arange(1503)))
 checks['inner_group_disjoint']=all(not(set(g[a[tr]])&set(g[a[te]])) for a,b in outer for tr,te in five_splits(X[a],y[a],g[a]))
 z=np.array([[-1.,0.],[.2,.8],[1.,-1.],[12.,-20.]])
 C=np.array([[-1.,0.],[1.,1.]])
 S=np.array([[.5,1.],[1.,.5]])
 W,loga=fire(z,C,S)
 checks['log_space_weights_normalize']=bool(np.allclose(W.sum(axis=1),1))
 checks['out_of_support_log_firing']=bool(float(loga[-1])<float(loga[0]))
 P=design(z,W); beta=np.arange(P.shape[1])*0.5
 Wp,_=fire(z,C[::-1],S[::-1]);Pp=design(z,Wp)
 checks['rule_permutation_invariance']=bool(np.allclose(P@beta,Pp@beta.reshape(2,3)[::-1].ravel()))
 duplicate=np.column_stack((P[:,:3],P[:,:3]))
 checks['duplicated_rule_rank_deficient']=bool(np.linalg.matrix_rank(duplicate)<duplicate.shape[1])
 LS=coef_lstsq(P, np.arange(4,dtype=float),0)
 rLS=coef_svd(P/2,np.arange(4,dtype=float)/2,0)
 checks['least_squares_parity']=bool(np.allclose(LS,rLS,atol=1e-8,rtol=1e-8))
 ridge=coef_lstsq(duplicate,np.arange(4,dtype=float),.01)
 ridge2=coef_svd(duplicate/2,np.arange(4,dtype=float)/2,.01)
 checks['ridge_solver_parity']=bool(np.allclose(ridge,ridge2,atol=1e-8))
 checks['ridge_full_rank_augmented']=bool(np.linalg.matrix_rank(np.vstack((duplicate/2,np.sqrt(.01)*np.eye(6))))==6)
 sim=np.array([[0.8,0.7],[.75,.01]])
 a=assign(sim,.75)
 checks['dummy_matching_threshold']=bool((a>=0).sum()==1 and a.tolist()==[0,-1])
 sim=np.array([[.1,.4],[.2,.1]])
 checks['dummy_matching_unmatched']=bool(np.array_equal(assign(sim,.5),[-1,-1]))
 b=np.array([1.,2.,-3.]); pm=np.array([4.,3.]);ps=np.array([2.,4.]);bm=np.array([3.,5.]);bs=np.array([1.,2.]);mu=10.; zz=np.array([[-1.,.5],[.6,1.]])
 bc=same_coords_beta(b,bm,bs,mu,pm,ps)[0]
 checks['affine_coordinate_change']=bool(np.allclose(mu+b[0]+((zz*ps+pm-bm)/bs)@b[1:],bc[0]+zz@bc[1:]))
 gr=np.array(['a','a','b']);yy=np.array([0,0,0]);pp=np.array([1.,3.,2.]);
 checks['equal_group_metric']=bool(np.isclose(group_rmse(yy,pp,gr),np.sqrt((5+4)/2)))
 return checks

def verify_outputs():
 result={}
 for feature in ('F4','F5'):
  base=ROOT/'results'/'tsk'/f'group_log_{feature}'
  result[f'{feature}_TSK_folds']=len(list(base.glob('fold_??/COMPLETE.json')))
  if not (base/'OUTER_PREDICTIONS.csv').exists():continue
  pp=pd.read_csv(base/'OUTER_PREDICTIONS.csv')
  for model,gg in pp.groupby('model'):
   result[f'{feature}_{model}_OOF']=len(gg)==1503 and gg.row_id.nunique()==1503
  result[f'{feature}_TSK_inner_candidates']=int(pd.read_csv(base/'ALL_INNER_CANDIDATES.csv').shape[0])
  for model in ('linear_ridge','svr','extra_trees','xgboost'):
   folder=ROOT/'results'/'baselines'/f'group_log_{feature}'/model
   result[f'{feature}_baseline_{model}_done']=bool((folder/'COMPLETE.json').exists())
  result[f'{feature}_boot_complete']=len(list((ROOT/'results'/'bootstrap').glob(f'{feature}_fold*_*/COMPLETE.json')))
 return result

def main():
 with threadpool_limits(1):
  c=verify_math();o=verify_outputs()
  doc={'math_tests':c,'passed':sum(c.values()),'total':len(c),'output_coverage':o}
  save_json(ROOT/'audit'/'TEST_REPORT.json',doc)
  print('MATH',sum(c.values()),'/',len(c),'OUTPUT',o)
  assert all(c.values())

if __name__=='__main__':main()
