#!/usr/bin/env python3
"""Source-locked Bristol benchmark; summary metrics only, no raw data release."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,io,json,hashlib,urllib.parse
from pathlib import Path
import h5py,numpy as np,requests
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path('57_H3_Airfoil_R5/code').resolve()))
from common import fcm,fire,design,coef_lstsq
ROOT='https://data.bris.ac.uk/datasets/30c8q39wi1mxf205flnun3l5rl'
SOURCE={'NACA 0012':('n0012_ff_sound.h5','1e6b4482b46452a35a93a59252c4eb03c1bf05d32f23e8625a4af16152ac2c60'),
'NACA 0024':('n0024_ff_sound.h5','e5c772c9a130ebf9bc19b2ac2a0c1a9982ecb528ed1c7703b1c1ced78d755373'),
'NACA 16-506':('n16506_ff_sound.h5','468014d130c0a0a9f20df25aa9f197ac874f3e8ec06678187176075aebc42c5b'),
'NACA 16-616':('n16616_ff_sound.h5','40901ef2c635c6cc9d4adec60f6df265db20e1c87a8c34d4fcbeaf6f48c63e0e')}
OUTPUT=Path('research/bristol_external_spl/experiment_results.json')
def load():
 rows=[];session=requests.Session()
 for geom,(name,expected) in SOURCE.items():
  u=ROOT+'/'+urllib.parse.quote(geom)+'/'+name;r=session.get(u,timeout=(15,85));r.raise_for_status()
  assert hashlib.sha256(r.content).hexdigest()==expected,'SOURCE_SHA_CHANGED '+geom
  with h5py.File(io.BytesIO(r.content),'r') as h:
   spl=np.asarray(h['spl'][()]);f=np.asarray(h['f'][()]);angles=np.asarray(h['angles'][()]).ravel()
  assert spl.shape==f.shape==(len(angles),12) and np.all(np.isfinite(spl)) and np.all(f>0)
  for i,a in enumerate(angles):
   for j in range(12):rows.append((geom,float(a),float(f[i,j]),float(spl[i,j])))
 assert len(rows)==1464
 geom=np.array([r[0] for r in rows]);angle=np.array([r[1] for r in rows]);freq=np.array([r[2] for r in rows]);y=np.array([r[3] for r in rows])
 return geom,angle,freq,y,np.c_[np.log10(freq),angle]
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def tsk(Xtr,ytr,Xte):
 sc=StandardScaler().fit(Xtr);z=sc.transform(Xtr);q=sc.transform(Xte)
 fc,records=fcm(z,6)
 if fc is None:return None,'FCM_NONCONVERGENCE'
 P=design(z,fire(z,fc['C'],fc['S'])[0]);Q=design(q,fire(q,fc['C'],fc['S'])[0])
 beta=coef_lstsq(P,ytr-ytr.mean(),0.01)
 return ytr.mean()+Q@beta,'OK'
def model_predict(Xtr,ytr,Xte):
 p0=np.full(len(Xte),ytr.mean());pt,status=tsk(Xtr,ytr,Xte)
 et=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=1,max_features=1.0,random_state=20261007,n_jobs=1)
 et.fit(Xtr,ytr)
 return {'mean':p0,'TSK':pt,'ET':et.predict(Xte)},status
def main():
 geom,ang,freq,y,X=load()
 report={'source_doi':'10.5523/bris.30c8q39wi1mxf205flnun3l5rl',
 'source_authentication':'4 original official SHA256-locked HDF5 n*_ff_sound.h5',
 'source_license':'Non-Commercial Government Licence for public sector information',
 'target':'Bristol beamformed integrated source SPL in dB, not harmonized with NASA SPL',
 'features':['log10(frequency_Hz)','geometric_angle_deg'],
 'controls':{'TSK':{'R':6,'h':1,'lambda':0.01,'FCM_seeds':[11,23,47,71,101]},
 'ExtraTrees':{'n_estimators':400,'min_samples_leaf':1,'max_features':1.0,'random_state':20261007}},
 'selection':'fixed prior-style controls; no selection based on any external holdout',
 'counts':{'rows':int(len(y)),'angle_spectra':int(len(set(zip(geom,ang)))),'geometries':4},
 'within_geometry_angle_heldout':[],'leave_geometry_out':[]}
 key=np.array([f'{a:g}|{f:.6f}' for a,f in zip(ang,freq)])
 keys,inv=np.unique(key,return_inverse=True);counts=np.bincount(inv);mn=np.bincount(inv,weights=y)/counts
 spread=np.array([float(np.ptp(y[inv==i])) for i in range(len(keys))])
 drift=np.array([float(np.ptp(freq[inv==i])) for i in range(len(keys))])
 shared=counts>=2
 report['geometry_blind_collision']={
 'unique_frequency_angle_pairs':int(len(keys)),
 'pairs_with_multiple_geometries':int(shared.sum()),
 'max_frequency_drift_Hz_in_6dec_join':float(drift.max()),
 'minimum_pooled_training_RMSE_for_any_geometry_blind_predictor_dB':rmse(y,mn[inv]),
 'max_same_input_SPL_spread_dB':float(spread.max()),
 'median_shared_input_SPL_spread_dB':float(np.median(spread[shared])),
 'n_shared_inputs_with_gt_1dB_spread':int((spread[shared]>1).sum()),
 'scope':'pooled observed-input interpolation lower bound, not a LOGO test lower bound'}
 with threadpool_limits(limits=1):
  for name in SOURCE:
   ix=np.where(geom==name)[0]
   pred={k:np.full(len(ix),np.nan) for k in ['mean','TSK','ET']}
   fstatus=[]
   for f,(tr,te) in enumerate(GroupKFold(5).split(X[ix],y[ix],ang[ix])):
    assert not set(ang[ix][tr])&set(ang[ix][te])
    outputs,status=model_predict(X[ix][tr],y[ix][tr],X[ix][te])
    for k,v in outputs.items():
     if v is not None:pred[k][te]=v
    fstatus.append(status)
   z={'geometry':name,'n_rows':int(len(ix)),'n_heldout_angles':int(len(set(ang[ix]))),'fold_status':fstatus}
   for k,v in pred.items():z[k+'_RMSE_dB']=rmse(y[ix],v) if np.isfinite(v).all() else None
   report['within_geometry_angle_heldout'].append(z)
   print('WITHIN_ANGLE',name,z,flush=True)
  for name in SOURCE:
   tr=np.where(geom!=name)[0];te=np.where(geom==name)[0]
   outputs,status=model_predict(X[tr],y[tr],X[te])
   z={'heldout_geometry':name,'n_training_rows':int(len(tr)),'n_test_rows':int(len(te)),'n_test_spectra':int(len(set(ang[te]))),'TSK_status':status}
   for k,v in outputs.items():z[k+'_RMSE_dB']=rmse(y[te],v) if v is not None else None
   report['leave_geometry_out'].append(z)
   print('LEAVE_GEOMETRY_OUT',name,z,flush=True)
 report['within_geometry_contiguous_angle_blocks']=[]
 with threadpool_limits(limits=1):
  for name in SOURCE:
   ix=np.where(geom==name)[0];sorted_angles=np.sort(np.unique(ang[ix]));blocks=np.array_split(sorted_angles,5)
   pred={k:np.full(len(ix),np.nan) for k in ['mean','TSK','ET']};fstatus=[];ranges=[]
   for fold,block in enumerate(blocks):
    te=np.where(np.isin(ang[ix],block))[0];tr=np.where(~np.isin(ang[ix],block))[0]
    assert not (set(ang[ix][tr])&set(ang[ix][te])) and len(te)>0
    outs,status=model_predict(X[ix][tr],y[ix][tr],X[ix][te])
    for k,v in outs.items():
     if v is not None:pred[k][te]=v
    fstatus.append(status)
    ranges.append({'fold':fold,'min_angle':float(block.min()),'max_angle':float(block.max()),'n_test_angles':len(block)})
   z={'geometry':name,'n_rows':int(len(ix)),'blocked_angle_ranges':ranges,'fold_status':fstatus}
   for k,v in pred.items():z[k+'_RMSE_dB']=rmse(y[ix],v) if np.isfinite(v).all() else None
   report['within_geometry_contiguous_angle_blocks'].append(z)
   print('CONTIGUOUS_BLOCKS',name,z,flush=True)

 for arm in ['within_geometry_angle_heldout','within_geometry_contiguous_angle_blocks','leave_geometry_out']:
  report[arm+'_equal_geometry_RMSE_dB']={key:(float(np.sqrt(np.mean([z[key]**2 for z in report[arm]]))) if all(z[key] is not None for z in report[arm]) else None) for key in ['mean_RMSE_dB','TSK_RMSE_dB','ET_RMSE_dB']}
 report['scientific_status']='CONDITIONAL_BRISTOL_EXTERNAL_DB_EXPERIMENT; NOT_NASA_TO_BRISTOL_MODEL_TRANSFER'
 OUTPUT.parent.mkdir(parents=True,exist_ok=True);OUTPUT.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
 print('SOURCE-LOCKED EXPERIMENT DONE',json.dumps(report['geometry_blind_collision']),flush=True)
if __name__=='__main__':main()
