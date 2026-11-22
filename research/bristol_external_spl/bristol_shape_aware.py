#!/usr/bin/env python3
"""Bristol shape-aware LOGO; release only derived metrics, not original profiles."""
import io,hashlib,json,urllib.parse
from pathlib import Path
import h5py,numpy as np,requests
from bristol_model_audit import SOURCE,ROOT,load,rmse,model_predict
SHAPE={
'NACA 0012':('n0012_aerofoil_coords.h5','f806d37c030f64b7df56c2adfec52b9aaef2155e245bd26333e8d29d6cdcddae'),
'NACA 0024':('n0024_aerofoil_coords.h5','7ac196d8d9eb2c084e900ba1535b94e77ba517112b1f2110ccfc3d10a7221baa'),
'NACA 16-506':('n16506_aerofoil_coords.h5','c95656e8c2d482091e20aeab197ea8f3955284c1ee3d18836de354d7b7aedf36'),
'NACA 16-616':('n16616_aerofoil_coords.h5','0ed5f25970a5de88565563bbc5fee1a15bb9527ff4c8813e5bea27d23c621fcd')}
def descriptor(x,y):
 x=np.asarray(x).ravel();y=np.asarray(y).ravel()
 assert x.size==y.size and x.size>=30
 k=int(np.argmin(x));assert k>10 and k<len(x)-10
 sides=[];grid=np.linspace(.01,.99,99)
 for ax,ay in ((x[:k+1],y[:k+1]),(x[k:],y[k:])):
  ids=np.argsort(ax);xx=ax[ids];yy=ay[ids]
  xx,inds=np.unique(xx,return_index=True);yy=yy[inds]
  assert xx.max()>=.99 and xx.min()<=.01
  sides.append(np.interp(grid,xx,yy))
 a,b=sides;up=np.maximum(a,b);lo=np.minimum(a,b)
 t=up-lo;camber=.5*(up+lo);i=int(np.argmax(t));j=int(np.argmax(np.abs(camber)))
 return {'tmax_over_c':float(t[i]),'x_over_c_at_tmax':float(grid[i]),'camber_at_absmax_over_c':float(camber[j]),'absmax_camber_over_c':float(abs(camber[j])),'section_area_over_c2':float(np.trapezoid(t,grid)),'surface_profile_points':int(x.size)}
def main():
 geo,ang,freq,y,X=load();shape={}
 for name,(filename,sha) in SHAPE.items():
  uri=ROOT+'/'+urllib.parse.quote(name)+'/'+filename
  r=requests.get(uri,timeout=(15,85));r.raise_for_status()
  assert hashlib.sha256(r.content).hexdigest()==sha,'SOURCE_SHAPE_CHANGED '+name
  with h5py.File(io.BytesIO(r.content),'r') as h:
   x=np.asarray(h['xoverc'][()]);yy=np.asarray(h['yoverc'][()])
  shape[name]=descriptor(x,yy)
  print('PROFILE',name,shape[name],flush=True)
 features=np.array([[shape[g]['tmax_over_c'],shape[g]['camber_at_absmax_over_c']] for g in geo])
 XS=np.c_[X,features]
 report={'source_doi':'10.5523/bris.30c8q39wi1mxf205flnun3l5rl',
  'shape_source_hashes':{k:v[1] for k,v in SHAPE.items()},
  'geometry_descriptors':shape,
  'features':['log10(frequency_Hz)','geometric_angle_deg','tmax/chord','max_abs_signed_camber/chord'],
  'loo_rows':[],'no_geometry_id':True,'shape_descriptors_known_without_acoustic_test_response':True,
  'metric':'held-out-geometry RMSE dB',
  'model_settings':'R6,h1,lambda.01 and 400-tree ET fixed controls, no geometry test tuning'}
 for name in SOURCE:
  tr=np.where(geo!=name)[0];te=np.where(geo==name)[0]
  out,status=model_predict(XS[tr],y[tr],XS[te])
  row={'heldout_geometry':name,'TSK_status':status,'n_train_rows':int(len(tr)),'n_test_rows':int(len(te)),
   'inside_shape_training_box':bool(np.all((features[te][0]>=features[tr].min(0)) & (features[te][0]<=features[tr].max(0))))}
  for k,v in out.items():row[k+'_RMSE_dB']=rmse(y[te],v) if v is not None else None
  report['loo_rows'].append(row)
  print('GEOMETRY_AWARE_LOGO',name,row,flush=True)
 report['equal_geometry_RMSE_dB']={k:float(np.sqrt(np.mean([r[k]**2 for r in report['loo_rows']]))) if all(r[k] is not None for r in report['loo_rows']) else None for k in ('TSK_RMSE_dB','ET_RMSE_dB','mean_RMSE_dB')}
 report['warning']='Only 4 distinct geometry descriptors; 3 geometries in each train fold. Does not establish robust population transfer, structural causality, or NASA-to-Bristol SPL equivalence.'
 Path('research/bristol_external_spl/shape_aware_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
if __name__=='__main__':main()
