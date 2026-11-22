#!/usr/bin/env python3
"""Source inventory for geometry shapes; metadata only, no raw shape rehosting."""
import hashlib,io,json,urllib.parse
from pathlib import Path
import requests,h5py,numpy as np
ROOT='https://data.bris.ac.uk/datasets/30c8q39wi1mxf205flnun3l5rl'
FILES={'NACA 0012':'n0012_aerofoil_coords.h5','NACA 0024':'n0024_aerofoil_coords.h5','NACA 16-506':'n16506_aerofoil_coords.h5','NACA 16-616':'n16616_aerofoil_coords.h5'}
out={}
for geom,filename in FILES.items():
 url=ROOT+'/'+urllib.parse.quote(geom)+'/'+filename;entry={'url':url}
 try:
  r=requests.get(url,timeout=(15,85));r.raise_for_status()
  entry.update({'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content)})
  with h5py.File(io.BytesIO(r.content),'r') as h:
   nodes=[]
   def visit(k,v):
    if isinstance(v,h5py.Dataset):
     d=np.asarray(v[()]);node={'path':k,'shape':list(d.shape),'dtype':str(d.dtype)}
     if d.dtype.kind in 'fiub':
      f=d[np.isfinite(d)];node['range']=[float(f.min()),float(f.max())] if f.size else None
      node['first_10']=d.reshape(-1)[:10].tolist()
     nodes.append(node)
   h.visititems(visit);entry['nodes']=nodes;entry['status']='HDF5_OK'
 except Exception as e:entry.update({'status':'DOWNLOAD_OR_SCHEMA_ERROR','error':f'{type(e).__name__}: {str(e)[:220]}'})
 out[geom]=entry
 print('SHAPE_COORDS',geom,entry['status'],entry.get('nodes'),flush=True)
p=Path('research/bristol_external_spl/shape_probe.json')
p.write_text(json.dumps(out,indent=2),encoding='utf-8')
