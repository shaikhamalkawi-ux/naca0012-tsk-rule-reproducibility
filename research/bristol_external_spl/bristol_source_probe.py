#!/usr/bin/env python3
"""Source-first admission probe. Never commits Bristol raw measurements."""
import hashlib, io, json, pathlib, urllib.parse
import h5py, numpy as np, requests
ROOT='https://data.bris.ac.uk/datasets/30c8q39wi1mxf205flnun3l5rl'
SOURCES={'NACA 0012':'n0012_ff_sound.h5','NACA 0024':'n0024_ff_sound.h5','NACA 16-506':'n16506_ff_sound.h5','NACA 16-616':'n16616_ff_sound.h5'}
OUT=pathlib.Path('research/bristol_external_spl/probe_inventory.json')
def h5_metadata(blob):
    rows=[]
    def norm(v):
        if isinstance(v,np.ndarray):return {'shape':list(v.shape),'sample':v.ravel()[:12].tolist()}
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,bytes):return v.decode('utf-8','replace')
        if isinstance(v,(str,int,float,bool)) or v is None:return v
        return str(v)
    with h5py.File(io.BytesIO(blob),'r') as f:
        def walk(k,v):
            row={'path':k,'type':'dataset' if isinstance(v,h5py.Dataset) else 'group','attributes':{s:norm(x) for s,x in v.attrs.items()}}
            if isinstance(v,h5py.Dataset):
                row['shape']=list(v.shape);row['dtype']=str(v.dtype)
                if v.size<=100000 and v.dtype.kind in 'fiub':
                    x=np.asarray(v[()]);y=x[np.isfinite(x)]
                    row['finite']=int(y.size);row['values']=int(x.size)
                    row['range']=[float(y.min()),float(y.max())] if y.size else None
                    if x.size<=120:row['value_preview']=x.reshape(-1)[:120].tolist()
            rows.append(row)
        f.visititems(walk)
        attrs={s:norm(v) for s,v in f.attrs.items()}
    return {'root_attributes':attrs,'nodes':rows}
def main():
    report={'admission_status':'PENDING_SOURCE_DOWNLOAD','source_doi':'10.5523/bris.30c8q39wi1mxf205flnun3l5rl','license':'Non-Commercial Government Licence for public sector information','sources':{},'operation':'read-only source acquisition; no raw data included in version control'}
    session=requests.Session();session.headers['User-Agent']='AcademicResearchDataAudit/1.0 (non-commercial methods validation)'
    for geom,name in SOURCES.items():
        report['sources'][geom]={'filename':name,'status':'not_downloaded','attempts':[]}
        urls=[ROOT+'/'+urllib.parse.quote(geom)+'/'+name,ROOT.replace('data.bris.ac.uk','data.bristol.ac.uk')+'/'+urllib.parse.quote(geom)+'/'+name]
        for url in urls:
            try:
                r=session.get(url,timeout=(15,85),allow_redirects=True)
                report['sources'][geom]['attempts'].append({'url':url,'status':r.status_code,'bytes':len(r.content)})
                if r.status_code!=200 or r.content[:8]!=b'\x89HDF\r\n\x1a\n':continue
                md=h5_metadata(r.content)
                report['sources'][geom].update({'status':'hdf5_verified','download_url':url,'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content),'hdf5':md})
                break
            except Exception as e:
                report['sources'][geom]['attempts'].append({'url':url,'error':f'{type(e).__name__}: {str(e)[:240]}'})
        print('BRISTOL_SOURCE',geom,'=',report['sources'][geom]['status'],flush=True)
        for node in report['sources'][geom].get('hdf5',{}).get('nodes',[]):
            print('H5_NODE',geom,node['path'],node.get('shape'),node.get('range'),flush=True)
    valid=sum(x['status']=='hdf5_verified' for x in report['sources'].values())
    report['verified_files']=valid
    report['admission_status']='HDF5_ACQUIRED_READY_FOR_SCIENTIFIC_UNIT_AUDIT' if valid==4 else 'PARTIAL_OR_FAILED_DOWNLOAD_NO_NUMERIC_CLAIMS'
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('FINAL_ADMISSION_STATUS',report['admission_status'])
    print('REPORT_JSON',OUT,flush=True)
if __name__=='__main__':main()
