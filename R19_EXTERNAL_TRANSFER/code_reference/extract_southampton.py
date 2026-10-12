from __future__ import annotations
from pathlib import Path
import hashlib,re,zipfile,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ZIP=ROOT/'source/DataFiles.zip'
OUT=ROOT/'data/SOUTHAMPTON_CANONICAL_REEXTRACTED.csv'
# Source-condition mapping transcribed from Lau et al. case table / figure captions before modelling outcomes.
META={
3: ('NACA643-418',0,0.6,70.0,[0.0]),
4: ('NACA643-418',1,0.6,70.0,[0.0,3.0]),
5: ('NACA0012',1,0.4,None,[0.0,4.0]), # velocity differs by angle below
6: ('DU96-180',1,0.3,60.0,[4.0]),
7: ('RisoB1-15',0,0.6,70.0,[3.0,6.0,9.0,11.0]),
8: ('RisoB1-15',1,0.6,70.0,[3.0,6.0,9.0,11.0]),
9: ('RisoB1-18',0,0.6,70.0,[3.4,5.1,7.1,8.9]),
10:('RisoB1-18',1,0.6,70.0,[3.4,5.1,7.1,8.9]),
11:('Aerofoil-18',0,0.6,70.0,[2.0,4.0,6.0,9.0]),
12:('Aerofoil-18',1,0.6,70.0,[2.0,4.0,6.0,9.0]),
13:('Aerofoil-21',0,0.6,70.0,[2.7,4.6,6.5,8.9]),
14:('Aerofoil-21',1,0.6,70.0,[2.7,4.6,6.5,8.9]),
}

def blocks(text, fig):
    lines=text.splitlines(); out=[]
    if fig==3:
        # The file opens directly with the sole experimental spectrum, before model columns.
        i=next(i for i,l in enumerate(lines) if 'Freq,Hz' in l)
        cur=[]
        for l in lines[i+1:]:
            if not l.strip(): break
            vals=l.split()
            if len(vals)>=2:
                try: cur.append((float(vals[0]),float(vals[1])))
                except: pass
        return [(0.0,cur)]
    # Experimental blocks precede model-prediction blocks and start with 'Expt LWT'.
    i=0
    while i<len(lines):
        l=lines[i]
        if 'Expt LWT' in l:
            m=re.search(r'AoA\s*=\s*([+-]?[0-9.]+)deg',l)
            angle=float(m.group(1)) if m else None
            j=i+1
            while j<len(lines) and 'Freq,Hz' not in lines[j]: j+=1
            cur=[]; j+=1
            while j<len(lines) and lines[j].strip():
                vals=lines[j].split()
                if len(vals)>=2:
                    try: cur.append((float(vals[0]),float(vals[1])))
                    except: pass
                j+=1
            out.append((angle,cur)); i=j
        else: i+=1
    return out

def main():
    sha=hashlib.sha256(ZIP.read_bytes()).hexdigest()
    rows=[]
    with zipfile.ZipFile(ZIP) as z:
        for fig in range(3,15):
            geom,trip,chord,vel,angles=META[fig]
            text=z.read(f'Figure_{fig}_Data.dat').decode('utf-8','replace')
            bs=blocks(text,fig)
            if len(bs)!=len(angles): raise RuntimeError((fig,len(bs),angles))
            for k,((a,xy),expected) in enumerate(zip(bs,angles)):
                if a is None: a=expected
                if abs(a-expected)>1e-9: raise RuntimeError((fig,a,expected))
                v=vel
                if fig==5: v=56.0 if abs(a)<1e-12 else 54.8
                for f,y in xy:
                    rows.append({'source_figure':fig,'geometry_id':geom,'trip_flag':trip,'angle_deg':a,'chord_m':chord,
                                 'velocity_m_s':v,'frequency_Hz':f,'response_norm':y,
                                 'group_id':f"{geom}|{'trip' if trip else 'clean'}|{a:g}"})
    d=pd.DataFrame(rows); d.insert(0,'row_id',range(1,len(d)+1)); d.to_csv(OUT,index=False)
    print(json.dumps({'source_sha256':sha,'rows':len(d),'groups':d.group_id.nunique(),'geometries':d.geometry_id.nunique(),
                      'figures':sorted(d.source_figure.unique().tolist())},indent=2))
if __name__=='__main__': main()
