#!/usr/bin/env python3
"""Regenerate spectra from locked OOF records; plot paired support sensitivity."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10,'axes.labelsize':10})

def main(root,out):
 fdir=out/'manuscript/figures';fdir.mkdir(parents=True,exist_ok=True)
 data=pd.read_csv(root/'results/ALL_PRIMARY_OOF.csv');records=[]
 for group,name,label in [('G059','Fig4_MedianSpectrum','Near-median-error configuration'),('G106','Fig4_HighestErrorSpectrum','Highest-error configuration')]:
  original=pd.read_csv(root/'data/airfoil_F5_SEMANTICALLY_VERIFIED.csv')
  subset=data.query('feature_set=="F5" and group_id==@group')
  a=subset.query('model=="TSK-Ridge-oneSE"')[['row_id','y','pred']].rename(columns={'pred':'TSK'})
  b=subset.query('model=="extra_trees"')[['row_id','pred']].rename(columns={'pred':'ET'})
  t=a.merge(b,on='row_id',validate='one_to_one').merge(original[['row_id','frequency_Hz']],on='row_id',validate='one_to_one').sort_values('frequency_Hz')
  r1=float(np.sqrt(np.mean((t.y-t.TSK)**2)));r2=float(np.sqrt(np.mean((t.y-t.ET)**2)))
  fig,ax=plt.subplots(figsize=(5.25,3.65))
  ax.plot(t.frequency_Hz,t.y,marker='o',label='Measured')
  ax.plot(t.frequency_Hz,t.TSK,marker='s',linestyle='--',label=f'TSK-ridge ({r1:.3f} dB)')
  ax.plot(t.frequency_Hz,t.ET,marker='^',linestyle='-.',label=f'Extra Trees ({r2:.3f} dB)')
  ax.set_xscale('log');ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('Scaled sound pressure level (dB)');ax.set_title(group)
  ax.legend(loc='best',fontsize=9);fig.tight_layout();fig.savefig(fdir/(name+'.pdf'));fig.savefig(fdir/(name+'.png'),dpi=170);plt.close(fig)
  t.to_csv(out/'analysis'/f'SPECTRUM_{group}_LOCKED_OOF.csv',index=False);records.append({'group':group,'rows':len(t),'TSK_RMSE_dB':r1,'ExtraTrees_RMSE_dB':r2})
 (out/'audit/SPECTRAL_FIGURE_QA.json').write_text(json.dumps(records,indent=2))
 fp=out/'analysis/SHARED_SUPPORT_FOLDS.csv'
 if fp.exists():
  d=pd.read_csv(fp);fig,ax=plt.subplots(figsize=(7,4.2))
  for j,(fs,s,label) in enumerate([('F4','compact','F4 one-SE'),('F5','compact','F5 one-SE'),('F4','reference','F4 fixed R=6'),('F5','reference','F5 fixed R=6')]):
   t=d.query('feature==@fs and structure==@s and cutoff==0.1').sort_values('fold')
   a=t.median_D_parent_dB.to_numpy();b=t.median_D_common_dB.to_numpy();x=3*j
   ax.plot(np.array([[x,x+1]]*len(t)).T,np.c_[a,b].T,marker='o',alpha=.35,linewidth=.8,markersize=3)
   ax.plot([x,x+1],[np.median(a),np.median(b)],marker='D',linewidth=2,markersize=5)
   ax.text(x+.5,-.17,label,transform=ax.get_xaxis_transform(),ha='center',fontsize=10)
  ax.set_xticks([3*j+k for j in range(4) for k in (0,1)],['Parent','Shared']*4)
  ax.set_ylabel('Median local-function disagreement (dB)');ax.set_ylim(bottom=0)
  ax.set_title('Within-fold paired evaluation on parent and shared activation sets')
  ax.text(.01,.97,'Each thin segment is one fold; diamonds show fold medians.',ha='left',va='top',transform=ax.transAxes,fontsize=9)
  fig.subplots_adjust(bottom=.23,left=.12,right=.99,top=.86);fig.savefig(fdir/'Fig9_SharedActivationSupport.pdf');fig.savefig(fdir/'Fig9_SharedActivationSupport.png',dpi=170);plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();main(a.root,a.out)
