"""Generate numeric figures from R5 OOF and bootstrap data, no invented values."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from common import ROOT,load_data

FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def save(name):
 plt.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.11)
 plt.savefig(FIG/(name+'.png'),bbox_inches='tight',dpi=200,pad_inches=.11)
 plt.close()

def main():
 d=load_data()
 pp=pd.read_csv(ROOT/'results'/'ALL_PRIMARY_OOF.csv')
 tab=pd.read_csv(ROOT/'results'/'TABLE_PRIMARY_BENCHMARK.csv')
 # fig 1 group size spectrum count
 groups=d.groupby('group_id').agg(n=('row_id','size'),angle=('angle_deg','first'),chord=('chord_m','first'),velocity=('velocity_m_s','first')).reset_index()
 fig,ax=plt.subplots(figsize=(5.8,3.1))
 ax.hist(groups.n,bins=np.arange(7.5,20.6,1.),edgecolor='white');ax.set_xlabel('Frequency observations per operating configuration');ax.set_ylabel('Number of configurations')
 ax.set_title(f'1,503 observations grouped into {len(groups)} configurations')
 fig.tight_layout();save('Fig1_GroupStructure')
 # fig 2 primary performance
 morder=['linear_ridge','svr','xgboost','extra_trees','TSK-LS-oneSE','TSK-Ridge-oneSE']
 f4=[float(tab[(tab.feature_set=='F4')&(tab.model==m)].equal_config_RMSE.iloc[0]) for m in morder]
 f5=[float(tab[(tab.feature_set=='F5')&(tab.model==m)].equal_config_RMSE.iloc[0]) for m in morder]
 fig,ax=plt.subplots(figsize=(7.3,3.6));x=np.arange(len(morder));w=.38
 ax.bar(x-w/2,f4,width=w,label='F4 (no displacement thickness)');ax.bar(x+w/2,f5,width=w,label='F5 (all five inputs)')
 ax.set_xticks(x);ax.set_xticklabels(['Linear/Ridge','SVR','XGBoost','Extra Trees','TSK-LS','TSK-Ridge'],rotation=18,ha='right')
 ax.set_ylabel('Equal-configuration RMSE (dB), lower is better');ax.legend(fontsize=8);fig.tight_layout();save('Fig2_ModelComparison')
 # fig 3 condition vs selected rule number
 q=pd.concat([pd.read_csv(ROOT/'results'/'tsk'/f'group_log_{f}'/'FOLD_DIAGNOSTICS.csv') for f in ('F4','F5')])
 q=q[q.name=='TSK-Ridge-oneSE'];fig,ax=plt.subplots(figsize=(5.2,3.6))
 for feat in ('F4','F5'):
  z=q[q.feature_set==feat];ax.scatter(z.R,z.condition,label=feat,alpha=.8)
 ax.set_yscale('log');ax.set_xlabel('Number of selected TSK rules');ax.set_ylabel('Condition number of scaled design matrix');ax.legend();fig.tight_layout();save('Fig3_ConditionRuleCount')
 # fig4 held-out spectrum: representative median and worst, ordered by predetermined group RMSE F5 TSK Ridge
 a=pp[(pp.feature_set=='F5')&(pp.model=='TSK-Ridge-oneSE')]
 error=a.assign(se=lambda x:(x.y-x.pred)**2).groupby('group_id').se.mean().pow(.5).sort_values()
 chosen=[error.index[len(error)//2],error.index[-1]]
 for kind,gid in zip(('Median','HighestError'),chosen):
  z=a[a.group_id==gid].merge(d[['row_id','frequency_Hz']],on='row_id').sort_values('frequency_Hz')
  fig,ax=plt.subplots(figsize=(5.8,3.1));ax.plot(z.frequency_Hz,z.y,'o-',label='Measured NASA SPL');ax.plot(z.frequency_Hz,z.pred,'s--',label='Group-held-out TSK-Ridge')
  ax.set_xscale('log');ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('Scaled SPL (dB)');ax.set_title(f'{kind} F5 TSK case: {gid}, RMSE = {error[gid]:.2f} dB');ax.legend(fontsize=8)
  fig.tight_layout();save('Fig4_'+kind+'Spectrum')
 # fig5 conditional stability and matched rule fraction
 sbfile=ROOT/'results'/'TABLE_BOOTSTRAP_FOLD_SUMMARY.csv'
 if sbfile.exists():
  sb=pd.read_csv(sbfile)
  z=sb[(sb.fit=='Ridge')&(sb.structure=='compact')]
  fig,ax=plt.subplots(figsize=(5.7,3.5))
  for branch,mark in [('fixed','o'),('relearn','s')]:
   for feature in ('F4','F5'):
    zz=z[(z.branch==branch)&(z.feature==feature)]
    ax.scatter(zz.pred_disp_rms,zz.local_disp_rms_median,marker=mark,label=f'{feature} {branch}',alpha=.75)
  ax.set_xlabel('Global prediction dispersion across bootstraps (dB)');ax.set_ylabel('Median local-rule prediction dispersion (dB)');ax.legend(fontsize=8);fig.tight_layout();save('Fig5_GlobalVsLocalDispersion')
  fig,ax=plt.subplots(figsize=(5.8,3.1))
  for feature in ('F4','F5'):
   zz=z[(z.branch=='relearn')&(z.feature==feature)]
   ax.scatter(zz.fold,zz.matched_rule_fraction,label=feature)
  ax.set_ylim(0,1.02);ax.set_xlabel('Outer group fold');ax.set_ylabel('Fraction of matched parent fuzzy rules');ax.set_xticks(np.arange(10));ax.legend();fig.tight_layout();save('Fig6_RuleMatchCoverage')
 print('FIGURES',len(list(FIG.glob('*.pdf'))))
if __name__=='__main__':main()
