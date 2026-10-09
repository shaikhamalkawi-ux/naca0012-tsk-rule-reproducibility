#!/usr/bin/env python3
"""Render publication-quality, derived-only R6 figures from audited pair outputs."""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
from matplotlib.colors import LogNorm
R=Path(__file__).resolve().parents[1]
A=R/'analysis';F=R/'figures';F.mkdir(exist_ok=True)
all_pairs=pd.read_csv(A/'R6_MATCHED_PAIRS_0p5.csv')
folds=pd.read_csv(A/'R6_FOLD_LEVEL_DESCRIPTIVES.csv')
for fs in ('F4','F5'):
 p=all_pairs[(all_pairs.feature==fs)&(all_pairs.structure=='compact')]
 fig,ax=plt.subplots(figsize=(6.5,3.8))
 bins=ax.hexbin(p.Jaccard,p.functional_disagreement_rmse_dB,
      gridsize=46, xscale='linear', yscale='log', mincnt=1, norm=LogNorm(),cmap='viridis',linewidths=0)
 ax.set_xlim(.49,1.005); ax.set_ylim(.75,380)
 ax.set_xlabel('Parent-relearned membership similarity (weighted Jaccard)')
 ax.set_ylabel('Matched local-function RMSE (dB; log axis)')
 r=folds[(folds.feature==fs)&(folds.structure=='compact')]
 cbar=fig.colorbar(bins,ax=ax,pad=.022,shrink=.82)
 cbar.set_label('Matched pairs per bin (log color scale)')
 ax.text(.025,.97,f'{fs} | {len(p):,} successful matched pairs\nMedian of 10 fold-wise Spearman rho = {r.rho.median():.3f}',
  ha='left',va='top',transform=ax.transAxes,fontsize=8.5,
  bbox=dict(facecolor='white',edgecolor='0.7',alpha=.93,boxstyle='round,pad=0.45'))
 ax.grid(alpha=.14,axis='y')
 fig.tight_layout()
 fig.savefig(F/f'Fig7_{fs}_SimilarityVsLocalFunctionalDifference.pdf',bbox_inches='tight')
 fig.savefig(F/f'Fig7_{fs}_SimilarityVsLocalFunctionalDifference.png',dpi=180,bbox_inches='tight')
 plt.close(fig)

tr=pd.read_csv(A/'R6_MATCH_THRESHOLD_AUDIT.csv')
fig,ax=plt.subplots(figsize=(6.5,3.3))
for i,fs in enumerate(('F4','F5')):
 part=tr[(tr.feature==fs)&(tr.structure=='compact')]
 xs=np.array([.3,.5,.7])
 for _,f in part.groupby('fold'):
  ys=f.sort_values('threshold').matched_fraction_success.to_numpy()
  ax.plot(xs,ys,color=('0.82' if fs=='F4' else '0.70'),lw=.45,alpha=.75)
 meds=part.groupby('threshold').matched_fraction_success.median().reindex(xs).to_numpy()
 ax.plot(xs,meds,marker=('o' if fs=='F4' else 's'),lw=2.1,label=f'{fs}: fold medians')
ax.set_xlabel('Minimum weighted-Jaccard similarity for accepting a match')
ax.set_ylabel('Matched parent-rule fraction (successful draws)')
ax.set_ylim(.05,1.02); ax.set_xticks([.3,.5,.7]); ax.grid(axis='y',alpha=.2)
ax.legend(loc='lower left',frameon=True)
fig.tight_layout()
fig.savefig(F/'FigS1_Threshold_Sensitivity.pdf',bbox_inches='tight')
fig.savefig(F/'FigS1_Threshold_Sensitivity.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print('Created R6 vector figures F4, F5, and S1')
