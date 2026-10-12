#!/usr/bin/env python3
"""Prespecified descriptive LOGO coverage/error audit, no test-set cutoff tuning.
Absolute fuzzy activation is exp(-0.5 * squared width-scaled rule distance).
Training-only activation cutoffs at 1%, 5%, 10%; fixed, not outcome-selected.
"""
from pathlib import Path
import json,numpy as np,pandas as pd
from scipy.stats import spearmanr
BASE=Path(__file__).resolve().parent
DATA=BASE/'R17_Southampton_External_Reproducibility'/'data'/'SOUTHAMPTON_CANONICAL.csv'
OUT=BASE/'R19_Clean_12grid'
D=pd.read_csv(DATA).set_index('row_id')
P=pd.read_csv(OUT/'LOGO_PREDICTIONS.csv').set_index('row_id')
P['abs_TSK_error']=(P.actual-P.TSK_pred).abs();P['abs_ET_error']=(P.actual-P.ET_pred).abs()
P['TSK_sq_error']=P.abs_TSK_error**2;P['ET_sq_error']=P.abs_ET_error**2
summ=[]; rows=[]
for geom in sorted(P.heldout_geometry.unique()):
 z=np.load(OUT/'states'/f'{geom}.npz',allow_pickle=False)
 assert set(z['test_rows'].tolist())==set(P[P.heldout_geometry==geom].index.tolist())
 t=D.loc[z['train_rows']]
 x=np.c_[np.log10(t.frequency_Hz.to_numpy(float)),t.angle_deg.to_numpy(float),t.trip_flag.to_numpy(float),t.chord_m.to_numpy(float),t.velocity_m_s.to_numpy(float)]
 Z=(x-z['mean'])/z['scale']
 q=(Z[:,None,:]-z['C'][None,:,:])/z['S'][None,:,:]
 train_lm=(-.5*np.sum(q*q,axis=2)).max(axis=1)
 assert len(train_lm)==len(z['train_rows'])
 thresholds={f'train_logq{v:02d}':float(np.quantile(train_lm,v/100)) for v in (1,5,10)}
 gg=P[P.heldout_geometry==geom].copy()
 for v in (1,5,10):
  key=f'below_train_q{v:02d}';gg[key]=gg.log_best_firing < thresholds[f'train_logq{v:02d}']
  P.loc[gg.index,key]=gg[key].astype(int)
 # ordinary exponential is numerically defined on this dataset
 gg['max_absolute_activation']=np.exp(gg.log_best_firing)
 P.loc[gg.index,'max_absolute_activation']=gg.max_absolute_activation
 for rid,r in gg.iterrows():rows.append({'row_id':int(rid),'geometry':geom,'spectrum':r.group_id,'best_log_activation':r.log_best_firing,'best_absolute_activation':r.max_absolute_activation,'max_normalized_firing_weight':r.max_normalized_rule_weight,'TSK_absolute_error':r.abs_TSK_error,'ET_absolute_error':r.abs_ET_error,'below_train_q05':bool(r.below_train_q05)})
 record={'geometry':geom,'rows':len(gg),'spectra':gg.group_id.nunique(),'TSK_RMSE':float(np.sqrt(gg.TSK_sq_error.mean())),'ET_RMSE':float(np.sqrt(gg.ET_sq_error.mean())),'median_log_best_firing':float(gg.log_best_firing.median()), 'median_absolute_max_activation':float(gg.max_absolute_activation.median()),'median_max_normalized_weight':float(gg.max_normalized_rule_weight.median()),'train_q01_log_activation':thresholds['train_logq01'],'train_q05_log_activation':thresholds['train_logq05'],'train_q10_log_activation':thresholds['train_logq10']}
 for v in (1,5,10): record[f'frac_heldout_below_train_q{v:02d}']=float(gg[f'below_train_q{v:02d}'].mean())
 record['frac_weight90_and_below_train_q05']=float(np.mean((gg.max_normalized_rule_weight>=.9)&gg.below_train_q05))
 summ.append(record)
P.reset_index().to_csv(OUT/'COVERAGE_PREDICTIONS.csv',index=False,float_format='%.17g')
pd.DataFrame(rows).to_csv(OUT/'COVERAGE_ROW_METRICS.csv',index=False,float_format='%.17g')
G=pd.DataFrame(summ);G.to_csv(OUT/'COVERAGE_GEOMETRY.csv',index=False,float_format='%.17g')
# Average row sq errors to spectrum-level; no row-wise independent tests.
spec=P.groupby(['heldout_geometry','group_id']).agg(nrows=('actual','size'),mean_log_activation=('log_best_firing','mean'),median_log_activation=('log_best_firing','median'),mean_abs_error_TSK=('abs_TSK_error','mean'),mean_abs_error_ET=('abs_ET_error','mean'),mean_sq_error_TSK=('TSK_sq_error','mean'),mean_sq_error_ET=('ET_sq_error','mean'),below_q05_share=('below_train_q05','mean'),median_max_weight=('max_normalized_rule_weight','median')).reset_index()
spec['spectrum_rmse_TSK']=np.sqrt(spec.mean_sq_error_TSK);spec['spectrum_rmse_ET']=np.sqrt(spec.mean_sq_error_ET)
spec.to_csv(OUT/'COVERAGE_SPECTRUM.csv',index=False,float_format='%.17g')
# Within-geometry: each geometry mean-center the spectrum-level values; report only descriptive association.
for col in ('median_log_activation','spectrum_rmse_TSK','spectrum_rmse_ET'):
 spec[f'centered_{col}']=spec[col]-spec.groupby('heldout_geometry')[col].transform('mean')
res={
 'design':'descriptive, seven nested heldout geometry tests with 38 correlated spectra; no row-level p-values',
 'cutoffs':'train-only 1%, 5%, 10% quantiles of parent-model absolute best log activation',
 'n_geometries':int(len(G)),'n_spectra':int(len(spec)),'n_rows':int(len(P)),
 'geometry_level_spearman_log_activation_vs_TSK_RMSE':float(spearmanr(G.median_log_best_firing,G.TSK_RMSE).statistic),
 'spectrum_level_naive_spearman_log_activation_vs_TSK_RMSE_descriptive_only':float(spearmanr(spec.median_log_activation,spec.spectrum_rmse_TSK).statistic),
 'within_geometry_centered_pearson_log_activation_vs_TSK_RMSE_descriptive_only':float(np.corrcoef(spec.centered_median_log_activation,spec.centered_spectrum_rmse_TSK)[0,1]),
 'low_coverage_test_rows_train_q05':int(P.below_train_q05.sum()),
 'rows_maxweight_ge_0_9_and_low_coverage_q05':int(((P.max_normalized_rule_weight>=.9)&(P.below_train_q05==1)).sum()),
 'fraction_low_coverage_q05_while_normalized_weight_ge_0_9':float(np.mean(P.loc[P.below_train_q05==1,'max_normalized_rule_weight']>=.9)),
 'normalized_target_not_dB':True,
 'caveat':'Geometry-level N=7 is too small for confident transfer effects or universal coverage-error thresholds.'}
(OUT/'COVERAGE_AUDIT.json').write_text(json.dumps(res,indent=2))
print(G.to_string(index=False)); print(json.dumps(res,indent=2))
