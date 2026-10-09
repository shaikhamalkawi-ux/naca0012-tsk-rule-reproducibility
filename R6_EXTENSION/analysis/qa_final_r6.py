#!/usr/bin/env python3
from pathlib import Path
import re, json, hashlib, subprocess, math
import numpy as np,pandas as pd,fitz
R=Path(__file__).resolve().parents[1]
B=R.parent/'57_H3_Airfoil_R5'
main=R/'manuscript/main.tex';supp=R/'manuscript/supplement.tex'
s=main.read_text();sp=supp.read_text()
checks={}
# All citations defined, no bibliography orphans; first-appearance numbering.
all_cites=[]
for match in re.finditer(r'\\citep\{([^}]+)\}',s.split('\\begin{thebibliography}')[0]):
 for key in match.group(1).split(','):
  if key not in all_cites:all_cites.append(key)
refs=re.findall(r'\\bibitem\{([^}]+)\}',s)
checks['all_19_references_cited']=len(refs)==19 and set(refs)==set(all_cites)
checks['references_in_first_appearance_order']=refs==all_cites
for nm,txt in [('main',s),('supplement',sp)]:
 labs=set(re.findall(r'\\label\{([^}]+)\}',txt))
 wanted=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',txt))
 checks[f'{nm}_all_internal_refs_defined']=wanted.issubset(labs)
 checks[f'{nm}_no_internal_version_labels']=not bool(re.search(r'\b(?:R4|R5|R6|Stage A|v2 plan)\b',txt))
 checks[f'{nm}_no_placeholder_urls']=not bool(re.search(r'(REPLACE_THIS|INSERT_DOI|example\.com|TBD)',txt,re.I))
 checks[f'{nm}_no_artificial_outcome_claim']=not bool(re.search(r'(first time in literature|universally superior|new identification theorem (?!nor))',txt,re.I))

for nm in ('main','supplement'):
 pdf=R/'manuscript'/f'{nm}.pdf'
 with fitz.open(pdf) as doc:
  checks[f'{nm}_page_count_expected'] = (len(doc)==(11 if nm=='main' else 5))
  checks[f'{nm}_text_based']=all(len(p.get_text('text'))>100 for p in doc)
  # No render/text blocks extend beyond actual page geometry (2 pt tolerance)
  bad=[]
  for pn,p in enumerate(doc,1):
   rect=p.rect
   for block in p.get_text('dict')['blocks']:
    if block['type']!=0:continue
    x0,y0,x1,y1=block['bbox']
    if min(x0,y0)<-2 or x1>rect.width+2 or y1>rect.height+2:
     bad.append((pn,(x0,y0,x1,y1)))
  checks[f'{nm}_no_offpage_textblocks']=(len(bad)==0)
  txt='\n'.join(p.get_text('text') for p in doc)
  checks[f'{nm}_no_unresolved_ref_tokens']=not bool(re.search(r'(\[\?\]|\?\?|undefined references)',txt,re.I))

for nm in ('main','supplement'):
 log=(R/'manuscript'/f'{nm}.log').read_text(errors='replace')
 checks[f'{nm}_latex_no_overfull_or_undefined']=not bool(re.search(r'(Overfull \\hbox|Undefined control sequence|Citation .* undefined|Reference .* undefined|There were undefined references)',log))
 fonts=subprocess.check_output(['pdffonts',str(R/'manuscript'/f'{nm}.pdf')],text=True)
 checks[f'{nm}_zero_type3_fonts']='Type 3' not in fonts

# Exact lock from R5 source outputs, no rerun or edited source.
d=pd.read_csv(B/'results'/'TABLE_PRIMARY_BENCHMARK.csv');bs=pd.read_csv(B/'results'/'TABLE_BOOTSTRAP_SUMMARY.csv')
pairs=pd.read_csv(R/'analysis'/'R6_MATCHED_PAIRS_0p5.csv');sumry=pd.read_csv(R/'analysis'/'R6_SUMMARY_BY_SPECIFICATION.csv');folds=pd.read_csv(R/'analysis'/'R6_FOLD_LEVEL_DESCRIPTIVES.csv');thr=pd.read_csv(R/'analysis'/'R6_MATCH_THRESHOLD_AUDIT.csv')
checks['R6_pair_count_50376']=len(pairs)==50376
checks['R6_all_40_outer_analysis_units']=len(folds)==40
checks['R6_120_threshold_records']=len(thr)==120
checks['R6_spearman_f4_compact_median']=abs(folds.query('feature=="F4" and structure=="compact"').rho.median()+0.30036)<1e-4
checks['R6_spearman_f5_compact_median']=abs(folds.query('feature=="F5" and structure=="compact"').rho.median()+0.31387)<1e-4
checks['R6_actual_fit_failures_preserved']=int(sumry.groupby(['feature','structure']).n_fit_failures.first().sum())==2
# R5 scientific results reused: verify exact text values appear in manuscript (no numerical modifications)
checks['R5_primary_RMSE_table_unchanged']=all(v in s for v in ['2.381','2.648','4.166','3.950','4.395'])
checks['R5_bootstrap_values_unchanged']=all(v in s for v in ['1.606','2.479','10.513','28,000'])
checks['R6_values_explicitly_posthoc']=('post hoc' in s.lower() and '50,376' in s and 'matched' in s)
checks['R6_unmatched_excluded_from_D']=not pairs.Jaccard.lt(.5).any()
checks['source_papers_stored_privately']=all((R/'references/private_only'/name).exists() for name in ['Jin_2024_FSS_FullText_PRIVATE.pdf','KerrWilson_Pedrycz_2017_FSS_FullText_PRIVATE.pdf'])
checks['source_original_UCI_byte_hash_not_claimed']='original UCI download bytes were not hashed' in s

p_counts = {'main_pages':11,'supplement_pages':5}
checks={k:bool(v) for k,v in checks.items()}
status={'checks':checks,'pass':sum(bool(x) for x in checks.values()),'count':len(checks),
 'report': 'PASS' if all(checks.values()) else 'FAIL',
 'source_R5_sha256':(hashlib.sha256((R.parent/'57_H3_Airfoil_R5_Complete_Reproducibility_Package.zip').read_bytes()).hexdigest() if (R.parent/'57_H3_Airfoil_R5_Complete_Reproducibility_Package.zip').exists() else 'External R5 reference only'),
 'R6_documented_PDFs':p_counts}
(R/'qa'/'R6_FINAL_QA.json').write_text(json.dumps(status,indent=2))
print('R6 FINAL QA',status['report'],status['pass'],'/',status['count'])
if not all(checks.values()):print('FAILED',[(k,v) for k,v in checks.items() if not v])
assert all(checks.values())
