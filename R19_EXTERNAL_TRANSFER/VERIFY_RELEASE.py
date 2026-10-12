#!/usr/bin/env python3
"""R19 GitHub snapshot verifier using only the Python standard library.

Verifies archived bytes, folds, train/test disjointness, search-grid enumeration,
source identity, and OOF aggregate metrics. This is NOT a model-refitting test.
"""
from __future__ import annotations
from collections import defaultdict
from io import StringIO
import csv, hashlib, json, math, pathlib, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parent
FOLDER = 'R19_Executable_Reproduction/'
SOURCE_REL = FOLDER + 'R17_Southampton_External_Reproducibility/data/SOUTHAMPTON_CANONICAL.csv'
SRCZIP_REL = FOLDER + 'R17_Southampton_External_Reproducibility/source/DataFiles.zip'
OUTPUT = FOLDER + 'R19_Clean_12grid/'

def digest(data): return hashlib.sha256(data).hexdigest()

def csv_rows(data): return list(csv.DictReader(StringIO(data.decode('utf-8'))))

def rms_group(rows, group, model):
    sums=defaultdict(lambda:[0.,0])
    for row in rows:
        e=(float(row['actual'])-float(row[f'{model}_pred']))**2
        z=sums[row[group]]; z[0]+=e; z[1]+=1
    return math.sqrt(sum(ss/n for ss,n in sums.values())/len(sums))

def check_close(actual, expected, name, eps=2e-12):
    assert math.isfinite(actual) and abs(actual-expected)<eps,(name,actual,expected)

def main():
    pack = ROOT / 'R19_Southampton_ExternalTransfer_SourceToResult_Public.zip'
    with zipfile.ZipFile(pack) as z:
        assert z.testzip() is None, 'CRC failed'
        names=set(z.namelist())
        sha=json.loads(z.read('SHA256_PAYLOADS.json'))['entries']
        for n,hexdigest in sha.items():
            assert n in names and digest(z.read(n))==hexdigest, f'SHA256 mismatch: {n}'
        lock=json.loads(z.read('LOCK.json')) if 'LOCK.json' in names else json.loads((ROOT/'LOCK.json').read_text())
        assert lock['n_rows']==343 and lock['n_spectra']==38 and lock['n_geometry']==7
        assert digest(z.read(SRCZIP_REL)) == lock['source_data_sha256']
        assert digest(z.read(SOURCE_REL)) == lock['canonical_data_sha256']
        assert len(zipfile.ZipFile(__import__('io').BytesIO(z.read(SRCZIP_REL))).namelist())==17
        inp=csv_rows(z.read(SOURCE_REL)); out=csv_rows(z.read(OUTPUT+'LOGO_PREDICTIONS.csv'))
        assert len(inp)==len(out)==343
        canon={int(r['row_id']):r for r in inp}
        assert len(canon)==343
        seen=set()
        for r in out:
            key=int(r['row_id'])
            assert key not in seen and key in canon;seen.add(key)
            assert r['heldout_geometry']==canon[key]['geometry_id']
            assert r['group_id']==canon[key]['group_id']
            check_close(float(r['actual']),float(canon[key]['response_norm']),f'target row {key}')
        geom=set(r['heldout_geometry'] for r in out)
        groups=set(r['group_id'] for r in out)
        assert len(geom)==7 and len(groups)==38
        assert {r['geometry_id'] for r in inp}==geom
        expected=json.loads(z.read(OUTPUT+'SUMMARY.json'))
        assert expected['n_rows']==343 and expected['n_geometries']==7 and expected['n_spectra']==38
        scores={}
        for model in ('TSK','ET'):
            scores[model+'_row_RMSE']=math.sqrt(sum((float(r['actual'])-float(r[model+'_pred']))**2 for r in out)/len(out))
            scores[model+'_equal_geometry_RMSE']=rms_group(out,'heldout_geometry',model)
            scores[model+'_equal_spectrum_RMSE']=rms_group(out,'group_id',model)
        for name,v in scores.items():check_close(v,expected[name],name)
        check_close(scores['TSK_equal_geometry_RMSE'],0.0593961028381699,'locked TSK')
        check_close(scores['ET_equal_geometry_RMSE'],0.04783880803883461,'locked ET')
        assert scores['ET_equal_geometry_RMSE'] < scores['TSK_equal_geometry_RMSE']
        for h in geom:
            ec=csv_rows(z.read(OUTPUT+'candidates/ET_holdout_'+h+'.csv'))
            tc=csv_rows(z.read(OUTPUT+'candidates/TSK_holdout_'+h+'.csv'))
            assert len(ec)==12 and len(tc)==45, (h,len(ec),len(tc))
            assert {int(float(r['min_samples_leaf'])) for r in ec}=={1,3,5}
            assert {float(r['max_features']) for r in ec}=={0.7,1.0}
            assert (OUTPUT+'states/'+h+'.npz') in names
            assert zipfile.ZipFile(__import__('io').BytesIO(z.read(OUTPUT+'states/'+h+'.npz'))).testzip() is None
            split=json.loads(z.read(OUTPUT+'splits/'+h+'.json'))
            train=set(split['outer_train_rows']);test=set(split['outer_test_rows'])
            assert not (train & test) and train|test==seen
            assert test=={i for i,r in canon.items() if r['geometry_id']==h}
            for inner in split['inner_folds']:
                a=set(inner['train_row_ids']); b=set(inner['validation_row_ids'])
                assert not (a & b) and a|b==train
                assert not {canon[i]['group_id'] for i in a}&{canon[i]['group_id'] for i in b}
        cover=csv_rows(z.read(OUTPUT+'COVERAGE_PREDICTIONS.csv'))
        assert len(cover)==343
        low=sum(int(float(r['below_train_q05'])) for r in cover)
        low_and_high=sum(int(float(r['below_train_q05'])) and float(r['max_normalized_rule_weight'])>=.9 for r in cover)
        assert (low,low_and_high)==(30,10),(low,low_and_high)
        aq=json.loads(z.read(OUTPUT+'INDEPENDENT_REPLAY_QA.json'))
        assert aq['pass'] is True and aq['corrected_clean_grid_per_fold']==12
        assert aq['max_TSK_saved_vs_independent_replay']<1e-12 and aq['max_ET_saved_vs_selected_retraining']<1e-12
    print(json.dumps({'status':'PASS','note':'snapshot and OOF replay, not model training','source_sha256':lock['source_data_sha256'],'canonical_sha256':lock['canonical_data_sha256'],'payload_files_verified':len(sha),'rows':len(out),'spectra':len(groups),'geometries':len(geom),'candidates_per_geometry':{'ExtraTrees':12,'TSK':45},'low_absolute_activation_rows':low,'normalized_weight_ge_0_9_among_low':low_and_high,'scores':scores},indent=2))
    return 0

if __name__=='__main__':sys.exit(main())
