import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/e3'))
import summarize_cv_audit as audit
METRICS=('ssim','pcc','ssim_raw','ssim_tl_trivial','pcc_tl_trivial')
FIELDS=('split','fov','study','site','mod','z','nz','org')+METRICS


def fixture(root):
    # Explicit row list and hand-calculated oracle, independent of expected_rows().
    index = {
        'A': dict(study=1,site='S',fold=0,tl={'one':['x'],'three':['x']*3},orgs=['Nucleus']),
        'B': dict(study=1,site='S',fold=0,tl={'one':['x']},orgs=['Nucleus']),
        'C': dict(study=2,site='S',fold=0,tl={'one':['x']},orgs=['Nucleus']),
        'D': dict(study=3,site='S',fold=1,tl={'one':['x']},orgs=['Nucleus']),
        'O': dict(study=4,site='S',fold='ood',tl={'one':['x']},orgs=['Nucleus']),
    }
    rows = {f:[] for f in range(5)}
    def add(fold,fov,mod,z,nz,value,split='val'):
        r = dict(split=split,fov=fov,study=index[fov]['study'],site='S',mod=mod,z=z,nz=nz,org='Nucleus')
        r.update({k:value for k in METRICS})
        rows[fold].append(r)
    add(0,'A','one',0,1,0)
    for z in (0,1,2): add(0,'A','three',z,3,1)
    add(0,'B','one',0,1,1)
    add(0,'C','one',0,1,0)
    add(1,'D','one',0,1,.9)
    for f in range(5): add(f,'O','one',0,1,(f+1)/10,'ood')
    def save():
        for fold, rs in rows.items():
            target = root / f'fold{fold}'
            target.mkdir(parents=True,exist_ok=True)
            with (target/'metrics_rows.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=FIELDS)
                writer.writeheader(); writer.writerows(rs)
    save()
    return index,rows,save

def test_hierarchical_weighting_and_ood(tmp_path):
    index,_,_=fixture(tmp_path)
    out=audit.aggregate(tmp_path,index)
    assert out['status']=='complete'
    # A=(0+mean(1,1,1))/2=.5; B=1; C=0; D=.9.
    assert out['per_fold']['val|Nucleus|0']['fov_mean']['ssim']==pytest.approx(.5)
    assert out['per_fold']['val|Nucleus|0']['study_mean']['ssim']==pytest.approx(.375)
    v=out['val']['Nucleus']
    assert v['n_fov']==4 and v['n_studies']==3
    assert v['fov_mean']['ssim']==pytest.approx(.6)
    assert v['study_mean']['ssim']==pytest.approx(.55)
    assert v['fold_macro_mean']['ssim']==pytest.approx(.7)
    o=out['ood']['Nucleus']
    assert o['n_unique_fov']==1 and o['n_model_evaluations']==5
    assert o['model_mean']['ssim']==pytest.approx(.3)
    assert o['model_sd']['ssim']==pytest.approx(math.sqrt(.025))

def test_z_average_uses_every_selected_plane(tmp_path):
    index,rows,save=fixture(tmp_path)
    for row in rows[0]:
        if row['fov']=='A' and row['mod']=='three':
            for metric in METRICS:row[metric]=row['z']/2
    save();out=audit.aggregate(tmp_path,index)
    # A modality mean=(0+.5+1)/3=.5, then A FOV=(0+.5)/2=.25.
    assert out['val']['Nucleus']['fov_mean']['ssim']==pytest.approx((.25+1+0+.9)/4)
    assert out['per_fold']['val|Nucleus|0']['fov_mean']['ssim']==pytest.approx((.25+1)/3)

@pytest.mark.parametrize('fault',['missing','duplicate','unexpected','metadata','bad_metric'])
def test_coverage_faults_block_final(tmp_path,fault):
    index,rows,save=fixture(tmp_path)
    if fault=='missing': rows[0].pop(0)
    if fault=='duplicate': rows[0].append(dict(rows[0][0]))
    if fault=='unexpected': rows[0].append({**rows[0][0],'z':99})
    if fault=='metadata': rows[0][0]['study']=999
    if fault=='bad_metric': rows[0][0]['ssim']='not_a_number'
    save(); out=audit.aggregate(tmp_path,index)
    c=out['coverage']['0']
    assert out['status']=='INCOMPLETE_DO_NOT_REPORT_AS_FINAL'
    if fault=='missing': assert c['missing']==1
    if fault=='duplicate':
        assert c['duplicates']==1
        assert out['val']['Nucleus']['fov_mean']['ssim']==pytest.approx(.6)
    if fault=='unexpected': assert c['unexpected']==1
    if fault in ('metadata','bad_metric'): assert c['invalid']==1 and c['missing']==1

@pytest.mark.parametrize('metric',METRICS)
def test_nan_reported_per_metric_without_zero_imputation(tmp_path,metric):
    index,rows,save=fixture(tmp_path)
    # B is wholly missing for one metric, independent metrics retain B=1.
    next(r for r in rows[0] if r['fov']=='B')[metric]='nan'
    save();out=audit.aggregate(tmp_path,index)
    assert out['status']=='complete_with_metric_warnings'
    assert len(out['metric_warnings'])==1
    assert out['metric_warnings'][0]['metric']==metric
    v=out['val']['Nucleus']
    assert v['valid_fov_per_metric'][metric]==3
    assert v['fov_mean'][metric]==pytest.approx((.5+0+.9)/3)
    for other in set(METRICS)-{metric}:
        assert v['valid_fov_per_metric'][other]==4
        assert v['fov_mean'][other]==pytest.approx(.6)
    json.dumps(out,allow_nan=False)

def test_all_nan_metric_is_null_and_metric_only(tmp_path):
    index,rows,save=fixture(tmp_path)
    for rs in rows.values():
        for r in rs:r['ssim_raw']='nan'
    save();out=audit.aggregate(tmp_path,index)
    v=out['val']['Nucleus']
    assert v['valid_fov_per_metric']['ssim_raw']==0
    assert v['fov_mean']['ssim_raw'] is None
    assert v['fov_mean']['ssim']==pytest.approx(.6)
    assert out['ood']['Nucleus']['model_mean']['ssim_raw'] is None
    json.dumps(out,allow_nan=False)

def test_constant_gt_excludes_all_metrics_and_preserves_csv(tmp_path):
    index,_,_=fixture(tmp_path)
    paths=list(tmp_path.glob('fold*/metrics_rows.csv'))
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out=audit.aggregate(tmp_path,index,constant_gt={('B','Nucleus')})
    v=out['val']['Nucleus']
    for metric in METRICS:
        assert v['valid_fov_per_metric'][metric]==3
        assert v['fov_mean'][metric]==pytest.approx((.5+0+.9)/3)
    assert len(out['metric_warnings'])==6
    assert sum('constant reference' in w.get('reason','') for w in out['metric_warnings'])==1
    assert {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}==before
    json.dumps(out,allow_nan=False)
