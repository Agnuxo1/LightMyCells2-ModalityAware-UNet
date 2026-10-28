"""One fixed evaluation of two arms, frozen old model and baselines on 46 new FOV."""
import csv
import ctypes
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path('/home/ai/proyectos/light-my-cells-2')
OUT=ROOT/'final-20261027'
sys.path.insert(0,str(OUT/'build'))
os.environ['LMC_DEVICE']='cuda:0';os.environ['LMC_MODEL_DIR']=str(OUT/'build/model')
import numpy as np
import torch
from lmc_common import MODS,ORGS,read_ome
from lmc_model import UNet,rescale
from projection_model import projection_features,predict_projection,FEATURE_VERSION
from eval_math import prepare,score,independent
import inference

torch.set_num_threads(1)
receipt=json.loads((OUT/'supervised-mip/training_receipt.json').read_text());assert receipt['complete'] and receipt['iterations']==2000
assert (OUT/'holdout_download_receipt.json').exists()
old=inference.init_model();models=[]
for name in ('arm_a_masked_l1.pt','arm_b_l1_ssim.pt'):
    ck=torch.load(OUT/'supervised-mip'/name,map_location='cpu',weights_only=True)
    assert ck['feature_version']==FEATURE_VERSION
    model=UNet(cin=6).to('cuda:0').eval();model.load_state_dict(ck['model']);models.append(model)
index={}
for st in (76,79,80,81,83,85,87):
    paths={p.name:p for p in (OUT/'ed2-holdout'/f'Study_{st:02d}').rglob('*.tiff')}
    for r in csv.DictReader(next((OUT/'metadata-ed2').rglob(f'Study_{st:02d}_all_files.tsv')).open(),delimiter='\t'):
        index.setdefault((st,int(r['image_id'])),{}).setdefault(r['channel'],{})[int(r['z'])]=paths[r['filename']]
assert len(index)==46
directory=OUT/'holdout-evaluation';directory.mkdir(exist_ok=True)
fields=['study','fov','modality','organelle','method','ssim','pcc','n_z','pixel_um']
methods=['supervised_a','supervised_b','ed1_ensemble','tl_max','tl_mean']
started=time.monotonic();worst=0;checked=0;count_rows=0;timings=[]
for (st,fov),channels in sorted(index.items()):
    f=directory/f'S{st}_{fov}.csv'
    expected=len(set(channels)&set(MODS))*len(set(channels)&set(ORGS))*len(methods)
    if f.exists():
        rows=list(csv.DictReader(f.open()))
        if len(rows)==expected:count_rows+=len(rows);continue
        f.rename(f.with_suffix('.partial-before-recovery'))
    t=time.monotonic();rows=[]
    targets={}
    for org in ORGS:
        if org not in channels:continue
        result=None
        for z,p in sorted(channels[org].items()):
            gt,_=read_ome(p);result=gt.copy() if result is None else np.maximum(result,gt)
        targets[org]=result
    references={org:prepare(gt,'cuda:1') for org,gt in targets.items()}
    for mod in MODS:
        if mod not in channels:continue
        planes=sorted(channels[mod].items());tl,px=read_ome(planes[0][1]);shape=tl.shape
        features,info=projection_features((read_ome(p)[0] for z,p in planes),px)
        new=[predict_projection(m,features,shape,mod,'cuda:0') for m in models]
        historical=maximum=total=None
        for z,p in planes:
            raw,_=read_ome(p)
            x=(rescale(raw,px)-info['low'])/max(info['high']-info['low'],1e-6)
            pr=inference.predict_normalized(old,x,shape,mod,'cuda:0')
            historical=pr if historical is None else np.maximum(historical,pr)
            maximum=raw.copy() if maximum is None else np.maximum(maximum,raw)
            total=raw.copy() if total is None else total+raw
        baseline={'tl_max':maximum,'tl_mean':total/len(planes)}
        cache={m:prepare(a,'cuda:1') for m,a in baseline.items()}
        for org,gt in targets.items():
            i=ORGS.index(org)
            preds={'supervised_a':new[0][i],'supervised_b':new[1][i],'ed1_ensemble':historical[i]}|baseline
            for method,array in preds.items():
                prepared=cache[method] if method in cache else prepare(array,'cuda:1')
                values=score(prepared,references[org])
                if checked<5 or count_rows%100==0:
                    ref=independent(array,gt)
                    errors=[abs(ref[k]-values[k]) for k in ('ssim','pcc') if ref[k] is not None]
                    assert all((ref[k] is None)==(values[k] is None) for k in ('ssim','pcc'))
                    worst=max([worst]+errors);assert worst<1e-6,worst;checked+=1
                rows.append(dict(study=st,fov=fov,modality=mod,organelle=org,method=method,n_z=len(planes),pixel_um=px,**values));count_rows+=1
    assert len(rows)==expected
    with f.open('x',newline='') as dst:
        writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    timings.append({'study':st,'fov':fov,'seconds':time.monotonic()-t,'rows':len(rows)})
    (directory/f'S{st}_{fov}.receipt.json').write_text(json.dumps(timings[-1]))
    print(json.dumps(timings[-1]),flush=True)
    del targets,references,cache,new,historical,maximum,total,rows,prepared,features,tl,pr,raw,gt,preds,array
    gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0);torch.cuda.empty_cache()

all_rows=[]
for p in sorted(directory.glob('S*.csv')):all_rows.extend(csv.DictReader(p.open()))
assert len(all_rows)==650
assert len({(r['study'],r['fov'],r['modality'],r['organelle'],r['method']) for r in all_rows})==650
final=OUT/'supervised_holdout_metrics.csv'
with final.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_rows)
result={'complete':True,'n_fov':46,'n_studies':7,'rows':650,'modality_cases':54,'primary_candidate':'supervised_a',
        'no_holdout_selection':True,'seconds':time.monotonic()-started,'checks_vs_cpu':checked,'max_cpu_difference':worst,
        'source_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'training_receipt_sha256':hashlib.sha256((OUT/'supervised-mip/training_receipt.json').read_bytes()).hexdigest(),
        'peak_gpu0_mib':torch.cuda.max_memory_allocated(0)/1024**2,'peak_gpu1_mib':torch.cuda.max_memory_allocated(1)/1024**2}
(OUT/'supervised_holdout_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
