"""CPU-only physical projection features and real MIP labels, memory mapped."""
import argparse
import csv
import ctypes
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/home/ai/proyectos/light-my-cells-2')
OUT=ROOT/'final-20261027'
sys.path.insert(0,str(OUT/'build'))
from lmc_common import MODS,ORGS,read_ome,pnorm
from lmc_model import rescale
sys.path.insert(0,str(Path(__file__).parent))
from projection_model import projection_features,FEATURE_VERSION
import torch
torch.set_num_threads(1)
ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['development','holdout'],required=True);a=ap.parse_args()
studies=[77,78,82,84,86] if a.kind=='development' else [76,79,80,81,83,85,87]
data=OUT/('ed2-cohort' if a.kind=='development' else 'ed2-holdout')
cache=OUT/('mip-cache-'+a.kind+'-v2');cache.mkdir(exist_ok=True)
records=[]
for st in studies:
    tsv=next((OUT/'metadata-ed2').rglob(f'Study_{st:02d}_all_files.tsv'))
    paths={p.name:p for p in (data/f'Study_{st:02d}').rglob('*.tiff')}
    fovs={}
    for r in csv.DictReader(tsv.open(),delimiter='\t'):
        fovs.setdefault(int(r['image_id']),{}).setdefault(r['channel'],{})[int(r['z'])]=paths[r['filename']]
    focus={int(r['image_id']):r for r in csv.DictReader(next((OUT/'metadata-ed2').rglob(f'Study_{st:02d}_best_focus.tsv')).open(),delimiter='\t')}
    for fov,channels in sorted(fovs.items()):
        for mod in MODS:
            if mod not in channels:continue
            destination=cache/f'S{st}_{fov}_{mod}.npy'
            sidecar=destination.with_suffix('.json')
            if sidecar.exists():records.append(json.loads(sidecar.read_text()));continue
            planes=sorted(channels[mod].items())
            raw,px=read_ome(planes[0][1]);shape=raw.shape
            features,feature_info=projection_features((read_ome(p)[0] for z,p in planes),px)
            target=np.zeros((4,*features.shape[-2:]),np.float32);mask=[]
            for i,org in enumerate(ORGS):
                mask.append(int(org in channels))
                if org not in channels:continue
                mip=None
                for z,p in sorted(channels[org].items()):
                    gt,_=read_ome(p);assert gt.shape==shape
                    mip=gt.copy() if mip is None else np.maximum(mip,gt)
                target[i]=pnorm(rescale(mip,px),sub=1)
            array=np.concatenate([features,target]).astype(np.float32)
            assert np.isfinite(array).all()
            np.save(destination,array)
            record={'study':st,'fov':fov,'modality':mod,'pixel_um':px,'native_shape':list(shape),'working_shape':list(features.shape[-2:]),'feature_version':FEATURE_VERSION,
                    'n_z':len(planes),'mask':mask,'file':str(destination),'kind':a.kind,
                    'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
            sidecar.write_text(json.dumps(record));records.append(record)
            print(json.dumps({'kind':a.kind,'study':st,'fov':fov,'modality':mod,'cache_ready':True}),flush=True)
            del array,features,target,raw
            if 'gt' in globals():del gt,mip
            gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
(OUT/('mip-cache-'+a.kind+'-v2.json')).write_text(json.dumps(records,indent=2))
print(json.dumps({'kind':a.kind,'records':len(records),'unique_fov':len({(r['study'],r['fov']) for r in records})}),flush=True)
