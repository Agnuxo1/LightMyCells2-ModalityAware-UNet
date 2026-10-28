"""Three fixed MIP aggregation interventions on a hash-selected real-data pilot."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['LMC_DEVICE']='cuda:0'
import argparse,csv,json,hashlib,sys,time,gc,datetime
from pathlib import Path
import numpy as np
import torch

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/home/ai/proyectos/light-my-cells-2/final-20261027'));args=ap.parse_args();base=args.root
 out=Path('output');out.mkdir(exist_ok=True)
 # Import the already-executed frozen model. Never modify it or the input dataset.
 sys.path[:0]=[str(base/'build'),str(base)]
 os.environ['LMC_MODEL_DIR']=str(base/'build/model')
 from lmc_common import MODS,ORGS,read_ome
 from lmc_model import rescale
 import inference
 from importlib.util import spec_from_file_location,module_from_spec
 math_path=Path(__file__).with_name('eval_math.py')
 if not math_path.exists():math_path=Path(__file__).resolve().parents[1]/'src/mip_supervised/eval_math.py'
 spec=spec_from_file_location('audit_eval_math',math_path);em=module_from_spec(spec);spec.loader.exec_module(em)
 torch.set_num_threads(1);assert torch.cuda.is_available()
 models=inference.init_model();assert len(models)==5
 index={}
 for study in [76,79,80,81,83,85,87]:
  paths={p.name:p for p in (base/'ed2-holdout'/f'Study_{study:02d}').rglob('*.tiff')}
  meta=list((base/'metadata-ed2').rglob(f'Study_{study:02d}_all_files.tsv'));assert len(meta)==1
  for r in csv.DictReader(meta[0].open(),delimiter='\t'):
   index.setdefault((study,int(r['image_id'])),{}).setdefault(r['channel'],{})[int(r['z'])]=paths[r['filename']]
 assert len(index)==46
 chosen=[]
 for study in [76,79,80,81,83,85,87]:
  ids=[k for k in index if k[0]==study]
  chosen.append(min(ids,key=lambda k:hashlib.sha256(f'LMC-independent-pilot-20261028:{k[0]}:{k[1]}'.encode()).hexdigest()))
 # This manifest is written before reading pixels or any score in this run.
 (out/'lmc_pilot_selection.json').write_text(json.dumps({'rule':'one minimum SHA256 ID per study; fixed salt LMC-independent-pilot-20261028','selected':chosen,'cohort':'Previously evaluated public 46-FOV cohort; exploratory pilot, not a fresh holdout.'},indent=2))
 metrics=[];timings=[];start=time.monotonic();sources={}
 for (study,fov) in chosen:
  t=time.monotonic();channels=index[study,fov];targets={}
  for org in ORGS:
   if org not in channels:continue
   gt=None
   for z,p in sorted(channels[org].items()):
    a,_=read_ome(p);gt=a.copy() if gt is None else np.maximum(gt,a)
   targets[org]=gt
  for mod in MODS:
   if mod not in channels:continue
   planes=sorted(channels[mod].items());samples=[];px=None;shape=None
   # Reproduce baseline common-stack percentiles from rescaled per-plane samples.
   for z,p in planes:
    raw,px=read_ome(p);shape=raw.shape;x=rescale(raw,px);samples.append(x[::4,::4].ravel())
   low,high=np.percentile(np.concatenate(samples),[2,99.8]);del samples
   maximum=total=soft_m=soft_s=None;tau=.1
   for z,p in planes:
    raw,_=read_ome(p);x=(rescale(raw,px)-low)/max(high-low,1e-6)
    pred=inference.predict_normalized(models,x,shape,mod,'cuda:0')
    assert np.isfinite(pred).all()
    if maximum is None:
     maximum=pred.copy();total=pred.astype(np.float64);soft_m=pred.copy();soft_s=np.ones_like(pred,dtype=np.float64)
    else:
     maximum=np.maximum(maximum,pred);total+=pred
     nm=np.maximum(soft_m,pred);soft_s=soft_s*np.exp((soft_m-nm)/tau)+np.exp((pred-nm)/tau);soft_m=nm
   mean=(total/len(planes)).astype(np.float32)
   soft=(soft_m+tau*np.log(soft_s/len(planes))).astype(np.float32)
   methods={'control_max':maximum,'L4_mean':mean,'L5_mix_half':.5*maximum+.5*mean,'L6_logmeanexp':soft}
   for org,gt in targets.items():
    oi=ORGS.index(org)
    for method,prediction in methods.items():
     sc=em.independent(prediction[oi],gt)
     metrics.append(dict(study=study,fov=fov,modality=mod,organelle=org,method=method,n_z=len(planes),ssim=sc['ssim'],pcc=sc['pcc']))
   del maximum,total,soft_m,soft_s,mean,soft,methods,pred,x,raw
   gc.collect();torch.cuda.empty_cache()
  timings.append({'study':study,'fov':fov,'seconds':time.monotonic()-t})
  with (out/'lmc_pilot_metrics.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
  print(json.dumps(timings[-1]),flush=True)
 # Sanity check against the independently archived original predictions on the same FOV.
 oldfile=base/'supervised_holdout_metrics.csv'
 if oldfile.exists():
  old={(int(r['study']),int(r['fov']),r['modality'],r['organelle']):r for r in csv.DictReader(oldfile.open()) if r['method']=='ed1_ensemble'}
  diffs=[abs(float(r['ssim'])-float(old[r['study'],r['fov'],r['modality'],r['organelle']]['ssim'])) for r in metrics if r['method']=='control_max' and r['ssim'] is not None]
  baseline_max_diff=max(diffs) if diffs else None
 else:baseline_max_diff=None
 # Keep summarisation dependency-free: the original completed GPU run saved
 # every measurement, then failed here because its frozen image lacks pandas.
 from collections import defaultdict
 from statistics import mean as average
 bins=defaultdict(list)
 for row in metrics:
  for metric in ['ssim','pcc']:
   if row[metric] is not None:
    bins[row['study'],row['fov'],row['organelle'],row['method'],metric].append(row[metric])
 fov={key:average(values) for key,values in bins.items()}
 summ=[]
 for org in sorted({r['organelle'] for r in metrics}):
  for method in ['control_max','L4_mean','L5_mix_half','L6_logmeanexp']:
   summ.append(dict(organelle=org,method=method,**{metric:average(v for (s,f,o,m,t),v in fov.items() if (o,m,t)==(org,method,metric)) for metric in ['ssim','pcc']}))
 paired=[]
 for metric in ['ssim','pcc']:
  for method in ['L4_mean','L5_mix_half','L6_logmeanexp']:
   by_fov=defaultdict(list)
   for (s,f,o,m,t),value in fov.items():
    if (m,t)==(method,metric):by_fov[s,f].append(value-fov[s,f,o,'control_max',metric])
   v=[average(values) for values in by_fov.values()]
   paired.append({'metric':metric,'method':method,'n_studies':len(v),'equal_fov_mean_delta':average(v),'improved_fov':sum(x>0 for x in v),'worsened_fov':sum(x<0 for x in v)})
 result={'kind':'new_GPU_MIP_aggregation_interventions','complete':True,'selected_fov':chosen,'n_fov':7,'rows':len(metrics),'summary':summ,'paired_deltas':paired,'baseline_max_abs_ssim_difference_from_archived':baseline_max_diff,'seconds':time.monotonic()-start,'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,'source_sha256':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [base/'build/inference.py',base/'build/model_manifest.json',math_path]},'versions':{'torch':torch.__version__,'numpy':np.__version__},'limits':['Exploratory seven-FOV pilot, one per study, not a fresh holdout.','Only stack aggregation changes; no new training.','Primary macro SSIM delta gives each available organelle equal weight within FOV.','No channel selection or promotion on this cohort.'],'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (out/'lmc_pilot_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'complete':True,'rows':len(metrics),'paired_deltas':paired}),flush=True)
if __name__=='__main__':main()
