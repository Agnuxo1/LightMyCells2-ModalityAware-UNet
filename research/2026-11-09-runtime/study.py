"""Execute the frozen, paired transport-only optimization protocol."""
import hashlib,json,math,os,resource,statistics,sys,time
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(os.environ['LMC_BASELINE_ROOT'])
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
sys.path.insert(1,str(SOURCE))
import numpy as np
import torch
import inference as baseline
import streaming

OUT=Path(os.environ['LMC_STUDY_OUTPUT'])
P=json.loads((SOURCE/'PROTOCOL.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,value):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(value,indent=2));t.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
def timed(fn,models,stack,case):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 out=fn(models,stack,case['px'],case['mod'])
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 return out,elapsed,torch.cuda.max_memory_allocated()

assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert sha(BASE/'inference.py')==P['baseline_inference_sha256']
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
manifest=json.loads((BASE/'model_manifest.json').read_text())
assert len(manifest['checkpoints'])==5
model_dir=Path(os.environ['LMC_MODEL_DIR'])
for rec in manifest['checkpoints']:assert sha(model_dir/rec['file'])==rec['sha256']
models=baseline.init_model()
torch.set_num_threads(4)
source_hashes={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()}
result={'started_utc':now(),'protocol':P,'source_sha256':source_hashes,'torch':torch.__version__,
 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'visible_device_count':torch.cuda.device_count(),
 'baseline_source_verified':True,'all_five_weight_hashes_verified':True,'cases':[],'complete':False}
save('STARTED.json',result)
for si,seed in enumerate(P['seeds']):
 for ci,case in enumerate(P['cases']):
  rng=np.random.default_rng(seed)
  shape=tuple(case['shape'])
  if case['kind']=='constant':stack=np.full(shape,42,np.float32)
  elif case['kind']=='strided':
   backing=rng.standard_normal((shape[0],shape[1],shape[2]*2),dtype=np.float32)
   stack=backing[:,:,::2]
   assert not stack.flags.c_contiguous
  else:stack=rng.standard_normal(shape,dtype=np.float32)
  for fn in (baseline.predict_stack,streaming.predict_stack):
   warm=fn(models,stack,case['px'],case['mod']);del warm
  torch.cuda.synchronize()
  row={'seed':seed,'case':case,'pairs':[]}
  for rep in range(P['paired_repetitions']):
   order=['baseline','streaming'] if (si+ci+rep)%2==0 else ['streaming','baseline']
   values={};elapsed={};peak={}
   for name in order:
    fn=baseline.predict_stack if name=='baseline' else streaming.predict_stack
    values[name],elapsed[name],peak[name]=timed(fn,models,stack,case)
   a,b=values['baseline'],values['streaming']
   assert a.shape==b.shape==(4,*shape[-2:])
   assert np.isfinite(a).all() and np.isfinite(b).all()
   err=float(np.max(np.abs(a-b)))
   close=bool(np.allclose(a,b,atol=P['output_gate']['allclose_atol'],rtol=P['output_gate']['allclose_rtol']))
   row['pairs'].append({'repeat':rep,'order':order,'seconds':elapsed,'gpu_peak_bytes':peak,
    'shape':list(a.shape),'finite':True,'max_abs_error':err,'bitwise_equal':bool(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()),
    'allclose':close,'baseline_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
    'streaming_sha256':hashlib.sha256(b.tobytes()).hexdigest()})
   del values,a,b
   save('PROGRESS.json',{'current_seed':seed,'case':case['id'],'completed_pairs':len(row['pairs']),'completed_cases':len(result['cases']),'last_pair':row['pairs'][-1]})
  row['median_seconds']={n:statistics.median(v['seconds'][n] for v in row['pairs']) for n in ['baseline','streaming']}
  row['median_speedup']=row['median_seconds']['baseline']/row['median_seconds']['streaming']
  result['cases'].append(row)
  save('PARTIAL.json',result)
  print(json.dumps({'seed':seed,'case':case['id'],'speedup':row['median_speedup'],'max_abs_error':max(p['max_abs_error'] for p in row['pairs'])}),flush=True)
  del stack
  if case['kind']=='strided':del backing
pairs=[p for r in result['cases'] for p in r['pairs']]
g=P['promotion_gate']
geo=math.exp(statistics.mean(math.log(r['median_speedup']) for r in result['cases']))
gpu_peak=max(v for p in pairs for v in p['gpu_peak_bytes'].values())
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
checks={'nine_cases':len(result['cases'])==9,'twentyseven_pairs':len(pairs)==27,
 'all_output_equivalent':all(p['allclose'] for p in pairs),
 'geomean_speedup':geo>=g['geomean_of_nine_case_median_speedups_at_least'],
 'no_case_regression_over_five_percent':all(r['median_speedup']>=g['every_case_median_speedup_at_least'] for r in result['cases']),
 'gpu_memory':gpu_peak<g['gpu_peak_bytes_below'],'rss_memory':rss<g['process_peak_rss_bytes_below']}
result.update({'complete':True,'finished_utc':now(),'geomean_speedup':geo,'max_abs_error':max(p['max_abs_error'] for p in pairs),
 'all_bitwise_equal':all(p['bitwise_equal'] for p in pairs),'gpu_peak_bytes':gpu_peak,'process_peak_rss_bytes':rss,
 'checks':checks,'candidate_promoted':all(checks.values()),'biological_accuracy_tested':False,'t4_tested':False})
save('RESULT.json',result)
print(json.dumps({k:result[k] for k in ['complete','geomean_speedup','max_abs_error','all_bitwise_equal','checks','candidate_promoted']}),flush=True)
