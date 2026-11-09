"""Three synthetic full-shape IYX/ZYX cases; fixed weights, no mock/target data."""
import hashlib,importlib.util,json,os,sys,tempfile,time
from pathlib import Path
os.environ['LMC_DEVICE']='cuda:0';os.environ['LMC_THREADS']='4'
root=Path('/mnt/datos/proyectos/lmc2-runtime-rescue-20261109');base=root/'frozen-image-runtime'
os.environ['LMC_MODEL_DIR']=str(base/'model');sys.path.insert(0,str(base))
import numpy as np
import torch,tifffile
import inference as original
path=root/'iyx-reader-build/inference.py'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='019be8bc00d2e6af0b2aa95e65ae255c70a24d300db1f6b9bf29d6bd148641d7'
spec=importlib.util.spec_from_file_location('fixed_reader',path);fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
models=original.init_model();torch.set_num_threads(4);rows=[]
with tempfile.TemporaryDirectory(prefix='reader-gpu-',dir=root) as tmp:
 for seed,mod in [(20261109,'BF'),(7,'PC'),(1234,'DIC')]:
  arr=np.random.default_rng(seed).standard_normal((28,2048,2048),dtype=np.float32)
  xml=f'<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="float" SizeX="2048" SizeY="2048" SizeZ="28" SizeC="1" SizeT="1" PhysicalSizeX="0.2" PhysicalSizeY="0.2"><Channel ID="Channel:0:0" Name="{mod}" SamplesPerPixel="1"/></Pixels></Image></OME>'
  p=Path(tmp)/'legacy.tiff';tifffile.imwrite(p,arr,photometric='minisblack',description=xml,metadata=None)
  del arr
  with tifffile.TiffFile(p) as tf:assert tf.series[0].axes=='IYX'
  try:original.read_case(p)
  except ValueError as e:assert 'Expected one time and channel' in str(e)
  else:raise AssertionError('Original error not reproduced')
  stack,px,mode,meta=fixed.read_case(p)
  torch.cuda.synchronize();start=time.monotonic();a=original.predict_stack(models,stack,px,mode)
  b=fixed.predict_stack(models,stack,px,mode);torch.cuda.synchronize()
  assert a.shape==b.shape==(4,2048,2048) and np.isfinite(b).all() and a.tobytes()==b.tobytes()
  rows.append({'seed':seed,'modality':mod,'input_axes':'IYX','input_shape':[28,2048,2048],'old_rejection_reproduced':True,'prediction_bytes_equal':True,'max_abs_error':float(np.max(np.abs(a-b))),'seconds_for_pair':time.monotonic()-start,'sha256':hashlib.sha256(b.tobytes()).hexdigest()})
  Path('READER_GPU_PROGRESS.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
  del stack,a,b
  p.unlink()
receipt={'passed':len(rows)==3,'rows':rows,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'single_assigned_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'torch':torch.__version__,'no_mock_or_target_data':True,'t4_tested':False}
Path('READER_GPU_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'passed':receipt['passed'],'cases':len(rows)}),flush=True)
