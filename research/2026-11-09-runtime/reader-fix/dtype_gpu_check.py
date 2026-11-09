"""Preregistered synthetic integer TIFF coverage; frozen source/weights, no fitting."""
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
fixtures=[('uint16','random','BF',20261109),('uint16','zero','PC',7),('uint16','max','DIC',1234),('uint8','random','DIC',20261109),('uint8','zero','BF',7),('uint8','max','PC',1234)]
Path('PREREGISTRATION.json').write_text(json.dumps({'fixtures':fixtures,'shape':[28,2048,2048],'gates':['all6finite','prediction_bytes_equal','four_output_tiff_bytes_equal_to_prediction','output_shape_and_OME_spacing'],'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'no_quality_or_T4_claim':True},indent=2))
with tempfile.TemporaryDirectory(prefix='reader-dtype-',dir=root) as tmp:
 for dtype,kind,mod,seed in fixtures:
  start=time.monotonic();dt=np.dtype(dtype);hi=np.iinfo(dt).max
  arr=np.random.default_rng(seed).integers(0,hi+1,size=(28,2048,2048),dtype=dt) if kind=='random' else np.full((28,2048,2048),hi if kind=='max' else 0,dtype=dt)
  xml=f'<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="{dtype}" SizeX="2048" SizeY="2048" SizeZ="28" SizeC="1" SizeT="1" PhysicalSizeX="0.2" PhysicalSizeY="0.2"><Channel ID="Channel:0:0" Name="{mod}" SamplesPerPixel="1"/></Pixels></Image></OME>'
  p=Path(tmp)/'fixture.tiff';tifffile.imwrite(p,arr,photometric='minisblack',description=xml,metadata=None);del arr
  with tifffile.TiffFile(p) as tf:assert tf.series[0].axes=='IYX'
  stack,px,mode,meta=fixed.read_case(p)
  a=original.predict_stack(models,stack,px,mode);b=fixed.predict_stack(models,stack,px,mode);torch.cuda.synchronize()
  assert a.tobytes()==b.tobytes() and b.shape==(4,2048,2048) and np.isfinite(b).all()
  out=Path(tmp)/f'{dtype}-{kind}';fixed.write_outputs(b,meta,'out.ome.tiff',out)
  files=sorted(out.rglob('*.tiff'));assert len(files)==4
  for idx,org in enumerate(fixed.ORGS):
   op=out/fixed.OUTPUTS[org]/'out.ome.tiff'
   with tifffile.TiffFile(op) as tf:
    assert tf.series[0].shape==(2048,2048) and tf.asarray().tobytes()==b[idx].tobytes()
    assert tf.ome_metadata and 'PhysicalSizeX="0.2"' in tf.ome_metadata and 'PhysicalSizeY="0.2"' in tf.ome_metadata
  row={'dtype':dtype,'kind':kind,'modality':mod,'seed':seed,'input_shape':[28,2048,2048],'finite':True,'prediction_bytes_equal':True,'serialized_four_outputs_equal':True,'shape_and_spacing_pass':True,'seconds':time.monotonic()-start,'prediction_sha256':hashlib.sha256(b.tobytes()).hexdigest()}
  rows.append(row);Path('DTYPE_PROGRESS.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
  del stack,a,b;p.unlink()
receipt={'passed':len(rows)==6,'rows':rows,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'torch':torch.__version__,'single_assigned_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'synthetic_only':True,'t4_tested':False}
Path('DTYPE_OUTPUT_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'passed':receipt['passed'],'cases':len(rows)}),flush=True)
