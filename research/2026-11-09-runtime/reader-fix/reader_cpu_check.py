"""Real synthetic TIFF regression: reproduce IYX, validate canonical equivalence and ambiguity guards."""
import importlib.util,json,os,sys,tempfile,xml.etree.ElementTree as ET
from pathlib import Path
os.environ['LMC_DEVICE']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/opt/app')
import numpy as np
import tifffile,torch
import inference as fixed
spec=importlib.util.spec_from_file_location('old_reader','/old/inference.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
torch.set_num_threads(4);models=fixed.init_model();torch.set_num_threads(4)
def xml(z,mod,c=1,t=1):
 return f'<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="float" SizeX="37" SizeY="23" SizeZ="{z}" SizeC="{c}" SizeT="{t}" PhysicalSizeX="0.2" PhysicalSizeY="0.15"><Channel ID="Channel:0:0" Name="{mod}" SamplesPerPixel="1"/></Pixels></Image></OME>'
rows=[]
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 for idx,(z,mod) in enumerate([(3,'BF'),(5,'PC'),(28,'DIC')]):
  arr=np.random.default_rng(20261109+idx).standard_normal((z,23,37),dtype=np.float32)
  p=root/f'legacy-{mod}.tiff';tifffile.imwrite(p,arr,photometric='minisblack',metadata=None,description=xml(z,mod))
  with tifffile.TiffFile(p) as tf:assert len(tf.series)==1 and tf.series[0].axes=='IYX' and tf.ome_metadata
  try:old.read_case(p)
  except ValueError as e:assert 'Expected one time and channel' in str(e)
  else:raise AssertionError('Original reader did not reproduce IYX rejection')
  got,px,mode,meta=fixed.read_case(p);assert np.array_equal(got,arr) and px==.2 and mode==mod
  q=root/f'canonical-{mod}.ome.tiff';tifffile.imwrite(q,arr,photometric='minisblack',ome=True,metadata={'axes':'ZYX','Channel':{'Name':[mod]},'PhysicalSizeX':.2,'PhysicalSizeY':.15})
  standard,spx,smode,smeta=old.read_case(q);assert np.array_equal(standard,got) and spx==px and smode==mode
  ref=old.predict_stack(models,standard,spx,smode);actual=fixed.predict_stack(models,got,px,mode)
  assert ref.shape==actual.shape==(4,23,37) and ref.tobytes()==actual.tobytes()
  output=root/('out-'+mod);fixed.write_outputs(actual,meta,p.name,output)
  for org in fixed.ORGS:
   with tifffile.TiffFile(output/fixed.OUTPUTS[org]/p.name) as tf:
    assert tf.is_ome and tf.series[0].axes=='YX' and tf.asarray().shape==(23,37)
    pix=next(n for n in ET.fromstring(tf.ome_metadata).iter() if n.tag.endswith('Pixels'))
    assert float(pix.get('PhysicalSizeX'))==.2 and float(pix.get('PhysicalSizeY'))==.15
  rows.append({'mode':mod,'depth':z,'real_tiff_axes':'IYX','old_rejection_reproduced':True,'canonical_pixels_equal':True,'predictions_bitwise_equal':True,'four_ome_outputs':True})
 for name,z,c,t,ome in [('multiple-channels',3,2,1,True),('multiple-times',3,1,2,True),('depth-mismatch',4,1,1,True),('missing-ome',3,1,1,False)]:
  p=root/(name+'.tiff');a=np.ones((3,23,37),np.float32)
  tifffile.imwrite(p,a,photometric='minisblack',metadata=None,description=xml(z,'PC',c,t) if ome else None)
  try:fixed.read_case(p)
  except ValueError:rows.append({'rejection_case':name,'rejected':True})
  else:raise AssertionError('Ambiguous/metadata-missing input accepted: '+name)
result={'passed':True,'valid_real_iyx_cases':3,'rejection_cases':4,'rows':rows,'no_mock_or_target_pixels_used':True,'model_and_normalization_unchanged':True,'gpu_used':False}
Path('/receipt/READER_CPU_RECEIPT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
