"""Task-specific direct MIP adapter. Primary arm fixed before holdout outcomes."""
import json
import os
import time
from pathlib import Path
import torch
from lmc_model import UNet
from projection_model import projection_features,predict_projection,FEATURE_VERSION
from contract_io import read_case,write_outputs,INPUT_PATH,OUTPUT_PATH,INPUT_REL

DEVICE=torch.device(os.environ.get('LMC_DEVICE','cuda' if torch.cuda.is_available() else 'cpu'))
HERE=Path(__file__).parent
MODEL_DIR=Path(os.environ.get('LMC_MODEL_DIR','/opt/ml/model'))

def init_model():
    torch.set_num_threads(int(os.environ.get('LMC_THREADS','1')))
    manifest=json.loads((HERE/'model_manifest.json').read_text())
    assert manifest['task']=='mip' and manifest['primary_arm']=='A'
    checkpoint=torch.load(MODEL_DIR/manifest['weights'],map_location='cpu',weights_only=True)
    assert checkpoint['feature_version']==FEATURE_VERSION
    model=UNet(cin=6).to(DEVICE).eval();model.load_state_dict(checkpoint['model'])
    return model

def run(model):
    files=sorted(p for p in (INPUT_PATH/INPUT_REL).iterdir() if p.suffix.lower() in ('.tiff','.tif'))
    if not files:raise ValueError('No MIP input OME-TIFF')
    for path in files:
        start=time.monotonic();stack,px,mod,meta=read_case(path)
        features,info=projection_features(iter(stack),px)
        pred=predict_projection(model,features,stack.shape[-2:],mod,DEVICE)
        write_outputs(pred,meta,path.name)
        print(json.dumps({'task':'mip','file':path.name,'shape':list(stack.shape),'seconds':time.monotonic()-start,'device':str(DEVICE),'feature_version':FEATURE_VERSION}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(run(init_model()))
