"""Official four-output LMC2 adapter. Fixed five-fold ed1 ensemble.

Best-focus: historical per-image preprocessing. MIP: a preregistered heuristic
with ONE input normalisation per stack, and maximum of ensemble predictions.
This is a baseline to falsify against real fluorescence MIPs, not a claim that
best-focus supervision recovers calibrated per-plane fluorescence intensities.
"""
import json
import math
import os
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from lmc_common import MODS, ORGS, UNIT_TO_UM, modality_from, pnorm
from lmc_model import UNet, rescale

HERE = Path(__file__).parent
INPUT_PATH = Path(os.environ.get('LMC_INPUT', '/input'))
OUTPUT_PATH = Path(os.environ.get('LMC_OUTPUT', '/output'))
MODEL_DIR = Path(os.environ.get('LMC_MODEL_DIR', '/opt/ml/model'))
INPUT_REL = 'images/organelles-transmitted-light-ome-tiff'
OUTPUTS = {o: f'images/{o.lower()}-fluorescence-ome-tiff' for o in ORGS}
DEVICE = torch.device(os.environ.get('LMC_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))


def read_case(path):
    """Validate OME dimensions without silently flattening time or channels."""
    with tifffile.TiffFile(path) as tf:
        if len(tf.series) != 1:
            raise ValueError('Exactly one image series is required')
        s = tf.series[0]
        axes, shape = s.axes, s.shape
        if axes[-2:] != 'YX' or len(axes) != len(shape):
            raise ValueError(f'Unsupported axes {axes} / {shape}')
        for axis, n in zip(axes[:-2], shape[:-2]):
            if axis not in 'ZTC' or axis != 'Z' and n != 1:
                raise ValueError(f'Expected one time and channel, got {axes}/{shape}')
        xml = tf.ome_metadata
        if not xml:
            raise ValueError('OME metadata is required to identify TL modality')
        root = ET.fromstring(xml)
        pixels = next(e for e in root.iter() if e.tag.endswith('Pixels'))
        channels = [e for e in pixels if e.tag.endswith('Channel')]
        if len(channels) != 1:
            raise ValueError('Exactly one transmitted-light channel is required')
        channel = channels[0].get('Name', '')
        mod = modality_from(channel)
        if mod is None:
            raise ValueError(f'Unknown transmitted-light modality: {channel!r}')
        meta = {}
        for a in ('X', 'Y'):
            val, unit = pixels.get('PhysicalSize'+a), pixels.get('PhysicalSize'+a+'Unit', 'µm')
            if val is not None:
                if unit not in UNIT_TO_UM or not math.isfinite(float(val)) or float(val) <= 0:
                    raise ValueError(f'Invalid physical pixel size {val} {unit}')
                meta['PhysicalSize'+a] = float(val)
                meta['PhysicalSize'+a+'Unit'] = unit
        px = meta.get('PhysicalSizeX')
        px = px * UNIT_TO_UM[meta['PhysicalSizeXUnit']] if px else None
        if px is None or 'PhysicalSizeY' not in meta:
            raise ValueError('PhysicalSizeX/Y are required for valid Grand Challenge voxel spacing')
        arr = s.asarray().reshape((-1, *shape[-2:])).astype(np.float32)
    if not np.isfinite(arr).all() or min(arr.shape[-2:]) < 8:
        raise ValueError('Input must be finite and at least 8x8')
    return arr, px, mod, meta


def init_model():
    torch.set_num_threads(int(os.environ.get('LMC_THREADS', '1')))
    manifest = json.loads((HERE/'model_manifest.json').read_text())
    models = []
    for record in manifest['checkpoints']:
        ck = torch.load(MODEL_DIR/record['file'], map_location='cpu', weights_only=True)
        model = UNet().eval().to(DEVICE)
        model.load_state_dict(ck['model'])
        models.append(model)
    if not models:
        raise ValueError('Empty ensemble')
    return models


@torch.inference_mode()
def predict_normalized(models, x, original_shape, mod, device=None):
    """Infer from an already rescaled/normalised plane, retaining training code."""
    device = DEVICE if device is None else torch.device(device)
    h,w=x.shape
    H,W=math.ceil(h/16)*16,math.ceil(w/16)*16
    x=np.pad(x,((0,H-h),(0,W-w)),mode='reflect')
    onehot=np.zeros((3,H,W),np.float32)
    onehot[MODS.index(mod)]=1
    inp=torch.from_numpy(np.concatenate((x[None],onehot))[None].astype(np.float32)).to(device)
    cuda=device.type=='cuda'
    # Fixed fp16 on CUDA is supported on the official T4 and on both 3090s.
    # CPU runs float32; differences are measured and reported, never assumed zero.
    total=None
    with torch.autocast('cuda' if cuda else 'cpu',dtype=torch.float16,enabled=cuda):
        for model in models:
            pred=model(inp).float()[:,:,:h,:w]
            total=pred if total is None else total+pred
    pred=F.interpolate(total/len(models),size=original_shape,mode='bilinear')
    out=pred[0].cpu().numpy()
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite prediction')
    return out


def predict_stack(models, stack, px, mod, normalization='common', device=None):
    """Baseline: max_z(mean_folds f(TL_z)); common affine input normalisation."""
    if stack.ndim != 3 or stack.shape[0] < 1:
        raise ValueError('Expected nonempty ZYX')
    # Only small samples retained to compute a single stack-level intensity scale.
    low=high=None
    if len(stack)>1 and normalization=='common':
        sample=np.concatenate([rescale(plane,px)[::4,::4].ravel() for plane in stack])
        low,high=np.percentile(sample,[2,99.8])
    out=None
    for plane in stack:
        x=rescale(plane,px)
        if low is None:
            x=pnorm(x)
        else:
            x=(x-low)/max(high-low,1e-6)
        pred=predict_normalized(models,x,plane.shape,mod,device)
        out=pred if out is None else np.maximum(out,pred)
    return out


def write_outputs(pred, meta, filename, output_path=None):
    output_path=OUTPUT_PATH if output_path is None else Path(output_path)
    if pred.ndim!=3 or pred.shape[0]!=4 or not np.isfinite(pred).all():
        raise ValueError('Expected four finite YX predictions')
    px_um=[float(meta['PhysicalSize'+a])*UNIT_TO_UM[meta.get('PhysicalSize'+a+'Unit','µm')] for a in ('X','Y')]
    for i,o in enumerate(ORGS):
        d=output_path/OUTPUTS[o]
        d.mkdir(parents=True,exist_ok=True)
        tifffile.imwrite(d/filename,pred[i].astype(np.float32),ome=True,photometric='minisblack',
                         metadata=dict(meta,axes='YX',Channel={'Name':[o]}),compression='zlib',tile=(128,128),
                         resolution=(10000/px_um[0],10000/px_um[1]),resolutionunit='CENTIMETER')


def run(model):
    start=time.monotonic()
    files=sorted(p for p in (INPUT_PATH/INPUT_REL).iterdir() if p.suffix.lower() in ('.tiff','.tif'))
    if not files:
        raise ValueError('No OME-TIFF input cases')
    for f in files:
        t=time.monotonic()
        stack,px,mod,meta=read_case(f)
        pred=predict_stack(model,stack,px,mod)
        write_outputs(pred,meta,f.name)
        print(json.dumps({'file':f.name,'shape':list(stack.shape),'modality':mod,'pixel_um':px,
                          'seconds':time.monotonic()-t,'device':str(DEVICE),'normalization':'common-stack' if len(stack)>1 else 'per-image'}),flush=True)
    print(json.dumps({'cases':len(files),'total_seconds':time.monotonic()-start}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(run(init_model()))
