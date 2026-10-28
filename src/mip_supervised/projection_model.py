"""Direct MIP baseline: three TL projections plus explicit modality."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from lmc_common import MODS
from lmc_model import UNet,rescale

FEATURE_VERSION='v2-computed-focus-common-stack'

def projection_features(planes,px):
    total=maximum=best=None; samples=[]; best_score=-float('inf');n=0;native_shape=None
    for raw in planes:
        raw=np.asarray(raw,np.float32)
        assert raw.ndim==2 and np.isfinite(raw).all()
        if native_shape is None:native_shape=raw.shape
        assert raw.shape==native_shape
        mean=float(raw.mean(dtype=np.float64))
        focus=float(raw.var(dtype=np.float64)/mean) if mean>0 else -float('inf')
        x=rescale(raw,px)
        samples.append(x[::4,::4].ravel())
        total=x.copy() if total is None else total+x
        maximum=x.copy() if maximum is None else np.maximum(maximum,x)
        if best is None or focus>best_score:best=x.copy();best_score=focus
        n+=1
    if n==0:raise ValueError('Empty TL stack')
    low,high=np.percentile(np.concatenate(samples),[2,99.8])
    features=(np.stack([best,total/n,maximum])-low)/max(high-low,1e-6)
    return features.astype(np.float32),{'low':float(low),'high':float(high),'n_z':n,'native_shape':list(native_shape),'version':FEATURE_VERSION}


@torch.inference_mode()
def predict_projection(model,features,shape,mod,device):
    h,w=features.shape[-2:];H,W=math.ceil(h/16)*16,math.ceil(w/16)*16
    x=np.pad(features,((0,0),(0,H-h),(0,W-w)),mode='reflect')
    onehot=np.zeros((3,H,W),np.float32);onehot[MODS.index(mod)]=1
    inp=torch.from_numpy(np.concatenate([x,onehot])[None]).to(device)
    cuda=torch.device(device).type=='cuda'
    with torch.autocast('cuda' if cuda else 'cpu',dtype=torch.float16,enabled=cuda):y=model(inp).float()[:,:,:h,:w]
    y=F.interpolate(y,size=shape,mode='bilinear')[0].cpu().numpy()
    assert np.isfinite(y).all()
    return y
