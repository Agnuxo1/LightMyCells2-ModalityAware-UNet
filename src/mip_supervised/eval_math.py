"""Provisional full-image SSIM/PCC, cached float64 CUDA and independent CPU check."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

def norm(array):
    x=np.asarray(array,np.float64);a,b=np.percentile(x,[2,99.8])
    return (x-a)/(b-a) if b>a else x.copy()

def prepare(array,device):
    x=torch.from_numpy(norm(array)).to(device)
    mean=F.avg_pool2d(x[None,None],7,1)[0,0]
    var=(F.avg_pool2d((x*x)[None,None],7,1)[0,0]-mean*mean)*(49/48)
    return x,mean,var

def score(p,g):
    x,ux,vx=p;y,uy,vy=g;dr=float(y.max()-y.min())
    if dr==0:return {'ssim':None,'pcc':None}
    cov=(F.avg_pool2d((x*y)[None,None],7,1)[0,0]-ux*uy)*(49/48)
    c1,c2=(.01*dr)**2,(.03*dr)**2
    ss=float((((2*ux*uy+c1)*(2*cov+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2))).mean())
    xc=x-x.mean();yc=y-y.mean();den=torch.sqrt((xc*xc).sum()*(yc*yc).sum())
    pc=float((xc*yc).sum()/den) if float(den)>0 else None
    return {'ssim':ss,'pcc':pc}

def independent(pred,gt):
    if np.ptp(gt)==0:return {'ssim':None,'pcc':None}
    p,g=norm(pred),norm(gt)
    return {'ssim':float(structural_similarity(g,p,data_range=float(np.ptp(g)),win_size=7,use_sample_covariance=True)),
            'pcc':float(pearsonr(g.ravel(),p.ravel()).statistic) if np.ptp(p)>0 else None}
