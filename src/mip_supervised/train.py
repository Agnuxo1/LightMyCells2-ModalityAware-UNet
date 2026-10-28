"""Two fixed scientific loss arms in one host process, one RTX3090 per arm."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MALLOC_ARENA_MAX','2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path('/home/ai/proyectos/light-my-cells-2')
OUT=ROOT/'final-20261027'
sys.path.insert(0,str(OUT/'build'))
import numpy as np
import torch
import torch.nn.functional as F
from lmc_model import UNet
from lmc_common import MODS,ORGS
from projection_model import FEATURE_VERSION

torch.set_num_threads(1);torch.manual_seed(20261027);torch.cuda.manual_seed_all(20261027)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
assert torch.cuda.device_count()==2
records=json.loads((OUT/'mip-cache-development-v2.json').read_text())
assert len({(r['study'],r['fov']) for r in records})==33
assert all(r['feature_version']==FEATURE_VERSION for r in records)
holdout=json.loads((OUT/'holdout_frozen.json').read_text())
assert set(holdout['studies'])=={76,79,80,81,83,85,87}
assert not (set(holdout['studies'])&{r['study'] for r in records})
train_dir=OUT/'supervised-mip';train_dir.mkdir(exist_ok=True)
assert not (train_dir/'training_receipt.json').exists(),'Do not repeat finished training'
init=torch.load(OUT/'build/model/fold0.pt',map_location='cpu',weights_only=True)['model']
prototype=UNet(cin=6)
state=prototype.state_dict()
for k,v in init.items():
    if k=='down.0.0.weight':
        state[k][:,:3]=v[:,:1].repeat(1,3,1,1)/3
        state[k][:,3:]=v[:,1:]
    else:state[k]=v.clone()
devices=[torch.device('cuda:0'),torch.device('cuda:1')]
models=[UNet(cin=6).to(d) for d in devices]
for model in models:model.load_state_dict(state);model.train()
optimizers=[torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4) for m in models]
scalers=[torch.amp.GradScaler('cuda') for _ in models]
maps=[np.load(r['file'],mmap_mode='r') for r in records]
eligible=[[i for i,r in enumerate(records) if r['mask'][org]] for org in range(4)]
assert all(eligible)
rng=np.random.default_rng(20261027);skipped=[0,0];loss_rows=[];start=time.monotonic()

def ssim_loss(pred,gt,mask):
    ux=F.avg_pool2d(pred,7,1);uy=F.avg_pool2d(gt,7,1)
    vx=(F.avg_pool2d(pred*pred,7,1)-ux*ux)*(49/48)
    vy=(F.avg_pool2d(gt*gt,7,1)-uy*uy)*(49/48)
    cov=(F.avg_pool2d(pred*gt,7,1)-ux*uy)*(49/48)
    sim=((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    return ((1-sim)*mask).sum()/(mask.sum().clamp_min(1)*sim.shape[-2]*sim.shape[-1])

for step in range(1,2001):
    xs=[];ys=[];ms=[]
    for _ in range(4):
        org=int(rng.integers(4));i=int(rng.choice(eligible[org]));arr=maps[i]
        h,w=arr.shape[-2:]
        if h<256 or w<256:arr=np.pad(arr,((0,0),(0,max(0,256-h)),(0,max(0,256-w))),mode='reflect');h,w=arr.shape[-2:]
        top=int(rng.integers(h-255));left=int(rng.integers(w-255))
        crop=np.array(arr[:,top:top+256,left:left+256],copy=True)
        crop=np.rot90(crop,int(rng.integers(4)),axes=(-2,-1))
        if rng.random()<.5:crop=crop[:,:,::-1]
        if rng.random()<.5:crop=crop[:,::-1,:]
        onehot=np.zeros((3,256,256),np.float32);onehot[MODS.index(records[i]['modality'])]=1
        xs.append(np.concatenate([crop[:3],onehot]));ys.append(crop[3:]);ms.append(records[i]['mask'])
    x=torch.from_numpy(np.stack(xs));y=torch.from_numpy(np.stack(ys));mask=torch.tensor(ms,dtype=torch.float32)[:,:,None,None]
    row={'step':step}
    for arm,(model,opt,scaler,device) in enumerate(zip(models,optimizers,scalers,devices)):
        xx,yy,mm=x.to(device),y.to(device),mask.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):pred=model(xx)
        pred=pred.float()
        l1=((pred-yy).abs()*mm).sum()/(mm.sum().clamp_min(1)*256*256)
        loss=l1 if arm==0 else l1+.2*ssim_loss(pred,yy,mm)
        assert torch.isfinite(loss), (step,arm)
        previous=scaler.get_scale();scaler.scale(loss).backward();scaler.step(opt);scaler.update()
        if scaler.get_scale()<previous:skipped[arm]+=1
        if step%100==0:row['loss_arm'+str(arm)]=float(loss.detach())
    if step%100==0:
        row.update(seconds=time.monotonic()-start,skipped_updates=skipped.copy())
        loss_rows.append(row);print(json.dumps(row),flush=True)
        (train_dir/'progress.json').write_text(json.dumps(row))

artifacts=[]
for arm,model in enumerate(models):
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    p=train_dir/('arm_a_masked_l1.pt' if arm==0 else 'arm_b_l1_ssim.pt')
    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'cin':6,'feature_version':FEATURE_VERSION,'iterations':2000,'skipped_updates':skipped[arm],'loss':'maskedL1' if arm==0 else 'maskedL1+0.2SSIM','seed':20261027},p)
    artifacts.append({'arm':arm,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
receipt={'complete':True,'iterations':2000,'batch':4,'crop':256,'learning_rate':1e-4,'weight_decay':1e-4,'seed':20261027,
         'feature_version':FEATURE_VERSION,'training_fov':33,'holdout_studies':holdout['studies'],'no_holdout_selection':True,
         'skipped_updates':skipped,'seconds':time.monotonic()-start,'torch':torch.__version__,'artifacts':artifacts,
         'training_cache_manifest_sha256':hashlib.sha256((OUT/'mip-cache-development-v2.json').read_bytes()).hexdigest(),
         'holdout_freeze_sha256':hashlib.sha256((OUT/'holdout_frozen.json').read_bytes()).hexdigest()}
(train_dir/'training_receipt.json').write_text(json.dumps(receipt,indent=2));(train_dir/'loss_history.json').write_text(json.dumps(loss_rows,indent=2))
print(json.dumps(receipt),flush=True)
