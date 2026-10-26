"""Matched local-only/contextual relation SFT; no external pretrained weights."""
from round22_common import ROOT,load,read,dump,sha
from round87_model import local_matrix,context,forward
from round82_relations import LABELS
from scipy import sparse
from collections import Counter
from threadpoolctl import threadpool_limits
from torch import nn
import torch,numpy as np,time,copy
OUT=ROOT/'experiments/round87'

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.local=nn.Linear(8192,32);self.local_norm=nn.LayerNorm(32);self.context=nn.Linear(1024,32);self.context_norm=nn.LayerNorm(32);self.hidden=nn.Linear(64,32);self.out=nn.Linear(32,6);self.drop=nn.Dropout(.2)
    def forward(self,x,c):
        a=self.local_norm(self.local(x));b=self.context_norm(self.context(c));return self.out(self.drop(self.hidden(torch.cat([a,b],1)).relu()))

def fit(x,c,y,parents,mask,path,use_context):
    path.parent.mkdir(parents=True,exist_ok=True);torch.manual_seed(87087);rng=np.random.default_rng(87087);ii=np.flatnonzero(mask);_,unique=np.unique(parents[mask],return_index=True);mean=c[ii[unique]].mean(0);scale=c[ii[unique]].std(0).clip(.01)
    xx=torch.tensor(x[ii].toarray(),device='cuda');cc=torch.tensor(np.clip((c[ii]-mean)/scale,-8,8) if use_context else np.zeros_like(c[ii]),device='cuda');yy=torch.tensor(y[ii],device='cuda');count=Counter(parents[mask]);weights=np.asarray([1/count[p] for p in parents[mask]],np.float32);weights/=weights.mean();ww=torch.tensor(weights,device='cuda')
    m=Model().cuda();ema=copy.deepcopy(m);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);history=[];tick=time.perf_counter()
    for epoch in range(60):
        m.train();loss_total=0
        for batch in np.array_split(rng.permutation(len(ii)),int(np.ceil(len(ii)/256))):
            logits=m(xx[batch],cc[batch]);loss=(nn.functional.cross_entropy(logits,yy[batch],reduction='none')*ww[batch]).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();loss_total+=float(loss.detach())*len(batch)
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),m.parameters()):ep.lerp_(mp,.02)
        history.append(loss_total/len(ii))
    ema.eval();w=dict(mean=mean,scale=scale,use_context=np.asarray(use_context))
    for key,value in ema.state_dict().items():w[key.replace('.','_')]=value.detach().cpu().numpy()
    with torch.no_grad():expected=ema(xx[:16],cc[:16]).cpu().numpy()
    actual=forward(x[ii[:16]],c[ii[:16]],w);assert np.allclose(actual,expected,atol=2e-4)
    np.savez(path,**w);dump(path.with_suffix('.json'),dict(training_parent_ids=sorted(map(int,set(parents[mask]))),training_pairs=len(ii),epochs=60,seed=87087,use_context=use_context,history=history,seconds=time.perf_counter()-tick,cpu_gpu_agreement=True));print('FIT',path.parent.name,path.name,'seconds',round(time.perf_counter()-tick,2),flush=True)
    return w

if __name__=='__main__':
    threadpool_limits(4);torch.set_num_threads(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['local','context'],epochs=60,seed=87087,lr=.001,weight_decay=.01,dropout=.2,ema=.98,local_dim=8192,width=32,test_opened=False,supervision='Unchanged round82 labels',context='Mean of L2-normalized supplied candidate embeddings, L2-normalized; no label-dependent pooling',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round87*.py')}))
    b,a,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)};rows=read(ROOT/'experiments/round82/labels.json');parents=np.asarray([r['train_index'] for r in rows]);f=np.asarray([r['fold'] for r in rows]);targets=np.asarray([r['label'] for r in rows]);x=local_matrix(sparse.load_npz(ROOT/'experiments/round82/pair_features.npz'));q=context(a);ctx=q[[index[p] for p in parents]]
    records=[]
    for arm in ('local','context'):
        scores=np.zeros((len(rows),6))
        for held in range(3):
            w=fit(x,ctx,targets,parents,f!=held,OUT/arm/f'oof{held}.npz',arm=='context');scores[f==held]=forward(x[f==held],ctx[f==held],w)
        for held in (0,1):fit(x,ctx,targets,parents,f==1-held,OUT/arm/f'nested{held}.npz',arm=='context')
        fit(x,ctx,targets,parents,np.ones(len(rows),bool),OUT/arm/'full.npz',arm=='context');np.savez(OUT/f'{arm}_oof.npz',logits=scores,labels=targets,parents=parents,folds=f);te=f==2;pred=scores[te].argmax(1);rec=dict(arm=arm,n=int(te.sum()),correct=int((pred==targets[te]).sum()),by_class={LABELS[k]:dict(n=int((targets[te]==k).sum()),correct=int(((targets[te]==k)&(pred==k)).sum())) for k in range(6)});records.append(rec);print('HELD',rec,flush=True)
    dump(OUT/'held_relation_results.json',records);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False,auxiliary_supervision_only=True))
