"""Complete OOF predictions for the actual frozen round25 eight-member ensemble."""
from round22_common import ROOT,load,read,dump,sha
from round22_selector import dense_forward
from round3_features import build
from round14_prepare import family
from torch import nn
import torch,numpy as np,copy,time
OUT=ROOT/'experiments/round68'

def fit(x,y,ex,cfg):
    torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);mean=x.mean((0,1));scale=x.std((0,1)).clip(.001)
    xx=torch.as_tensor((x-mean)/scale,device='cuda');yy=torch.as_tensor(y,device='cuda')
    m=nn.Sequential(nn.Linear(x.shape[-1],96),nn.ReLU(),nn.Dropout(.2),nn.Linear(96,32),nn.ReLU(),nn.Linear(32,1)).cuda()
    ema=copy.deepcopy(m);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.1)
    for epoch in range(35):
        m.train()
        for ix in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/128))):
            s=m(xx[ix]).squeeze(-1);t=yy[ix];group=-(s.log_softmax(1)*t/t.sum(1,keepdim=True)).sum(1)
            bce=nn.functional.binary_cross_entropy_with_logits(s,t,reduction='none').mean(1);loss=(group+.25*bce).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),m.parameters()):ep.lerp_(mp,.02)
    w=dict(mean=mean,scale=scale,layers=np.asarray(3))
    for i,l in enumerate(v for v in ema if isinstance(v,nn.Linear)):
        w[f'w{i}']=l.weight.detach().cpu().numpy().T.copy();w[f'b{i}']=l.bias.detach().cpu().numpy().copy()
    return w,dense_forward(ex,w)

if __name__=='__main__':
    OUT.mkdir();torch.set_num_threads(4);start=time.perf_counter();b,a,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)}
    sources={};source=ROOT/'experiments/round25/selected_bundle'
    for new,old in [(0,0),(1,1),(2,2),(3,4)]:
        src=ROOT/f'experiments/round22/full/baseline/member{old}'
        assert sha(src/'model.npz')==sha(source/f'member{new}'/'model.npz')
        sources[str(new)]=dict(round22_member=old,matching_full_weight_sha256=sha(src/'model.npz'))
    dump(OUT/'plan.json',dict(reused_sources=sources,new_oof_members=[4,5,6,7],seed_recipe='Exact frozen round25 member training recipe',test_opened=False,source_sha256=sha(__file__)))
    with np.load(ROOT/'experiments/round22/oof.npz') as z:
        assert np.array_equal(z['ids'],ids);oof=np.zeros((len(b),4,8));oof[:,:,:4]=z['components'][:,:,[0,1,2,4]]
    seen=np.zeros((len(b),4),int)
    for kind,members in [('ground',[4,5]),('answer',[6,7])]:
        x=build(b,a,kind)
        for fold in range(3):
            part=read(ROOT/f'experiments/round22/oof/fold{fold}/partition.json');tr=np.array([index[k] for k in part['fit_ids']]);te=np.array([index[k] for k in part['held_ids']])
            assert not {family(b[i]['question']) for i in tr}&{family(b[i]['question']) for i in te}
            for member in members:
                cfg=read(source/f'member{member}'/'config.json');w,s=fit(x[tr],y[tr],x[te],cfg)
                p=OUT/f'fold{fold}'/f'member{member}';p.mkdir(parents=True);np.savez(p/'model.npz',**w);dump(p/'training.json',dict(training_ids=ids[tr].tolist(),held_ids=ids[te].tolist(),config=cfg))
                oof[te,:,member]=s;seen[te,member-4]+=1;print('OOF',fold,member,'seconds',time.perf_counter()-start,flush=True)
    assert (seen==1).all();np.savez(OUT/'oof.npz',ids=ids,labels=y,components=oof)
    dump(OUT/'oof_complete.json',dict(seconds=time.perf_counter()-start,all_members_question_disjoint=True,full_models_unchanged=True,test_opened=False))
