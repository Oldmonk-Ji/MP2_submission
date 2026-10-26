from round22_common import ROOT,load,read,dump,sha
from round194_features import programs,build
from round192_trace import checks
from scipy import sparse
import time
OUT=ROOT/'experiments/round194'
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter()
    for src,h in read(ROOT/'experiments/round193/plan.json')['source_hashes'].items():assert sha(ROOT/src)==h
    dump(OUT/'plan.json',dict(checks=checks(),fit_audit=read(ROOT/'experiments/round193/fit_audit.json'),method='Freeze strict179 parser, then apply to all TRAIN/VAL inputs. Same local103 65536-dimensional structure features. Retain old104 selection evidence; this is an additional optional trace branch.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round194*.py')},test_opened=False))
    fit={r['train_index']:r['programs'] for r in read(ROOT/'experiments/round193/fit_programs.json')}
    for split in ('train','validation'):
        b,a,y,ids=load(split);pp=[]
        for i,row in enumerate(b):
            pp.append(fit[int(ids[i])] if split=='train' and int(ids[i]) in fit else programs(row))
            if i%500==0:print('PROGRAMS',split,i,flush=True)
        dump(OUT/f'{split}_programs.json',pp);x=build(b,pp,lambda i:print('FEATURES',split,i,flush=True));sparse.save_npz(OUT/f'{split}_features.npz',x);dump(OUT/f'{split}_metadata.json',dict(ids=ids.tolist(),rows=x.shape[0],columns=x.shape[1],nonzeros=x.nnz,test_opened=False))
    dump(OUT/'preparation_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
