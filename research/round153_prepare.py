"""Freeze markup repair after TRAIN-fit audit; apply to input-only feature caches."""
from round22_common import ROOT,load,read,dump,sha
from round153_features import build
from round152_markup import checks
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round153'
if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter()
    for src,h in read(ROOT/'experiments/round152/plan.json')['source_hashes'].items():assert sha(ROOT/src)==h
    dump(OUT/'plan.json',dict(checks=checks(),fit_audit=read(ROOT/'experiments/round152/fit_audit.json'),method='Union markup-recovered arithmetic roots with unchanged originals. Recompute all program-derived features. Keep all auxiliary models fixed; refit only fixed200-tree selector. No label read by extraction.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round153*.py')},test_opened=False))
    for split in ('train','validation'):
        b,a,y,ids=load(split);old=read(ROOT/f'experiments/round102/{split}_programs.json')
        r,e,f,p,ptr,packets,s=build(b,'local',lambda i:print('FEATURES',split,i,flush=True),old)
        np.savez(OUT/f'{split}_features.npz',root=r,extra=e,ids=ids)
        for name,x in [('flow',f),('pairs',p),('structure',s)]:sparse.save_npz(OUT/f'{split}_{name}.npz',x)
        np.save(OUT/f'{split}_ptr.npy',ptr);dump(OUT/f'{split}_packets.json',packets)
    dump(OUT/'preparation_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
