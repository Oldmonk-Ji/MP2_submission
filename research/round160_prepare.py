from round22_common import ROOT,load,read,dump,sha
from round159_factors import packet,matrix,checks
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round160'
if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(checks=checks(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round160*.py')},method='Same normalized candidate programs and27 summary as150; add source-only and constant-only input blocks for factorized probabilities. Scope packets must exactly match prior150 caches.',test_opened=False))
    for split in ('train','validation'):
        b,a,y,ids=load(split);programs=read(ROOT/f'experiments/round102/{split}_programs.json');fs=[];uf=[];cf=[];ps=[];ptr=[0]
        for i,(row,pp) in enumerate(zip(b,programs)):
            f,p,u,c=packet(row,pp);fs.extend(f);uf.extend(u);cf.extend(c);ps.append(p);ptr.append(len(fs))
            if i%500==0:print('PREPARE',split,i,flush=True)
        xx=matrix(fs);assert not (xx-sparse.load_npz(ROOT/f'experiments/round150/{split}_features.npz')).nnz;assert np.array_equal(ptr,np.load(ROOT/f'experiments/round150/{split}_ptr.npy'));assert ps==read(ROOT/f'experiments/round150/{split}_packets.json')
        for name,x in [('features',xx),('source',matrix(uf)),('constant',matrix(cf))]:sparse.save_npz(OUT/f'{split}_{name}.npz',x)
        np.save(OUT/f'{split}_ptr.npy',np.asarray(ptr));dump(OUT/f'{split}_packets.json',ps);dump(OUT/f'{split}_metadata.json',dict(ids=ids.tolist(),pairs=xx.shape[0],scope_features_and_packets_exactly_match150=True,test_opened=False))
    dump(OUT/'preparation_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
