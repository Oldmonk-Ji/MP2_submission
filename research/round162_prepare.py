from round22_common import ROOT,load,read,dump,sha
from round162_decisions import packet,checks
from round139_features import matrix
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round162'
if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(checks=checks(),method='Fixed40 per-candidate features: root availability/count, source/constant counts; mean/min/max across roots of separate4 source-use,4 literal-presence,4 conditional-scope statistics. Reuse frozen159 models; no new training hyperparameters.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round162*.py')},test_opened=False))
    for split in ('train','validation'):
        b,a,y,ids=load(split);programs=read(ROOT/f'experiments/round102/{split}_programs.json');fs=[];uf=[];cf=[];ps=[];ptr=[0];uptr=[0];cptr=[0]
        for i,(row,pp) in enumerate(zip(b,programs)):
            f,p,u,c=packet(row,pp);fs.extend(f);uf.extend(u);cf.extend(c);ps.append(p);ptr.append(len(fs));uptr.append(len(uf));cptr.append(len(cf))
            if i%500==0:print('PREPARE',split,i,flush=True)
        xx=matrix(fs);assert not (xx-sparse.load_npz(ROOT/f'experiments/round150/{split}_features.npz')).nnz
        old=read(ROOT/f'experiments/round150/{split}_packets.json');assert [[[r['scope'] for r in c] for c in row] for row in ps]==old
        for name,x in [('features',xx),('source',matrix(uf)),('constant',matrix(cf))]:sparse.save_npz(OUT/f'{split}_{name}.npz',x)
        for name,p in [('ptr',ptr),('uptr',uptr),('cptr',cptr)]:np.save(OUT/f'{split}_{name}.npy',np.asarray(p))
        dump(OUT/f'{split}_packets.json',ps);dump(OUT/f'{split}_metadata.json',dict(ids=ids.tolist(),scope_packets_match150=True,test_opened=False))
    dump(OUT/'preparation_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
