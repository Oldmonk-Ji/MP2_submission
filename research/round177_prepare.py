from round22_common import ROOT,load,read,dump,sha
from round176_types import packet,checks
from round139_features import matrix
from scipy import sparse
import numpy as np,time
OUT=ROOT/'experiments/round177'
if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(checks=checks(),method='Only auxiliary source and scope input type blocks change. Programs and root decision packets must exactly match162.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round177*.py')},test_opened=False))
    for split in ('train','validation'):
        b,a,y,ids=load(split);programs=read(ROOT/f'experiments/round102/{split}_programs.json');old=read(ROOT/f'experiments/round162/{split}_packets.json');fs=[];uf=[];ptr=[0];uptr=[0]
        for i,(row,pp) in enumerate(zip(b,programs)):
            f,p,u,c=packet(row,pp);assert p==old[i];fs.extend(f);uf.extend(u);ptr.append(len(fs));uptr.append(len(uf))
            if i%500==0:print('PREPARE',split,i,flush=True)
        assert np.array_equal(ptr,np.load(ROOT/f'experiments/round162/{split}_ptr.npy'));assert np.array_equal(uptr,np.load(ROOT/f'experiments/round162/{split}_uptr.npy'))
        for name,values in [('features',fs),('source',uf)]:sparse.save_npz(OUT/f'{split}_{name}.npz',matrix(values))
        dump(OUT/f'{split}_metadata.json',dict(ids=ids.tolist(),packets_and_source_slots_match162=True,test_opened=False))
    dump(OUT/'preparation_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
