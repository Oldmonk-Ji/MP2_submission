"""Remove surface/magnitude reliance from round235's contextual interaction."""
from round22_common import ROOT,dump,sha
from round227_quadratic import transform
from round240_inference import columns
import round236_answer_evidence as protocol
import numpy as np

def design(e,g,w,answers,arm):
    base=transform(e,g,w)
    q=np.clip(((e.mean(1)-w['qmean'])@w['qpc'].T)/w['qscale'],-6,6)
    g=np.clip((g-w['gmean'])/w['gscale'],-6,6)[:,:,columns(arm)]
    cross=(q[:,None,:,None]*g[:,:,None,:]).reshape(len(e),4,-1)/np.sqrt(32*82)
    return np.ascontiguousarray(np.concatenate([base,cross],2),dtype=np.float32)

def record(path,value):
    if path.name=='plan.json':
        value.update(source_hashes={n:sha(ROOT/'research'/n) for n in ('round240_trim_context.py','round240_inference.py','round236_answer_evidence.py')},
            method='Ablate only the new semantic interaction: no surface; arithmetic verified only; number binding only. Preserve original interaction scale, pairwise C1, base features and gate.',
            feature_columns={a:columns(a).tolist() for a in protocol.ARMS})
    dump(path,value)

if __name__=='__main__':
    # Reuse the identical TRAIN/held/VAL protocol and round235 comparisons.
    protocol.OUT=ROOT/'experiments/round240'
    protocol.ARMS=['no_surface','verified_only','binding_only']
    protocol.design=design
    protocol.dump=record
    protocol.main()
