"""Same frozen extraction, with an explicit empty source-pair matrix."""
import numpy as np
from scipy import sparse
from round104_features import extract,summarize,matrix,pair_matrix,structure_matrix
from round82_relations import DIM

def build(bank,arm,progress=None):
    rr=[];ee=[];ff=[];pf=[];packets=[];ptr=[0];structure=[]
    for i,row in enumerate(bank):
        r,e,f,p,packet,sf=extract(row,arm);structure.extend(sf);rr.append(r);ee.append(e);ff.extend(f);pf.extend(p);packets.append(packet);ptr.append(len(pf))
        if progress is not None and i%500==0:progress(i)
    pairs=pair_matrix(pf) if pf else sparse.csr_matrix((0,DIM),dtype=np.float32)
    return np.asarray(rr),np.asarray(ee),matrix(ff),pairs,np.asarray(ptr),packets,structure_matrix(structure)
