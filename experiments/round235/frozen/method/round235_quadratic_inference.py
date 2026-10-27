"""Input-only semantic/grounding quadratic expert; NumPy only."""
import numpy as np

def scores(e,g,w,arm):
    q=e.mean(1)
    r=e-q[:,None]
    r=np.clip(((r-w['rmean'])@w['rpc'].T)/w['rscale'],-6,6)
    q=np.clip(((q-w['qmean'])@w['qpc'].T)/w['qscale'],-6,6)
    g=np.clip((g-w['gmean'])/w['gscale'],-6,6)
    cross=(r[:,:,:,None]*q[:,None,None,:]).reshape(len(e),4,-1)
    a,b=np.triu_indices(32)
    quad=r[:,:,a]*r[:,:,b]
    blocks=[r/np.sqrt(128),cross/np.sqrt(4096),quad/np.sqrt(len(a)),
            g/np.sqrt(g.shape[-1]),(g-g.mean(1,keepdims=True))/np.sqrt(g.shape[-1])]
    if arm in ('context_ground','both_ground'):
        blocks.append((q[:,None,:,None]*g[:,:,None,:]).reshape(len(e),4,-1)/np.sqrt(32*g.shape[-1]))
    if arm in ('residual_ground','both_ground'):
        blocks.append((r[:,:,:32,None]*g[:,:,None,:]).reshape(len(e),4,-1)/np.sqrt(32*g.shape[-1]))
    return np.ascontiguousarray(np.concatenate(blocks,2),dtype=np.float32)@w['weight']
