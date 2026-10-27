"""Input-only reduced semantic/numeric interaction expert."""
import numpy as np

def columns(arm):
    if arm=='no_surface':return np.arange(14,82)
    if arm=='verified_only':return np.r_[14:19,30:48,78:82]
    if arm=='binding_only':return np.r_[26:28,30:36,40:48,78:82]
    raise ValueError(arm)

def scores(e,g,w,arm):
    q=e.mean(1);r=e-q[:,None]
    r=np.clip(((r-w['rmean'])@w['rpc'].T)/w['rscale'],-6,6)
    q=np.clip(((q-w['qmean'])@w['qpc'].T)/w['qscale'],-6,6)
    g=np.clip((g-w['gmean'])/w['gscale'],-6,6)
    cross=(r[:,:,:,None]*q[:,None,None,:]).reshape(len(e),4,-1)
    a,b=np.triu_indices(32);quad=r[:,:,a]*r[:,:,b]
    local=g[:,:,columns(arm)]
    context=(q[:,None,:,None]*local[:,:,None,:]).reshape(len(e),4,-1)/np.sqrt(32*82)
    x=np.concatenate([r/np.sqrt(128),cross/np.sqrt(4096),quad/np.sqrt(len(a)),
                      g/np.sqrt(g.shape[-1]),(g-g.mean(1,keepdims=True))/np.sqrt(g.shape[-1]),context],2)
    return np.ascontiguousarray(x,dtype=np.float32)@w['weight']
