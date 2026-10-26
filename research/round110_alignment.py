"""Parent-balanced candidate ranking and reference-equivalent root decisions."""
import numpy as np
from scipy.special import expit
from round107_latent import pool,pairs

def supervision(y,ptr,gold,active):
    count=np.diff(ptr);owner=np.repeat(np.arange(len(count)),count)
    ng=np.bincount(owner,weights=gold,minlength=len(count))
    assert np.all((~active)|((ng>0)&(ng<count)&(y.ravel()>0)))
    nactive=active.reshape(-1,4).sum(1);parent=nactive>0
    pos,neg,pw=pairs(y);pw=pw*np.where(parent[pos//4],.5,1.)
    coefficient=np.zeros(len(active));coefficient[active]=1/nactive[np.flatnonzero(active)//4]
    # Both objective branches together retain original parent mass 2.
    mass=np.bincount(pos//4,weights=2*pw,minlength=len(y))+coefficient.reshape(-1,4).sum(1)
    mixed=(y.min(1)==0)&(y.max(1)>0)
    assert np.allclose(mass[mixed],2.)
    return (pos,neg,pw),coefficient

def loss_gradient(w,x,ptr,gold,pair,coefficient):
    raw=np.asarray(x@w).ravel();scores,rw,owner=pool(raw,ptr)
    pos,neg,pw=pair;delta=scores[pos]-scores[neg]
    loss=2*np.dot(pw,np.logaddexp(0,-delta))+.5*np.dot(w,w)
    d=-2*pw*expit(-delta)
    cg=np.bincount(pos,weights=d,minlength=len(scores))-np.bincount(neg,weights=d,minlength=len(scores))
    rg=cg[owner]*rw
    active=coefficient>0;use=gold&active[owner]
    if np.any(active):
        maximum=np.full(len(scores),-np.inf);np.maximum.at(maximum,owner[use],raw[use])
        gw=np.zeros(len(raw));gw[use]=np.exp(raw[use]-maximum[owner[use]])
        total=np.bincount(owner,weights=gw,minlength=len(scores));gw[use]/=total[owner[use]]
        all_lse=scores[active]+np.log(np.diff(ptr)[active])
        gold_lse=maximum[active]+np.log(total[active])
        loss+=np.dot(coefficient[active],all_lse-gold_lse)
        rg+=coefficient[owner]*(rw-gw)
    return float(loss),np.asarray(x.T@rg).ravel()+w

def checks():
    from scipy import sparse
    from round107_latent import loss_gradient as original
    rng=np.random.default_rng(110110);x=sparse.csr_matrix(rng.normal(size=(11,7)))
    ptr=np.array([0,2,3,3,5,7,8,10,11]);y=np.array([[1,0,1,0],[0,1,0,1]],float)
    gold=np.zeros(11,bool);gold[0]=True;active=np.zeros(8,bool);active[0]=True
    pair,coef=supervision(y,ptr,gold,active);w=rng.normal(size=7)*.2
    _,g=loss_gradient(w,x,ptr,gold,pair,coef);numeric=[]
    for j in range(len(w)):
        step=np.zeros_like(w);step[j]=1e-5
        numeric.append((loss_gradient(w+step,x,ptr,gold,pair,coef)[0]-loss_gradient(w-step,x,ptr,gold,pair,coef)[0])/2e-5)
    assert np.allclose(g,numeric,atol=2e-7,rtol=2e-6)
    pair,coef=supervision(y,ptr,gold,np.zeros(8,bool));a,ag=loss_gradient(w,x,ptr,gold,pair,coef);b,bg=original(w,x,ptr,pairs(y))
    assert np.allclose(a,b) and np.allclose(ag,bg)
    return dict(analytic_alignment_gradient=True,inactive_matches_original=True,mixed_parent_mass_preserved=True)
