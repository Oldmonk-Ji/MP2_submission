"""Marginal correct-set objective: selecting any correct candidate is sufficient."""
import numpy as np
from scipy.special import logsumexp

def loss_gradient(w,x,y):
    s=np.asarray(x@w).reshape(-1,4);valid=y>0;assert valid.any(1).all()
    all_lse=logsumexp(s,axis=1);positive=np.where(valid,s,-np.inf);good_lse=logsumexp(positive,axis=1)
    loss=2*np.sum(all_lse-good_lse)+.5*np.dot(w,w)
    ds=2*(np.exp(s-all_lse[:,None])-np.exp(positive-good_lse[:,None]))
    return float(loss),np.asarray(x.T@ds.ravel()).ravel()+w

def checks():
    from scipy import sparse
    rng=np.random.default_rng(126126);x=sparse.csr_matrix(rng.normal(size=(12,7)));w=rng.normal(size=7)*.2;y=np.array([[1,0,1,0],[1,1,1,1],[0,1,0,0]])
    loss,g=loss_gradient(w,x,y);num=[]
    for k in range(len(w)):
        d=np.zeros_like(w);d[k]=1e-5;num.append((loss_gradient(w+d,x,y)[0]-loss_gradient(w-d,x,y)[0])/2e-5)
    assert np.allclose(g,num,atol=2e-7,rtol=2e-6)
    a,ag=loss_gradient(w,x[4:8],y[1:2]);assert np.allclose(a,.5*np.dot(w,w)) and np.allclose(ag,w)
    order=np.array([3,0,2,1]);ix=(np.arange(3)[:,None]*4+order).ravel();b,bg=loss_gradient(w,x[ix],y[:,order]);assert np.allclose(loss,b) and np.allclose(g,bg)
    return dict(analytic_gradient=True,all_correct_parent_has_no_ranking_gradient=True,candidate_permutation=True)
