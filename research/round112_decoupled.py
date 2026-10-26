"""Frozen process preference pools program features; a separate scorer ranks candidates."""
import numpy as np
from scipy import sparse
from round107_latent import pool

def aggregate(x,ptr,teacher):
    _,weight,owner=pool(np.asarray(x@teacher).ravel(),ptr)
    projection=sparse.csr_matrix((weight.astype(np.float32),(owner,np.arange(x.shape[0]))),shape=(len(ptr)-1,x.shape[0]))
    return (projection@x).tocsr()

def predict(x,ptr,teacher,ranker):
    # Algebraically identical to aggregate(x) @ ranker, without materializing
    # a second large feature matrix during inference.
    _,weight,owner=pool(np.asarray(x@teacher).ravel(),ptr)
    return np.bincount(owner,weights=weight*np.asarray(x@ranker).ravel(),minlength=len(ptr)-1).reshape(-1,4)

def checks():
    rng=np.random.default_rng(112112);x=sparse.csr_matrix(rng.normal(size=(11,7)));ptr=np.array([0,2,3,3,5,7,8,10,11]);teacher=rng.normal(size=7);ranker=rng.normal(size=7)
    assert np.allclose(predict(x,ptr,teacher,ranker).ravel(),aggregate(x,ptr,teacher)@ranker,atol=1e-6)
    from round107_latent import subset
    xx,pp=subset(x,ptr,np.array([1,0]));assert np.allclose(predict(xx,pp,teacher,ranker),predict(x,ptr,teacher,ranker)[[1,0]])
    assert predict(x,ptr,teacher,ranker)[0,2]==0
    return dict(direct_score_matches_weighted_features=True,question_permutation=True,empty_candidate_zero=True)
