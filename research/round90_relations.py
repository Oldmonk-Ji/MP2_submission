"""Factor operation-family evidence from conditional operand direction."""
import numpy as np
from scipy import sparse
from scipy.special import expit, softmax

FAMILY=np.asarray([0,1,1,2,3,3])

def direction_matrix(x,is_div):
    return sparse.hstack((x,x.multiply((~is_div)[:,None]),x.multiply(is_div[:,None])),format='csr')

def probabilities(x,w):
    family=softmax(np.asarray(x@w['family_weight'].T)+w['family_bias'],axis=1)
    d=x.shape[1];weight=w['direction_weight'].reshape(3,d)
    common=np.asarray(x@weight[0]).ravel()+float(w['direction_bias'][0])
    sub=expit(common+np.asarray(x@weight[1]).ravel())
    div=expit(common+np.asarray(x@weight[2]).ravel())
    return np.column_stack((family[:,0],family[:,1]*(1-sub),family[:,1]*sub,family[:,2],family[:,3]*(1-div),family[:,3]*div))
