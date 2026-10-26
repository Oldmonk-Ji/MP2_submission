import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from round29_programs import quantities
from round4_quantities import quantity_documents
from scipy import sparse
DIM=16384

def documents(q):
    qs=quantities(q);counts=Counter(x['value'] for x in qs)
    qs=[dict(x,value=float(__import__('fractions').Fraction(x['value'])),occurrences=counts[x['value']],left=re.findall(r'[a-z]+',q[:x['start']].lower())[-8:],right=re.findall(r'[a-z]+',q[x['end']:].lower())[:8]) for x in qs]
    return [dict(Counter(d.split())) for d in quantity_documents(q,qs)]

def matrix(rows):
    return FeatureHasher(n_features=DIM,input_type='dict',alternate_sign=False,dtype=np.float32).transform(rows) if rows else sparse.csr_matrix((0,DIM),dtype=np.float32)

def build(bank):
    ds=[];ptr=[0]
    for r in bank:ds.extend(documents(r['question']));ptr.append(len(ds))
    return matrix(ds),np.asarray(ptr)
