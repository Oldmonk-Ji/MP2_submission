"""Meaningful invariants: group equality, ranking boundaries, order, and isolation."""
from data_access import ROOT,CODE,forbid_test
import json,hashlib
import numpy as np
from round9_group_features import build,group_labels,subset,candidate_scores
from round10_fusion import arrays,fuse

row=dict(id='synthetic-unseen',question='A worker makes 10 dollars per hour for 2 hours.',candidates=[dict(text=t,index=i) for i,t in enumerate(['10*2=20\n#### 20','10+2=12\n#### 12','20 dollars\n#### 20','#### 2'])])
bank=[row];raw=np.array([[[2.,1.],[0.,0.],[1.,2.],[-1.,-1.]]]);base=np.array([[2.,0.,1.,-1.]])
x,b,mapping,sizes,meta=build(bank,raw,base,None,'groups');assert sizes.tolist()==[3];assert group_labels(np.array([[1,0,1,0]]),mapping).tolist()==[1,0,0]
perm=np.array([2,0,3,1]);bp=[dict(row,candidates=[row['candidates'][j] for j in perm])];xp,bpbase,mp,sp,_=build(bp,raw[:,perm],base[:,perm],None,'groups')
by_answer={meta['answers'][0][g[0]]:f for f,(i,g) in zip(x,mapping)}
from features import metadata
for f,(i,g) in zip(xp,mp):assert np.allclose(f,by_answer[metadata(bp)['answers'][0][g[0]]])
s=candidate_scores(b,mapping,base,bank);ps=candidate_scores(bpbase,mp,base[:,perm],bp);assert row['candidates'][int(s.argmax(1)[0])]['text']==bp[0]['candidates'][int(ps.argmax(1)[0])]['text']
# Unsorted fitting IDs must not scramble LightGBM question group boundaries.
xx=np.zeros((9,2));bb=np.zeros(9);yy=np.zeros(9);mm=[(0,[0]),(0,[1]),(1,[0]),(1,[1]),(1,[2]),(2,[0]),(2,[1]),(2,[2]),(2,[3])]
_,_,_,gg=subset(xx,bb,yy,mm,np.array([2,3,4]),np.array([2,0]));assert gg.tolist()==[2,4]
z,eq,valid=arrays(bank,raw,base,False);params=np.zeros(2*z.shape[2])
for pooling in ('probability','logit'):
 for grouping in ('sum','mean'):
  values=fuse(z,eq,valid,params,pooling,grouping);assert np.allclose((np.exp(values)/np.maximum(eq.sum(2),1)).sum(1),1)
for name,record in json.loads((ROOT/'experiments/round4/asset_manifest.json').read_text()).items():assert hashlib.sha256((ROOT/'experiments/round4/selected_bundle'/name).read_bytes()).hexdigest()==record['sha256']
try:forbid_test('open',(str(CODE/'data/gsm8k_test.jsonl'),))
except PermissionError:pass
else:raise AssertionError('test guard missing')
print('PASS: group equivalence/permutation, stable representative, unsorted fold boundaries, probability normalization, original assets unchanged, test guard (no test open).')
