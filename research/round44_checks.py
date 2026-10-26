"""Pre-training executor checks and reviewed equal-number semantic contrasts."""
from data_access import ROOT
from round22_common import read,dump,sha
import round44_prepare as prep
from round44_programs import *
from round44_policy import loss,decisions
from scipy import sparse
import torch,datetime

OUT=ROOT/'experiments/round44'

def main():
    assert not (OUT/'training_freeze.json').exists()
    # Resolve the three recorded ambiguities only when the reviewed source
    # explicitly supplies an indefinite singleton. No numeric reference guess.
    old=prep.fact_expr
    def explicit(f,qs):
        if f['kind']=='item_count' and f['value']=='1' and f.get('rule') in ('indefinite_singleton_purchase','explicit_indefinite_purchase'):
            assert re.search(r'\ba\s+',f['evidence']['text']);return ('L','1')
        return old(f,qs)
    prep.fact_expr=explicit
    programs=read(OUT/'reviewed_programs.json');pairs=read(OUT/'constraint_pairs.json');resolved=[]
    for unresolved in read(OUT/'unresolved_annotation_bindings.json'):
        r=next(x for x in read(ROOT/'experiments/round40/relation_supervision.json') if x['train_index']==unresolved['train_index'])
        p=prep.money_program(r);qs=quantities(r['question']);assert str(value(p,qs))==r['answer']
        programs.append(dict(train_index=r['train_index'],question=r['question'],program=p,answer=r['answer'],provenance='reviewed round40 facts; explicit indefinite article bound to constant1'))
        resolved.append(r['train_index']);seen=defaultdict(set)
        for action,n,d in edits(p,qs):
            a=valid(n,qs);key=repr(canonical(n))
            if a is not None and a!=r['answer'] and key not in seen[action] and len(seen[action])<4:
                seen[action].add(key);pairs.append(dict(train_index=r['train_index'],question=r['question'],positive=p,negative=n,action=action,decision=d,annotation='Reviewed positive relation, execution-invalid one-step negative.'))
    # These negatives receive the SAME final-number reward as the positive.
    # They are negative ONLY for the separately supervised relation objective.
    examples={670:'(n0*(1+n2/100)+n1)*(1+n3/100)',3635:'(n0-n2*n3-n4*n5)/(n1-n2-n5)'}
    for i,e in examples.items():
        r=next(x for x in programs if x['train_index']==i);negative=expression(e);qs=quantities(r['question']);positive=tree(r['program'])
        assert value(positive,qs)==value(negative,qs)
        if i==670:qs[1]['value']='10'
        else:qs[5]['value']='3'
        assert value(positive,qs)!=value(negative,qs)
        pairs.append(dict(train_index=i,question=r['question'],positive=positive,negative=negative,action='object_binding',decision=dict(same_numeric_answer=True),
            annotation='Reviewed roles conflict despite equal source numbers: interest percentage vs purchase dollars, or notebook count vs dollars/notebook. Candidate final-answer label is NOT changed. Counterfactual source-value separation confirms distinct binding.'))
    dump(OUT/'reviewed_programs.json',programs);dump(OUT/'constraint_pairs.json',pairs)
    def csr(side):
        ii=[];jj=[];vv=[]
        for i,p in enumerate(pairs):
            for j,v in features(p['question'],quantities(p['question']),p[side]).items():ii.append(i);jj.append(j);vv.append(v)
        return sparse.csr_matrix((np.asarray(vv,np.float32),(ii,jj)),shape=(len(pairs),DIM))
    sparse.save_npz(OUT/'constraint_positive.npz',csr('positive'));sparse.save_npz(OUT/'constraint_negative.npz',csr('negative'));np.save(OUT/'constraint_parent_ids.npy',[p['train_index'] for p in pairs])
    s=torch.tensor([[.2,-.4]],requires_grad=True);r=torch.tensor([[1.,0.]]);mask=torch.ones_like(s,dtype=torch.bool)
    l=loss(s,r,mask,s.detach(),'reward');l.backward();assert s.grad[0,0]<0 and s.grad[0,1]>0
    assert valid(('DIV',('N',0),('L','0')),[dict(value='2')]) is None
    q='Lena has 8 red beads and 2 blue beads. How many beads does she have?'
    texts=['8 + 2 = 10\n#### 10','8 - 2 = 6\n#### 6','8 * 2 = 16\n#### 16','8 / 2 = 4\n#### 4']
    row=dict(question=q,candidates=[dict(text=t) for t in texts]);a=proposals(row);assert a['hypotheses']
    row['candidates']=list(reversed(row['candidates']));b=proposals(row)
    assert [(x['program'],x['answer']) for x in a['hypotheses']]==[(x['program'],x['answer']) for x in b['hypotheses']]
    for p in pairs:
        qs=quantities(p['question']);assert np.isfinite(list(features(p['question'],qs,p['positive']).values())).all()
    empty_mask=np.zeros((1,LIMIT),bool);empty_match=np.zeros((1,LIMIT,4),bool);base=np.asarray([2]);bs=np.asarray([[0.,1.,2.,3.]])
    hard,res,p=decisions(np.zeros((1,LIMIT)),empty_mask,empty_match,base,bs);assert hard.tolist()==[2] and np.all(res==0)
    dump(OUT/'executor_checks.json',dict(exact_reward_gradient=True,division_by_zero_rejected=True,proposal_candidate_permutation_invariant=True,
        unknown_program_retains_baseline=True,finite_features=True,equal_numeric_but_wrong_binding_contrasts=2,
        recovered_explicit_singletons=resolved,reviewed_questions=len(programs),relation_pairs=len(pairs),actions=dict(Counter(p['action'] for p in pairs))))
    files=[ROOT/'research'/f for f in ('round44_programs.py','round44_prepare.py','round44_policy.py','round44_train.py','round44_checks.py')]
    files += [OUT/f for f in ('reviewed_programs.json','constraint_pairs.json','constraint_positive.npz','constraint_negative.npz','constraint_parent_ids.npy','executor_checks.json','plan.json')]
    dump(OUT/'training_freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        amendments_before_policy_training='Resolved explicit indefinite singleton counts; added two reviewed same-answer wrong-binding contrasts. No policy held/validation scores used.',
        hashes={str(p.relative_to(ROOT)):sha(p) for p in files},test_opened=False))
    print(read(OUT/'executor_checks.json'),flush=True)

if __name__=='__main__':main()
