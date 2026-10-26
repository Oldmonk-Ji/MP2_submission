"""Train-only supervision checks; no test-bank access."""
from data_access import ROOT
from round22_common import read,dump,sha
from round29_programs import value,parse,vocabulary
from round32_graph import labels,OPERATORS
from round33_text import question_tokens
from round14_prepare import family
from fractions import Fraction
import numpy as np
import json


def main():
    rows=[json.loads(s) for s in open(ROOT/'experiments/round31/training_programs.jsonl',encoding='utf-8')]
    ids=[r['train_index'] for r in rows];assert ids==read(ROOT/'experiments/round22/plan.json')['training_ids']
    count=0
    for r in rows:
        ref=r['reference']
        if ref['clean']:assert value(ref['program'],ref['quantities'])==Fraction(ref['answer'])
        for negative,mask in zip(r['contrast_negatives'],r['contrast_mask']):
            if mask:
                assert value(negative['program'],ref['quantities'])!=Fraction(ref['answer']);count+=1
            else:assert negative['program']==ref['program']
    assert count==7463
    with np.load(ROOT/'experiments/round29/diagnostic_split.npz') as z:fit=z['fit'];held=z['held']
    groups=[family(r['question']) for r in rows]
    assert not {groups[i] for i in fit}&{groups[i] for i in held}
    assert set(fit)|set(held)==set(range(4743))
    fitv=vocabulary([rows[i] for i in fit]);fullv=vocabulary(rows)
    for stage,expected in [('diagnostic',fitv),('full',fullv)]:
        for variant in ['reference_only','reference_and_candidate_ranking']:
            assert read(ROOT/f'experiments/round29/{stage}/{variant}/vocabulary.json')==expected
    item=dict(informative=True,clean=True,quantities=[dict(value='5'),dict(value='3')],program=['SUB',['N',0],['N',1]])
    role,pair=labels(item,'');assert role[:2].tolist()==[1,2] and pair[0,1]==OPERATORS.index('SUB')
    item['program']=['DIV',['N',1],['N',0]];role,pair=labels(item,'');assert role[:2].tolist()==[2,1] and pair[0,1]==OPERATORS.index('RDIV')
    item['informative']=False;role,pair=labels(item,'');assert (role==-100).all() and (pair==-100).all()
    duplicate=parse('Liam has 4 apples and Maya has 4 apples. How many apples altogether?','4 + 4 = 8\n#### 8')
    assert len(duplicate['quantities'])==2 and duplicate['ambiguous_leaves']>0 and not duplicate['clean']
    qa='Liam has 5 apples. Maya has 3 apples. How many apples does Liam have left after giving 2 to Maya?'
    qb=qa.replace('Liam','Zariel').replace('Maya','Qevina')
    assert question_tokens(qa)==question_tokens(qb)
    historical={}
    for number in range(22,29):
        p=ROOT/f'experiments/round{number}/completion_manifest.json'
        assert p.exists(),p
        manifest=read(p);assert all(sha(ROOT/k)==v for k,v in manifest['hashes'].items());historical[str(number)]=len(manifest['hashes'])
    report=dict(training_n=4743,verified_wrong_programs=count,reference_clean_count=sum(r['reference']['clean'] for r in rows),
        caveat='Clean is an extraction heuristic, not an independently certified semantic label; arithmetic execution alone does not prove wording entailment.',
        diagnostic_group_disjoint=True,diagnostic_fit=len(fit),diagnostic_held=len(held),diagnostic_vocab_fit_only=True,
        duplicate_quantity_ambiguity_preserved=True,entity_renaming_invariant=True,operation_direction_and_role_fixtures=True,
        unknown_not_contradiction=True,historical_manifests_unchanged=historical,test_opened=False)
    dump(ROOT/'experiments/semantic_supervision_audit_20261016.json',report);print(report)


if __name__=='__main__':main()
