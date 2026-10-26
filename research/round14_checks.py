"""Structural/semantic invariants, not evidence of natural-language accuracy."""
from data_access import forbid_test
from fractions import Fraction as F
from round14_local import equation, status, extract, inspect, target


def main():
    fact=equation('age_fact','a',None,F(10))
    assert status([fact],equation('age_fact','a',None,F(11)))=='contradicted'
    assert status([fact],equation('age_fact','b',None,F(10)))=='insufficient'
    assert status([fact],equation('age_fact','a',None,F(11),F(1)))=='entailed'
    # Partial systems can determine one target without determining everybody.
    r=equation('age_ratio','b','c',F(2))
    assert status([fact,r],fact)=='entailed'
    assert status([r],equation('age_ratio','c','b',F(1,2)))=='entailed'
    assert status([r],equation('age_ratio','c','b',F(2)))=='insufficient'
    assert status([fact,equation('age_fact','a',None,F(11))],fact)=='inconsistent_source'
    # Same-value quantity occurrences with different people are NOT merged.
    q='Alice is 10 years old. Bob is 10 years old. How old is Alice?'
    rels,_=extract(q);assert len(rels)==2
    assert {r['left'] for r in rels}=={'alice','bob'}
    for r in rels:
        for key in ('evidence','claim_span'):
            s=r[key];assert q[s['start']:s['end']]==s['text']
    q='In twenty years, Ramon will be twice as old as Loui today. Loui is currently 23 years old. How old is Ramon now?'
    d=inspect(q,['Ramon is 26 years old.','Ramon is 46 years old.'],['26','46'])
    finals=[c['status'] for c in d['claims'] if c['claim'].get('final_target_claim')]
    assert finals==['entailed','contradicted'],finals
    assert d['source_relations'][0]['left_time']=='20' and d['source_relations'][0]['right_time']=='0'
    # Event time, negation, ambiguous ratio, and unresolved pronouns abstain.
    for s in ['Alice was 20 years old when Bob was 8 years old.', 'Alice is not 20 years old.', 'Alice will be 20 years old.']:
        assert not extract(s)[0],s
    d=inspect('Alice is 5 years old and Bob is 3 times older than Alice. How old is Bob?', ['Bob is 15 years old.'],['15'])
    assert d['ambiguous'] and all(c['status']=='ambiguous' for c in d['claims'])
    assert target('Alice is 5 years old. Bob is 9 years old. How old is she?', ['alice','bob']) is None
    # Candidate permutation changes only candidate IDs, not support decisions.
    q='Alice is 10 years old. How old is Alice?';texts=['Alice is 10 years old.','Alice is 11 years old.']
    x=inspect(q,texts,['10','11']);y=inspect(q,texts[::-1],['11','10'])
    for j in (0,1):assert sorted(c['status'] for c in x['claims'] if c['candidate']==j)==sorted(c['status'] for c in y['claims'] if c['candidate']==1-j)
    # Test protection is checked without opening any test file.
    try:forbid_test('open',('gsm8k_test.jsonl',))
    except PermissionError:pass
    else:raise AssertionError('Missing test guard')
    print('PASS: exact rational constraints, partial/unknown/inconsistent systems, separate entity/time scope, original spans, ambiguity abstention, permutation, test guard.')


if __name__=='__main__':main()
