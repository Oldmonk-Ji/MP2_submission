"""Conservative source-span event extractor; unresolved references stay unknown."""
import re
from functools import lru_cache
NAME=r'[A-Z][a-z]+'
NUMBER=r'(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)'
VALUES=dict(zip('one two three four five six seven eight nine ten'.split(),map(str,range(1,11))))
PATTERNS=[
 ('money_gift',re.compile(rf'\b(?P<receiver>{NAME}) received \$(?P<amount>\d+(?:\.\d+)?) as a (?P<entity>gift)\b')),
 ('give_to',re.compile(rf'\b(?P<giver>{NAME}) (?:gave|gives) (?P<amount>{NUMBER}) (?P<entity>[a-z]+(?: of [a-z]+)?) to (?P<receiver>{NAME})\b')),
 ('give_person',re.compile(rf'\b(?P<giver>{NAME}) (?:gave|gives) (?P<receiver>{NAME}) (?P<amount>{NUMBER}) (?P<entity>[a-z]+(?: of [a-z]+)?)\b')),
 ('receive',re.compile(rf'\b(?P<receiver>{NAME}) (?:received|receives) (?P<amount>\$?{NUMBER}) (?P<entity>[a-z]+(?: of [a-z]+)?)\b'))]
EXCLUDED={'He','She','They','It','Each','Every','If','When','After','Before','His','Her','The','A','An'}
BAD=set('more less fewer times percent half quarter third first second third rest change discount raise bonus salary paycheck wage wages fee fees'.split())

@lru_cache(maxsize=12000)
def extract(q):
    events=[]
    for rule,pattern in PATTERNS:
        for m in pattern.finditer(q):
            d=m.groupdict();giver=d.get('giver');receiver=d['receiver'];entity=d['entity'];amount=d['amount']
            if re.search(r'\b[A-Z][a-z]+\s+$',q[max(0,m.start()-40):m.start()]):continue
            if rule=='money_gift':entity='dollars'
            if giver in EXCLUDED or receiver in EXCLUDED or set(entity.split())&BAD:continue
            tail=q[m.end():m.end()+60]
            # Reject rates, implicit repeated quantities, partial noun phrases and currency idioms.
            if re.match(r'\s+(?:each|per|every|a|an|more|less|fewer|times|of|in|for|and|to)\b',tail):continue
            if rule!='money_gift' and not re.match(r'\s*(?:[.,;?!]|$)',tail):continue
            if amount.startswith('$'):continue
            before=q[max(q.rfind('.',0,m.start()),q.rfind('?',0,m.start()))+1:m.start()]
            if re.search(r'\b(?:if|would|could|every|each)\b',before,re.I):continue
            events.append(dict(rule=rule,giver=giver,receiver=receiver,entity=entity,
                amount=VALUES.get(amount,amount),source_span=list(m.span()),evidence=m.group(),
                giver_delta='negative' if giver else 'unknown',receiver_delta='positive',
                time_before='event_before',time_after='event_after',
                status='partial_event_hypothesis',unknown=['absolute_balances','calendar_time']+(['giver'] if giver is None else [])))
    return sorted(events,key=lambda e:e['source_span'])

if __name__=='__main__':
    from round22_common import ROOT,load,dump,sha
    import numpy as np,sys
    out=ROOT/'experiments/round58';out.mkdir(exist_ok=True)
    mode=sys.argv[1];assert mode in ('fit','held')
    if mode=='held':
        freeze=__import__('json').loads((out/'rule_freeze.json').read_text(encoding='utf-8'))
        assert sha(__file__)==freeze['extractor_sha256']
    b,_,_,ids=load('train')
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:folds=z['folds'];assert np.array_equal(ids,z['ids'])
    records=[];den=0;cue=0
    for r,i,f in zip(b,ids,folds):
        if (f==2)!=(mode=='held'):continue
        den+=1;cue+=bool(re.search(r'\b(gave|gives|received|receives)\b',r['question']))
        events=extract(r['question'])
        if events:records.append(dict(train_index=int(i),fold=int(f),question=r['question'],events=events))
    dump(out/f'{mode}_extractions.json',records)
    dump(out/f'{mode}_coverage.json',dict(n=den,lexical_cue_parents=cue,covered_parents=len(records),events=sum(len(r['events']) for r in records),test_opened=False))
    for r in records:print(r['train_index'],r['question'],r['events'])
