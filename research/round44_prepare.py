from data_access import ROOT,CODE,read_jsonl,parse_answer
from round22_common import load,read,dump,sha
from round44_programs import *
import json,datetime,time

OUT=ROOT/'experiments/round44'
MANUAL={
3307:'n1/n0+n1+n1/n3',23:'n1*(1+n0/100)',
1905:'(n0-n0*n1-n0*n2)*n3',3259:'(n2-n4-n3*n0)/n1',
670:'(n0*(1+n1/100)+n2)*(1+n3/100)',
1202:'(n2/100*n3)/(n0/100*n1)',3635:'(n0-n2*n3-n4*n5)/(n1-n2-n4)',2172:'(n2+n1)/(1-n0)'}

def op(k,a,b):return canonical((k,a,b))
def const(v):return ('L',str(Fraction(v)))
def plus(ps):
    p=const(0)
    for x in ps:p=op('ADD',p,x)
    return p

def fact_expr(f,qs):
    e=f.get('evidence');v=Fraction(f['value'])
    if not e:return const(v)
    inside=[i for i,q in enumerate(qs) if e['start']<=q['start']<e['end']]
    exact=[i for i in inside if Fraction(qs[i]['value'])==v]
    if f.get('quantity_multiplier',1)!=1:
        raw=[i for i in inside if qs[i]['raw'].lower()!='dozen']
        if len(raw)==1:return op('MUL',('N',raw[0]),const(f['quantity_multiplier']))
    if exact:
        # Each-price evidence includes the preceding count. Price occupies the
        # last numeric occurrence; count occupies the first. Preserve identity.
        return ('N',exact[-1] if f['kind']=='unit_price' else exact[0])
    if f.get('rule')=='banknotes' and len(inside)==2:return op('MUL',('N',inside[0]),('N',inside[1]))
    if f.get('rule')=='half_previous_item_count':
        e=f['antecedent_evidence'];before=[i for i,q in enumerate(qs) if e['start']<=q['start']<e['end']]
        if len(before)==1 and len(inside)==1:return op('MUL',('N',before[0]),('N',inside[0]))
    if f['kind']=='per_relation' and len(inside)==2:
        p=op('DIV',('N',inside[0]),('N',inside[1]))
        if value(p,qs)==v:return p
    if f['kind']=='target_period':return const(1)
    raise ValueError('ambiguous supervised fact binding')

def money_program(r):
    qs=quantities(r['question']);fs=r['relation_labels'];target=r['target'];rows={};whole={}
    allscopes={f['scope'] for f in fs if f['kind'] in ('item_count','unit_price') and f['scope']}
    if target.get('scope') and target['kind']!='average_unit_price':allscopes.add(target['scope'])
    keys=[s for s in allscopes if not any(set(s.split())<set(t.split()) for t in allscopes)]
    def key(s):
        ks=[k for k in keys if set(s.split())<=set(k.split())]
        if len(ks)!=1:raise ValueError('scope not unique')
        return ks[0]
    for f in fs:
        p=fact_expr(f,qs)
        if f['kind'] in ('item_count','unit_price'):rows.setdefault(key(f['scope']),{})[f['kind']]=p
        else:whole[f['kind']]=p
    for k in keys:rows.setdefault(k,{})
    missing=[k for k in keys if 'item_count' not in rows[k]]
    if 'total_item_count' in whole and len(missing)==1:
        rows[missing[0]]['item_count']=op('SUB',whole['total_item_count'],plus([v['item_count'] for v in rows.values() if 'item_count' in v]))
    spent=whole.get('total_spent_money')
    if spent is None and 'available_money' in whole and 'change_money' in whole:spent=op('SUB',whole['available_money'],whole['change_money'])
    known=plus([op('MUL',v['item_count'],v['unit_price']) for v in rows.values() if 'item_count' in v and 'unit_price' in v]);kind=target['kind']
    if kind=='total_spent_money':p=known
    elif kind=='change_money':p=op('SUB',whole['available_money'],known)
    elif kind=='shortfall_money':p=op('SUB',known,whole['available_money'])
    elif kind=='average_unit_price':p=op('DIV',known,plus([v['item_count'] for v in rows.values()]))
    elif kind=='item_total_money':p=op('SUB',spent,known)
    else:
        slot='unit_price' if kind=='item_count' else 'item_count';p=op('DIV',op('SUB',spent,known),rows[key(target['scope'])][slot])
    return p

def chain_program(r):
    q=r['question'];qs=quantities(q);proof=r['labels'];edges=[];bases=[]
    for f in proof['facts']:
        if f['kind'] in ('per_relation','unit_conversion'):
            if f.get('rule')=='explicit_activity_period':
                source=next(x for x in proof['facts'] if x['kind']=='activity_count_per_period' and x['unit']==f['child']);p=fact_expr(source,qs)
            else:p=fact_expr(f,qs)
            edges.append((f['parent'],f['child'],p))
        elif f['kind'] in ('total_count','target_period'):bases.append((f['unit'],fact_expr(f,qs)))
    target=proof['equations'][0]['target'];out=[]
    for eq in proof['equations']:
        source=[p for u,p in bases if u==eq['source']]
        if len(source)!=1:raise ValueError('base mismatch')
        p=source[0]
        for i in eq['rate_edges']:p=op('MUL',p,edges[i][2])
        out.append(p)
    return plus(out)

def supervision(bank,ids):
    reviewed={};failures=[];annotations=read(ROOT/'experiments/round38/root_cause_annotations.json')['records']
    byid={int(i):r for i,r in zip(ids,bank)}
    for number,func in [(40,money_program),(42,chain_program)]:
        for r in read(ROOT/f'experiments/round{number}/relation_supervision.json'):
            i=r['train_index'];assert r['question']==byid[i]['question']
            try:
                p=func(r);answer=r['answer'] if number==40 else r['labels']['answer'];assert str(value(p,quantities(r['question'])))==answer
                reviewed[i]=dict(train_index=i,question=r['question'],program=p,answer=answer,provenance=f'reviewed round{number} source facts, explicit occurrence bindings')
            except (ValueError,KeyError,StopIteration,AssertionError) as e:failures.append(dict(train_index=i,source=number,reason=str(e)))
    for i,e in MANUAL.items():
        a=next(r for r in annotations if r['train_index']==i);q=byid[i]['question'];p=expression(e)
        from features import safe_arithmetic
        assert float(value(p,quantities(q)))==safe_arithmetic(a['target_expression'])
        reviewed[i]=dict(train_index=i,question=q,program=p,answer=str(value(p,quantities(q))),provenance='round38 reviewed source relation; manually explicit quantity-occurrence bindings',root_cause=a['root_cause'])
    pairs=[]
    for i,r in sorted(reviewed.items()):
        qs=quantities(r['question']);group=defaultdict(list)
        for action,p,d in edits(r['program'],qs):
            # The positive relation structure is adjudicated. A generated
            # negative must additionally fail the TRAIN target execution.
            a=valid(p,qs)
            if a is not None and a!=r['answer']:group[action].append((p,d))
        for action,ps in group.items():
            seen=set()
            for p,d in ps:
                key=repr(canonical(p))
                if key in seen:continue
                seen.add(key);pairs.append(dict(train_index=i,question=r['question'],positive=r['program'],negative=p,action=action,decision=d,
                    annotation='Positive source relations adjudicated; one-step negative also execution-checked against TRAIN target. Not independent human gold.'))
                if len(seen)>=4:break
    dump(OUT/'reviewed_programs.json',list(reviewed.values()));dump(OUT/'constraint_pairs.json',pairs);dump(OUT/'unresolved_annotation_bindings.json',failures)
    return pairs

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists()
    dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        primary='A finite learned relation-edit policy: source-object binding, fraction base, time/event ordering, operator direction. Fraction executor, then select an original candidate.',
        proposal_limit=LIMIT,feature_dimensions=DIM,policy='Hashed question/source-context/AST relations; 64-hidden-unit scratch MLP. No candidate embeddings or text generation.',
        arms=['sft','reward','reward_relations'],warm_epochs=10,continuation_epochs=20,seeds=[44044,44045],
        reward='1 iff executed program equals official TRAIN final answer and is selectable among original candidates; otherwise0. Exact expectation over bounded hypotheses, not a Monte Carlo estimate.',
        constraint='Reviewed positive executable relation programs vs execution-invalid single relation changes; auxiliary pair logistic loss weight .2. Unknown annotation bindings are skipped and recorded, never fabricated.',
        kl=.02,learning_rate=.0003,weight_decay=.01,auxiliary_weight=.2,
        diagnostic='Same grouped fold2 as DPO; vocabulary-free hashing; auxiliary parents in held fold excluded. All derived examples remain with parent.',
        final_training_n=4743,validation_n=798,test_opened=False,
        deployments=['top executed hypothesis with baseline fallback','fixed .25 normalized answer-mass residual on round25; no alpha sweep'],
        selection='Full validation, correctness then fewer harms then smaller assets; old control retained on ties.',
        limitations='Bounded candidate-derived program proposals, not unrestricted synthesis. Numeric reward alone does not prove semantics. Historical TRAIN/schema exploration means held tests are not pristine.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research/round44_programs.py',ROOT/'research/round44_prepare.py']}))
    b,a,y,ids=load('train');dump(OUT/'training_ids.json',ids.tolist());start=time.perf_counter();packets=[]
    for k,r in enumerate(b):
        packets.append(proposals(r))
        if (k+1)%500==0:print(dict(prepared=k+1,seconds=round(time.perf_counter()-start,1)),flush=True)
    (OUT/'train_proposals.jsonl').write_text(''.join(json.dumps(dict(train_index=int(i),**p))+'\n' for i,p in zip(ids,packets)),encoding='utf-8')
    sparse.save_npz(OUT/'train_features.npz',matrix(b,packets))
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');rewards=np.zeros((len(b),LIMIT),np.float32);mask=np.zeros_like(rewards,dtype=bool)
    matches=np.zeros((len(b),LIMIT,4),bool);original_oracle=0
    for k,(i,p) in enumerate(zip(ids,packets)):
        gold=parse_answer(refs[int(i)]['answer'])
        for j,h in enumerate(p['hypotheses']):
            mask[k,j]=True;matches[k,j,h['matches']]=True;rewards[k,j]=h['answer']==gold and bool(h['matches'])
        original_oracle+=any(h['action']=='candidate_program' and h['answer']==gold and h['matches'] for h in p['hypotheses'])
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:folds=z['folds']
    np.savez(OUT/'train_targets.npz',rewards=rewards,mask=mask,matches=matches,folds=folds,ids=ids,labels=y)
    pairs=supervision(b,ids);px=[];nx=[];pairids=[]
    for p in pairs:
        qs=quantities(p['question']);px.append(features(p['question'],qs,p['positive']));nx.append(features(p['question'],qs,p['negative']));pairids.append(p['train_index'])
    def csr(ds):
        rr=[];cc=[];vv=[]
        for i,d in enumerate(ds):
            for j,v in d.items():rr.append(i);cc.append(j);vv.append(v)
        return sparse.csr_matrix((np.asarray(vv,np.float32),(rr,cc)),shape=(len(ds),DIM))
    sparse.save_npz(OUT/'constraint_positive.npz',csr(px));sparse.save_npz(OUT/'constraint_negative.npz',csr(nx));np.save(OUT/'constraint_parent_ids.npy',pairids)
    report=dict(training_n=len(b),has_proposal=int(mask.any(1).sum()),original_program_oracle=original_oracle,expanded_program_oracle=int(rewards.any(1).sum()),
        reviewed_questions=len(read(OUT/'reviewed_programs.json')),constraint_pairs=len(pairs),action_counts=dict(Counter(p['action'] for p in pairs)),
        unresolved_reviewed_bindings=read(OUT/'unresolved_annotation_bindings.json'),seconds=time.perf_counter()-start,test_opened=False)
    dump(OUT/'preparation_result.json',report);print(report,flush=True)

if __name__=='__main__':main()
