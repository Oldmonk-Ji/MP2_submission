"""Source-grounded, partial purchase ledger. No IDs, references or learned answer table.

Amounts, item counts and prices occupy different typed slots. A complete proof
is required; unsupported amounts/scopes cause an internal unknown and the
selector retains its baseline answer. This is not a general word-problem solver.
"""
import re
from fractions import Fraction
from round4_quantities import TOKEN,WORDS

NUM=r'(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)'
MONEY=re.compile(r'\$\s*(?P<a>'+NUM+r')|(?P<b>'+NUM+r')\s+dollars?\b',re.I)
STOP=set('a an the each every some any of those these other ones one all total same price at for in with their his her its and she he it they them i if then also to on either both much many'.split())
BAD_SCOPE=set('bought buys buy purchased purchases spent spend paid pay cost costs costing dollar dollars change money total sale saving saves enough remaining left have has had is are was were been get gets can should could would will make makes gives gave received receive wanted wants want last time day days week weeks year years hour hours month months'.split())
UNSUPPORTED=re.compile(r'%|\b(?:percent|discount|tip|tax|ratio|profit|sell|sells|sold|resell|interest|increased|decreased|last time|per day|per week|per month|per year|salary|earns|earned|hour|hours)\b',re.I)


def number(raw):
    raw=raw.lower().strip()
    return Fraction(str(WORDS[raw])) if raw in WORDS else Fraction(raw.replace(',',''))


def singular(word):
    special={'glasses':'glass','clothes':'clothes','pants':'pants','shoes':'shoe','feet':'foot','children':'child','movies':'movie','cookies':'cookie','paintbrushes':'paintbrush'}
    if word in special:return special[word]
    if word.endswith('ies') and len(word)>4:return word[:-3]+'y'
    if word.endswith('s') and not word.endswith(('ss','us')) and len(word)>3:return word[:-1]
    return word


def scope(text):
    ws=[singular(w) for w in re.findall(r'[a-z]+',text.lower()) if w not in STOP]
    if not ws or any(w in BAD_SCOPE for w in ws) or len(ws)>5:return None
    return ' '.join(ws)


def phrase_after(text):
    return re.split(r'[,.;?!]|\b(?:at|for|and|that|which|who|where|when|with|but|are|is|costs?|has|have|leaves|left|all|as|will|were)\b',text,1,flags=re.I)[0].strip()


def compile(question):
    text=question;low=text.lower();facts=[];problems=[];covered=[]
    result=dict(status='unknown',facts=facts,issues=problems,target=None,answer=None,equations=[])
    if not re.search(r'\b(?:buy|buys|bought|purchase|purchased|purchases|shopping)\b',low):problems.append('outside_purchase_domain');return result
    if UNSUPPORTED.search(text):problems.append('unsupported_scope_or_operation');return result
    qm=list(re.finditer(r'\bhow (?:many|much)\b|\bwhat is\b',text,re.I))
    if not qm:problems.append('unknown_target');return result
    target_start=qm[-1].start();target_text=text[target_start:];body=text[:target_start]
    if re.search(r'\b(?:than|instead|difference|less|cheaper|expensive|twice|times|percent|ratio)\b',target_text,re.I):problems.append('unsupported_target_relation');return result
    if re.search(r'\bmore\b',target_text,re.I) and not re.search(r'\bmore money\b',target_text,re.I):problems.append('unsupported_target_relation');return result
    if MONEY.search(target_text) or TOKEN.search(target_text):problems.append('quantity_inside_target_clause');return result
    money=list(MONEY.finditer(body));count_mentions=[];prices=[];totals=[];budget=[];change=[];total_counts=[]
    def fact(kind,value,item,start,end,rule):
        f=dict(kind=kind,value=str(value),scope=item,evidence=dict(start=start,end=end,text=text[start:end]),rule=rule)
        facts.append(f);return f
    def add_count(value,item,start,end,rule):
        f=fact('item_count',value,item,start,end,rule);count_mentions.append(f);return f
    # Preserve occurrence identity; identical numbers never establish scope.
    for m in TOKEN.finditer(body):
        if any(x.start()<=m.start()<x.end() for x in money):continue
        raw=m.group();right=body[m.end():];left=body[max(0,m.start()-65):m.start()]
        if raw.lower() in ('first','second','third','fourth','fifth'):problems.append('ordinal_scope');continue
        if re.match(r'\s*\$',right):continue # banknote multiplicity handled with its cash amount
        if raw.lower()=='half' and re.match(r'\s+as many\s+',right,re.I) and count_mentions and re.fullmatch(r'\s*and\s*',body[count_mentions[-1]['evidence']['end']:m.start()],re.I):
            match=re.match(r'\s+as many\s+([^.;,?!]+)',right,re.I);item=scope(phrase_after(match.group(1))) if match else None
            if item:
                source=count_mentions[-1];f=add_count(number(source['value'])/2,item,m.start(),m.end()+match.end(),'half_previous_item_count')
                f['antecedent_evidence']=source['evidence'];covered.append((m.start(),m.end()));continue
        phrase=phrase_after(right);item=scope(phrase)
        if not item:continue
        end=m.end()+right.find(phrase)+len(phrase);v=number(raw)
        if re.search(r'\btotal(?: number)?(?: of)?\s*$',left,re.I):
            total_counts.append(fact('total_item_count',v,item,m.start(),end,'explicit_total_count'))
        else:add_count(v,item,m.start(),end,'number_item_phrase')
        covered.append((m.start(),m.end()))
    # Money roles are distinct. Never infer unit price solely from "$".
    for m in money:
        v=number(m.group('a') or m.group('b'));left=body[:m.start()];right=body[m.end():]
        clause=re.split(r'[.!?;]|\band\b',left,flags=re.I)[-1];short=clause[-100:]
        start=m.start();end=m.end();item=None
        if re.match(r'\s*bills?\b',right,re.I):
            pm=list(TOKEN.finditer(left));prev=pm[-1] if pm else None
            if prev and not left[prev.end():].strip() and re.search(r'\b(?:paid|gave|pay|cashier)\b',clause,re.I):
                budget.append(fact('available_money',number(prev.group())*v,None,prev.start(),m.end()+re.match(r'\s*bills?\b',right,re.I).end(),'banknotes'))
                covered.append((prev.start(),prev.end()));covered.append((m.start(),m.end()));continue
        residual=re.match(r'\s+(?:(?:in\s+)?change|remaining|left)\b',right,re.I)
        if residual:
            change.append(fact('change_money',v,None,start,m.end()+residual.end(),'explicit_change'));covered.append((m.start(),m.end()));continue
        per=re.match(r'\s*(?:per|a|an)\s+([a-z]+(?:\s+of\s+[a-z]+)?)',right,re.I)
        each=re.match(r'\s*each\b',right,re.I)
        if per:
            item=scope(per.group(1));end=m.end()+per.end();rule='explicit_per_price'
        elif each:
            # Prefer subject attached to cost, otherwise the nearest quantified
            # object in the SAME local clause (not an arbitrary equal number).
            subject=re.search(r'(?:^|\bthe\s+)([a-z][a-z ]{0,45}?)\s+costs?\s*$',short,re.I)
            if subject:item=scope(subject.group(1));start=len(left)-len(short)+subject.start(1)
            if not item:
                earlier=[f for f in count_mentions if f['evidence']['start']<m.start() and m.start()-f['evidence']['end']<45
                         and not re.search(r'[.;?!]|\band\b',body[f['evidence']['end']:m.start()],re.I)]
                if earlier:
                    f=max(earlier,key=lambda f:f['evidence']['start']);item=f['scope'];start=f['evidence']['start']
            end=m.end()+each.end();rule='explicit_each_price'
        else:
            subject=re.search(r'\beach\s+([a-z][a-z ]{0,40}?)\s+(?:costs?|at|is)\s*$',short,re.I)
            if subject:item=scope(subject.group(1));start=len(left)-len(short)+subject.start();rule='each_subject_price'
            else:
                after=re.match(r'\s+for\s+((?:small|large|medium|red|blue|green)\s+[a-z]+)',right,re.I)
                if after and re.search(r'\bcost\b',body[max(0,m.start()-100):m.start()],re.I):
                    item=scope(after.group(1));end=m.end()+after.end();rule='contrasted_object_prices'
        if item:
            prices.append(fact('unit_price',v,item,start,end,rule));covered.append((m.start(),m.end()));continue
        if per or each:problems.append('price_scope_unresolved');continue
        if re.search(r'\b(?:spent|spend|paid)\s+(?:a\s+)?(?:total\s+)?(?:of\s+)?$',short,re.I) or re.search(r'\btotal cost of\s*$',short,re.I):
            totals.append(fact('total_spent_money',v,None,max(0,len(left)-len(short)),m.end(),'explicit_total_spending'));covered.append((m.start(),m.end()));continue
        if re.search(r'\b(?:has|had|have|budget of)\s+(?:a total of\s+)?$',short,re.I) or re.search(r'\bgave\s+(?:him|her|them)\s*$',short,re.I):
            budget.append(fact('available_money',v,None,max(0,len(left)-len(short)),m.end(),'explicit_available_money'));covered.append((m.start(),m.end()));continue
        if re.search(r'\bfor\s*$',short,re.I) and re.match(r'\s+(?:she|he|they|[A-Z][a-z]+)\s+bought\b',right):
            totals.append(fact('total_spent_money',v,None,m.start()-4,m.end(),'fronted_total_spending'));covered.append((m.start(),m.end()));continue
        # Explicit singleton purchase: "a pattern for $15", "a box ... cost $5".
        single=re.search(r'\ba\s+([a-z][a-z ]{0,45}?)\s+(?:for|that cost|costs?)\s*$',short,re.I)
        if single:
            item=scope(single.group(1))
            if item:
                ss=len(left)-len(short)+single.start();f=add_count(Fraction(1),item,ss,m.start(),'indefinite_singleton_purchase')
                prices.append(fact('unit_price',v,item,ss,m.end(),'singleton_purchase_price'));covered.append((m.start(),m.end()));continue
        problems.append('unassigned_money_role')
    # A purchased indefinite article supplies an explicit quantity of one even
    # when this item's price is the requested unknown. Do not count intentions.
    for m in re.finditer(r'\b(?:bought|buys|purchased)\s+(.+?)(?:[.;?!]|$)',body,re.I):
        for one in re.finditer(r'(?:^|,\s*|\band\s+)a\s+([a-z][a-z -]*?)(?=\s+(?:for|at|that|which)\b|[,.;?!]|$)',m.group(1),re.I):
            item=scope(one.group(1));ss=m.start(1)+one.start();ee=m.start(1)+one.end()
            if item and not any(f['scope']==item for f in count_mentions):add_count(Fraction(1),item,ss,ee,'explicit_indefinite_purchase')
    # Resolve "small ones" from a unique explicitly named object head.
    scopes={f['scope'] for f in facts if f['scope']};heads={s.split()[-1] for s in scopes if s.split()[-1] not in ('small','large','medium','red','blue','green')}
    if len(heads)==1:
        head=next(iter(heads))
        for f in facts:
            if f['scope'] in ('small','large','medium','red','blue','green'):f['scope']+=' '+head
    def match_scope(a,b):
        if a==b:return True
        aa=set(a.split());bb=set(b.split())
        # More-specific noun phrase may resolve a bare unit, but conflicting
        # modifiers never match (red versus blue, small versus large).
        return aa<=bb or bb<=aa
    all_scopes={f['scope'] for f in count_mentions+prices if f['scope']}
    target_low=target_text.lower();kind=None;target_scope=None
    if re.search(r'\baverage\b',target_low):
        m=re.search(r'\bper\s+([a-z]+)',target_low)
        if m:kind='average_unit_price';target_scope=scope(m.group(1))
        else:problems.append('average_denominator_unknown')
    elif re.search(r'\bmore money\b',target_low):kind='shortfall_money'
    elif re.search(r'\bchange\b|\b(?:left with|money.*get back)\b',target_low):kind='change_money'
    elif re.search(r'^how many\b',target_low):
        m=re.match(r'how many\s+(.+?)\s+(?:did|does|do|can|will|should|has|have|is|are|was|were)\b',target_text,re.I)
        if m:target_scope=scope(m.group(1));kind='item_count'
    elif re.search(r'\beach\b',target_low):
        m=re.search(r'\beach\s+(.+?)\s+(?:cost|is|was|does|did|be|sell|selling)\b',target_text,re.I)
        if m:target_scope=scope(m.group(1));kind='unit_price'
    elif re.search(r'\b(?:spend|spent|cost|pay|paid)\b',target_low):
        if re.search(r'\b(?:on|for)\s+\w+',target_low) and not re.search(r'\b(?:in all|everything|total|entire)\b',target_low):
            m=re.search(r'\bon\s+(?:the\s+)?([a-z][a-z -]+?)[?.]?$',target_low)
            if m:kind='item_total_money';target_scope=scope(m.group(1))
            else:problems.append('target_specific_batch_cost_not_total')
        else:kind='total_spent_money'
    if target_scope in ('small','large','medium','red','blue','green') and len(heads)==1:target_scope+=' '+next(iter(heads))
    if not kind or (kind in ('item_count','unit_price') and not target_scope):problems.append('unknown_target');return result
    result['target']=dict(kind=kind,scope=target_scope,evidence=dict(start=target_start,end=len(text),text=target_text))
    if target_scope and kind!='average_unit_price':all_scopes.add(target_scope)
    # Canonical slots are the most specific observed item phrases.
    keys=[s for s in all_scopes if not any(s!=t and set(s.split())<set(t.split()) for t in all_scopes)]
    table={k:{} for k in keys}
    for f in count_mentions+prices:
        matches=[k for k in keys if match_scope(f['scope'],k)]
        if len(matches)!=1:problems.append('ambiguous_item_scope');continue
        key=matches[0];slot=f['kind'];v=Fraction(f['value'])
        if slot=='item_count' and slot in table[key]:problems.append('repeated_item_scope_requires_event_binding')
        if slot in table[key] and table[key][slot]!=v:problems.append('conflicting_item_facts')
        table[key][slot]=v
    def single_value(fs,name):
        values={Fraction(f['value']) for f in fs}
        if len(values)>1:problems.append('conflicting_'+name);return None
        return next(iter(values)) if values else None
    spend=single_value(totals,'total_spent');cash=single_value(budget,'available_money');returned=single_value(change,'change')
    total_n=single_value(total_counts,'total_count')
    if spend is None and cash is not None and returned is not None:spend=cash-returned;result['equations'].append('spent=available-change')
    elif spend is None and cash is not None and kind in ('item_count','unit_price'):
        # A budget is not evidence that every cent was spent.
        problems.append('budget_does_not_prove_spending')
    target_keys=[k for k in keys if target_scope and match_scope(target_scope,k)]
    target_key=target_keys[0] if len(target_keys)==1 else None
    if target_scope and not target_key and kind!='average_unit_price':problems.append('ambiguous_target_scope')
    unknown_counts=[k for k,v in table.items() if 'item_count' not in v]
    if total_n is not None and len(unknown_counts)==1:
        if any(not all(match_scope(f['scope'],k) for k in keys) for f in total_counts):problems.append('total_count_scope_mismatch')
        k=unknown_counts[0];table[k]['item_count']=total_n-sum(v.get('item_count',0) for v in table.values());result['equations'].append('remaining_count=total_count-sum(known_counts)')
    # Do not silently omit a source number (duration, quantity, batch or state).
    for m in TOKEN.finditer(body):
        if not any(a<=m.start()<b for a,b in covered):problems.append('unconsumed_quantity');break
    if not table:problems.append('no_line_items')
    if any(v.get('item_count',1)<0 or v.get('unit_price',1)<0 for v in table.values()):problems.append('negative_quantity')
    if problems:return result
    known_cost=sum(v['item_count']*v['unit_price'] for v in table.values() if set(v)=={'item_count','unit_price'})
    incomplete=[k for k,v in table.items() if set(v)!={'item_count','unit_price'}]
    if not incomplete and spend is not None and spend!=known_cost:problems.append('stated_spending_disagrees_with_ledger');return result
    answer=None
    if kind=='total_spent_money' and not incomplete:answer=known_cost;result['equations'].append('spent=sum(item_count*item_unit_price)')
    elif kind=='average_unit_price' and not incomplete and target_scope and all(target_scope in k.split() for k in keys):
        denominator=sum(v['item_count'] for v in table.values())
        if denominator>0:answer=known_cost/denominator;result['equations'].append('average_price=sum(item_costs)/sum(item_counts)')
    elif kind=='change_money' and cash is not None and not incomplete:answer=cash-known_cost;result['equations'].append('change=available-sum(item_count*item_unit_price)')
    elif kind=='shortfall_money' and cash is not None and not incomplete:answer=max(Fraction(0),known_cost-cash);result['equations'].append('shortfall=max(0,sum(item_costs)-cash)')
    elif kind=='item_total_money' and spend is not None and incomplete==[target_key]:answer=spend-known_cost;result['equations'].append('target_item_cost=spent-other_item_costs')
    elif target_key and spend is not None and incomplete==[target_key]:
        row=table[target_key];needed='unit_price' if kind=='item_count' else 'item_count'
        if needed in row and row[needed]>0:
            answer=(spend-known_cost)/row[needed];result['equations'].append(kind+'=(spent-known_item_costs)/target_'+needed)
    if answer is None:problems.append('incomplete_connected_ledger');return result
    if answer<0 or (kind=='item_count' and answer.denominator!=1):problems.append('infeasible_target');return result
    result.update(status='proved_within_grammar',answer=str(answer),items={k:{s:str(v) for s,v in row.items()} for k,row in table.items()})
    return result
