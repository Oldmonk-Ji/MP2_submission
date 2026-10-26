"""New manual quantity bindings plus explicitly marked old-label alignment.

Codes: C=count, P=unit price/rate in money, T=total price/payment, B=balance,
D=additive change/difference, R=ratio, A=age, H=time amount, O=other.
Stages: N=current/unspecified, P=past, F=future, I=initial, E=final.
Price scope points to the numeric count mention of the same item/batch, or -1
when its denominator has no explicit count mention. This does NOT identify all
units, target variables, or both endpoints of every temporal relation.
"""
from data_access import ROOT,CODE,read_jsonl
import json
from round7_core import quantities

OUT=ROOT/'experiments/round7'
# Each entry is ROLE/STAGE/OWNER[/PRICE_COUNT_INDEX], in source occurrence order.
MANUAL={
4261:'H/P/NIL R/P/bethany H/F/NIL A/F/bethany\'s_sister',
3545:'T/N/percy D/N/percy D/N/percy P/F/percy/-1',
2930:'R/N/omi R/N/arlette A/N/kimiko C/N/NIL',
3888:'H/F/NIL R/F/frank D/N/ty R/N/ty D/N/carla A/N/karen H/F/NIL',
412:'R/N/alexa O/N/jasmine',
4021:'T/N/troy B/I/troy T/F/troy',
4412:'H/N/connor D/N/luke R/N/connor\'s_puppy',
3790:'C/N/NIL R/N/puffy R/N/puffy C/N/NIL C/N/juniper',
822:'P/N/joanie/-1 T/I/joanie H/F/NIL',
730:'C/N/NIL P/N/mitch/0 C/N/jam P/N/jam/2 C/N/jay P/N/jay/4 C/N/NIL',
4001:'C/N/brad P/N/brad/2 C/N/brad P/N/brad/-1 D/N/brad B/E/brad',
1585:'P/N/neil/2 P/N/neil/3 C/N/neil C/N/neil',
3874:'P/N/NIL/3 P/N/NIL/4 C/N/NIL C/N/NIL C/N/NIL C/F/NIL',
2807:'P/N/NIL/2 P/N/NIL/3 C/N/NIL C/N/NIL',
4458:'T/I/mark R/N/mark P/N/mark/-1 T/N/mark',
1343:'C/N/NIL P/N/NIL/0 C/N/NIL',
4117:'C/N/jackson C/N/NIL C/N/NIL P/N/jackson/-1 R/N/jackson',
146:'P/N/NIL/-1 B/N/emmy B/N/gerry',
1197:'R/N/trent P/I/trent/3 P/I/trent/4 C/N/NIL C/N/NIL',
2062:'R/N/axel T/I/axel R/N/axel',
967:'A/N/joel A/N/joel\'s_dad R/F/joel\'s_dad',
989:'R/N/ernie R/N/jack B/P/ernie',
1693:'D/N/erick R/N/erick P/I/erick/4 P/I/erick/5 C/N/erick C/N/erick',
610:'C/N/bruce P/N/bruce/0 C/N/bruce P/N/bruce/2 C/N/bruce P/N/bruce/4 B/I/bruce C/N/bruce P/N/bruce/7',
}
def unpack(s):
    r=[]
    for x in s.split():
        p=x.split('/');r.append(dict(role=p[0],stage=p[1],owner=p[2].replace('_',' '),scope=int(p[3]) if len(p)>3 else -1))
    return r
def manual_rows(mapping):
    refs=read_jsonl(CODE/'data/gsm8k_train.jsonl');result=[]
    for i,s in mapping.items():
        q=refs[i]['question'];qs=quantities(q);labels=unpack(s);assert len(qs)==len(labels),(i,len(qs),len(labels))
        for j,l in enumerate(labels):assert l['scope']==-1 or (0<=l['scope']<len(qs) and l['scope']!=j)
        result.append(dict(index=i,question=q,quantities=qs,labels=labels,source='manual question-only quantity annotation',mask=[True]*len(qs)))
    return result
def old_rows():
    rows=[]
    roles={'age':'A','age_ratio':'R','age_difference':'D','age_sum':'A','count':'C','balance':'B','change':'D','unit_price':'P','total_price':'T','payment':'T','expense':'T','earn_rate':'P','work_rate':'C','discount':'R','increase':'R','target_cost':'T','duration':'H','threshold':'A','partition':'C','group_size':'C','received':'C','received_ratio':'R','retained_fraction':'R','consumed_fraction':'R','total_ratio':'R','weekly_ratio':'R','duration_ratio':'R','repetition':'C','price_age_ratio':'R','count_difference':'D','distance_difference':'D','distance':'C','unit_price_difference':'D','daily_reserved':'C'}
    for filename in ('development_gold.json','audit_gold.json'):
        for r in json.loads((ROOT/'experiments/round6'/filename).read_text(encoding='utf-8'))['rows']:
            q=r['question'];qs=quantities(q);labels=[];mask=[]
            for j,x in enumerate(qs):
                v=x['value']/100 if q[x['end']:].lstrip().startswith('%') else x['value']
                matches=[f for f in r['facts'] if abs(float(f['value'])-v)<1e-7 or (f['kind'] in ('change','age_difference','distance_difference','count_difference') and abs(abs(float(f['value']))-v)<1e-7)]
                unique=len(matches)==1 and sum(abs(z['value']-x['value'])<1e-7 for z in qs)==1
                if unique:
                    f=matches[0];tm=f['time'];stage=('F' if tm>0 else 'P' if tm<0 else 'N') if isinstance(tm,(int,float)) else {'initial':'I','final':'E','before_discount':'I','after_discount':'E','future':'F'}.get(tm,'N');role=roles.get(f['kind'],'O');scope=-1
                    if role=='P':
                        counts=[f2 for f2 in r['facts'] if f2['owner']==f['owner'] and f2['item']==f['item'] and f2['kind']=='count'];hits=[k for k,z in enumerate(qs) if len(counts)==1 and abs(z['value']-counts[0]['value'])<1e-7 and k!=j]
                        if len(hits)==1:scope=hits[0]
                    owner=f['owner'] if f['owner'] not in ('NIL',) else 'NIL';labels.append(dict(role=role,stage=stage,owner=owner,scope=scope))
                else:labels.append(dict(role='O',stage='N',owner='NIL',scope=-1))
                mask.append(unique)
            rows.append(dict(index=r['index'],question=q,quantities=qs,labels=labels,mask=mask,source='weak: unambiguous numeric alignment of round6 manual facts; excluded collisions are masked, not negative labels'))
    return rows
def main():
    rows=old_rows()+manual_rows(MANUAL);plan=json.loads((OUT/'plan.json').read_text());assert {r['index'] for r in rows}==set(plan['existing_train_questions'])|{i for g in plan['groups'].values() for i in g['new_train']}
    (OUT/'training_annotations.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8');print(dict(questions=len(rows),manual_questions=len(MANUAL),quantity_mentions=sum(len(r['labels']) for r in rows),supervised_mentions=sum(sum(r['mask']) for r in rows)))
if __name__=='__main__':main()
