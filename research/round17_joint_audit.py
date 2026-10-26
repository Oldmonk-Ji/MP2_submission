"""Post-run metric clarification: correct edge topology versus correct fields.

Two wrong-but-equal predicted labels can connect a true pair. Such an edge is
not a fully correct semantic record. Preserve initial results; report both.
"""
from data_access import ROOT
from round17_model import OUT
import json

def main():
    rows=json.loads((OUT/'audit_annotations.json').read_text(encoding='utf-8'))
    details=json.loads((OUT/'audit_details.json').read_text(encoding='utf-8'))
    pred={(d['record'],d['anchor']):d['prediction'] for d in details}
    groups={}
    for r in rows:
        if r['kind']=='candidate':groups.setdefault(r['train_index'],[]).append(r)
    c=dict(proposed=0,edge_correct=0,both_objects_correct=0,both_objects_predicates_correct=0,full_fields_correct=0,full_fields_unequal_values=0)
    for _,(l,r) in groups.items():
        for i,(a,ga) in enumerate(zip(l['anchors'],l['labels'])):
            pa=pred[l['id'],i]
            for j,(b,gb) in enumerate(zip(r['anchors'],r['labels'])):
                pb=pred[r['id'],j]
                proposed=pa['valid'] and pb['valid'] and pa['object']==pb['object'] and pa['predicate']==pb['predicate'] and pa['predicate']!='other'
                if not proposed:continue
                c['proposed']+=1
                gold=ga['valid'] and gb['valid'] and all(g[k] is not None for g in (ga,gb) for k in ('object','predicate','role')) and ga['object']==gb['object'] and ga['predicate']==gb['predicate'] and ga['predicate']!='other'
                c['edge_correct']+=gold
                obj=gold and all(g['object']==p['object'] for g,p in ((ga,pa),(gb,pb)))
                rel=obj and all(g['predicate']==p['predicate'] for g,p in ((ga,pa),(gb,pb)))
                full=rel and all(g['role']==p['role'] for g,p in ((ga,pa),(gb,pb)))
                c['both_objects_correct']+=obj;c['both_objects_predicates_correct']+=rel;c['full_fields_correct']+=full
                c['full_fields_unequal_values']+=full and a['value']!=b['value']
    c['note']='Additional stricter audit after original results. No retraining/threshold changes. Even all these fields exclude owner, batch, time and RHS correctness.'
    (OUT/'joint_metric_clarification.json').write_text(json.dumps(c,indent=2));print(json.dumps(c,indent=2))

if __name__=='__main__':main()
