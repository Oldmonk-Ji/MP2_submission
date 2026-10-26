"""Training-derived semantic contrast programs, no rewritten text/embeddings."""
from data_access import ROOT
from round22_common import dump,read,sha
from round29_programs import canonical,serialize,value
from fractions import Fraction
import json,copy,datetime
import numpy as np

OUT=ROOT/'experiments/round31'


def variants(program,qs):
    result=[]
    def visit(p,path=()):
        if p[0]=='N':
            for j in range(len(qs)):
                if j!=p[1]:result.append(('binding',path,('N',j)))
        elif p[0]!='L':
            for op in ['ADD','SUB','MUL','DIV']:
                if op!=p[0]:result.append(('operator',path,(op,p[1],p[2])))
            if p[0] in ('SUB','DIV'):result.append(('direction',path,(p[0],p[2],p[1])))
            result.extend([('omission',path,p[1]),('omission',path,p[2])])
            visit(p[1],path+(1,));visit(p[2],path+(2,))
    def replace(p,path,new):
        if not path:return new
        out=list(p);out[path[0]]=replace(p[path[0]],path[1:],new);return tuple(out)
    visit(program)
    for kind,path,new in result:yield kind,canonical(replace(program,path,new))


def main():
    OUT.mkdir();rng=np.random.default_rng(31031)
    rows=[json.loads(s) for s in open(ROOT/'experiments/round29/training_programs.jsonl',encoding='utf-8')]
    counts={};eligible=0
    for r in rows:
        ref=r['reference'];negative=[];kinds=[]
        if ref['clean']:
            pool={};target=Fraction(ref['answer']);seen=set()
            for kind,p in variants(ref['program'],ref['quantities']):
                try:
                    toks=serialize(p)+['EOS'];v=value(p,ref['quantities'])
                    if len(toks)>96 or v==target:continue
                except (ValueError,ZeroDivisionError,OverflowError):continue
                key=tuple(toks)
                if key in seen:continue
                seen.add(key);pool.setdefault(kind,[]).append(dict(tokens=toks,program=p,value=str(v)))
            for kind in rng.permutation(list(pool)):
                if len(negative)==3:break
                item=pool[kind][int(rng.integers(len(pool[kind])))]
                assert Fraction(item['value'])!=target
                negative.append(item);kinds.append(str(kind));counts[str(kind)]=counts.get(str(kind),0)+1
        r['contrast_mask']=[1]*len(negative)+[0]*(3-len(negative))
        eligible+=bool(negative)
        while len(negative)<3:negative.append(dict(tokens=ref['tokens'],program=ref['program'],value=ref['answer']))
        r['contrast_negatives']=negative;r['contrast_kinds']=kinds
    (OUT/'training_programs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    import shutil
    shutil.copyfile(ROOT/'experiments/round29/diagnostic_split.npz',OUT/'diagnostic_split.npz')
    dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=29029,mutation_seed=31031,
        training_n=4743,validation_n=798,test_opened=False,variants=['semantic_contrast'],epochs=[15,40,80],alphas=[.1,.25,.5],
        method='Same pointer model with executable reference-vs-wrong-program contrasts: entity binding, operator, direction, omitted quantity',
        sources='Only supplied eligible training questions/reference programs; question text unchanged; no new embeddings claimed',
        training_questions_with_contrast=eligible,negative_kind_counts=counts,
        label_check='Each negative executes to a different numeric answer from its own official training reference; no variant is an independent natural question',
        grouped='All variants share their original question/template split; final model still fits all 4743',
        control='round30 reference_and_candidate_ranking; same question exposures/epochs/seed/architecture',
        source_hash=sha(ROOT/'experiments/round29/training_programs.jsonl')))
    dump(OUT/'contrast_audit_sample.json',[dict(train_index=r['train_index'],question=r['question'],reference=r['reference'],
        negatives=r['contrast_negatives'],kinds=r['contrast_kinds']) for r in rows if r['contrast_kinds']][:20])
    print(dict(questions=eligible,negative_counts=counts),flush=True)


if __name__=='__main__':main()
