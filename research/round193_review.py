from round22_common import ROOT,load,read,dump
if __name__=='__main__':
    old={r['train_index']:r['programs'] for r in read(ROOT/'experiments/round179/fit_programs.json')};new=read(ROOT/'experiments/round193/fit_programs.json');b,a,y,ids=load('train');idx={int(v):i for i,v in enumerate(ids)};counts=dict(changed_candidates=0,newly_supported=0,lost_support=0,fewer_alternatives=0,more_alternatives=0);examples=[]
    for r in new:
        k=r['train_index']
        for j,(p,q) in enumerate(zip(old[k],r['programs'])):
            if p==q:continue
            counts['changed_candidates']+=1;counts['newly_supported']+=not p and bool(q);counts['lost_support']+=bool(p) and not q;counts['fewer_alternatives']+=len(q)<len(p);counts['more_alternatives']+=len(q)>len(p)
            if len(examples)<15:examples.append(dict(train_index=k,candidate=j,question=b[idx[k]]['question'],before=len(p),after=len(q)))
    dump(ROOT/'experiments/round193/fit_changes.json',dict(counts=counts,examples=examples,test_opened=False));print(counts)
