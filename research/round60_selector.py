"""Section-aware textual verification: separate restatements from calculation."""
import re,json
from pathlib import Path
import numpy as np
from round3_features import lexical_documents
from round25_selector import scores as baseline
from round22_selector import normalized
from round4_gates import apply_gate

def documents(bank,sections=True):
    if not sections:return lexical_documents(bank)
    result=[]
    for row in bank:
        for c in row['candidates']:
            t=c['text'];lines=t.splitlines();parts={'intro':[],'work':[],'answer':[]};stage='intro'
            for line in lines:
                low=line.lower()
                if re.search(r'final answer|final total|####|boxed',low):stage='answer'
                elif re.search(r'\bstep\s*\d|step.by.step|\bcalculate\b|\blet.s\s+(?:solve|find|break)',low) or '=' in line:stage='work'
                parts[stage].append(line)
            docs=lexical_documents([dict(question=row['question'],candidates=[dict(text='\n'.join(parts[k])) for k in parts])])
            result.append(' '.join(' '.join(k+'_'+w for w in d.split()) for k,d in zip(parts,docs)))
    return result

def text_scores(bank,path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    cfg=json.loads((path/'config.json').read_text());vocab=json.loads((path/'vocabulary.json').read_text())
    with np.load(path/'model.npz') as w:
        vec=TfidfVectorizer(vocabulary=vocab,ngram_range=(1,2),sublinear_tf=True,dtype=np.float32);vec.idf_=w['idf']
        return np.asarray(vec.transform(documents(bank,cfg['sections']))@w['weight']).reshape(-1,4)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    p=Path(checkpoint);cfg=json.loads((p/'config.json').read_text());s=text_scores(bank,p)
    return apply_gate(bank,normalized(baseline(bank,arrays,p/'baseline'))+cfg['alpha']*normalized(s),'parse_and_repair',4).astype(int).tolist()
