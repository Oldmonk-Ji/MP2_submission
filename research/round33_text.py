"""Per-question entity identities; no names/IDs memorized across questions."""
import re
import numpy as np
from round29_programs import quantities
from round4_quantities import WORDS

EXCLUDE=set('A An The There How What If When While After Before During In On At For Each Every All Both Some One Two Three Four Five Six Seven Eight Nine Ten He She His Her They Their It Its This That Last Next Now Then First Second Third Mr Mrs Ms Dr Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August September October November December'.lower().split())|set(WORDS)


def names(question):
    out={}
    for m in re.finditer(r'\b[A-Z][a-z]+\b',question):
        w=m.group().lower()
        if w not in EXCLUDE and w not in out and len(out)<16:out[w]='entity'+str(len(out))
    return out


def question_tokens(question):
    qs=quantities(question);entities=names(question)
    def tokenize(s):
        out=[]
        for token in re.findall(r"[a-z]+(?:['’][a-z]+)?|[?.!,;%$]",s.lower()):
            stem=token.replace('’',"'").removesuffix("'s")
            if stem in entities:
                out.append(entities[stem])
                if stem!=token:out.append('possessive')
            else:out.append(token)
        return out
    parts=[];last=0
    for i,q in enumerate(qs):
        parts+=tokenize(question[last:q['start']])+[f'numslot{i}']+tokenize(q['raw'])
        parts+=['digit'+c if c.isdigit() else 'fractionbar' for c in q['value']];last=q['end']
    parts+=tokenize(question[last:]);return parts


def encode_questions(bank,vocab,max_length=256):
    x=np.zeros((len(bank),max_length),np.int64)
    for i,r in enumerate(bank):
        ts=question_tokens(r['question'])
        if len(ts)>max_length:ts=ts[:max_length//2]+ts[-max_length//2:]
        x[i,:len(ts)]=[vocab.get(t,1) for t in ts]
    return x
