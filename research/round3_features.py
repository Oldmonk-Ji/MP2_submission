"""Answer-aware comparisons and bounded numeric evidence; input-only features.

These features measure textual consistency, not a proof that a word problem
was solved correctly. They do not override the official numeric parser.
"""
import re
from fractions import Fraction
import numpy as np
from features import prepare, matrix, metadata, safe_arithmetic, NUMBER


def normalize_math(text):
    s=text.replace('\\times','*').replace('\\cdot','*').replace('×','*')
    s=s.replace('\\div','/').replace('÷','/').replace('−','-')
    s=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'\1',s)
    s=s.replace('\\left','').replace('\\right','')
    for _ in range(4):
        s=re.sub(r'\\(?:d?frac)\{([\d.\s+*/()\-]+)\}\{([\d.\s+*/()\-]+)\}',r'((\1)/(\2))',s)
        s=re.sub(r'\\boxed\{([\d.\s+*/()\-]+)\}',r'(\1)',s)
    return s.replace('\\,','').replace('\\!','').replace('\\$','').replace('$','')


def equations(text):
    """Extract only fully numeric adjacent sides; no symbolic evaluation."""
    found=[]
    for line in normalize_math(text).splitlines():
        for match in re.finditer('=',line):
            left=re.search(r'([\d().+*/ \t\-]+)$',line[:match.start()])
            right=re.match(r'([\d().+*/ \t\-]+)',line[match.end():])
            if left is None or right is None:continue
            l,r=left.group(1).strip(),right.group(1).strip().rstrip('.')
            # Never reinterpret the numeric suffix of "x + 2" as unary "+2".
            prefix=line[:match.start()][:left.start()].rstrip()
            if l.startswith(('+','-','*','/')) and prefix and (prefix[-1].isalnum() or prefix[-1] in ')}]'):
                continue
            suffix=line[match.end():][right.end():]
            if suffix and suffix[0].isalnum():continue
            if not any(c in l+r for c in '+-*/'):continue
            try:a,b=safe_arithmetic(l),safe_arithmetic(r)
            except (ValueError,SyntaxError,OverflowError,ZeroDivisionError):continue
            found.append((l,r,a,b))
    return found


def close(a,b):
    return abs(a-b)<=1e-5*max(1,abs(a),abs(b))


def number_set(text):
    return set(float(x.replace(',','')) for x in NUMBER.findall(text))


def grounding(bank):
    rows=[]
    cues=[r'percent|percentage|%',r'\beach\b|\bper\b',r'\bmore\b|\bincrease',
          r'\bless\b|\bfewer\b|\bdecrease',r'\btotal\b|\ball\b|\btogether\b',
          r'\bleft\b|\bremain',r'\btwice\b|\bdouble\b|\btwo\b',
          r'\bhalf\b|\bquarter\b',r'\bminute|\bhour|\bsecond',r'\baverage\b']
    meta=metadata(bank)
    for i,row in enumerate(bank):
        q=row['question'].lower();qn=number_set(q)
        for j,c in enumerate(row['candidates']):
            t=c['text'];tl=t.lower();eq=equations(t)
            final=meta['answers'][i][j];ans=float(Fraction(final)) if final is not None else 0.
            operands=set();results=[];bad=[];reused=0
            for l,r,a,b in eq:
                ln=number_set(l);operands.update(ln)
                reused+=sum(any(close(n,p) for p in results) for n in ln)
                results.append(b)
                bad.append(float(abs(a-b)/max(1,abs(a),abs(b))>.001 and abs(a-b)>.02))
            miss=sum(not any(close(n,v) for v in operands) for n in qn)
            unseen=sum(not any(close(n,v) for v in qn) for n in operands)
            body=t.rsplit('####',1)[0] if '####' in t else t
            bn=number_set(body)
            f=[np.log1p(len(eq)),np.mean(bad) if bad else 0.,
               float(bad[-1]) if bad else 0.,np.log1p(reused),
               miss/max(1,len(qn)),unseen/max(1,len(operands)),
               float(any(close(ans,v) for v in results)),
               float(bool(results) and close(ans,results[-1])),
               min([abs(ans-v)/max(1,abs(ans),abs(v)) for v in results] or [1.]),
               float(any(close(ans,v) for v in bn)),
               np.log1p(len(operands)),float(len(qn)>0 and miss==0)]
            # Generic condition/cue coverage; never interpreted as a hard rule.
            for cue in cues:
                qc=len(re.findall(cue,q));tc=len(re.findall(cue,tl))
                f.extend([float(qc>0),float(qc>0 and tc==0),np.log1p(tc)])
            # Operation frequencies conditional on words in the question.
            ops=np.asarray([sum(l.count(op) for l,_,_,_ in eq) for op in '+-*/'],float)
            f.extend(np.log1p(ops))
            rows.append(f)
    return np.asarray(rows,np.float32).reshape(len(bank),4,-1)


def build(bank,arrays,kind,prepared=None,ground=None):
    p=prepare(bank,arrays) if prepared is None else prepared
    base=matrix(p,'context_checked')
    parts=[base]
    if kind in ('ground','answer_ground'):
        g=grounding(bank) if ground is None else ground
        parts.extend([g,g-g.mean(1,keepdims=True)])
    if kind in ('answer','answer_ground'):
        e=p['embedding'];answers=metadata(bank)['answers']
        same=np.asarray([[[a is not None and a==b for b in row] for a in row] for row in answers],np.float32)
        other=1-same
        own=np.einsum('nij,njk->nik',same,e)/np.maximum(same.sum(2,keepdims=True),1)
        alt=np.einsum('nij,njk->nik',other,e)/np.maximum(other.sum(2,keepdims=True),1)
        parts.extend([own-alt,own*alt])
        # Pool arithmetic/style evidence by the competing final answers too.
        small=np.concatenate([p['base'],p['numeric'][:,:,5:],p['checks2']],2)
        so=np.einsum('nij,njk->nik',same,small)/np.maximum(same.sum(2,keepdims=True),1)
        sa=np.einsum('nij,njk->nik',other,small)/np.maximum(other.sum(2,keepdims=True),1)
        parts.extend([so,so-sa])
    if kind not in ('base','ground','answer','answer_ground'):raise ValueError(kind)
    return np.ascontiguousarray(np.concatenate(parts,2),dtype=np.float32)


def lexical_documents(bank,mode='reasoning'):
    """Normalize numeric values, keeping their bindings to question numbers.

    Vocabulary/IDF/coefficients are fitted on TRAIN only. Question IDs are never
    included. This is a local n-gram representation, not a semantic verifier.
    """
    docs=[]
    for row in bank:
        q=row['question'].lower()
        nums=list(dict.fromkeys(float(s.replace(',','')) for s in NUMBER.findall(q)))
        mapping={v:f' qnum{k} ' for k,v in enumerate(nums)}
        qw=set(re.findall(r'[a-z]{3,}',q))
        for c in row['candidates']:
            text=c['text'].lower()
            if mode=='answer_only':
                # Deliberately omit the reasoning for the text-information ablation.
                from features import parse_answer
                a=parse_answer(text)
                text=q+' answer '+('invalid' if a is None else str(a))
            text=NUMBER.sub(lambda m:mapping.get(float(m.group().replace(',','')),' newnum '),text)
            text=text.replace('\\times',' multiply ').replace('*',' multiply ').replace('×',' multiply ')
            text=text.replace('\\div',' divide ').replace('/',' divide ').replace('+',' plus ').replace('=',' equal ')
            words=re.findall(r'[a-z]{2,}\d*',text)
            overlap=['qword_'+w for w in words if w in qw]
            docs.append(' '.join(words+overlap))
    return docs
