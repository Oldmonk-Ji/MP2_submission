"""Input-only quantity extraction and arithmetic dependency differentiation.

This is a conservative, partial trace parser, not a natural-language theorem
prover. Constants, source-number collisions, and incomplete traces are exposed
as coverage features. The official final-answer parser remains untouched.
"""
import ast,re
from fractions import Fraction
import numpy as np
from features import parse_answer,safe_arithmetic

WORDS={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,
       'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,
       'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,'thirty':30,
       'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,
       'once':1,'twice':2,'double':2,'doubled':2,'triple':3,'tripled':3,'half':.5,
       'quarter':.25,'dozen':12,'first':1,'second':2,'third':3,'fourth':4,'fifth':5}
NUM=r'(?:\d[\d,]*\s+\d+/\d+|(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?)'
TOKEN=re.compile(r'(?<![\w.])'+NUM+r'(?!\w)|\b(?:'+ '|'.join(WORDS)+r')\b',re.I)
STOP=set('a an the of for in on at to and or is are was were be has have had he she it his her they them their with how many much what does do did will would can each per than as by from that this'.split())


def quantities(question):
    result=[];lookup={}
    for m in TOKEN.finditer(question):
        raw=m.group().lower()
        if raw in WORDS:v=float(WORDS[raw])
        else:
            try:
                parts=raw.replace(',','').split();v=float(sum(Fraction(p) for p in parts))
            except (ValueError,ZeroDivisionError):continue
        if not np.isfinite(v):continue
        if v in lookup:
            result[lookup[v]]['occurrences']+=1;continue
        if len(result)>=24:break
        lookup[v]=len(result)
        left=re.findall(r'[a-z]+',question[:m.start()].lower())[-8:]
        right=re.findall(r'[a-z]+',question[m.end():].lower())[:8]
        result.append(dict(value=v,start=m.start(),end=m.end(),raw=raw,left=left,right=right,occurrences=1))
    return result


def math_text(text):
    s=text.replace('×','*').replace('÷','/').replace('−','-').replace('–','-')
    s=s.replace('\\times','*').replace('\\cdot','*').replace('\\div','/')
    s=re.sub(r'\\(?:text|mathrm|operatorname)\{[^{}]*\}',' ',s)
    s=s.replace('\\left','').replace('\\right','').replace('\\,','').replace('\\!','')
    s=re.sub(r'\*\*([^*]+)\*\*',r'\1',s)
    s=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',s)
    s=s.replace('\\$','').replace('$','')
    for _ in range(6):
        s=re.sub(r'\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}',r'((\1)/(\2))',s)
        s=re.sub(r'\\boxed\s*\{([^{}]+)\}',r'(\1)',s)
    s=re.sub(r'(\d+(?:\.\d+)?)\\?%',r'(\1/100)',s)
    # Math annotations in GSM8K references are parsed separately; remove the
    # adjacent rendered answer here so it cannot concatenate with the RHS.
    s=re.sub(r'<<([^<>]+)>>\s*[-+]?\d*(?:\.\d+)?',r'\1 ',s)
    return s


def arithmetic_steps(text,reference=False,strict=False):
    source='\n'.join(re.findall(r'<<([^<>]+)>>',text)) if reference else text
    if strict:
        lines=[]
        for line in source.splitlines():
            bullet=re.match(r'^\s*[-+*]\s+(.*)$',line)
            if bullet:
                rest=bullet.group(1)
                # A Markdown list marker before a math span is not unary minus.
                if rest.startswith(('$','\\','**')):line=rest
                elif rest[:1].isdigit():continue  # ambiguous plain-text prefix
            lines.append(line)
        source='\n'.join(lines)
    steps=[]
    for line in math_text(source).splitlines():
        for eq in re.finditer('=',line):
            left=re.search(r'([\d().+*/ \t\-]+)$',line[:eq.start()])
            right=re.match(r'([\d().+*/ \t\-]+)',line[eq.end():])
            if not left or not right:continue
            l=left.group().strip();r=right.group().strip().rstrip('.')
            prefix=line[:eq.start()][:left.start()].rstrip()
            if prefix and prefix[-1].isalnum() and l.startswith(('+','-','*','/')):continue
            suffix=line[eq.end():][right.end():]
            if suffix and suffix[0].isalnum():continue
            try:
                tree=ast.parse(l,mode='eval');a=safe_arithmetic(l);b=safe_arithmetic(r)
            except (ValueError,SyntaxError,OverflowError,ZeroDivisionError):continue
            if not any(isinstance(n,ast.BinOp) for n in ast.walk(tree)):continue
            steps.append((l,r,a,b))
    return steps


def trace(question,text,reference=False,repair_subexpressions=False,strict_math=None):
    qs=quantities(question);values=np.asarray([q['value'] for q in qs]);n=len(qs)
    strict=repair_subexpressions if strict_math is None else strict_math
    direct={v:i for i,v in enumerate(values)};cache={};steps=arithmetic_steps(text,reference,strict=strict)
    collisions=0;unexplained=0;resolved=0;roots=[];bad=0;expression_cache={}
    def constant(v):
        nonlocal collisions,unexplained,resolved
        # Prefer the latest earlier computation when a value is also a source
        # quantity. This ambiguity is counted and can be excluded in supervision.
        if v in cache:
            if v in direct:collisions+=1
            resolved+=1;g,program=cache[v];return v,g.copy(),program
        if v in direct:
            g=np.zeros(n);g[direct[v]]=1;resolved+=1;return v,g,('q',direct[v])
        # Explicit percentages often become decimals in the solution.
        if 'percent' in question.lower() or '%' in question:
            for i,q in enumerate(qs):
                if abs(q['value']/100-v)<1e-9 and abs(v)>1e-10:
                    g=np.zeros(n);g[i]=.01;resolved+=1;return v,g,('/',('q',i),('c',100))
        if v not in (0,1,2,3,4,5,7,10,12,24,60,100,1000):unexplained+=1
        return v,np.zeros(n),('c',v)
    def visit(node):
        if isinstance(node,ast.Expression):return visit(node.body)
        if repair_subexpressions and isinstance(node,ast.BinOp):
            key=ast.dump(node)
            if key in expression_cache:
                v,g,p=expression_cache[key];return v,g.copy(),p
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return constant(float(node.value))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            v,g,p=visit(node.operand)
            return (-v,-g,('-',('c',0),p)) if isinstance(node.op,ast.USub) else (v,g,p)
        if not isinstance(node,ast.BinOp):raise ValueError('symbolic')
        a,ga,pa=visit(node.left);b,gb,pb=visit(node.right)
        if isinstance(node.op,ast.Add):return a+b,ga+gb,('+',pa,pb)
        if isinstance(node.op,ast.Sub):return a-b,ga-gb,('-',pa,pb)
        if isinstance(node.op,ast.Mult):return a*b,ga*b+gb*a,('*',pa,pb)
        if isinstance(node.op,ast.Div) and abs(b)>1e-10:return a/b,(ga*b-gb*a)/(b*b),('/',pa,pb)
        raise ValueError('unsupported')
    for l,r,a,claimed in steps:
        try:value,grad,program=visit(ast.parse(l,mode='eval'))
        except (ValueError,OverflowError,ZeroDivisionError):continue
        if not np.isfinite(grad).all() or np.max(np.abs(grad),initial=0)>1e12:continue
        bad+=int(abs(a-claimed)>.02 and abs(a-claimed)/max(1,abs(a),abs(claimed))>.001)
        # Intermediate values use what the candidate actually asserted, while
        # derivatives retain the expression's dependency on question quantities.
        cache[claimed]=(grad,program);roots.append((claimed,grad,program))
        if repair_subexpressions:
            expression_cache[ast.dump(ast.parse(r,mode='eval').body)]=(claimed,grad,program)
    ans=parse_answer(text);answer=float(Fraction(ans)) if ans is not None else None
    matching=[r for r in roots if answer is not None and abs(r[0]-answer)<1e-5*max(1,abs(answer))]
    root=matching[-1] if matching else None
    gradient=root[1] if root else np.zeros(n)
    classes=np.where(np.abs(gradient)<1e-7,0,np.where(gradient>0,1,2)).astype(int)
    # Independently expose mismatches caused by earlier #### headings. This
    # does not change or repair the official parser's answer.
    boxes=re.findall(r'\\boxed\{([^{}]+)\}',text)
    last_box=parse_answer('#### '+boxes[-1]) if boxes else None
    return dict(quantities=qs,gradient=gradient,classes=classes,answer=answer,root_found=root is not None,
        program=root[2] if root else None,steps=len(steps),bad=bad,collisions=collisions,
        unexplained=unexplained,resolved=resolved,
        boxed_parse_disagreement=float(last_box is not None and ans is not None and last_box!=ans))


def quantity_documents(question,qs):
    target=re.split(r'[.?!]',question)[-2] if question.rstrip().endswith('?') else question
    tail=re.findall(r'[a-z]+',target.lower())[-25:]
    target_words=[w for w in tail if w not in STOP]
    docs=[]
    for q in qs:
        local=q['left'][-5:]+q['right'][:5]
        content=[w for w in local if w not in STOP]
        tokens=['l_'+w for w in q['left']]+['r_'+w for w in q['right']]+['t_'+w for w in tail]
        tokens+=['x_'+a+'_'+b for a in content for b in target_words]
        tokens+=['left_'+'_'.join(q['left'][-2:]),'right_'+'_'.join(q['right'][:2])]
        tokens+=['fraction' if abs(q['value'])<1 else 'whole','repeat_'+str(min(q['occurrences'],3))]
        docs.append(' '.join(tokens))
    return docs


def semantic_features(question,candidates,probabilities):
    qs=quantities(question);n=len(qs);out=[]
    for c in candidates:
        t=trace(question,c['text']);cl=t['classes'];p=probabilities
        informative=np.asarray([q['value'] not in (0,1) for q in qs])
        m=informative & (p.max(1)>.5) if n else np.zeros(0,bool)
        if n:
            agree=np.log(np.clip(p[np.arange(n),cl],1e-5,1))
            chosen=p.argmax(1);used=cl!=0;expected_used=1-p[:,0]
            elastic=t['gradient']*np.asarray([q['value'] for q in qs])/max(1,abs(t['answer'] or 0))
            f=[float(agree.mean()),float(agree[m].mean()) if m.any() else 0.,float((cl==chosen).mean()),
               float((cl[m]==chosen[m]).mean()) if m.any() else 0.,float((expected_used*(~used)).mean()),
               float((p[:,1]*(cl==2)+p[:,2]*(cl==1)).mean()),float((p[:,0]*used).mean()),
               float(used.mean()),float(np.mean(np.minimum(np.abs(elastic),10))),float(p.max(1).mean())]
        else:f=[0.]*10
        f += [float(t['root_found']),np.log1p(t['steps']),np.log1p(t['bad']),
              np.log1p(t['collisions']),np.log1p(t['unexplained']),np.log1p(t['resolved']),
              t['boxed_parse_disagreement'],np.log1p(n)]
        if not t['root_found']:f[:10]=[0.]*10
        out.append(f)
    return np.asarray(out,np.float32)
