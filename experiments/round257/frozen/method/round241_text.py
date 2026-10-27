"""TRAIN-fitted word tokenizer and question-relative numeric bindings."""
from collections import Counter
from fractions import Fraction
import re
import json
from pathlib import Path
import numpy as np

QLEN=80
CLEN=192
VOCAB=2048
SPECIAL=['<pad>','<unk>','<number>','<answer>']+[f'<qnum{i}>' for i in range(16)]
TOKEN=re.compile(r'\d[\d,]*(?:\.\d+)?(?:/\d+)?|[a-zA-Z]+(?:\x27[a-zA-Z]+)?|####|[+*/=<>%-]')
NUMBER=re.compile(r'^\d')

def tokens(question,text):
    mapping={}
    for t in TOKEN.findall(question):
        if NUMBER.match(t):
            try:v=Fraction(t.replace(',',''))
            except (ValueError,ZeroDivisionError):continue
            if v not in mapping and len(mapping)<16:mapping[v]=len(mapping)
    scale=max([abs(float(v)) for v in mapping]+[1.])
    result=[]
    for t in TOKEN.findall(text):
        if NUMBER.match(t):
            try:v=Fraction(t.replace(',',''))
            except (ValueError,ZeroDivisionError):v=Fraction(0)
            known=v in mapping
            word=f'<qnum{mapping[v]}>' if known else '<number>'
            number=[1.,min(3.,np.log1p(abs(float(v)))/np.log1p(scale))/3.,float(v<0),float(known)]
        else:
            word='<answer>' if t=='####' else t.lower()
            number=[0.,0.,0.,0.]
        result.append((word,number))
    return result or [('<unk>',[0.,0.,0.,0.])]

def fit_vocab(bank,rows):
    count=Counter()
    for i in rows:
        row=bank[i]
        for text in [row['question']]+[c['text'] for c in row['candidates']]:
            count.update(word for word,_ in tokens(row['question'],text) if word not in SPECIAL)
    words=sorted((w for w,n in count.items() if n>=2),key=lambda w:(-count[w],w))[:VOCAB-len(SPECIAL)]
    return {w:i for i,w in enumerate(SPECIAL+words)}

def encode(bank,vocab):
    n=len(bank)
    result=dict(q=np.zeros((n,QLEN),np.int64),c=np.zeros((n,4,CLEN),np.int64),
                qn=np.zeros((n,QLEN,4),np.float32),cn=np.zeros((n,4,CLEN,4),np.float32))
    def fill(text,question,length,ids,numbers):
        ts=tokens(question,text)
        if len(ts)>length:
            head=length*2//3;ts=ts[:head]+ts[-(length-head):]
        for j,(word,number) in enumerate(ts):
            ids[j]=vocab.get(word,1);numbers[j]=number
    for i,row in enumerate(bank):
        fill(row['question'],row['question'],QLEN,result['q'][i],result['qn'][i])
        for j,c in enumerate(row['candidates']):
            fill(c['text'],row['question'],CLEN,result['c'][i,j],result['cn'][i,j])
    return result

def save_vocab(path,vocab):
    Path(path).write_text(json.dumps(vocab,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
