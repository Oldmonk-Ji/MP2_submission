"""Small scratch-initialized, permutation-equivariant real-candidate comparator.

No pretrained tokenizer/weights, reference reasoning, question IDs, or labels
enter the network. Frozen official candidate embeddings are optional inputs.
"""
from round19_common import OUT, read_plan
from collections import Counter
import json, math, re
from pathlib import Path
import numpy as np
import torch
from torch import nn

TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?|[a-z]+(?:'[a-z]+)?|[^\w\s]", re.I)
NUM = re.compile(r"^\d[\d,]*(?:\.\d+)?$")
RESERVED = ['<pad>', '<unk>', '<number>']
LEFT = [0, 0, 0, 1, 1, 2]
RIGHT = [1, 2, 3, 2, 3, 3]


def tokens(text):
    text = text.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/')
    return TOKEN.findall(text.lower())


def vocabulary(bank, limit):
    count = Counter()
    for row in bank:
        for text in [row['question']] + [c['text'] for c in row['candidates']]:
            count.update(t for t in tokens(text) if not NUM.fullmatch(t))
    words = sorted(count, key=lambda t: (-count[t], t))[:limit-len(RESERVED)]
    return {s: i for i, s in enumerate(RESERVED + words)}


def encode_text(text, vocab, length):
    ts = tokens(text)
    original = len(ts)
    # Keep problem setup and concluding calculation; never label-guided cropping.
    if len(ts) > length:
        ts = ts[:length//2] + ts[-(length-length//2):]
    ids = np.zeros(length, np.int64)
    nums = np.zeros((length, 5), np.float32)
    for j, t in enumerate(ts):
        if NUM.fullmatch(t):
            value = float(t.replace(',', ''))
            value = min(value, 1e12)
            ids[j] = vocab['<number>']
            nums[j] = [1, min(math.log1p(value)/10, 3), value/(1+value),
                       float(value.is_integer()), value % 1]
        else:
            ids[j] = vocab.get(t, vocab['<unk>'])
    if not len(ts):
        ids[0] = vocab['<unk>']
    return ids, nums, dict(original=original, retained=len(ts), unknown=int((ids==1).sum()))


def encode_bank(bank, arrays, vocab, cfg):
    q, qn, c, cn, lengths = [], [], [], [], []
    for row in bank:
        qi, qv, qm = encode_text(row['question'], vocab, cfg['question_tokens'])
        q.append(qi); qn.append(qv); qi_list, qv_list = [], []
        lengths.append(dict(kind='question', **qm))
        for candidate in row['candidates']:
            ci, cv, cm = encode_text(candidate['text'], vocab, cfg['candidate_tokens'])
            qi_list.append(ci); qv_list.append(cv)
            lengths.append(dict(kind='candidate', **cm))
        c.append(qi_list); cn.append(qv_list)
    data = dict(q=np.asarray(q), qn=np.asarray(qn), c=np.asarray(c), cn=np.asarray(cn),
                e=np.asarray(arrays['embeddings'], np.float32))
    stats = {}
    for kind in ('question', 'candidate'):
        ls = [m for m in lengths if m['kind']==kind]
        stats[kind] = dict(n=len(ls), truncated=sum(m['original']>m['retained'] for m in ls),
                          retained_fraction=sum(m['retained'] for m in ls)/max(1,sum(m['original'] for m in ls)),
                          unknown_fraction=sum(m['unknown'] for m in ls)/max(1,sum(m['retained'] for m in ls)))
    return data, stats


def tensor_batch(data, idx, device):
    selected={k:v[idx] for k,v in data.items()}
    qlen=max(1,int((selected['q']!=0).sum(-1).max()))
    clen=max(1,int((selected['c']!=0).sum(-1).max()))
    selected['q']=selected['q'][:,:qlen];selected['qn']=selected['qn'][:,:qlen]
    selected['c']=selected['c'][:,:,:clen];selected['cn']=selected['cn'][:,:,:clen]
    if 'qk' in selected:
        selected['qk']=selected['qk'][:,:qlen];selected['ck']=selected['ck'][:,:,:clen]
    return {k: torch.as_tensor(v, device=device) for k, v in selected.items()}


def pool(x, mask):
    avg = (x*mask[..., None]).sum(-2)/mask.sum(-1, keepdim=True).clamp_min(1)
    maximum = x.masked_fill(~mask[..., None], -1e4).amax(-2)
    return torch.cat([avg, maximum], -1)


class Comparator(nn.Module):
    def __init__(self, cfg, vocab_size, mean=None, scale=None):
        super().__init__()
        self.cfg = cfg
        w = cfg['width']
        self.word = nn.Embedding(vocab_size, w, padding_idx=0)
        self.position = nn.Embedding(max(cfg['question_tokens'], cfg['candidate_tokens']), w)
        self.numeric = nn.Linear(5, w, bias=False)
        self.norm = nn.LayerNorm(w)
        self.conv = nn.Conv1d(w, w, 3, padding=1)
        self.align_q = nn.Linear(w, w, bias=False)
        self.align_c = nn.Linear(w, w, bias=False)
        self.question_compare = nn.Sequential(nn.Linear(4*w, w), nn.ReLU())
        self.other_compare = nn.Sequential(nn.Linear(4*w, w), nn.ReLU())
        self.emb_project = nn.Sequential(nn.Linear(1024, cfg['embedding_projection']), nn.ReLU())
        self.utility = nn.Sequential(nn.Linear(6*w+cfg['embedding_projection'], 64), nn.ReLU(),
                                     nn.Dropout(cfg['dropout']), nn.Linear(64, 1))
        self.drop = nn.Dropout(cfg['dropout'])
        self.register_buffer('mean', torch.zeros(1024) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.ones(1024) if scale is None else torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer('left', torch.tensor(LEFT))
        self.register_buffer('right', torch.tensor(RIGHT))
        incidence = np.zeros((6, 4), np.float32)
        for k,(i,j) in enumerate(zip(LEFT, RIGHT)):
            incidence[k,i] = 1/3; incidence[k,j] = -1/3
        self.register_buffer('incidence', torch.from_numpy(incidence))
        nn.init.normal_(self.word.weight, std=.08)
        nn.init.normal_(self.position.weight, std=.02)
        with torch.no_grad(): self.word.weight[0].zero_()

    def encode(self, ids, numbers):
        mask = ids.ne(0)
        x = self.word(ids)+self.numeric(numbers)+self.position(torch.arange(ids.shape[-1], device=ids.device))
        x = self.norm(x)*mask[..., None]
        x = self.drop(x+torch.tanh(self.conv(x.transpose(1,2)).transpose(1,2)))
        return x*mask[..., None], mask

    def forward(self, q, qn, c, cn, e):
        n, _, length = c.shape
        qx, qm = self.encode(q, qn)
        cx, cm = self.encode(c.reshape(n*4,length), cn.reshape(n*4,length,5))
        qx = qx.repeat_interleave(4,0); qm = qm.repeat_interleave(4,0)
        aq = torch.matmul(self.align_q(cx), self.align_q(qx).transpose(-1,-2))/math.sqrt(self.cfg['width'])
        q_context = aq.masked_fill(~qm[:,None,:], -1e4).softmax(-1)@qx
        conditioned = self.question_compare(torch.cat([cx, q_context, cx-q_context, cx*q_context], -1))
        own = torch.cat([pool(cx,cm), pool(conditioned,cm),
                         self.emb_project(((e-self.mean)/self.scale).clamp(-10,10)).reshape(n*4,-1)], -1).reshape(n,4,-1)
        h = conditioned.reshape(n,4,length,-1); cm = cm.reshape(n,4,length)
        a, b = h[:,self.left], h[:,self.right]
        ma, mb = cm[:,self.left], cm[:,self.right]
        if self.cfg['mode']=='joint':
            alignment = self.align_c(a)@self.align_c(b).transpose(-1,-2)/math.sqrt(self.cfg['width'])
            ab = alignment.masked_fill(~mb[...,None,:],-1e4).softmax(-1)@b
            ba = alignment.transpose(-1,-2).masked_fill(~ma[...,None,:],-1e4).softmax(-1)@a
        else:
            # Same layers and fitting targets, with the other candidate hidden.
            ab, ba = torch.zeros_like(a), torch.zeros_like(b)
        fa = self.other_compare(torch.cat([a,ab,a-ab,a*ab],-1))
        fb = self.other_compare(torch.cat([b,ba,b-ba,b*ba],-1))
        ua = self.utility(torch.cat([own[:,self.left],pool(fa,ma)],-1)).squeeze(-1)
        ub = self.utility(torch.cat([own[:,self.right],pool(fb,mb)],-1)).squeeze(-1)
        margin = ua-ub
        return margin@self.incidence, margin


def objective(scores, margin, labels):
    diff = labels[:,LEFT]-labels[:,RIGHT]
    eligible = diff.ne(0)
    count = eligible.sum(1)
    per = (nn.functional.softplus(-diff*margin)*eligible).sum(1)/count.clamp_min(1)
    group = torch.logsumexp(scores,1)-torch.logsumexp(scores.masked_fill(labels<.5,-1e4),1)
    keep = count>0
    return (per[keep]+.25*group[keep]).mean() if keep.any() else scores.sum()*0


@torch.no_grad()
def predict_data(model, data, device='cpu', batch=32):
    model.eval()
    scores, margins = [], []
    for start in range(0,len(data['q']),batch):
        s,m = model(**tensor_batch(data,slice(start,start+batch),device))
        scores.append(s.cpu().numpy()); margins.append(m.cpu().numpy())
    return np.concatenate(scores), np.concatenate(margins)


def save_model(model, path):
    np.savez(Path(path)/'model.npz', **{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})


def load_model(path, device='cpu'):
    path = Path(path)
    cfg = json.loads((path/'config.json').read_text())
    vocab = json.loads((path/'vocabulary.json').read_text(encoding='utf-8'))
    if cfg.get('anchored'):
        from round19_anchored import AnchoredComparator
        model=AnchoredComparator(cfg,len(vocab))
    else:model = Comparator(cfg, len(vocab))
    with np.load(path/'model.npz',allow_pickle=False) as z:
        model.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    return model.to(device).eval(), vocab, cfg
