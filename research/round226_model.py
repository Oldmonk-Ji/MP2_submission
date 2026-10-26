"""Scratch-trained, answer-group-aware conditional comparison network."""
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from features import prepare, metadata
from round3_features import grounding

SEED = 226226
EPOCHS = 120
WIDTH = 128


def inputs(bank, arrays):
    p = prepare(bank, arrays)
    g = np.concatenate([p["base"], p["numeric"], p["checks2"], grounding(bank)], 2)
    meta = metadata(bank)
    same = np.asarray([[[a is not None and a == b for b in row] for a in row]
                       for row in meta["answers"]], np.float32)
    same[:, np.arange(4), np.arange(4)] = 1
    return dict(e=np.asarray(arrays["embeddings"], np.float32),
                g=g.astype(np.float32), same=same, valid=meta["valid"])


class Comparator(nn.Module):
    def __init__(self, data, rows):
        super().__init__()
        e, g = data["e"][rows], data["g"][rows]
        q = e.mean(1, keepdims=True)
        residual = e - q
        for name, value in dict(qmean=q.mean((0, 1)), qscale=q.std((0, 1)).clip(1e-3),
                                 rscale=residual.std((0, 1)).clip(1e-4),
                                 gmean=g.mean((0, 1)), gscale=g.std((0, 1)).clip(.05)).items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.residual = nn.Linear(e.shape[-1], WIDTH)
        self.condition = nn.Linear(e.shape[-1], 2*WIDTH)
        self.numeric = nn.Linear(g.shape[-1], WIDTH)
        self.norm = nn.LayerNorm(WIDTH)
        self.dropout = nn.Dropout(.15)
        self.head = nn.Sequential(nn.Linear(7*WIDTH, 128), nn.GELU(approximate="tanh"),
                                  nn.Dropout(.15), nn.Linear(128, 32),
                                  nn.GELU(approximate="tanh"), nn.Linear(32, 1))

    def forward(self, e, g, same, valid):
        q = e.mean(1, keepdim=True)
        r = (e-q)/self.rscale
        q = (q-self.qmean)/self.qscale
        g = (g-self.gmean)/self.gscale
        gamma, beta = self.condition(q).chunk(2, -1)
        h = self.residual(r) * (1 + .5*torch.tanh(gamma)) + beta + self.numeric(g)
        h = self.dropout(F.gelu(self.norm(h), approximate="tanh"))
        own = (same @ h) / same.sum(2, keepdim=True).clamp_min(1)
        other_mask = 1-same
        other = (other_mask @ h) / other_mask.sum(2, keepdim=True).clamp_min(1)
        center = h.mean(1, keepdim=True).expand_as(h)
        comparison = torch.cat([h, own, other, h-own, h-other, own-other, center], -1)
        raw = self.head(comparison).squeeze(-1)
        return raw.masked_fill(~valid, -1e4), raw


def tensors(data):
    return {k: torch.as_tensor(v) for k, v in data.items()}


def batch(t, indices):
    return {k: v[indices] for k, v in t.items()}


def scores(model, data, rows=None):
    model.eval()
    if rows is None:
        rows = np.arange(len(data["e"]))
    t = tensors(data)
    result = []
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            s, _ = model(**batch(t, rows[start:start+128]))
            result.append(s.numpy())
    return np.concatenate(result)


def loss(scores, raw, y):
    mixed = y.sum(1) < 4
    correct_set = torch.logsumexp(scores, 1) - torch.logsumexp(
        scores.masked_fill(y < .5, -1e4), 1)
    margin = scores[:, :, None] - scores[:, None, :]
    mask = y[:, :, None] * (1-y[:, None, :])
    pair = (F.softplus(-margin)*mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
    rank = (correct_set[mixed] + .5*pair[mixed]).mean() if mixed.any() else scores.sum()*0
    return rank + .1*F.binary_cross_entropy_with_logits(raw, y)


def train(data, y, rows, path):
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = Comparator(data, rows)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.05)
    t = tensors(data)
    target = torch.as_tensor(y)
    tick = time.perf_counter()
    history = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        order = rng.permutation(rows)
        total = 0.
        for start in range(0, len(order), 128):
            ix = order[start:start+128]
            s, raw = model(**batch(t, ix))
            value = loss(s, raw, target[ix])
            optimizer.zero_grad()
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            with torch.no_grad():
                for a, b in zip(ema.parameters(), model.parameters()):
                    a.mul_(.99).add_(b, alpha=.01)
            total += float(value.detach()) * len(ix)
        if epoch % 20 == 0:
            rec = dict(epoch=epoch, loss=total/len(rows), seconds=time.perf_counter()-tick)
            history.append(rec)
            print("TRAIN", path.name, rec, flush=True)
    state = {k: v.detach().numpy() for k, v in ema.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **state)
    return ema, history
