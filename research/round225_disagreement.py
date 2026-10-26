"""Question-conditioned lexical disagreement, fitted only on supplied TRAIN."""
from collections import Counter
from fractions import Fraction
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

DIM = 65536
STOP = set("a an the to of in on at by with and or is are was were be been has have had "
           "he she it they his her their them this that these those we you i for as "
           "then so therefore answer final boxed text frac left right".split())
TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?|[A-Za-z]+|[+*/=<>-]")


def normalized_tokens(text, number_map):
    text = text.replace("**", "").replace(r"\times", "*").replace("×", "*")
    text = text.replace(r"\div", "/").replace(r"\cdot", "*")
    out = []
    for t in TOKEN.findall(text):
        if t[0].isdigit():
            value = str(Fraction(t.replace(",", "")))
            if value in number_map:
                t = number_map[value]
            elif Fraction(value).denominator == 1 and abs(int(value)) <= 100:
                t = "k" + value
            else:
                t = "newnum"
        else:
            t = t.lower()
        out.append(t)
    return out


def row_documents(row):
    q = row["question"]
    nums = [str(Fraction(t.replace(",", ""))) for t in TOKEN.findall(q) if t[0].isdigit()]
    mapping = {v: "n" + str(i) for i, v in enumerate(dict.fromkeys(nums))}
    qt = normalized_tokens(q, mapping)
    question_parts = [s for s in re.split(r"[.!?](?:\s|$)", q) if s.strip()]
    target = normalized_tokens(question_parts[-1] if question_parts else q, mapping)
    # No labels, training corpus statistics, or candidate order in extraction.
    conditions = sorted(set(t for t in qt if t not in STOP and len(t) > 2))[:32]
    requested = sorted(set(t for t in target if t not in STOP and len(t) > 2))[:16]
    candidates = [normalized_tokens(c["text"], mapping) for c in row["candidates"]]
    grams = []
    for tokens in candidates:
        g = set()
        for width in (1, 2, 3):
            for i in range(len(tokens) - width + 1):
                part = tokens[i:i + width]
                if all(t in STOP for t in part):
                    continue
                g.add(" ".join(part))
        grams.append(g)
    frequency = Counter(g for group in grams for g in group)
    results = []
    for group in grams:
        distinct = [g for g in group if frequency[g] < 4]
        # Prioritize local numeric claims, then uncommon within-question phrases.
        distinct.sort(key=lambda g: (-int(any(t[:1] in ("n", "k") and
                            t[1:].isdigit() for t in g.split())),
                            frequency[g], -len(g.split()), g))
        selected = distinct[:48]
        doc = {"g:" + g: 1.0 for g in distinct[:384]}
        for g in selected:
            scale = (4 - frequency[g]) / 3
            for t in requested:
                doc["target:" + t + "|" + g] = scale
            for t in conditions:
                doc["condition:" + t + "|" + g] = .5 * scale
        results.append(doc)
    return results


def build(bank):
    hasher = FeatureHasher(n_features=DIM, input_type="dict", dtype=np.float32,
                           alternate_sign=True)
    x = hasher.transform(doc for row in bank for doc in row_documents(row))
    return normalize(x, norm="l2", copy=False).tocsr()


def pairwise_fit(x, y, rows):
    positive, negative, weights = [], [], []
    for i in rows:
        p, n = np.flatnonzero(y[i] == 1), np.flatnonzero(y[i] == 0)
        count = len(p) * len(n)
        for a in p:
            for b in n:
                positive.append(4 * i + a)
                negative.append(4 * i + b)
                weights.append(1 / count)
    delta = x[positive] - x[negative]
    train = sparse.vstack([delta, -delta], format="csr")
    model = LogisticRegression(C=1, fit_intercept=False, solver="liblinear",
                               max_iter=500, tol=1e-5, random_state=225225)
    model.fit(train, np.r_[np.ones(len(delta.indptr)-1), np.zeros(len(delta.indptr)-1)],
              sample_weight=np.tile(np.asarray(weights) / 2, 2))
    return model.coef_[0].astype(np.float32)


def score(x, weight):
    return np.asarray(x @ weight).reshape(-1, 4)
