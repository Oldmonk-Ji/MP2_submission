---
title: "Verification-Aware Comparative Ranking for Selecting Reliable LLM Reasoning"
subtitle: "DASE7506 · Advanced Machine Learning Methods · MP2 Final Report"
author: "Student ID: (to be filled)"
date: "31 October 2026"
geometry: margin=2.4cm
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \setlength{\parskip}{2pt}
---

# Summary

This report describes a system that selects one of four LLM-generated solutions for each GSM8K problem. The final method combines (i) an ensemble of eight base scorers over question/candidate features, (ii) comparative fusion features derived from the four candidates' derivations — question-context influence, conditional quantity-flow scores, arithmetic relation scores, and local program-structure scores — fused by a LambdaRank model, (iii) a parse-and-repair gate that reconciles the ranked choice with the boxed answer actually present in the text, and (iv) a strict numerical verification score with a conservative gap-based override rule.

On the official evaluation protocol the method selects the correct candidate on **760/798 validation questions (95.24%)** and **391/467 test questions (83.73%)**, improving the historical baseline test score of 386 (82.66%) by five questions. Inference assets total **15.09 MiB** (limit 16 MiB) and a fresh test prediction plus scoring process completes in **25.5 s** on the local machine using four threads (limit 30 s on the reference Xeon 8457C). All 137 verifier checks of the official validation harness pass: frozen hashes unchanged, batch and candidate-permutation invariance, no reference-answer reads on the prediction path, and no test use before the method was frozen.

# Method

The system receives a question, four candidate solutions (free text with reasoning and a final boxed answer), and fixed numerical representations of shape [4, 1024]. It outputs exactly one candidate index per question. All components are trained only on the 4,743 training questions (twelve defective/ambiguous questions excluded by a standing convention; 3-fold split used for auxiliary models, with fold 2 held out as a 1,582-question internal sanity set). The validation set (798 questions) is used for development and method selection only. The test set remained sealed until the method was frozen.

## Base expert scorers

Eight base members score each candidate independently, then their scores are combined with learned weights ($w$ in the bundle configuration):

1. **MLP-base** (width 128, mixed loss, EMA 0.98, seed 20261014, 180 epochs): embedding features derived from the fixed representations plus 14 hand-crafted text features (answer parses, answer-agreement counts, length, equation counts, formatting markers, word overlap with the question, revision markers); weight 0.307.
2. **MLP-ground** (width 128, 35 epochs): numeric/grounded features (arithmetic correctness of the derivation's equation chain, answer magnitude/fraction properties, number reuse between question and candidate); weight 0.495.
3. **MLP-answer** (width 128, 35 epochs): answer-focused features; weight 0.620.
4. **TF-IDF text scorer** (ngram 1–2, sublinear TF, C=100): logistic scorer over the raw reasoning text; weight 0.160.
5. **Four seed-diverse MLPs** (width 96, uniform-positive loss, seeds 25025/25026 over ground/answer features): diversity members with fixed weight 0.125 each.

The member weights are learned by a small fusion fit on training data. The heterogeneous families (MLP vs. linear text model, mixed vs. uniform-positive losses, multiple seeds) provide decorrelated evidence, which matters more for ranking than any single member's accuracy.

## Comparative fusion features

A candidate can look plausible in isolation and still be wrong; the strongest evidence comes from comparing the four derivations against each other and against the question. The fusion model consumes a 708-dimensional per-candidate feature surface built from:

- **Expert component scores**: the eight base members' out-of-fold scores (three-fold cross-fit on training data).
- **Question-context influence (PCA-64)**: a projection of the question's fixed representation; the feature measures how strongly a candidate's conclusion vector aligns with the question context.
- **Conditional quantity-flow scores (16,384 sparse dims)**: a contrastive logistic regression over hashed features of how quantities flow between equations (which quantity increases/decreases/is unused, conditioned on the operation), trained on candidate pairs from training data.
- **Arithmetic relation scores (16,384 sparse dims)**: a six-class classifier (ADD/SUB/SUB_REVERSE/MUL/DIV/DIV_REVERSE) over normalized derivation chains, adapted from the relation extraction approach of Roy and Roth (2015). It scores the plausibility of each candidate's operation sequence; it is auxiliary evidence, never a hard validity rule.
- **Local program-structure scores (65,536 sparse dims)**: hashed local structure features over a parsed program representation of each derivation (local arm).
- **Row base score**: the combined base expert score, also used as the ranking model's init score.

The surface is fused by a **LambdaRank gradient-boosted model** (LightGBM, 200 trees, 7 leaves, min data 100/leaf, learning rate 0.03, $\ell_2$=50, truncation level 1, seed 68068). Because the final answer only matters, ranking loss is more appropriate than pointwise classification.

## Parse-and-repair gate

The ranked choice is passed through a deterministic gate (margin limit 4). The gate parses each candidate's final boxed answer and, when the top-ranked candidate's boxed answer does not match its parsed final numeric answer, proposes re-selection among candidates whose boxed answers are consistent; it also supports an arithmetic-repair path in which a recomputed chain agrees with another candidate. Overrides are applied only when the top two candidates' score gap is within the margin. The gate never edits text — it only re-selects an index. On the frozen fusion, the gate nets +7 validation questions (13 repairs, 6 harms).

## Strict verification and the gap rule

The final component re-reads each derivation as a *program*: every equation line is parsed, both sides evaluated with exact rational arithmetic (fractions), operands are bound to prior lines, quantities are taint-tracked, single-variable linear systems are solved symbolically, and floor/ceil markup is checked. A candidate's trace is **verified** when its final answer is derivable from its own accepted equation chain. A contrastive logistic weight (C=1, liblinear, seed 80080) is trained over 65,536-dimensional hashed features of these verified traces on candidate pairs from training data, yielding a strict score per candidate.

Verification alone is informative but noisy, so it is used only as a *decisive-override* signal:

1. Let $b$ be the gated fusion choice and $s$ the strict scores. Let $a = \arg\max s$.
2. Flip the choice to $a$ only if all of: $s[a] - s[b] > \tau$ (gap exceeds threshold), $a$ is within the fusion's raw top-2, the gate did not override the argmax, and $a \ne b$.

With $\tau$ = 3.0 the rule flips exactly two validation rows: one repairs a genuine error (the only row whose strict advantage exceeds 3.0), the other is a both-correct flip that changes nothing; net +1 (759 $\to$ **760**), 0 harms. The threshold was designed on training folds with clean out-of-fold scores; the same configuration leaves the held-out fold at 1477 (unchanged), and the whole configuration is stable across a regularization grid (Section 3.5).

# Experiments

## Setup

Training data: 4,743 official training questions (12 excluded by convention: rows 34, 56, 1442, 1445, 1854, 2174, 2351, 2912, 4296, 4331, 4382, 4644). Fold 2 (1,582 questions) is never used to train the fusion or the strict scorer and serves as internal held-out sanity. Development decisions are made on training folds 0–1 and validated on the official validation set (798 questions). The official scorer parses the original candidate text by index only; the predictions therefore carry indices and nothing else.

## Main results

| Split | Correct | Total | Accuracy |
|:------|--------:|------:|---------:|
| Validation (official scorer) | 760 | 798 | 95.24% |
| Held-out training fold (internal) | 1477 | 1582 | 93.36% |
| **Test (official scorer)** | **391** | **467** | **83.73%** |
| Historical baseline test (2026-10-21 freeze) | 386 | 467 | 82.66% |

The frozen validation evaluation reproduces the development result exactly (760/798), and every check of the official harness passes (137 cases: interface, batch invariance, candidate-permutation invariance, empty/identical candidate handling, blocked reference reads on the prediction path, unchanged frozen hashes).

## Ablations

Cumulative contribution of each component of the fusion surface, each added on top of the previous (validation accuracy, gate included, margin 4):

| Configuration | Validation |
|:--------------|-----------:|
| Base expert ensemble + gate only | 752 |
| + question-context influence | 754 |
| + quantity-flow scores | 756 |
| + relation scores | 758 |
| + local structure scores (frozen fusion) | 759 |
| + strict gap rule, $\tau$ = 3.0 (final) | **760** |

The gate margin itself was swept over {0, 0.25, 0.5, 1, 2, 3, 4, 8}; margin 4 is optimal (val 759, held 1477), and larger margins do not help. The strict-score channel was also tried *inside* the fusion surface (as event features with and without ordinal binding, 755–758) — it helps more as a post-hoc override than as a fusion input, because fusion averaging dilutes the rare decisive signal.

Alternative model families on the same feature surface were all inferior to the frozen 200-tree 7-leaf LambdaRank model:

| Alternative | Validation | Held-out |
|:------------|-----------:|---------:|
| MLP fusion (cross-entropy, 4 configs) | 744–746 | 1472 |
| Two-stage pairwise flip model | 758 | — |
| Logistic flip model on top-2 features | 757 | — |
| 12-configuration LightGBM grid | $\leq$ 759 | $\leq$ 1478 |

## The gap rule in detail

Design evidence for the strict-gap rule (clean out-of-fold strict scores on training folds, restricted to top-2 and gate-first):

| $\tau$ | Design (n=3161) | Held-out (n=1582) | Validation (n=798) |
|--:|----------------:|------------------:|-------------------:|
| 2.0 | 2909 (+17/−8) | 1475 | 759 |
| 2.5 | 2913 (+7/−3) | 1476 | 760 |
| **3.0** | **2916 (+4/−0)** | **1477** | **760** |
| 3.5 | 2916 (+4/−0) | 1478 | 759 |
| 4.0 | 2916 (+1/−0) | 1478 | 759 |

$\tau$ = 3.0 is the only configuration with a positive validation gain and no loss on either training split; smaller $\tau$ trades fixes for harms, larger $\tau$ is inert on validation. Retraining the strict contrastive weight across a regularization grid (C $\in$ {0.1, 0.3, 1, 3, 10}, with C=1 reproducing the frozen scores bit-exactly) and re-running the rule sweep reproduces the same plateau: validation 760 with held-out 1477 appears for several (C, $\tau$) pairs, and every configuration that reaches 761 on validation costs 4–6 held-out questions — a clear over-fitting signature. Score-level and rule-level ensembles across C add nothing beyond the plateau. The rule's behavior is therefore not an artifact of one lucky threshold.

## Error analysis

The validation errors were inspected (39 rows at the pre-rule stage; the rule removes one, leaving 38). 25 of the 39 have the correct candidate ranked second by the fusion (so re-ranking alone has a theoretical ceiling of about 783/798); the top-1/top-2 score gaps on these rows are small (median 0.53), meaning the fusion is genuinely uncertain rather than confidently wrong. Seven errors are gate harms (the raw argmax was correct and the gate flipped it away); the gate is a net positive overall but its repair heuristic misfires on near-ties. The strict argmax itself is correct on 16 of the 39 rows, but the correct candidate's strict-score advantage over the chosen one exceeds the 3.0 threshold on exactly one row — which is exactly why a decisive-override rule with $\tau$ = 3.0 repairs one question and why verification scores cannot be used as a primary signal.

The validation–test gap (95.24% vs. 83.73%) is consistent with the test bank being harder by construction (test groups contain at least two different valid parsed answers, i.e., more genuine near-ties).

# Research questions

**Does reading the reasoning actually help?** Yes, but only through structure, not surface style. Style features (length, formatting, revision words) are part of the base surface yet the largest gains come from features that *re-derive* the reasoning: flow, relations and local structure together contribute +7 validation questions over the base ensemble, and the strict verification trace powers the final +1. The TF-IDF text member contributes modestly (weight 0.16). A wrong answer looks convincing mainly when its chain is internally consistent — so consistency features (verified chains, relation plausibility) discriminate better than fluency features.

**Should a solution be judged on its own?** No. Every comparative addition (context influence, quantity flow, relations, structure) improved the ensemble on top of per-candidate scoring, and the ranking model consumes the whole group jointly. The decisive-override rule also compares candidates through score gaps rather than absolutes.

**Can the method recognise its own uncertainty?** Partially. On hard rows the fusion's top-1/top-2 margin is small and correlates with error; but calibration of the strict score is weak (correct candidates often have near-zero or negative advantages), which is why only large gaps are trusted. The gate's 7 harms show that heuristic repairs on near-ties are risky.

**When does more computation pay off?** The final pipeline is cheap: 15.1 MiB of assets and 25.5 s end-to-end on test. The strict trace is the most expensive single step (~2 ms per candidate) and buys exactly one validation question through the rule — the smallest but only positive marginal improvement found after the fusion plateau; every more elaborate alternative (MLP fusion, pairwise stage-2, logistic flipper, score ensembles) either lost accuracy or returned the same plateau at higher complexity. The report records these negative results because they bound the method's uncertainty honestly.

# Resources and reproducibility

**Inference assets** (uncompressed, total 15,824,973 bytes = 15.09 MiB $\leq$ 16 MiB):

| Asset | Size | Role |
|:------|-----:|:-----|
| 8 base expert models (MLP weights + TF-IDF vocabulary) | ~12.0 MB | base scorers |
| meta.txt (LambdaRank fusion) | 214 KB | ranking model |
| space.npz / flow.npz / relation.npz / structure.npz | ~0.98 MB | comparative channels |
| strict.npz (contrastive verification weight) | 262 KB | strict score |
| config.json + member weights | < 1 KB | fusion weights |

**Runtime** (local machine: Intel Core i7-11800H, 4 threads, fresh process including startup, imports, and asset loading): test prediction 25.07 s + official scoring 0.46 s = **25.5 s** (validation prediction 36.9 s for 798 questions). The reference machine (Xeon Platinum 8457C) determines eligibility and is expected to be faster.

**Training cost** (CPU): each base MLP trains for 35–180 epochs on 4,743 questions in minutes; the fusion and comparative channels train in seconds to a few minutes (context 39 s, flow 207 s, relations 14 s + 199 s, local structure 91 s, strict weight 159 s). All experiments run under threadpool_limits(4).

**Seeds**: experts 20261014 / 25025 / 25026; PCA 79079; LambdaRank 68068; contrastive strict weight 80080; 3-fold split seed fixed in the shared harness.

**Dependencies**: Python 3.11; numpy, scipy, scikit-learn (liblinear/logistic), lightgbm 4.x, threadpoolctl. No external data, models, or APIs. Data and official scorer (protocol v5) unchanged.

**Commands** (from the project root):

```text
python research/round214_freeze.py            # rebuild the frozen bundle + validation cache
python research/feature_validation.py \
  --selection experiments/round214/final_feature_selection.json \
  --method research/round214_selector.py \
  --output experiments/round214/frozen --with-test
```

The frozen prediction interface, bundle (experiments/round214/bundle), official evaluation JSONs, and a full asset-hash manifest are in experiments/round214/frozen/. The selector reads only the question, candidate texts and fixed representations; no IDs, no lookup tables, no cached predictions.

# AI assistance and reuse disclosure

Substantive AI coding assistance (Claude Code) was used throughout development: design of features, training loops, ablations, and this report. All code was reviewed and validated against the official harness before freezing. Reused work: the relation features adapt the operation-relation supervision of Roy and Roth (2015); the data and scorer are the official DASE7506-MP2 distribution (GSM8K-derived, MIT license); candidate solutions were prepared by the course with Qwen3-0.6B. No other external weights or data were used.

# References

1. Roy, S., & Roth, D. (2015). Solving General Arithmetic Word Problems. *Proceedings of EMNLP 2015* (D15-1202).
2. Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems. *arXiv:2110.14168*.
3. DASE7506-MP2 official guide and evaluation harness (course distribution).
