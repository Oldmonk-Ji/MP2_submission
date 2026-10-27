---
title: "Verification-Aware Ranking with a Small Text Expert"
subtitle: "DASE7506 Advanced Machine Learning Methods - MP2 Final Report"
author: "Student ID: (to be filled)"
date: "27 October 2026"
papersize: a4
geometry: margin=2cm
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \setlength{\parskip}{3pt}
---

# Summary

The final system selects one of four supplied LLM solutions for each GSM8K question. It combines comparative ranking, a conditional quadratic expert, a small text network trained from random initialization, and conservative numerical consistency rules. It outputs a candidate index; it never rewrites a candidate or generates a replacement answer.

The frozen final method obtains **763/798 validation questions (95.61%)** and **394/467 test questions (84.37%)** under the unchanged official parser and scorer. The internal TRAIN-held result is **1481/1582 (93.62%)**. Required inference files total **16,686,526 bytes (15.9135 MiB)**. Fresh test prediction and official scoring take **27.59 seconds** on four Xeon Platinum 8352V CPU threads. These measurements are from the deployed server; they do not certify timing on the course reference Xeon 8457C.

The final candidate was selected using TRAIN/VAL before its final authorized TEST evaluation. All 89 frozen file hashes remained unchanged, and all 146 interface samples passed. Optimization stopped after this test, as requested.

# Task and data protocol

Each input consists of a question, four candidate solution texts, and four supplied 1024-dimensional representations. The official label is whether the unchanged numeric parser extracts the same final value from the selected candidate and the reference answer. Multiple candidates can therefore be correct. The system must still return exactly one index in $\{0,1,2,3\}$ for every question.

Only course-provided data are used. Final experts are fitted on the 4,743 eligible TRAIN questions, following the standing exclusion of twelve previously identified defective or ambiguous TRAIN rows. No validation or test question is excluded from evaluation. Three-fold cross-fitting supplies training predictions for downstream learning and development. The 1,582-question fold-2 score is computed with the corresponding fitting models excluding that fold; it is a repeatedly used development measure, not an untouched statistical test. Final expert fitting uses all eligible TRAIN questions.

VAL is a reused development and selection set. For the final candidate, VAL tied the numeric-only pipeline at 763 while TRAIN-held improved from 1479 to 1481. This comparison motivated a small text correction, not a claim of statistically established generalization improvement. The user explicitly authorized a final test despite the validation tie. The checkpoint, source, weights and selection were fixed before that run; TEST outcomes did not determine parameters or a subsequent method choice. Research did not inspect test references or per-question test errors.

\newpage

# Final method

## Base experts and comparative fusion

Eight base scorers use the supplied representations and deterministic evidence extracted from the question and candidate texts. They include an embedding MLP, numeric-grounding and answer-focused MLPs, a TF-IDF text scorer, and four additional MLP members. Their fixed weights are approximately 0.3071, 0.4946, 0.6196, 0.1596, followed by four weights of 0.125.

Comparative features describe each candidate relative to its alternatives. The 708-dimensional ranking surface includes base-expert outputs, representation context, conditional quantity-flow scores, arithmetic relation probabilities and local program-structure scores. The relation channel distinguishes addition, subtraction and its reversal, multiplication, division and its reversal. These channels measure evidence in a proposed derivation; they are not independent proofs that the word problem has been interpreted correctly.

A LightGBM LambdaRank model learns a correction to the weighted base score. Its fixed configuration uses 200 trees, seven leaves, at least 100 examples per leaf, learning rate 0.03, L2 penalty 50, truncation level one, and seed 68068. The model ranks candidates within their question instead of treating the four decisions as unrelated examples.

## Conditional quadratic expert

The quadratic expert uses the shared and differing parts of the supplied representations. For candidate $i$, let $e_i$ be its representation, $\bar e$ the mean of the four representations, and $r_i=e_i-\bar e$. PCA projections fitted on the relevant TRAIN partition produce 128 whitened residual coordinates and 32 whitened shared-context coordinates. Their values are clipped to a bounded range for stability.

The feature map contains residual coordinates, all residual/context products, upper-triangular quadratic products of the first 32 residual coordinates, and 82 standardized deterministic text/numeric/grounding features with row-centered counterparts. It also crosses the 32 shared-context coordinates with those 82 features. This final interaction allows the importance of a numerical check to depend on the semantic context of the question. The resulting map has 7,540 coordinates.

A regularized pairwise logistic model, C=1 and seed 227227, compares correct and incorrect candidates from the same TRAIN question. Pair weights are normalized per question. PCA, whitening and standardization are fitted only on the corresponding training partition. The final expert fits all eligible TRAIN questions and occupies 697,208 bytes.

## Scratch-trained text expert

The text expert receives the original question and reasoning texts, rather than the manually designed arithmetic features. Its 2,048-token vocabulary is fitted on TRAIN. Numeric occurrences are bound to matching quantities in the question, using question-relative quantity tokens; other numerical indicators describe their input magnitudes. There are no question-ID features or stored answer lookups.

Questions use at most 80 tokens and candidates at most 192. Longer inputs retain both an initial segment and a final segment, preserving question targets and candidate conclusions. The network has learned 24-dimensional word embeddings, a stride-two local CNN, layer normalization, and bidirectional question/candidate attention. The attention projection has 16 dimensions. Aligned vectors are compared using their values, differences and products, then projected to 32 dimensions. Mean/max pooling over both sequences yields a 128-dimensional candidate vector. Its concatenation with the difference from the four-candidate mean is scored by a 48-unit head.

Training starts from random initialization, with seed 241241, 60 fixed epochs, AdamW, learning rate 0.001 with cosine decay, weight decay 0.02, and an exponential moving average of 0.98. The objective combines within-question pairwise ranking, probability mass assigned to the correct candidate set, binary correctness loss, and a small in-batch question/solution matching auxiliary loss. That auxiliary uses supplied TRAIN text only. No external corpus or pretrained text weights are loaded.

The network has about 67,000 parameters. Its final weights and vocabulary require 164,256 bytes. Weights are stored in FP16 and evaluated with float32 arithmetic. Training uses PyTorch; inference is implemented in NumPy and does not require Torch.

## Score combination and consistency rules

Let $u_i$ denote the base-plus-LambdaRank score, $q_i$ the conditional quadratic score, and $t_i$ the text-network score. Before applying the consistency rules, the final score is

$$
s_i=u_i+0.25\,(q_i-\bar q)+0.01\,(t_i-\bar t),
$$

where each mean is over the four candidates. The text coefficient is deliberately small: the text network supplies complementary evidence rather than replacing the stronger numerical/comparative ranker.

The deterministic parse-and-repair policy proposes a candidate using parser alignment and arithmetic consistency. It accepts a proposed replacement only when its score deficit relative to the current winner is at most four. This policy selects an existing index and never edits the text or changes the official parser.

A strict verifier additionally parses numerical derivation traces, tracks bindings to prior quantities and scores local program compatibility. Exact rational calculations, supported symbolic equations and bounded floor/ceiling handling reduce extraction errors. Its contrastive TRAIN model produces a strict score $v_i$. Verification is evidence of internal consistency, not a guarantee that the story was translated correctly.

Let $b$ be the parse-and-repair result and $a=\arg\max_i v_i$. The strict rule changes $b$ to $a$ only when $a\ne b$, $v_a-v_b>3$, $a$ lies in the raw top two under $s$, and $b$ still equals the raw argmax under $s$. Otherwise it keeps $b$.

# Evaluation and interpretation

| Evaluation | Correct / total | Accuracy |
|:---|---:|---:|
| Official VAL | 763 / 798 | 95.61% |
| TRAIN-held development fold | 1481 / 1582 | 93.62% |
| Official final TEST | 394 / 467 | 84.37% |

The final candidate's fresh official validation predictions match its frozen development predictions exactly. The 146 interface samples cover batch consistency, candidate permutation at the parsed-answer level, unseen IDs, empty inputs and identical candidates. Prediction-time attempts to open reference files are blocked. All 89 frozen hashes are unchanged before and after the final test.

The text contribution has a modest development ablation: disabling it gives VAL763 and held1479; enabling its fixed weight gives VAL763 and held1481. This result supports retaining it as a small complementary signal, while emphasizing uncertainty about its marginal benefit. The final test observation did not trigger any further model selection or tuning.

The remaining errors include plausible but semantically incorrect interpretations, incorrect condition scope, and consistent arithmetic applied to the wrong quantities. Numerical trace consistency cannot eliminate these errors, and the small text model is not a complete mathematical reasoner. The validation score is a development estimate and should not be read as an unbiased forecast of test accuracy. Different candidate-bank geometries further limit direct comparison between the two percentages.

\newpage

# Resources and reproducibility

| Required asset group | Bytes |
|:---|---:|
| Base, fusion, verification assets and configuration | 15,825,062 |
| Conditional quadratic expert | 697,208 |
| Text expert weights and vocabulary | 164,256 |
| **Total required inference files** | **16,686,526** |

The total is measured directly from the final bundle, including configuration and file-container overhead. Archival checkpoint copies and research-only checkpoints are not loaded for prediction. The final deployment is CPU only, using four library threads. Fresh TEST prediction took 27.44065 seconds and official scoring took 0.15071 seconds, for **27.59135 seconds** including process startup, imports and checkpoint loading.

Inference dependencies are Python, NumPy, SciPy, scikit-learn, LightGBM and threadpoolctl. PyTorch 2.8 is needed only for neural training. Representative fixed seeds are 20261014 and 25025/25026 for base experts, 68068 for ranking, 80080 for strict verification, 227227 for the quadratic projection/ranker, and 241241 for the text network. The complete training configuration and frozen source closure are archived with the candidate.

With the provided course files placed in their original data directory and an activated inference environment, reproduce VAL from the repository root:

```text
python DASE7506-MP2-main/code/predict.py \
  --method experiments/round257/frozen/method/entrypoint.py \
  --checkpoint experiments/round257/bundle \
  --split validation --output remote-work/validation_predictions.json
python DASE7506-MP2-main/code/score_predictions.py \
  --predictions remote-work/validation_predictions.json \
  --split validation --output remote-work/validation_score.json
```

Expected VAL is 763/798. The canonical checkpoint is experiments/round257/bundle. Its immutable validation/interface certificate is experiments/round257/frozen. The separately authorized final TEST certificate is experiments/round257/official_test_20261027. The public repository contains the source, required checkpoint and evaluation provenance. No additional test rerun is needed for routine validation reproduction. Optimization is complete.

# AI assistance and reuse disclosure

Claude Code and OpenAI Codex provided substantive assistance with feature design, training code, inference implementations, experiments, packaging and this report. The data, fixed representations, candidate texts, parser and scorer are from the official DASE7506-MP2 distribution. Candidate solutions were supplied by the course. Arithmetic-relation supervision follows the operation-relation approach of Roy and Roth (2015). No additional external datasets, pretrained weights or inference APIs were used.

# References

1. DASE7506-MP2 official guide, data manifest and evaluation harness, course distribution.
2. Roy, S., and Roth, D. (2015). Solving General Arithmetic Word Problems. Proceedings of EMNLP, D15-1202.
3. Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.
