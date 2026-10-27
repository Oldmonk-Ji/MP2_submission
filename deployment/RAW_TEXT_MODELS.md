# Independent raw-text model research

These experiments train new text networks from random initialization. They
use only the supplied TRAIN questions, candidate reasoning and official
TRAIN correctness labels. No external corpus, tokenizer, pretrained weights
or inference API is used. They are research checkpoints, separate from the
selected round235 submission bundle.

All vocabularies fit the corresponding training partition. Numeric strings
are bound to their positions in the question, so repeated quantities can be
compared across question and solution text. The network never receives a
question ID. Shared candidate encoders and group pooling preserve candidate
permutation invariance.

| Experiment | Architecture / training |
| --- | --- |
| 241 | 2048-word learned embeddings, local CNN, bidirectional question/candidate attention, group comparison; 60 supervised epochs |
| 242 | Same architecture, preceded by six masked-word epochs on fitting TRAIN text only |
| 243 | Same locally pretrained architecture, trained to correct cross-fitted TRAIN rankings |
| 244 | Explicit adjudication under the original user held floor1477, replacing the extra conservative1479 research filter before any VAL evaluation of these networks |
| 245 | New joint two-layer Transformer48, four attention heads, FF96, local stride4 encoder; masked-word/number-binding pretraining plus40 supervised epochs |

Transformer inputs have64 question tokens and160 candidate tokens. Learned
word embeddings are stored as int8 with per-row FP16 scales; other parameters
use FP16. The earlier network stores all parameters in FP16. Decisions use
the exported, quantized weights, not an unexported higher-precision model.

Training uses PyTorch2.8 in the isolated training environment, four CPU
threads, no GPU. Inference implementations are NumPy only. Numerical export
checks compare Torch and NumPy, batch invariance and candidate permutations
using TRAIN inputs. A small runtime check is a TRAIN-prefix timing proxy,
not an official TEST evaluation.

Run scripts from research/ with the dedicated training Python, for example:

    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 ../../.train-venv/bin/python round245_train.py

The exact source hashes, seed, fixed epoch count, all blend coefficients and
results are recorded in each experiments/roundNNN/ directory. Research data
access rejects every TEST input/reference path. New candidates require VAL
above763 and held at least1477, followed by resource/interface checks and a
separate user authorization for one official TEST evaluation.

## Outcomes

Initial independent held accuracy was1306/1582 (241),1307 (242) and1335 (245).
243 produces a residual correction, so its raw argmax is not a standalone
selector. Initial blending did not clear the user floor except241/alpha0.1,
which reached held1478 and VAL759 after all-TRAIN fitting.

Round246 then selected weaker score scales on TRAIN-held only. Of24 fixed
configurations, only241/raw0.01 showed strict held improvement:1481/1582,
two fixes and zero harms versus round235. Its VAL remained763/798. No new
method was selected or tested. Full results and checkpoints are archived in
experiments/round241–246; summary: experiments/round246/continuation_summary.json.

The exported attention network costs164,305 bytes; the Transformer costs
241,071 bytes. Trained-model Torch/NumPy maximum score differences were
below1e-5 on all1582 TRAIN-held questions. A467-question TRAIN-held prefix
took about0.8 seconds for the attention network and1.31 seconds for the
Transformer alone. These are component timing proxies, not full official
TEST timings. The selected round235 bundle remains unchanged.
