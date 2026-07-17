# Experiment record

This file keeps durable conclusions from one-off experiments. The training
scripts and large experimental checkpoints are intentionally not part of the
maintained codebase.

All results below used the same 871,433 training transitions, 46,993
validation transitions, and 46,751-transition held-out test set as the frozen
1x baseline. The baseline is the immutable `baseline` contestant used by
`benchmark.py`.

## Capacity scaling

| model | parameters | best validation loss | test top-1 | recall@6 | recall@16 | perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x, d256/L6/ff1024 | 5.8M | — | 0.057 | 0.199 | 0.342 | 142.921 |
| 10x, d640/L12/ff2560 | 61.9M | 5.3283 | 0.078 | 0.266 | 0.427 | 102.718 |
| 100x, d1536/L20/ff6144 | 573.4M | 5.8176 | 0.051 | 0.180 | 0.313 | 167.139 |

The 10x model demonstrated that capacity can materially improve the learned
prior. The 100x model overfit this dataset: training loss continued to fall
while validation loss rose after epoch 3, and its held-out policy metrics were
worse than the 1x baseline. Future scaling work should start near 10x and
spend additional compute on data quality/quantity, regularization, or search
evaluation rather than raw parameter count.

## Damage-token ablation

All variants used the 1x transformer unless noted.

| input/model | best validation loss | validation top-1 |
| --- | ---: | ---: |
| damage suffix replaced with `DMG_UNK` | 5.4909 | 0.062 |
| damage suffix removed | 5.3659 | 0.065 |
| suffix removed, 5.8M residual MLP | 5.3579 | 0.066 |

Removing the damage suffix did not harm validation policy accuracy and reduced
transformer epoch time from roughly 230 seconds to 107 seconds in these runs.
This suggests the expensive damage matrix was not providing useful supervised
policy signal in the current representation. The residual MLP trained in
roughly 4 seconds per epoch after compilation, but it was not evaluated in
search. Treat it as evidence that architecture/representation efficiency is
worth testing, not as a replacement agent.

## Baseline identity

- Checkpoint SHA-256: `9685d185c7f30166eebbc0f3d62beaf8660783aef7fc3eb1b48b861d26533138`
- Vocabulary SHA-256: `62e8ec965085814f088b43540929e3ac26a07bdd67b20780bc0bb9c76023b96f`
- Architecture: joint policy head, tokenizer layout 3, 561 tokens
- Frozen source commit: `662211d`

## Value-head brick swap (exp/value-head) — pile-only lane

**What changed.** The baseline reads its leaf value from a single
`Linear(256,1)+tanh` on the CLS token, trained jointly with the policy by MSE
against the ±1 final outcome. This experiment keeps the base model and its
policy/aux outputs bit-identical (same trunk forward, same heads) and swaps
only the value scalar, so any strength change is attributable to the value
brick alone. Candidates (`value_lab.py`, architectures in
`models/value_heads.py`): an MLP on CLS; a learned-query attention-pooling
head over all 561 token states (the value path sees mon blocks, belief
tokens, and the damage matrix directly instead of through the CLS
bottleneck); the same pooling head trained with the baseline's tanh+MSE loss
(loss-vs-pooling ablation); and a dedicated value network (baseline-initialized
trunk fine-tuned end-to-end on the value objective — one extra forward per
leaf). The default loss is cross-entropy on the win logit ("value as
classification"): tanh+MSE has vanishing gradients exactly on
confidently-wrong predictions, and CE calibrates better, which matters because
search mixes leaf values with exact ±1 terminal values. Training adds two
sidecar auxiliary targets (end-of-game faint and HP-sum differentials,
`value_labels.py`, aligned to the existing shards with recomputed
weight/outcome proofs — shared shards untouched), and the selected brick gets
a post-hoc calibration temperature fitted on the validation split (monotone:
changes calibration, not ordering).

**Shared files touched:** none with logic changes (pile-only; `agent_server.py`
gained the `search-vh` kind and `cli_help.py` two help entries — neither is a
behavior-hashed file). Selection is on validation Brier; the test split
(46,751 transitions, same fixed splits as above) is reported once.

**Numbers (fill from `value_lab.py eval` on the training box):**

| value brick | test Brier | test sign acc | test AUC | test ECE |
| --- | ---: | ---: | ---: | ---: |
| control (baseline head) | TBD | TBD | TBD | TBD |

**Search-strength result (fill from the round robin):** exported as
`exp-value-head` (`--agent search-vh`); quick-10 vs the anchor, then a full
series. Offline value metrics without a search result are reported as such.

