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

## Team preview + forced switch-ins (exp/lead-switch, pile-only)

**Lane: pile-only.** Shared behavior files touched: **none** (the experiment
knobs live in `agents/lead_switch/lscfg.py`, not `config.py`; the frozen
chooser, model, and search are byte-identical to trunk).

**What changed.** The stock adapter answers team preview with `team 1234`
and forced switches with a random legal pick. `lead_switch_server.py`
subclasses it and routes exactly those two request kinds through selectors
(`agents/lead_switch/`): `expert` (damage-calc matchup scoring over all 90
bring/lead combos vs a softmax model of the opponent's 15 lead pairs, plus
Gen-IV-style switch-in scoring), `value` (the frozen baseline value head
evaluating hypothetical post-decision positions, expert-pruned, scored
mean/min over opponent leads), and `nn` (LeadNet, a ~1M-param transformer
imitating human preview choices, outcome/rating-weighted). Move requests are
untouched, so any Elo delta against the plain baseline bundle isolates
positioning alone.

**Method notes.**
- Team preview is simultaneous, so the value variant scores
  `0.5*mean + 0.5*min` over the opponent's expert-ranked lead pairs rather
  than best-responding to a single assumed lead.
- The value head cannot see which four are brought (turn-1 states encode
  leads only), so the back pair always comes from the expert coverage term;
  the payoff matrix only ranks lead pairs.
- Forced-switch hypotheticals need no sim stepping — switching only moves
  mons between slots — so a decision is one batched net call.
- Dataset fact (train split): humans led with their first two team-sheet
  mons in only a small minority of games (printed by `train_leads.py` as the
  `team 1234` floor), so the stock adapter plays a lead humans avoid in most
  games.

**LeadNet offline numbers** (same match-id splits as the frozen baseline;
one example per game per perspective, not per transition):
_pending — filled by `python train_leads.py` (val/test lead-pair top-1 and
top-3 vs the human choice, with the first-pair floor for context)._

**Elo** — the metric that matters — requires the big box (`ckpt_best.pt`):
export `rr-baseline` + the three variants and run 100-game series per
LEAD_SWITCH_RUNBOOK.md. _Results pending; offline metrics above are NOT
evidence of playing strength._

## Baseline identity

- Checkpoint SHA-256: `9685d185c7f30166eebbc0f3d62beaf8660783aef7fc3eb1b48b861d26533138`
- Vocabulary SHA-256: `62e8ec965085814f088b43540929e3ac26a07bdd67b20780bc0bb9c76023b96f`
- Architecture: joint policy head, tokenizer layout 3, 561 tokens
- Frozen source commit: `662211d`

