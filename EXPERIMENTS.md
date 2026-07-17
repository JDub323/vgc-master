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

## JEPA world model + latent matrix-game planner (exp/jepa-world-model)

**Lane: pile-only.** New feature layout, new action-selection algorithm, no
reuse of the layout-3 tokenizer or the policy/value transformer. Shared files
touched: **none** of the ten `BEHAVIOR_SOURCE_FILES`. The only edit outside the
experiment's own new files is one additive `if kind == "jepa"` branch in
`agent_server.build_chooser` (not a behavior file). Architecture label
`JEPA-WorldModel-MatrixSolve`; agent kind `jepa`; chooser
`agents.jepa_world_model.v1.JEPAWorldModelChooser`.

**What it is.** A learned latent one-ply world model replaces determinized DUCT
tree search. Per decision it encodes the position into 16 role-typed entity
latents (1 global + 6 ally + 6 foe + 2 opponent-intent + CLS) with a
transformer whose Q/K/V/O projections are *per role* (ally vs foe are different
maps) and whose attention is biased by an ally→foe damage edge. An
action-conditioned predictor maps `(latents, my joint action, opponent joint
action) → predicted next-state latents`, a value head reads a win probability
off each, and the resulting payoff matrix — averaged over belief
determinizations — is solved by regret matching into a **mixed strategy**. The
model is trained on human-replay transitions with a JEPA latent-matching loss
against an EMA target encoder, grounded next-state decoders (HP/faint/status/
field), value regression, opponent/my policy heads, and VICReg (four independent
anti-collapse mechanisms). Full design: `JEPA_DESIGN.md`.

**What moved.** Only a laptop-scale *pipeline validation* has been run, not a
full-scale fit: a 150-battle prep (`jepa_data.py --limit 150`, no damage edges)
→ 1,086 train / 98 val / 58 test transitions, then 4 epochs on CPU. Every loss
term fell monotonically (val total 13.88 → 10.41; value MSE 1.82 → 1.00; JEPA
latent ~0.57; my-action top-1 0.5% → 2.9%), the checkpoint round-trips into the
chooser, and the chooser emits a legal joint action with a solved mixed strategy
and valid `ChoiceInfo`. **These numbers certify the machinery, not strength** —
they are not comparable to the frozen 871,433/46,993/46,751 splits or to Elo.
No search-time (Elo) evaluation has been run yet; like the damage-ablation and
MLP rows above, treat the offline curve as "wired and learning," nothing more.

**Gates.** `tests/test_jepa.py`, `tests/test_documentation.py`, and
`tests/test_agents.py` pass. `scenarios.py` could not be run **on the dev
laptop only**: its damage bridge needs `@smogon/calc`, which is not installed in
this box's `artifacts/node` — the same failure the trunk DUCT agent hits here,
unrelated to this experiment (it touches no behavior files and `scenarios.py`
never imports the JEPA code). It should pass unchanged on the coordinator box.

**Open follow-ups (one blade each, do not widen this one).** Full-scale prep +
train on a big box and an Elo series vs `baseline`; feeding reverse (foe→ally)
damage edges; multi-ply latent unroll (`JEPAConfig.unroll` already supported in
the loss path); and a learned team-preview/lead head (currently first-four).

## Baseline identity

- Checkpoint SHA-256: `9685d185c7f30166eebbc0f3d62beaf8660783aef7fc3eb1b48b861d26533138`
- Vocabulary SHA-256: `62e8ec965085814f088b43540929e3ac26a07bdd67b20780bc0bb9c76023b96f`
- Architecture: joint policy head, tokenizer layout 3, 561 tokens
- Frozen source commit: `662211d`

