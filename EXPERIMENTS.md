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

This experiment has **two variants**. The **consequence variant (`jepa-c`)** is
the intended architecture; the **next-state variant (`jepa`)** was the first cut
and is kept because it is built and runnable.

### Consequence variant (`jepa-c`) — the intended architecture

**What it is.** For the current position and each legal OWN joint move, a
predictor outputs a single latent **consequence vector** that summarizes the
distribution over what happens after the opponent responds and chance resolves —
never decoded to an explicit state, no opponent-action axis, no matrix game. It
is trained JEPA-style: the taken move's consequence vector is matched (smooth-L1)
to an EMA target-encoder's embedding of the realized future position, so — since
one (position, move) maps to many futures — it learns the *expected* future
embedding and thereby the engine/opponent/luck dynamics implicitly. A policy
head ranks the candidate consequence vectors (behavior cloning on the human
move), a value head reads win probability off them (real outcome), and VICReg +
the EMA stop-grad prevent collapse. An optional luck latent `xi` lets the vector
represent the spread (default deterministic). Shares the role-typed encoder,
feature layout, and vocab with the next-state variant. Files:
`models/jepa_consequence.py`, `agents/jepa_world_model/v2.py`,
`train_consequence.py`, plus `jepa_data.py --consequence`.

**What moved (laptop pipeline validation only).** 150-battle consequence prep →
1,383 train / 120 val / 87 test transitions (each carries the future position
and up to 12 legal own-move candidates), 5 CPU epochs. The **JEPA latent loss
fell 0.571 → 0.129** (the predictor is genuinely learning to predict future
consequence embeddings), value MSE held ~1.0 (≈predicting 0 on this tiny set),
and the **policy head's validation top-1 rose 0.508 → 0.575**. The trained
checkpoint round-trips into the `jepa-c` chooser, which ranks ~60 candidate moves
by predicted consequence and returns a legal action with valid `ChoiceInfo`.
These certify the machinery, **not strength** — not comparable to the frozen
splits or to Elo, and no search/Elo evaluation has been run. (The larger
train/val policy-accuracy gap is dropout plus variable candidate counts on a
tiny set.) A simulated-counterfactual target path (replay a position, step the
env sidecar with different opponent moves/seeds to get many futures per move)
is **designed but not implemented** — it needs the Node sim, absent here.

**Bug found and fixed (candidate-set degeneracy).** The first full train reached
val policy top-1 0.917 but the exported agent lost 0-10 to `baseline` at
0.02 s/move. Cause: the behavior-cloning candidate set in prep was
`[human_move] + first-11-other-legal-joints`, and `legal_my_joints` listed
switches first — so the negatives were almost all *switches*. The policy head
only learned "human's attack vs a pile of switches" (trivial — hence the
implausible 0.917 vs the repo's ~0.06 joint top-1), never learned to rank *which*
attack/target, so its consequence vectors were nearly action-insensitive and at
temperature 0 it deterministically played the first legal joint (a double-switch)
every turn. Fixes: (1) prep now samples a *shuffled, diverse* subset of the full
legal own-joint set as negatives; (2) `legal_my_joints` uses each move's real dex
target semantics so candidates match play-time legality; (3) the chooser passes
`brought=None` to match prep and uses the model's stored config, and
`train_consequence` records whether the shards carried damage so the chooser
never adds off-distribution damage edges. After the fix, val top-1 drops to a
realistic ~0.22 (chance 0.083) on a 1,383-transition smoke, the predictor is
action-sensitive (distinct consequence vectors per move), and the agent plays
varied moves instead of a fixed double-switch. **A full re-prep + re-train is
required** — the old shards baked in the degenerate candidates.

**Tournament result + the scale/self-play iteration (2026-07-18).** The fixed
full-scale jepa-c (~8M params, trained on 896,649 human transitions) placed
~5th on the box leaderboard at ~0.01 s/move — roughly DUCT-baseline
strength at ~300x speed. Since it is BC-trained, that is about the imitation
ceiling; the next iteration targets *surpassing* it with two additions, both
implemented on this branch:

1. **~6x scale** (`jepa/config.scaled_consequence`, `train_consequence.py
   --large`): d448 / 5+3 role-typed layers / ff1792 ≈ 50M params (tested to
   land in 30–90M). Decision cost stays milliseconds — 16 tokens x ~60
   candidates.
2. **Outcome-driven self-play** (`selfplay_jepa.py`). Design rationale: jepa-c
   never touches the sim per decision, so self-play generation is *sim-bound* —
   the box that fed DUCT ~150 games/iter can feed jepa-c thousands. The loop is
   engineered against the known failure modes: advantage-weighted policy CE
   (weight `clip(exp((z - b)/beta))`, baseline = mean candidate value) instead
   of self-imitating BC; a league (mirror / past checkpoints / frozen anchor)
   against strategy cycling; a capped human-BC data mix against meta drift;
   temperature + eps-uniform exploration in generation only; teams sampled from
   the ~3k validated pool against pairwise memorization; samples recorded from
   the chooser's own plan (train=play identity — the bug class that sank the
   first export); and an argmax gate vs `spj_best` with promotion at >=55%.
   The JEPA latent loss continues on realized on-policy futures (last decision
   of a game has no future and is masked). Training half is dry-run-verified
   end-to-end on a recorder-built buffer; the generation half needs the sim
   and is untested on the dev laptop (no pokemon-showdown install) — first run
   on the box should start with `--iters 1 --games 40` as a smoke.

**Self-play regression + v3 stage 1 (2026-07-19).** 240+ iterations of the
v2 self-play loop made the agent *worse* externally: 38% -> 15% vs `bermuda`
(elo -85 -> -301) while the internal gate promoted steadily. Post-mortem in
`JEPA_V3_DESIGN.md` §1 (no improvement operator; whole-game credit; noisy
self-referential gate ratchet; 5-turn game collapse visible in samples/games).
Stage 1 of the redesign is implemented on this branch: recursive latent
dynamics `T(Z, a, b)` (`models/jepa_strategy.py`), multi-step JEPA training on
consecutive-transition windows (`jepa_data.py --seq`, `train_strategy.py`),
and the `jepa-s` chooser (`agents/jepa_world_model/v3.py`) — a depth-1 latent
matrix solve with depth-2 recursion available (`plan_depth`), the future
generation-time improvement operator. Laptop-validated (unit tests + smoke
prep/train below); big-box train + anchor series pending. Stages 2-4 (TD(λ) +
distributional value + SPRT anchor gate; trajectory encoder + strategy
bottleneck; what-if targets + exploiters) are specified in the design doc.

**Stage-1 full-scale result and the stage-2 fix set (2026-07-19).** The first
big-box `jepa-s` train (215k windows, 6 epochs) scored **18% vs bermuda** —
worse than v2's 38%. Three diagnosed causes, all visible in the training log:
(1) the JEPA dynamics loss ROSE every epoch (0.24 -> 0.34) because the summed
39-way policy CEs dominated the shared encoder ~20:1; (2) the value head
(MSE plateau ~0.8, barely better than predicting 0) was the *sole* decision
criterion — the matrix solve amplified value noise while the decent policy
head was demoted to candidate pruning; (3) seq data was ~4x thinner than v2's
(both-actions requirement + chain breaks on the ~24% unobservable-slot turns
+ stride 3). Fixes landed on this branch: **unknown-action marginalization**
(`AK_UNK` action kind — unobservable turns no longer break chains; the
dynamics learns the marginal transition, policy CE masks those steps via
`a_mask`/`b_mask`), **prior-anchored equilibrium** (`solve_matrix_anchored`:
p ∝ prior·exp(eta·Mq); eta->0 plays the BC prior, eta->inf plays Nash on the
value matrix — the value head only moves decisions where its differences beat
its noise floor), **encoder-trunk ownership** (`policy_detach=True`: policy
heads read a detached z0; per-head-mean masked CE), **distributional margin
value** (9-bin final-mon-differential CE, winner-consistent labels; scalar =
expectation), and **densified windows** (stride 1 + unbroken chains).
Requires re-running `jepa_data.py --seq` and `train_strategy.py --large`
(shard schema + head shapes changed).

**Stage-2 full-scale result (2026-07-20): still broken — overcorrection
diagnosed.** 1.12M windows trained cleanly (jepa 0.39 -> 0.23 falling, the
stage-1 pathology fixed; margin CE 1.25 -> 0.84) but the exported agent
started ~1-for-17 vs bermuda, losing across all matchups incl. mirrors —
systematically degenerate play. Cause: the full policy detach starved the
heads (a single Linear on frozen features; my_acc 0.14 vs stage-1's 0.23), so
BOTH the candidate pruning and the prior anchor went mushy, while the
now-confident value head steered `exp(4·Δv)` with its **off-policy bias** —
the matrix queries 36 counterfactual (a,b) pairs the value was never grounded
on. Fixes: (1) `policy_grad_scale=0.1` (gradient-scaled trunk access, not
detach) + MLP policy heads over CLS+pooled side summaries; (2) the anchored
solve now **standardizes** the payoff matrix per decision, so `solver_eta`
(default 1.5) is in units of that decision's own value spread — confidence
alone buys no influence; (3) `probe_strategy.py` measures the value head's
counterfactual ranking (v(T(z,a_true,b)) vs swapped own-actions) *before*
games: ~0.5 = no off-policy signal, keep eta near 0 and play the prior;
run it after every train, and pass `--eta`-adjusted exports accordingly.
The durable lesson for the record: a payoff matrix built from a value head
that was only ever supervised on the human-taken action is exploitable by
construction — counterfactual (what-if sim) grounding is not an optional
stage-3 nicety, it is what makes value-led decisions legitimate.

**Stage-3 result and the pruning-ceiling diagnosis (2026-07-20).** With the
grad-scale/standardized-solve fixes, the full retrain looked healthy on every
internal metric (jepa 0.39->0.22 monotone, my_acc 0.20, probe counterfactual
ranking 0.745, calibration correctly signed) — and still scored **18% vs
bermuda**, identical to broken stage 1. That coincidence is the diagnosis:
stage 1 (pure Nash on value) and stage 3 (anchored standardized eta=1.5) share
only one decision component — **top-6 own-candidate pruning through a
per-slot prior with top-1 ~0.21** — while v2, which scored the FULL legal set
every turn, got 38% from the same data. A 6-item menu that omits the human's
move roughly half the time is a ceiling no solver, value head, or eta can fix.
Change: `top_k_mine_s` 6 -> 64 (score all legal own joints; 64x6 = ~384
batched T applications, still ~0.02s/move), play-time only — **no retrain
required**. `probe_strategy.py` now also reports `candidate_coverage_top6/16`
(fraction of positions whose prior-ranked top-k contains the human joint), so
this failure class is measured before any series; validated against a
random-init model (coverage@6 ~= k/n_joints, cf ranking ~= 0.5).

**Stage-4 result: uncapping didn't help — the decision rule, not the menu,
is the bottleneck (2026-07-20).** `exp-jepa-s4` (same l3 checkpoint,
`top_k_mine_s=64`) scored **14% vs bermuda** (elo -315), statistically the
same as the top-6 runs, while the probe confirmed the coverage ceiling was
real (coverage@6 = 18%, @16 = 34%). Both facts together isolate the cause:
the **factorized per-slot prior cannot joint-rank** (per-slot top-1 0.19 ⇒
joint top-1 ~0.04; the human joint isn't in its top-16 two-thirds of the
time), and widening the menu to 64 only worsened the optimizer's curse —
argmax over more noisy-biased value estimates through a diffuse prior. This
is the head design the trunk repo already abandoned for a joint head in
phase 3; meanwhile v2's 38% came from scoring each candidate through a full
action-conditioned forward pass (candidate CE). Fix (this branch): **BC
scoring heads on the dynamics itself** — `score_head`/`opp_score_head` read
CLS of `T(z, action, other-side=AK_UNK)` ("my move's consequence, opponent
marginalized" = v2's primitive re-expressed through v3's dynamics), trained
with v2-style candidate CE against sampled legal negatives synthesized
on-the-fly from the stored window tensors (`jepa/candidates.py`, disk-cached
per shard under `<data>/negcache/` — no re-prep). At play time
(`prior_from_score=True`) these joint action-conditioned scores replace the
factorized prior in the anchored solve, so **eta=0 is functionally
v2-inside-v3** (BC argmax over the full legal set, expected ~38%) and any gap
isolates a v3-specific bug; the eta sweep upward then measures exactly what
the value matrix adds over BC play. `probe_strategy.py` now reports
`score_top1/3/6_joint` (joint-ranking quality of the score head over the
reconstructed full menu). Needs one big-box retrain (`train_strategy.py
--large`, shard schema unchanged); run the eta=0 export as the FIRST series,
then sweep eta 0 / 1.5 / 3.

### Next-state variant (`jepa`)

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
`tests/test_agents.py` pass. `scenarios.py` and the `round_robin.py` quick
series could not be run **on the dev laptop only**: this box's `artifacts/node`
has the Node runtime but not the `pokemon-showdown` sim or `@smogon/calc`
package, so any real game fails at `sidecar.js` / the damage bridge. That is a
whole-repo environment gap (the trunk DUCT agent hits the identical failure
here), unrelated to this experiment — it touches no behavior files and
`scenarios.py` never imports the JEPA code. Both should run unchanged on the
coordinator box, which supplies its own Node/sim via `$VGC_NODE_DIR`. The
exported bundle builds and plans correctly through the real `agent_server`
adapter (verified up to the point a live battle would feed protocol lines).

**Bundle.** `exp-jepa-wm-smoke` exported to the pile (dev box:
`../vgc-pile/exp-jepa-wm-smoke`, entrypoint
`python agent_server.py --agent jepa --ckpt artifacts/checkpoints/ckpt.pt`).
It carries the laptop-smoke weights; re-export after a full-scale train. To
collect it for the round robin, `rsync -a` the bundle into the coordinator
box's pile.

**Open follow-ups (one blade each, do not widen this one).** Full-scale prep +
train on a big box and an Elo series vs `baseline`; feeding reverse (foe→ally)
damage edges; multi-ply latent unroll (`JEPAConfig.unroll` already supported in
the loss path); and a learned team-preview/lead head (currently first-four).

## Baseline identity

- Checkpoint SHA-256: `9685d185c7f30166eebbc0f3d62beaf8660783aef7fc3eb1b48b861d26533138`
- Vocabulary SHA-256: `62e8ec965085814f088b43540929e3ac26a07bdd67b20780bc0bb9c76023b96f`
- Architecture: joint policy head, tokenizer layout 3, 561 tokens
- Frozen source commit: `662211d`

