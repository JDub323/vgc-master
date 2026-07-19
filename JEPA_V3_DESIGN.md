# Strategy-JEPA (v3): hierarchical latent planning for VGC

Design document for the third-generation JEPA agent. Motivated by the v2
self-play result: 240+ iterations of outcome-weighted self-play made the agent
*worse* against external opponents (38% -> 15% vs `bermuda`) while its internal
gate promoted steadily. This document (1) diagnoses that failure, (2) catalogs
every complexity the model must represent — especially long-horizon strategy,
(3) proposes solutions including deliberately unexplored ones, and (4) specifies
the v3 architecture: a hierarchical JEPA with a recursive latent dynamics core,
a strategy bottleneck, and a self-play loop built around a genuine policy
improvement operator.

---

## 1. Post-mortem of v2 self-play

Observed: gate hovered 45–70% (40 games, ±16% noise) with promotion at 55%;
`spj_best` ratcheted on noise. Games collapsed to ~5 decisions/side. Policy
self-consistency hit 0.87 with only eps=0.03 exploration. `adv_w` pinned near
1.78 — the mean-candidate-value baseline stayed ~0, so weighting degenerated to
"reinforce everything in wins, suppress everything in losses" (whole-game
REINFORCE, no per-move credit). JEPA loss plateaued: the state distribution
stopped being novel. External Elo fell ~200 while internal winrate "improved."

Root causes, ranked:

| # | cause | class |
|---|---|---|
| 1 | **No improvement operator**: training target = the raw policy's own choices, weighted by noisy outcomes. Nothing in the loop plays *better* than the current policy, so there is nothing better to learn from. AlphaZero's engine is `search > policy`; we had `policy ≈ policy`. | fatal |
| 2 | **Whole-game credit assignment**: one z for every decision; flat baseline. Winning blunders reinforced, losing brilliancies punished. | fatal |
| 3 | **Self-referential gate**: 40-game noise + ratchet + no external anchor = random walk labeled progress. | fatal |
| 4 | **Distribution collapse**: near-deterministic mirror games, 5-turn slugfests, JEPA/world-model starved of novelty. | severe |
| 5 | Human mix (25%) too weak an anchor at 240 x 2 epochs of drift. | moderate |

Every one of these is addressed structurally in §4, not by knob-tuning.

---

## 2. The complexity catalog: what a VGC agent must monitor

### 2.1 Turn-local (mostly covered by v2's features)

- **Damage pipeline**: types/STAB/items/abilities/weather/terrain/screens/
  spread-reduction/crits/rolls (covered via the damage-edge matrix).
- **Speed order under modifiers**: boosts, paralysis, Tailwind, Trick Room,
  Choice Scarf inference, priority brackets, mega-evolution speed-change
  timing (partially covered: scalar spe-ranges only — no pairwise order).
- **Targeting & redirection**: Follow Me / Rage Powder, Ally Switch mind
  games, spread vs single-target choice.
- **Protection layer**: consecutive-Protect odds, Fake Out pressure windows,
  Wide/Quick Guard coverage.
- **Immunity/absorb topology**: Lightning Rod, Storm Drain, absorb abilities,
  Levitate — which attacks are free vs punished.

### 2.2 Multi-turn strategic (largely NOT represented in v2)

1. **Win-condition identification and stewardship** — which surviving mon(s)
   close the game; keeping the wincon healthy; sacrifice sequencing to buy its
   setup; recognizing the *opponent's* wincon and denying it.
2. **Field-effect cycles as resources** — weather/terrain/Trick Room/Tailwind
   are 3–5–8 turn windows; strategy = setting windows, spending them
   efficiently, stalling out the opponent's (Protect through TR turns),
   fighting for the last word on weather (the "weather war").
3. **Tempo / momentum** — pivot cycling, keeping favorable matchups on the
   field, forcing the opponent to react; who is dictating pairings.
4. **Resource ledger across the game** — HP-as-resource, one-shot items
   (Sash/Berry/Herb), the single mega, Protect availability, the option value
   of unrevealed moves; spending vs conserving.
5. **Positioning / pairing control** — which 2v2 sub-battle is on the field;
   back-mon preservation; steering toward winnable endgames (endgame counting:
   2v2/1v2 lethal arithmetic several turns deep).
6. **Setup-vs-pressure timing** — when a setup turn (TR, Tailwind, stat boost)
   pays for its tempo cost; punish windows after the opponent's setup.
7. **Damage sequencing across turns** — chip now so priority finishes later;
   splitting damage to put two targets in KO range of one spread move.
8. **The prediction stack** — double switches, Protect-on-the-predicted-focus,
   Sucker Punch equilibria; being deliberately mixed at exploitable nodes.
9. **Information warfare** — what have I revealed vs what they revealed;
   playing around unrevealed coverage; *bluffing* sets you don't have;
   updating on the opponent's reveals mid-game; managing what my own moves
   teach the opponent.
10. **Risk posture as a function of score** — variance-seeking when behind
    (fish for the crit/miss line), variance-minimizing when ahead (the safest
    98% line beats the flashiest 85% line); explicit awareness of *which*
    distribution over outcomes each plan induces, not just its mean.
11. **Status/ability clocks** — burn/poison accumulation, sleep counters,
    Supreme Overlord stacks, Intimidate cycling on re-switches, weather
    re-trigger on switch-in.
12. **Team-preview planning** — bring-4 and lead selection *is* the game plan:
    archetype matchup (rain vs sun vs TR vs balance), assigning a wincon
    before turn 1, adapting when scouting contradicts the plan.
13. **Within-game opponent modeling** — this specific opponent's tendencies
    (Protect frequency, aggression, switch patterns) — exploitable inside one
    game and across a bo3.

### 2.3 Why v2 could not represent most of §2.2

v2 sees **one position snapshot**. Cycles, tempo, information history, risk
posture, and opponent tendencies are functions of the *trajectory*, not the
position. Durations were reserved-but-zero features. And the consequence
vector is one step deep: strategy lives at horizon 3–10, which v2's target
never reached.

---

## 3. Solutions, including deliberately unexplored ones

Mapped to the failures/complexities above. (U) = unexplored/creative bets.

1. **Recursive latent dynamics** (fixes: improvement operator, depth).
   Replace the terminal consequence vector with a latent transition the model
   can *apply repeatedly*: `T(z, a, b) -> z'` at the full entity-token level,
   JEPA-trained at multiple horizons. One-ply speed is preserved (heads can
   read one application) but planning can now recurse — the MuZero insight,
   with JEPA targets instead of reward-model reconstruction.
2. **Latent matrix-tree search as the improvement operator** (fixes: cause 1).
   At generation (and optionally at play), run a depth-2/3 search entirely in
   latent space: at each node solve the one-ply payoff matrix (v1's regret
   matching) over top-k own x top-k opponent actions, recurse through `T`.
   Sim-free, so ~milliseconds. Train the policy toward the *search's* mixed
   strategy (distillation) — search > policy is restored, so self-play has an
   actual gradient toward strength.
3. **TD(λ) per-decision credit** (fixes: cause 2). Value target = λ-mixture of
   n-step bootstrapped returns off the EMA target net; policy advantage =
   TD-style `V_target(z_{t+1}) − V_target(z_t)` per decision, not whole-game z.
4. **External-anchored SPRT gate** (fixes: cause 3). Gate vs a FIXED pool
   (frozen BC agent, DUCT baseline bundle, the anchor, top league member) with
   a sequential probability-ratio test; the agent's speed makes 300-game gates
   cost ~a minute. Promotion requires significance vs the pool, and the
   headline metric becomes Elo-vs-anchors over iterations — the curve that must
   go up.
5. **(U) Strategy bottleneck ("strategy = predictive information")**. A small
   discrete latent (VQ codebook, ~32 codes) trained so that *the code chosen at
   turn t makes the position at t+H predictable* while being temporally smooth.
   Information-theoretically: strategy is the minimal sufficient statistic of
   the present for the long-range future. The low-level policy conditions on
   the code; the high-level policy picks it. Interpretable for free: codes
   cluster into "set TR," "stall weather," "preserve wincon," etc.
6. **(U) Trajectory encoder — the game as a sequence**. Feed the position
   encoder's per-turn summaries through a causal transformer over turns.
   Unlocks cycles/tempo/history/opponent-tendency representation (§2.2) and
   within-game adaptation. The belief filter already summarizes *sets*; this
   summarizes *behavior*.
7. **(U) Distributional value head** (fixes: risk posture, §2.2-10).
   Predict a categorical distribution over outcome margin (e.g., final mon
   differential -4..+4), not a scalar. Risk posture falls out: behind -> pick
   plans with mass on +; ahead -> minimize left-tail mass. Margin labels are
   free from every game.
8. **(U) Win-condition auxiliary head**. From each game: which brought mons
   were alive at the end / scored KOs. Predicting "who will close this game"
   forces wincon representation — cheap labels, directly targets §2.2-1.
9. **Counterfactual futures from the sim ("what-if" prep)** (fixes: candidate
   JEPA-target leakage + on-policy echo chamber). Replay recorded positions on
   the real engine with *alternative* own moves and different RNG seeds ->
   real futures for non-taken candidates. The engine is ground truth; the
   agent is no longer trained only on its own echo.
10. **Exploiter league (PSRO-lite)** (fixes: cycling/robustness). Periodically
    fork a high-LR exploiter trained solely to beat `spj_best`; add it to the
    league. The main line must be robust to its own best-response — a Nash
    pressure, not a popularity contest.
11. **Diversity guards**: entropy floor in the policy loss; temperature floor
    in generation; **mean-game-length monitor** (the 5-turn collapse was
    visible in `samples/games` all along — alert on it); position-novelty
    bonus via JEPA loss magnitude (high latent surprise = worth training on).
12. **Duration/window features done honestly**: track set/elapsed turns for
    weather/terrain/TR/Tailwind from the event stream (the tracker sees the
    set events; durations are computable modulo item extensions, which the
    belief already guesses via item posterior).
13. **Pairwise speed-order matrix input**: P(my i outspeeds their j) under
    current modifiers — computable from belief spe ranges + boosts + TR/TW
    flags; the single most decision-relevant relational feature after damage.

## 4. v3 architecture: Strategy-JEPA

Three levels, one shared entity-token vocabulary (v2's 16-token layout plus
new inputs), all JEPA-trained, sim used only as ground truth.

### 4.1 Inputs (changes from v2)

- Per-mon features as in v2, **plus**: pairwise speed-order matrix [6x6]
  (belief-marginal P(outspeed) under current field), field-window countdowns
  (weather/terrain/TR/tailwind turns remaining, honest from event stream),
  revealed-information ledger (n moves revealed mine/theirs per mon, item
  revealed flags), once-per-game resources (mega available, sash-intact
  posterior), score margin (mon differential + HP-sum differential).
- **Trajectory context**: the last K=16 turns' pooled position summaries
  (d-dim each, from the same encoder, cached during play) fed to a small
  causal transformer; its output is one "history token" appended to the
  entity set. Cost: one extra token; the per-turn summaries are cached, so
  no re-encoding of history.

### 4.2 Encoders

- **Entity encoder** E: role-typed transformer (v2's, kept) over 17 tokens
  (16 + history token) -> entity latents Z_t; pooled CLS -> s_t (d=448).
- **Temporal encoder** H: causal transformer over the last K pooled summaries
  -> history token h_t (that's where tempo/cycles/tendencies live).
- **Strategy bottleneck** Q: VQ layer over a projection of (s_t, h_t) ->
  discrete code q_t from a 32-entry codebook, with commitment loss +
  temporal-consistency prior (penalize switching codes without cause;
  straight-through estimator).

### 4.3 Latent dynamics (the core change)

`T(Z, a, b) -> Z'` — a role-typed transformer block operating on the full
entity-latent set, conditioned on BOTH joint actions (mine and opponent's),
returning next-turn entity latents. JEPA losses:

- 1-step: `||T(Z_t, a_t, b_t) − sg(E_ema(x_{t+1}))||` (entity-wise).
- Multi-step: unroll T for n in {2,3,5} against EMA targets (discounted
  weights) — this is what pushes strategy-horizon structure into the latents.
- Opponent-action marginalization for *planning* happens via the opponent
  policy head + matrix solve (below), not by averaging inside T — T stays a
  clean two-sided dynamics model (v1's matrix insight and v2's speed insight
  finally compose instead of competing).

### 4.4 Heads (all read latents; none reconstruct observations)

- Opponent policy head (per-slot logits over the believed action space) — for
  the matrix solve and information-set robustness.
- Own policy head conditioned on strategy code q_t.
- **Distributional value**: categorical over final mon-differential bins
  [-4..+4]; scalar value = expectation (calibration falls out).
- Win-condition head: per brought mon, P(alive at end), P(scores a KO).
- Strategy predictive head: from (s_t, q_t) predict sg(s^ema_{t+H}), H=4 —
  the bottleneck's reason to exist.

### 4.5 Play-time decision (fast path unchanged, deep path optional)

Depth-1 (ladder-fast, ~v2 cost): encode, top-k own x top-k opp candidates,
one T application each, matrix solve on distributional values collapsed by
risk posture (behind -> optimistic quantile, ahead -> pessimistic), sample the
mixed strategy. Depth-2/3 (generation + big decisions): recurse T through the
matrix solve tree; still sim-free and ~10-30ms.

### 4.6 Training loop (replaces selfplay_jepa's inner objectives)

1. Generate with **depth-2 latent search + Dirichlet noise at the root +
   temperature schedule**; record search visit distributions.
2. Policy target = search mixed strategy (distillation), NOT the taken move.
3. Value target = TD(λ) over the recorded trajectory with EMA bootstrap;
   distributional (margin bins) at terminal.
4. JEPA targets: realized next positions (1..5-step) + **counterfactual
   futures** from the what-if sim replayer for a random 10% of decisions.
5. Anchors: human-BC mix (kept), frozen-anchor league (kept), exploiter forks
   every ~25 iterations (new).
6. Gate: SPRT vs fixed anchor pool, 300+ games (a minute of wall time),
   promotion only on significance; Elo-vs-anchor-pool is the tracked curve.
7. Health dashboard per iteration: mean game length (alert < 8), policy
   entropy, code-usage histogram (collapse = one code dominating), JEPA
   novelty, value calibration (Brier on held-out outcomes).

### 4.7 Why this can climb indefinitely (the theory)

- The improvement operator (latent search) restores the AlphaZero inequality:
  distillation target is provably at least as strong as the raw policy under
  the model, and the model itself keeps improving from real+counterfactual
  JEPA targets — two coupled ascent processes.
- The strategy bottleneck gives the search a small discrete space to be
  *consistent* in across turns — long plans stop being an accident of greedy
  per-turn choices.
- External anchoring makes "progress" mean progress: promotion is against a
  fixed reference frame, so gate inflation is structurally impossible.
- PSRO exploiters approximate a Nash pressure: the line must be unexploitable,
  not merely dominant in its own mirror meta.

## 5. Migration and scope

Reuses: vocab, feature extractor (extended), role-typed layers, VQ/temporal
modules are new but small, round-robin/export machinery unchanged (same
`jepa-c`-style kind, new `jepa-s` kind). The v2 checkpoint warm-starts E.
Order of implementation: (1) T + multi-step JEPA + depth-1 matrix play,
(2) TD(λ) + distributional value + new gate, (3) trajectory encoder + strategy
bottleneck + wincon head, (4) what-if replayer + exploiters. Each stage is
independently benchmarkable against `exp-jepa-c` and the anchor pool.
