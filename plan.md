# BERMUDA — pricing a VGC battle as a Bermudan option

**Bermudan-Exercise Regression Monte Carlo for Doubles Adversarial play.**
An agent architecture imported wholesale from a niche where it is *the*
production algorithm: pricing early-exercise (Bermudan/American) derivatives
on quant desks. Longstaff–Schwartz least-squares Monte Carlo (LSMC, 2001) and
its primal–dual refinements (Tsitsiklis–Van Roy, Andersen–Broadie) price
trillions of dollars of Bermudan swaptions every day. Nobody has brought this
machinery to game playing. This plan argues the fit is not a metaphor — it is
the same mathematical problem — and lays out the full pipeline: path
collection, regression training, the exercise-policy agent, evaluation, and
the tournament hook.

---

## 1. Why option pricing, of all things

A Bermudan option gives its holder the right to act at a *discrete set of
exercise dates*: at each date you either exercise now or continue holding,
and the underlying moves stochastically between dates. Pricing one means
solving an optimal-stopping/control problem under three demons:

| Demon in Bermudan pricing | The same demon in VGC doubles |
|---|---|
| The underlying diffuses **continuously** — every node has *infinitely many* chance children. A chance tree is unbuildable. | Each joint action fans into damage rolls (16), crit lotteries, accuracy checks, secondary effects, and speed ties — **thousands of chance children per turn**. A chance tree is unbuildable in practice. |
| Decisions happen only at **discrete exercise dates**; the value of acting now must be compared to a *continuation value* that depends on optimal future behavior. | Decisions happen only at **turn boundaries**; the value of Protecting, switching, or committing damage now must be compared to a continuation value that depends on optimal future play. |
| The pricing filtration 𝓕ₜ forbids regressing on information you do not have yet (future prices, latent volatility states). Exercise rules must be **measurable**. | The information set forbids conditioning on what you cannot see (opponent EVs, items, movesets). Policies must be measurable w.r.t. the *public* battle state plus your own sheet. |

Finance's answer to all three at once is **least-squares Monte Carlo**:

1. **Simulate an ensemble of full paths** of the underlying under a behavior
   ("path") measure. Never build a tree.
2. **Backward over exercise dates, regress realized continuation payoffs on
   basis functions of the *observable* state.** The regression's fitted value
   *is* the chance-node backup: conditional expectation recovered
   cross-sectionally from noisy samples. Chance branching stops being tree
   width and becomes statistical noise that the regression averages out.
3. **Derive the exercise policy** by comparing, at each date, the value of
   each available action against the regression-estimated continuation value.
   This policy is suboptimal-but-improvable: re-simulating paths *under the
   improved policy* and refitting tightens it (iterated LSM / policy
   iteration); the Andersen–Broadie dual gives a diagnostic upper bound via
   martingale ("hedging") residuals.

Every one of those mechanisms transfers, and each one lands exactly on a
classic VGC pain point.

## 2. The transfer, point by point

### 2.1 Chance nodes are never enumerated — regression is the backup operator

Search-based game agents (MCTS and friends) pay for randomness twice: chance
nodes multiply the tree's width, and value estimates at a node need many
visits before the noise cancels. LSMC pays once, in the training set: a state
that was reached along a simulated path carries its realized final outcome
(±1), and the fitted regression

&nbsp;&nbsp;&nbsp;&nbsp;`V(φ(s), phase) ≈ E[outcome | φ(s)]`

marginalizes *all* downstream randomness — RNG, and (see 2.3) hidden
information, and the behavior policy's own mixing — in one cross-sectional
fit. At play time BERMUDA does **no deep search at all**: it simulates each
candidate action exactly *one half-turn* into sampled scenarios and reads V
at the resulting states. Depth comes from the backward-propagated regression,
not from rollouts. This is the Longstaff–Schwartz exercise comparison:
"payoff of exercising now" vs "regressed continuation value," with the turn
index as the exercise date.

### 2.2 A battle is Bermudan, not American

Moves lock at turn boundaries — you cannot re-decide mid-resolution. Discrete
exercise dates are the *easier* Bermudan case and the one the algorithm was
built for. The phase-conditioned value net (`V(φ, phase)`, phase = turn
bucket) is the direct analogue of per-exercise-date regressions, sharing
statistical strength across dates the way modern LSMC implementations share
basis coefficients across tenors.

### 2.3 The filtration constraint *is* hidden information, solved for free

In pricing you may only regress on 𝓕ₜ-measurable quantities; under stochastic
volatility the latent vol state is simply *not in the basis*, and the fitted
conditional expectation automatically averages over its posterior given the
observables. BERMUDA enforces the same discipline: the feature map φ contains
**only public battle state plus the agent's own sheet** — species, HP
fractions, statuses, boosts, field, side conditions, faints, mega flags,
type-matchup summaries. Opponent EVs/items/movesets are deliberately *absent*
from φ. The regression then learns `E[outcome | public state]`, i.e. the
value already marginalized over the hidden-set posterior induced by the path
measure — belief inference never appears inside the value function. (A
particle filter still exists in the *scenario generator* at play time, purely
as plumbing to materialize concrete opponent sheets the simulator can step;
it is infrastructure, not the architecture.)

### 2.4 Simultaneous moves via smoothed fictitious play, not per-turn game solving

Pricing has one agent; VGC has an adversary choosing simultaneously. BERMUDA
folds the opponent into the stochastic driver of the path measure — exactly
how multi-factor models fold in a second Brownian motion — and then makes
that driver *self-consistent* across generations: generation *k* collects
paths with the generation-(k−1) exercise policy playing against opponents
drawn from a **reservoir of all earlier generations** (plus the heuristic
prior policy). This is smoothed fictitious play: best-responding to the
empirical mixture of past strategies, which damps the rock-paper-scissors
policy cycling that plain self-play exhibits in simultaneous-move games, and
steers the population toward equilibrium play without ever solving a matrix
game at a turn. Play-time mixing (softmax over risk-adjusted action values at
temperature τ) gives the quantal-response smoothing.

### 2.5 Risk is priced, not just averaged — and the sign flips with the score

VGC is an *incomplete market*: you cannot hedge a crit. Indifference pricing
handles unhedgeable risk with the entropic certainty equivalent, and BERMUDA
aggregates each candidate action's scenario values the same way:

&nbsp;&nbsp;&nbsp;&nbsp;`CE_λ(Q) = −(1/λ) · log E[ exp(−λ·Q) ]`   (λ→0 recovers the mean)

with **state-adaptive risk aversion** `λ = λ₀ · V̄`, where V̄ is the
current position's estimated value. Ahead (V̄ > 0) ⇒ λ > 0: risk-averse,
refuse lines that lose to a crit. Behind (V̄ < 0) ⇒ λ < 0: risk-*seeking*,
actively buy variance — the "when losing, play for the high roll" skill that
separates strong human VGC players from win-probability-mean maximizers.
One knob (λ₀), principled semantics, and an ablation switch (λ₀ = 0 is the
risk-neutral pricer).

### 2.6 Simulation-efficiency tricks ride along from the same literature

- **Common random numbers**: every candidate action is evaluated on the
  *same* frozen scenario (same sampled opponent sheet, same opponent action,
  same engine PRNG state via save/restore forking), so action comparisons
  difference away shared noise — the classic CRN variance reduction.
- **Two-stage screening and racing** (ranking & selection / OCBA, from
  discrete-event simulation): a cheap type-chart heuristic screens the ~10²
  legal joint actions down to A candidates, which then split the simulation
  budget of K scenarios each. Budget goes where the decision is close.
- **Martingale ("hedging") residuals** from the primal–dual literature as a
  *training diagnostic*: along held-out paths, V(s_{t+1}) − V(s_t) should be
  a mean-zero martingale increment; systematic drift per phase localizes
  where the value function is wrong — a far sharper signal than global MSE.

## 3. What BERMUDA is **not**

No game tree. No UCT/PUCT statistics. No policy network as the decision
maker. No per-turn equilibrium computation. No enumeration of chance
outcomes. The entire architecture is: *a path pile, one regression, and a
one-half-turn exercise comparison under a risk measure.*

## 4. Pipeline

```
gen 0                     gen k ≥ 1                         final
─────                     ─────────                         ─────
heuristic vs heuristic    exercise-policy(g k−1)            export bundle
paths (fast, no nets)     vs reservoir{h, g0..k−1} paths    round_robin.py
        │                          │                        vs the pile
        ▼                          ▼
   LSMC regression  ──►  refit on last B gens' paths (path reuse)
   V₀(φ, phase)          V_k(φ, phase)  + arena gate vs heuristic
```

1. **Path collection** (`bermuda/paths.py`) — full games on the Showdown
   sidecar; each side logs `(φ(public state), turn, final outcome ±1)` at
   every decision it faces, from its own viewpoint. Teams sampled from the
   replica registry or the mined self-play pool. Gen 0 uses the type-chart
   heuristic with a sampling temperature (a diffuse path measure, like the
   wide lognormal ensemble a pricer starts from); gen ≥ 1 uses the current
   exercise policy with exploration temperature against reservoir opponents.
2. **Regression** (`bermuda/train.py`) — fit `V(φ, phase)` to realized
   terminal payoff (the Longstaff–Schwartz "regress realized payoffs" form;
   `--target boot` switches to the Tsitsiklis–Van Roy bootstrapped form as an
   ablation). Group-split by game so val games are unseen. Reports MSE,
   sign-accuracy, per-phase calibration, and martingale residuals.
3. **Exercise policy** (`bermuda/chooser.py`) — the play-time agent:
   screen → freeze K scenarios (belief-sampled opponent sheet + heuristic
   opponent action + saved PRNG) → step each surviving candidate one
   half-turn on forked sims → V at the results → entropic CE with adaptive λ
   → argmax (or softmax at τ for exploration/quantal response).
4. **Evaluation** (`bermuda/eval.py`) — local arenas vs heuristic/random with
   Wilson intervals and s/move; calibration + residual diagnostics on shards.
5. **Generations driver** (`bermuda/loop.py`) — orchestrates 1–4 per
   generation, maintains the reservoir, appends a summary ledger.
6. **Tournament hook** — `agent_server.py` grows a `bermuda` agent kind
   (chooser behind the stdio protocol; forced switches routed through a
   matchup-aware picker instead of the default random). Export with
   `export_agent.py NAME --agent bermuda --ckpt …` and play it from the pile
   with `round_robin.py play/star/all` — no coordinator changes needed.

## 5. Files (all new, this worktree)

| File | Role |
|---|---|
| `bermuda/config.py` | every BERMUDA knob (budgets, λ₀, net size, paths) |
| `bermuda/typechart.py` | gen-9 type chart (features + heuristic) |
| `bermuda/features.py` | 𝓕ₜ-measurable feature map φ (public + own sheet) |
| `bermuda/heuristic.py` | type-chart policy: gen-0 paths, screening, opponent model |
| `bermuda/model.py` | phase-conditioned value MLP (the regression basis) |
| `bermuda/paths.py` | path collection (dataset) |
| `bermuda/train.py` | LSMC fit + diagnostics |
| `bermuda/chooser.py` | the exercise-policy MoveChooser (the agent) |
| `bermuda/eval.py` | arena + calibration/residual diagnostics |
| `bermuda/loop.py` | generation loop (paths → fit → gate) |
| `scripts/bermuda_setup.sh` | one-time worktree setup (artifacts, node) |
| `scripts/bermuda_smoke.sh` | tiny laptop end-to-end proof |
| `scripts/bermuda_full.sh` | big-box run: generations + export + round robin |

## 6. Commands

One-time setup (artifacts + node install are shared from the main checkout):

```bash
cd vgc-bot-bermuda
bash scripts/bermuda_setup.sh ../vgc-bot
```

Laptop smoke (tiny by design — proves every stage end-to-end):

```bash
bash scripts/bermuda_smoke.sh
```

Full experiment (big box):

```bash
bash scripts/bermuda_full.sh            # gens of paths→fit→gate, then export
python round_robin.py play bermuda baseline   # or: star / all
python round_robin.py standings
```

Stage-by-stage equivalents are printed by both scripts; every stage is also
runnable by hand (`python bermuda/paths.py --help`, etc.).

## 7. Evaluation plan

- **Per generation**: arena vs the heuristic prior (200 games, Wilson CI) —
  the policy-iteration gain curve should rise then plateau.
- **Value quality**: per-phase calibration (predicted vs realized win rate)
  and martingale residual drift on held-out games.
- **The real test**: export each gated generation and run
  `round_robin.py star` against the pile; Bradley–Terry standings decide.
- **Ablations** (flags, no code forks): λ₀ = 0 (risk-neutral); K and A
  budget sweeps; `--target boot` (TvR) vs realized-payoff (LS); reservoir off
  (pure self-play) to expose fictitious-play's anti-cycling value.

## 8. Approximations and risks, stated up front

- **Scenario reconstruction fidelity**: the sidecar's midgame reconstruction
  does not rebuild choice locks, encores, substitutes, or spent PP (a known
  sim-plumbing gap). Scenarios are therefore slightly optimistic about action
  freedom — shared equally by every candidate under CRN, so rankings suffer
  less than absolute values.
- **Opponent-brought inference**: the opponent's 4 brought mons are inferred
  as appeared ∪ first-unappeared; wrong guesses give scenarios phantom bench
  options. Mild, and it washes out under scenario averaging.
- **Heuristic opponent model at the root** (v1): scenario opponent actions
  come from the softmax type-chart policy, not from the reservoir nets
  (cost). If eval shows exploitable root behavior, promote the opponent model
  to the previous-generation value policy — the seam exists.
- **Screening bias**: a candidate the heuristic hates never reaches racing.
  Mitigated by a wide screen (A ≈ 12 of typically ~40–200 legal joints) and
  the exploration temperature during path collection.
- **One-half-turn horizon**: tactical traps deeper than one turn must live in
  V. That is the LSMC bet: backward-fitted value carries the depth. The
  martingale residuals tell us where the bet fails.

## 9. Why this can be the strongest VGC agent yet

The branching factor of VGC is not primarily *decision* branching — ~10²
joint actions is trivial — it is **chance × information branching**, ~10³–10⁴
per turn, which starves any tree search into evaluating a handful of noisy
leaves per candidate. BERMUDA moves that entire burden offline into one
regression fitted on millions of cheap path states, and spends its play-time
budget where variance actually bites: A×K CRN-paired half-turn simulations
aimed straight at the decision. It prices risk instead of averaging it, it
inherits an equilibrium-seeking training population instead of a cycling one,
and its every component has two decades of production hardening — in a
different industry.
