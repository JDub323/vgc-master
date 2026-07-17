# JEPA world model for VGC doubles — design

**Lane: pile-only.** This is a structural rewrite — new tokenizer/feature layout,
new action-selection algorithm, no reuse of the layout-3 tokenizer or the
policy/value transformer. It composes with everyone else through the round-robin
protocol, not through trunk. It touches none of the ten `BEHAVIOR_SOURCE_FILES`;
the only edit outside new files is one additive `if`-branch in
`agent_server.build_chooser` (not a behavior file).

Agent kind: `jepa`. Chooser: `agents.jepa_world_model.v1.JEPAWorldModelChooser`.
Architecture label: `JEPA-WorldModel-MatrixSolve`.

---

## 1. The idea, refined

The napkin idea was: *predict the consequences of each joint move you could
make, and pick the move with the best consequences (a distribution if needed).*
Taken literally against a simultaneous, hidden-information doubles game, three
things have to change before it is actually strong:

1. **A consequence is not a function of my move alone.** It depends on the
   opponent's simultaneous joint move `b`, on their hidden set (item/EVs/moves),
   and on damage rolls / accuracy / secondary effects. So the object we predict
   is a **payoff matrix** `V[a, b]`, not a vector `V[a]`, and we must
   marginalize the stochastic and hidden parts.

2. **"Pick the best" is wrong for a matrix.** Argmax over `a` of an expected
   value against a fixed opponent model collapses to a pure strategy that a
   human instantly exploits (the always/never-Sucker-Punch failure the repo's
   `scenarios.py` gate is built to catch). The correct object is a **mixed
   strategy** — the equilibrium of the one-shot matrix game `V[a, b]`. So the
   planner *solves* the matrix, it does not argmax it.

3. **Predicting in raw token space is brittle and slow.** JEPA's actual thesis
   is: predict in a **learned representation space**, supervised against a
   target-encoder's embedding of the true next state, not against reconstructed
   observations. We predict next-state *latents* and read the value (and
   grounded physical facts) off them.

So the architecture is a **latent one-ply world model + matrix-game planner**:

```
        encode s_t ──► Z (per-entity latents)
                         │
   for each (my a, opp b) candidate pair:
        Predictor(Z, a, b) ──► Ẑ'(a,b)   (predicted next-state latents)
        value_head(Ẑ') ─────► V[a,b]     (win prob of that consequence)
                         │
        average V over K belief determinizations   (marginalize hidden sets)
                         │
        solve the matrix game V ──► mixed strategy p*(a)
                         │
        sample p* (temperature) ──► joint action
```

This replaces the determinized-DUCT tree search over the real Node simulator
with **one forward sweep of a learned model** and a tiny game solve. There is no
sim forking at play time, which makes belief determinization almost free (the
usual cost — reconstructing a playable battle per sample — is gone).

## 2. Why this is a distinct, task-specific architecture

- **It is a world model, not a policy/value net.** The network's core is a
  *dynamics* function `Predictor(Z, a, b) → Ẑ'`; value is a readout of a
  predicted future, so the same net answers counterfactual "what if I click
  this" questions the BC policy net cannot.
- **The opponent is a first-class axis.** `V` is a matrix; the opponent has its
  own learned policy head and its own action axis; the decision is a game solve.
  This is the right primitive for simultaneous VGC, and it produces mixed
  strategies *by construction*.
- **Beliefs are consumed as determinizations of the model, not of a sim.**
  Averaging `V` over `belief.sample_sets(K)` marginalizes the hidden set exactly
  where it matters — inside the predicted payoff — and costs K cheap forward
  passes instead of K sim reconstructions.
- **Role-typed attention.** Ally and foe tokens carry asymmetric information (we
  know our sets exactly; we only believe theirs). The transformer uses
  **per-role Q/K/V/O projections** so "my mon" and "opponent mon" are literally
  different linear maps, while still attending across the board. Attention
  logits are further biased by a **damage edge feature** (best expected damage
  from ally *i* onto foe *k*), making the encoder a relational reasoner over the
  who-KOs-whom graph.

## 3. Tokens (fixed 16-entity layout)

| idx | entity | role | source |
|---|---|---|---|
| 0 | global field | `global` | weather/terrain/TR/screens/tailwind/turn (+reserved duration/gravity slots) |
| 1–6 | my 6 mons (preview order) | `ally` | full own set + live public state |
| 7–12 | opponent 6 mons (preview order) | `foe` | preview species + reveals + **belief** summary/particle features |
| 13–14 | opponent intent slot A/B | `intent` | learned query tokens → opponent-policy head |
| 15 | CLS | `cls` | pooled state value / VICReg anchor |

Per-mon features (identical schema for ally and foe; foe fields fall back to
belief where unobserved): species, item, ability, 4 move ids (foe: the assumed
moveset of the current determinization), status, 2 types, base stats, HP,
faint/active/bench flags, **brought-to-battle**, appeared, turns-active +
**fake-out-ready**, **protect counter → success prob** + can-protect, mega
availability/used, 7 boosts, item-consumed, and belief scalars (modal-item
prob, effective-speed lo/hi, bulk, nature prob). Everything the brief asked the
mon embedder to "keep track of" is a feature here.

Opponent move-slot semantics: a determinization assigns each foe a concrete
4-move set, so foe actions live in the **same 39-index slot-action space** as
ours (`actions.py`), and averaging `V` over determinizations marginalizes which
concrete move "slot k" was.

## 4. Losses (training)

Trained on parsed human replays as `(s_t, a_true, b_true) → s_{t+1}`, outcome:

- **JEPA latent loss** — `smooth_l1(Ẑ'(a_true,b_true), sg(EMA_encoder(s_{t+1})))`
  per entity. The representation-space prediction objective.
- **Value loss** — `mse(value_head(Ẑ'(a_true,b_true)), outcome)`; the value we
  act on is grounded in true game results through the predicted latent.
- **Grounded decoders** (anti-collapse anchors, read off `Ẑ'`): next HP per mon,
  faint, status, field conditions, and who-moved-first. A constant latent cannot
  predict varying HP/KOs, so these forbid collapse *and* make the value honest.
- **Opponent-policy CE** on `b_true` and **my-prior CE** on `a_true` (candidate
  generators; the my-prior also anchors the encoder).
- **VICReg** variance (hinge, keep per-dim std ≥ 1) + covariance (decorrelate)
  on the online encoder's entity latents.

Collapse prevention is therefore fourfold: EMA target + stop-grad asymmetry,
VICReg, grounded decoders, and the two policy heads reading the same latents.

## 5. Play-time planner

1. `belief.summary()` + `K` determinizations from `belief.sample_sets(K)`.
2. My candidates: top-`Ka` legal joint actions by the my-prior head
   (`legal_joint_actions` from the real request).
3. Opponent candidates: top-`Kb` joint actions by the opponent-policy head over
   a constructed foe action space (assumed moves × targets, believed switches,
   protect).
4. `V[a,b] = mean_det value_head(Predictor(Z_det, a, b))`.
5. Solve `V` for a mixed strategy `p*` with **regret matching** (row player
   maximizes, column player minimizes — VGC win prob is zero-sum). Game value is
   the saddle value.
6. Sample `p*` at the play temperature (0 ⇒ argmax). `ChoiceInfo` carries the
   value, `p*` as the mixed strategy, the opponent prediction, and per-action Q.

## 6. Honest limits (v1)

- One ply + learned value. No deep lookahead; depth-`d` latent unroll is a
  documented lever (the training loop already supports multi-step targets).
- Reverse (foe→me) damage edges are not fed as attention bias in v1 (only
  ally→foe, which `damage.damage_features` already produces); the net infers
  incoming threat from base stats/types.
- Weather/screen/TR **durations** and **gravity** are reserved feature slots set
  to "unknown": the shared tracker (`data.LogParser`) does not track them (a
  documented repo limitation), so they are honestly absent, not faked.
- A random-initialized net still runs and plays legally (weakly); it is the
  bundle's fallback so the contestant is always runnable.
