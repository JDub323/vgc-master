# vgc-bot

A Pokémon VGC (doubles) battling AI for the **Pokémon Champions ranked ladder**:
Regulation M-B (config-switchable to M-A), megas allowed, no Terastallization,
restricted legendaries banned, **closed team sheets** at play time.

## Architecture

The gameplay boundary is the small `agents.interfaces.MoveChooser` protocol.
Every Elo contestant is a complete chooser architecture; the current full bot
is versioned as
`agents.determinized_duct.v1.DeterminizedDUCTChooser`. Policy-only,
max-damage, and random choosers implement the same contract.

The full chooser orchestrates five injected, independently testable bricks:
the external battle-owned belief model, position encoder, policy prior, leaf
evaluator, and decoupled-UCT searcher. Their stable implementation IDs live in
`agents/ids.py`; behavior-changing replacements get a new versioned module and
ID while v1 remains loadable.

```text
game / benchmark / self-play
          │ passes tracker + externally owned belief
          ▼
MoveChooser (the Elo-rated agent)
    ├── PositionEncoder ──► LeafEvaluator + PolicyPrior
    ├── Searcher (selection, simulation, backup, aggregation)
    └── BeliefModel contract (constructed and updated by the game layer)
```

`search.mcts.DeterminizedDUCTChooser` is the concrete orchestration class;
archives address it through the versioned algorithmic ID under `agents/`.

For manual review, [contracts.py](contracts.py) defines the shared dictionary
and array shapes, while [DATA_CONTRACTS.md](DATA_CONTRACTS.md) catalogs every
production function/method with its input and output structure.

Three core systems cooperate inside those bricks:

1. **Policy/value network** ([models/policy_value.py](models/policy_value.py)) — a small
   from-scratch transformer (~6 layers, d=256) trained by behavior cloning on
   ~89k human Champions replays, then fine-tuned by self-play. Given a tokenized
   position from one player's perspective it outputs (a) a distribution over that
   player's **joint** action (both doubles slots at once, one masked 39×39
   softmax — phase 3; v1 checkpoints used two per-slot softmaxes and still load),
   (b) a value in [-1, 1], (c) an auxiliary prediction of the opponent's hidden
   sets (representation shaping + a sanity check against the particle filter).
   Run from both perspectives it proposes "my candidate moves" and "opponent's
   likely moves"; its job in search is pruning to top-k.
2. **Belief tracker** ([beliefs.py](beliefs.py)) — a particle filter over opponent sets,
   no neural net. Set priors come from train-split team sheets; objective
   nature/stat-point priors come from `spreads.json`. Reveals are hard
   constraints, while speed-order and observed-damage evidence eliminate
   inconsistent builds or narrow the feasible stat-point intervals used by
   the off-list fallback.
   Feeds summary tokens to the tokenizer and sampled sets to the search.
3. **Search** ([search/mcts.py](search/mcts.py), [search/node.py](search/node.py)) — decoupled UCT with
   policy priors on a forkable Showdown sim ([env.py](env.py) sidecar), determinized
   over belief-sampled opponent sets, solve-to-terminal in small endgames.
   Output is a mixed strategy (visit distribution), sampled with temperature.
   No alpha-beta: turns are simultaneous.

Data flow ([data.py](data.py) → [tokenizer.py](tokenizer.py)): replays are Open-Team-Sheet games, but
the bot plays Closed Team Sheets, so training inputs are **CTS-observable
reconstructions** (own team + opponent reveals only); the opponent sheet is
used strictly as oracle labels. Damage features ([damage.py](damage.py), a Node bridge to
`@smogon/calc`) are precomputed during prep so no bridge sits in the training
loop. Splits are by bo3 match, never by transition; samples are weighted by
rating, format (M-B > M-A) and recency ([config.py](config.py)).

## Setup

```bash
pip install -r requirements.txt
npm install github:smogon/pokemon-showdown#e440c4a18385274f10c405d0b158b6a962ce6d94
(cd node_modules/pokemon-showdown && npm install && node build)   # git install ships no dist/
npm install @smogon/calc@0.11.0
```

The npm release of `pokemon-showdown` (0.11.10) predates Pokémon Champions, so
the sim is pinned to a git commit that has the `gen9championsvgc2026*` formats
and the `champions` mod (new megas, e.g. Floettite → Floette-Mega). If a
different pin lacks the named format, the env falls back to
`gen9doublescustomgame`.

No root on the box? The notebook's setup cell installs Node from the official
standalone tarball into `artifacts/node-dist` instead (no apt needed) and puts
it on PATH. Point `Config.node_dir` at the directory containing `node_modules`
(default `artifacts/node`), and `Config.checkpoint_dir` at Google Drive when
on Colab (the notebook offers this automatically).

## Tests and model discovery

Install pytest in the same Python environment used to run the repo, then run:

```bash
python -m pip install pytest
python -m pytest -q
# or, without activating the checked-out virtualenv:
.venv/bin/python -m pytest -q
```

The suite covers chooser contracts, all v1 bricks, prior masking/pruning,
token equivalence, value orientation and terminal handling, belief updates,
search backup/aggregation, manifest validation, source-identity enforcement,
the tiny deterministic archived-agent fixture, strict stat inversion, and
real Showdown/calc naming and damage behavior. A documentation gate also checks
that every production function/module stays covered by the contract catalog and
blocks known stale architecture terminology. The Node fixtures are defined in
`tests/conftest.py`; `python tests/test_agents.py` remains a fast direct runner
for the modular tests.

Tests intentionally do **not** choose whichever checkpoint or archive has the
newest modification time. That would make CI slow and nondeterministic. Model
selection is explicit:

| command | model/archive selected |
|---|---|
| `python -m pytest -q` | No trained checkpoint; code/bricks plus deterministic fixtures. |
| `python scenarios.py` | `Config.checkpoint_dir/ckpt_best.pt` if it exists; otherwise uniform priors and terminal-only scenarios. |
| `python evaluate.py` | `ckpt_best.pt` by default, or the explicit positional checkpoint argument. |
| `python benchmark.py play current` | The live `ckpt_best.pt` against the frozen 1x reference agent (`baseline` is the default opponent). |
| `python benchmark.py list` | Every directory under `artifacts/benchmarks/` containing `meta.json`; it does not pick a winner or “latest” bundle. |
| `python benchmark.py archive NAME --ckpt PATH` | The explicitly supplied checkpoint, or `ckpt_best.pt` when `--ckpt` is omitted. |

A newly written `sp_iter_*.pt` or other checkpoint therefore is not tested or
added to the leaderboard automatically. Promote it deliberately by evaluating
it and archiving it with a unique name. Likewise, adding a new versioned Python
agent/brick requires a new ID and explicit registration in `agents/registry.py`;
filesystem discovery cannot silently redefine an archived implementation.

## Archiving and transferring agents

Turning the current trained model into a full portable contestant is one
command:

```bash
python benchmark.py archive my-agent-v1 --notes "what changed"
python benchmark.py list
```

This creates `artifacts/benchmarks/my-agent-v1/` with `agent.json`, checkpoint,
vocab, typed config, usage stats, dex, spreads, and metadata. Copy that entire
directory—not just `ckpt.pt`—to the same `artifacts/benchmarks/` location in
another checkout. `benchmark.py list` discovers it automatically from
`meta.json`; no database registration is needed. `registry.json` stores match
history and ratings rather than agent definitions, so copy or merge it only if
the results should move too.

Transfer is easy between machines with the same source and runtime, but it is
deliberately strict rather than universally plug-and-play. The destination
must retain every implementation ID in `agents/registry.py`, match the recorded
behavior-source hashes, and use the recorded Python, Torch, NumPy,
Showdown, calc, and format identities. Machine-local paths such as the Node
binary may differ. New archives record **AST-normalized (`ast-v1`) source
hashes** — docstring/comment/formatting churn no longer invalidates an
archive, while any logic edit still does. A real mismatch fails closed instead of
running the historical agent through newer behavior — unless
`benchmark.py play ... --allow-source-drift` is passed explicitly, which runs
the archive through current code, warns loudly, records the drifted files on
every result row, and marks the contestant with `*drift` in standings. To make
a behavior-changing improvement transferable alongside old agents, add v2
modules/IDs and keep v1 intact.

## Phase 1 — data + predictor

```bash
python env.py --dump-dex     # artifacts/dex.json from the sim's champions data
python data.py download      # HF: cameronangliss/vgc-battle-logs (~630 MB)
python data.py parse         # logs -> parsed battles, vocab, usage stats
python data.py prep          # beliefs + damage features + tokenize -> npz shards
python train.py              # behavior cloning; checkpoints + TensorBoard
python evaluate.py           # predictor benchmarks on held-out battles
python env.py --benchmark    # sidecar save/restore proof + steps/sec
```

`evaluate.py` reports top-1/3/5 joint-action accuracy, perplexity, calibration
(ECE), and the headline metric **pruned-set recall@k** — how often the human's
actual joint action is inside the model's top-k, i.e. whether search pruning is
safe — against max-damage and random floors.

Measured on a laptop CPU (sets the scale for search budgets; redo on your box):
~490 sim steps/s and ~930 state save/restore forks/s, with restored forks
replaying identically.

## Phase 2 — search

```bash
python env.py --selftest     # proves mid-battle state reconstruction — run first
python beliefs.py --audit    # particle-filter breakdown rates on held-out games
python scenarios.py          # endgame gates (mixed-strategy test) + midgame diagnostics
python scenarios.py --mine   # extract real-replay endgame candidates
python observe_game.py --step        # watch a search-vs-search game, turn by turn
python env.py --live [ckpt]          # play on a local Showdown server via poke-env
```

`scenarios.py`'s endgame gates run with or without a trained checkpoint
(endgames are solved to terminal); its earlygame/midgame diagnostic positions
and everything else in this phase need the phase-1 artifacts.
The current pipeline uses tokenizer layout 3 throughout. Any future layout
change **invalidates prepped shards and checkpoints: re-run `data.py prep` and
`train.py` before `evaluate.py` / `observe_game.py`**.

## Play against the bot

```bash
python teams.py --validate   # once per box: replica teams through the validator
python play.py               # interactive: pick your team, pick the bot
```

`play.py` spawns the pinned open-source pokemon-showdown server locally and
you battle in the **official Showdown client** (sprites, move buttons, HP
bars, statuses, battle text, animations — all from play.pokemonshowdown.com,
no custom battle GUI to maintain):

1. pick your team from 8–10 replica Reg M-B teams mirroring real tournament
   rosters (rain, sand, sun, snow, Trick Room, tailwind, balance, hyper
   offense — see [teams.py](teams.py)); the bot secretly picks from the same pool,
2. pick the opponent: `search` (full DUCT), `policy` (net only, no search),
   `max-damage`, or `random`,
3. open `https://play.pokemonshowdown.com/~~localhost:8000`, paste the
   printed team into Teambuilder → Import from text, and challenge the bot,
4. watch `http://localhost:8010` — a live dashboard of the bot's brain:
   the probability it assigns to **your** likely actions (with the bot's
   expected value for each of those branches and the search visits it spent
   there), its belief about your items/speed (with sprites), and its
   win-confidence sparkline. After every resolved turn it grades itself:
   was the move you actually played inside its top-6 predictions, at what
   rank/probability, and what it thought of the position if you played it —
   plus a running top-6 hit-rate for the game.

`python teams.py --mine` extracts the most common real high-rated team sheets
from the dataset — legal by construction — to swap in for any replica the
validator flags.

## Phase 3 — self-play, benchmarking, richer beliefs

The behavior-cloning pipeline above already produces the current layout-3,
joint-head `ckpt_best.pt`. Promote that explicit checkpoint to a static agent,
then use it as the starting point for self-play. Historical tokenizer layouts,
per-slot checkpoints, and checkpoint-only benchmark bundles are unsupported.

```bash
# 0. freeze the evaluated BC model and all of its behavior assets
python benchmark.py list                  # includes the frozen `baseline`
python benchmark.py list

# 0.5 widen the self-play team distribution (optional, recommended): real
#     teams so the net can't memorize the 10-replica pool's pairwise
#     artifacts. Two sources, both sim-validated into selfplay_teams.json:
python teams.py --fetch-pool         # ~1.2k Reg M-B tournament pastes (VGenC
                                     # index) with REAL EV/nature spreads
python teams.py --build-pool all     # + every distinct dataset sheet (~2.8k;
                                     # stat points redacted -> filled from the
                                     # Pikalytics objective prior)
python teams.py --pool               # inspect; --import-pool FILE adds more

# 1. self-play (the main event): fork the BC net, generate -> train -> gate
python selfplay.py --hours 10        # overnight; resumable, checkpoints each iter
python selfplay.py --iters 3         # or a fixed number of iterations
python benchmark.py archive sp-iter8 --ckpt artifacts/checkpoints/selfplay/sp_iter_008.pt

# 2. head-to-head: 100-game series (every ordered pairing of the 10 replica teams)
python benchmark.py play sp-iter8 baseline    # strength, 95% CI, and Elo
python benchmark.py play current              # live experiment vs frozen 1x
python benchmark.py standings              # Bradley-Terry ratings, segregated by era

# diagnostics for the two model-quality items
python evaluate.py --switches              # is switching underweighted in the prior?
python scenarios.py                        # endgame gates + midgame diagnostics

# prior ablation/capacity conclusions are preserved in EXPERIMENTS.md
```

**Self-play** ([selfplay.py](selfplay.py)) is AlphaZero adapted to simultaneous, hidden-info
doubles: the DUCT search's root visit distribution (a *joint* mixed strategy)
is the policy target, the game outcome is the value target, root Dirichlet
noise + a temperature schedule drive exploration — all confined to a self-play
path so live-play and evaluation behavior is untouched. Games are CTS-honest
(the same tracker + filter as everything else). Generation runs `sp_procs`
subprocesses × `sp_workers` game threads, and every leaf evaluation funnels
through one `BatchedEvaluator` per process that coalesces the threads' requests
into batched GPU `predict_batch` calls — the GPU parallelism the simulator
(real Node engine, CPU-bound) allows.

**Joint policy head.** The policy is one masked 39×39 softmax over both
slots' actions instead of two independent per-slot softmaxes, so a slot's
action is predicted in the context of its partner's (the factorized head could
not, e.g., condition an attack on the partner's Rage Powder). The frozen
baseline and all supported checkpoints use this head.

**Benchmarker** ([benchmark.py](benchmark.py)) freezes complete agents into immutable
bundles under `artifacts/benchmarks/`. Each new bundle contains checkpoint,
vocab, typed config, usage/dex/spread assets, and `agent.json`: an `AgentSpec`
with the top-level chooser ID, every brick ID/config/asset, source metadata,
and the exact Showdown/calc/Python/Torch/NumPy runtime identities. A series is
every ordered pairing of the replica teams; results are stored and rated, and
games from different eras are kept apart so a logic change never silently
pollutes the Elo. Standings group entries by chooser architecture while each
individual `AgentSpec` receives its own rating.

Archive contract:

- New `AgentSpec` archives resolve every chooser/brick through an explicit
  allow-list. Unknown implementation IDs fail closed; they never fall back to
  today's search code. Pre-AgentSpec checkpoint-only bundles are unsupported.
- Every behavior path in a new manifest must exist inside its bundle. Saved
  config and brick config are authoritative; current global behavior assets
  are never substituted. Machine-only paths (Node binary, data/checkpoint
  locations) remain local runtime overrides.
- Recorded runtime identities are checked before a static archive runs, so an
  engine or numerical-stack upgrade cannot silently change historical play.
  Resolved chooser/brick sources and behavior-critical shared modules are also
  hash-checked under the manifest's recorded scheme (`ast-v1` for new
  archives — docstrings/comments don't count).
  Mismatches fail closed by default; `--allow-source-drift` is the explicit,
  result-tainting override for exploratory runs. Preserve the recorded
  dependency environment and v1 source modules; add v2 modules for
  behavior-changing implementations.
- `agents/evaluation.py` provides append-only JSONL evaluations for policy
  prior recall/calibration/latency, token equivalence, leaf value calibration,
  belief deductions/depletion, and search scenario throughput. Results default
  to `artifacts/brick_evaluations/results.jsonl` for comparisons across brick
  versions.

**Nature/stat-point inference** ([beliefs.py](beliefs.py)) widens each redacted
train-sheet set with the top objective builds from `spreads.json`. Covered
builds carry concrete Champions stat points and nature; observed speed and
damage test those exact stats. An off-list `any` particle, and the hand-built
archetype fallback for uncovered species, retain feasible stat-point intervals
that evidence narrows instead of overconfidently killing the set. Archived
layout 2 encoded the archetype posterior; current layout 3 encodes the inferred
nature posterior from the objective spread prior.

**Protect counter** ([data.py](data.py), [env.py](env.py), [tokenizer.py](tokenizer.py)) tracks the
consecutive-protect (stall) counter for both sides — the "can I Protect again
risk-free" flag — verified against the pinned sim (the counter triples per
consecutive use; Wide/Quick Guard never fail from it but do increment it).
`env.reconstruct` now rebuilds the stall volatile, closing part of the
documented Protect-streak reconstruction gap; `env.py --selftest` asserts it.

---

# For the human reviewer — a control-flow primer

Everything below reads the repo in the order the pipeline actually runs.
Each numbered stage is one runnable script; the support modules are described
where control flow first enters them. Config knobs live in one dataclass —
[config.py](config.py) — so "change model size / format / search budget" is always a
one-line edit; every module takes `cfg=CFG` and reads only from it.

### 0. `python env.py --dump-dex | --benchmark | --selftest` — the simulator

[env.py](env.py) owns the battle engine. There is no Python port of the Showdown sim
worth trusting, so a ~150-line Node **sidecar** wraps the real
`pokemon-showdown` package and speaks JSON-lines over stdin/stdout:
`create / step / save / restore / reconstruct / dumpdex`. One persistent
process, no per-call startup cost.

- `--dump-dex` writes `artifacts/dex.json` (base stats, move priority,
  mega-stone mappings) straight from the sim's Champions data, so Python-side
  stat math ([beliefs.py](beliefs.py)) can never drift from the engine.
- `--benchmark` proves `save → restore → save` is byte-identical and that two
  restored forks replay identically, then measures steps/sec and forks/sec.
  Those two numbers bound the search budget (`sims_per_move`) — this was
  deliberately validated before any search code was written, because the
  whole phase-2 design leans on cheap forking.
- `--selftest` (phase 2) exercises the **reconstruct** op: rebuild a
  mid-battle position from public information only — team-preview choices put
  the right mons in the right slots, then engine-level mutations set HP,
  status, boosts, faints, consumed items, used megas, field and side
  conditions, and turns-on-field counters (Fake Out legality). Reconstruction
  is the keystone of hidden-information search: the search must be able to
  say "suppose the opponent's sets are X" and get a *playable* battle, both
  in self-play and on the live ladder where no true battle object exists.
  Run this once per machine/sim-pin before trusting search output.

Why reconstruct instead of forking the real battle and editing hidden fields?
One mechanism serves self-play *and* live play, and it enforces CTS hygiene
mechanically — the searcher literally cannot read the opponent's true sets,
because the battles it searches never contained them.

### 1. `python data.py download|parse|prep` — dataset pipeline

[data.py](data.py) turns raw battle logs into training shards in three steps:

- **download**: four Hugging Face log files (Reg M-A/M-B, bo1+bo3).
- **parse**: `LogParser` replays each log's protocol lines and emits, per
  turn, a **CTS-observable state from both players' perspectives** (own full
  team; opponent = preview species + revealed moves/items/abilities + visible
  HP%/status/boosts), the joint action each player actually chose (the
  labels), and an **event stream** (reveals, move order, observed damage)
  that later drives the belief filter. The full opponent sheets are kept
  only as oracle labels. Also written: the vocab name lists and per-species
  set **usage stats** from the train split (these become the belief prior).
  `parse()` is a thin loop over `feed(line)` — the same incremental method
  the live tracker uses at play time, so dataset parsing and live play
  share one battle-state implementation (that reuse is why phase 2 needed
  almost no new state-tracking code).
- **prep**: for every battle and both perspectives, run the belief filter
  turn by turn, compute damage features, tokenize, and write npz shards.
  Splitting is by **bo3 match id, never by transition** (two turns of one
  game in different splits is leakage); sample weights combine rating,
  format and recency.

Support modules entered here:

- [actions.py](actions.py) — the doubles action space. Each slot has 39 indexed actions
  (pass / 4 moves x 4 target codes x mega flag / 6 identity-based switches);
  a joint action is a pair flattened into the current 39x39 joint-policy head.
  The few globally illegal combinations (double mega, same switch target) are
  masked in the model, and position-specific legality is applied by the policy
  prior. Legacy two-slot-head checkpoints are recombined into the same joint
  inference contract.
  Switch actions index **team-preview order**, which is stable all game,
  and are translated to Showdown's volatile party positions only when a
  choice string is built.
- [beliefs.py](beliefs.py) — `OpponentBelief`, a particle filter over each opponent
  mon's full set. Particles are the distinct sets seen in train-split sheets,
  weighted by frequency. Update rules: reveals are hard constraints;
  same-priority move order implies effective-speed inequalities (this is
  what concentrates mass on choice-scarf/fast variants); damage they deal to
  us is a tight likelihood (our defenses are known exactly); damage we deal
  to them loosely constrains HP x defense. Sheets redact stat points and nature
  (the Champions 66-point / 32-per-stat system), so the filter crosses each set
  with objective builds from `spreads.json`. Exact builds are tested at their
  recorded stats; off-list and uncovered-species paths maintain feasible 0..32
  attack/speed intervals and conservative defensive bounds
  (`Config.investment_slack` is only the fallback when dex data is missing).
  If evidence kills every particle the
  weights rebuild from the prior (counted — see `--audit` below). A filter
  was chosen over a learned belief net because every update rule is
  physics-checkable against the damage calculator, and its failure mode
  (depletion) is observable and measurable.
- [damage.py](damage.py) — a persistent Node bridge to `@smogon/calc` (there is no
  maintained Python port; writing one would be a correctness tarpit). Used
  forward for the tokenizer's damage matrix and *in reverse* by the filter
  as a likelihood function. Requests are cached by canonical JSON — that
  cache is what makes per-particle likelihood evaluation and in-search
  damage features affordable. Requests are written/read in small chunks
  because Windows pipes deadlock on bulk writes.
- [tokenizer.py](tokenizer.py) — `PositionTokenizer`, CTS state → **fixed-length sequence
  of 561 tokens (current layout 3)** with a fixed layout (position i always means the same
  thing, so learned positional embeddings carry the structure and no
  schema/type embeddings are needed). Layout versions remain loadable from
  archived `vocab.json` files. Current blocks include
  field flags, 6+6 mon blocks with public Protect counters, per-opponent-mon
  belief tokens (modal item/nature, posterior mass, speed range, and bulk), and
  a 6x4x6 damage matrix as **(min, max)
  roll-bucket pairs** — two bounds fully describe a Showdown damage roll
  because the game draws uniformly from 16 evenly spaced multipliers in
  [0.85, 1.00]. `encode()` asserts the layout on every call.

### 2. `python train.py` — behavior cloning

Standard supervised training: weighted cross-entropy on the joint action
(both perspectives of every game), MSE on final outcome for the value head,
and an auxiliary set-prediction loss (~0.2 weight) on the oracle sheets.
bf16 autocast on CUDA, AdamW, cosine LR with warmup, grad clipping,
resumable checkpoints (`ckpt_last.pt` / `ckpt_best.pt`), TensorBoard + a
plain terminal table. [models/policy_value.py](models/policy_value.py) is a vanilla ~6-layer
pre-norm transformer encoder (~5.5M params — deliberately small: it must run
2x per node expansion *inside* the search). The aux head reads the six
opponent-species token positions; it exists to shape representations toward
hidden-set inference and to cross-check the particle filter, not to feed
search. `predict_batch()` is the single inference entry point everything
downstream uses.

### 3. `python evaluate.py` — is pruning safe?

Consumes the model's common joint-distribution inference contract (native joint
head), applies the same legality rules used by
search, and scores the checkpoint on held-out battles against [models/baselines.py](models/baselines.py)
(max-damage and random floors). The headline metric is **pruned-set
recall@k**: how often the human's actual joint action sits inside the model's
top-k. Search only expands the top-k joints per player, so this number *is*
the probability that pruning throws away the move a strong human would have
played. Perplexity and ECE matter because the raw probabilities seed the
search priors.

### 4. `python beliefs.py --audit` — measuring the filter's blind spot

The filter can only converge to sets that exist in its prior. The audit
replays held-out battles through the filter and reports: how often the true
(oracle) set was in the prior at all, how much posterior mass it ends with,
and how often evidence killed every particle (per species, with train-set
counts — rare mons deplete most). This quantifies the known blind spot
(novel sets / custom EV spreads); if the numbers are bad, the fix is a wider
prior (e.g. EV-spread variants inferred from damage residuals), not filter
patches. It also times the damage bridge to confirm where the cost lives.

### 5. `python scenarios.py` — search correctness gates

The versioned chooser orchestrator lives at
[agents/determinized_duct/v1.py](agents/determinized_duct/v1.py). Search mechanics
are split between the reusable versioned brick and node implementation:

- [search/node.py](search/node.py) — the **decoupled UCT** node. Simultaneous turns break
  alternating-move UCT and alpha-beta: modeling the opponent as moving
  *after* you converges to pure strategies that a human immediately exploits
  (always Sucker Punch / never Sucker Punch). Instead each node keeps two
  independent PUCT bandit tables, one per player, each seeded with that
  player's policy prior; both players "select" simultaneously and the joint
  pair indexes the child. Visit distributions then converge toward mixed
  strategies at equilibrium points. Values are stored from the searcher's
  perspective; the opponent table accumulates the negation.
- [agents/search/v1.py](agents/search/v1.py) — traversal, simulation stepping,
  rollout, backup, and root aggregation. [search/mcts.py](search/mcts.py) keeps
  reconstruction/orchestration in `DeterminizedDUCTChooser`. Per
  decision the chooser samples K opponent teams
  from the belief filter (**determinization**); reconstruct the public state
  once per sample with those sets as ground truth; run `sims_per_move / K`
  simulations per determinization on forks of that root. A simulation
  restores a fork, walks the tree applying both PUCT picks as real sim steps
  (the RNG rolls fresh each visit, so damage ranges/accuracy/crits are
  averaged implicitly — no explicit chance nodes), expands one leaf,
  evaluates it with the value head from the searcher's own perspective, and
  backs up both tables. **Leaf evaluation is the value head — there is no
  HP%-sum heuristic anywhere.** In endgames (≤ `solve_endgame_at` mons per
  side) the value head is ignored: no pruning, and every simulation runs to
  actual game end, so leaves are exact wins/losses. Root visit counts are
  aggregated across determinizations by action identity; that distribution
  is the output **mixed strategy**, sampled with `play_temperature`
  (0 = argmax for evaluation).

[scenarios.py](scenarios.py) is the acceptance test for all of that. The headline scenario
is **Metagross vs Kingambit 1v1** (Metagross at 70% so Sucker Punch's min
roll KOs): Bullet Punch outprioritizes and blanks Sucker Punch (target
already moved), Hammer Arm OHKOs at 4x but eats Sucker Punch first, Kowtow
Cleave punishes the Bullet Punch line — matching-pennies structure, so a
correct simultaneous-move search **must** output a mixed strategy
(assertion: both Metagross options ≥ 20%). A pure answer here is
the signature failure of sequential-move search. Two more authored endgames
assert won positions are recognized (value ≥ threshold, top action attacks),
and a Trick Room gate asserts the priority-chip line (Sucker Punch shrinking
a full-HP Eruption below Garchomp's bar) is found exactly in solve mode.

Scenarios are **not endgame-only**: a set of earlygame/midgame diagnostic
positions (full or near-full teams, back mons, megas, weather) probes model
understanding rather than search correctness — switching an endangered mon
out, weather-war control (snow Tailwind race, a predicted Pelipper switch-in
that flips sun to rain and third-cuts Heat Wave), and Contrary boost lines
(self-Tickle Mega Staraptor, including the Rage-Powder-vs-Follow-Me
redirection asymmetry: powder moves cannot redirect a Grass-type's Tickle,
Follow Me can). These need a checkpoint (value-head leaves), print NOTEs
instead of gating, and are meant to be tracked across checkpoints.
The runner prints the actual damage matrix via the calc bridge — with the
scenario's weather, mega formes, and fainted-ally state applied — so you can
verify each position is what the assertion assumes. `--mine` extracts real
2v2-or-smaller endgames from held-out replays into `artifacts/endgames.json`;
`--replay N` runs the search on one so vetted positions can be promoted into
the scenario list with documented expected behavior. Scenario "1v1" teams
carry a pre-fainted teammate because a doubles side that never had a second
mon leaves a null active slot the sim's choice code doesn't expect — and
real endgames always have fainted teammates anyway.

### 6. `python observe_game.py [--step]` — watch it think

Two bots share one sidecar battle. Each bot is honestly closed-sheet: it sees
its own team, the opponent's preview species, and nothing else — its tracker
(the same `LogParser.feed`) and particle filter learn the rest from the
protocol stream, even though the process holds both true teams. Per turn it
prints: the belief posterior per opponent mon ("kingambit: spe 90-142,
blackglasses 64%, focussash 22%"), the model's predicted opponent joint
actions, the bot's own mixed strategy with probabilities, and the value
estimate. `--step` pauses for Enter before each turn resolves; `--p2 random`
gives a floor opponent; `--temp 0` plays argmax.

### 7. `python env.py --live [ckpt]` — the ladder backend

`make_live_player` (in [env.py](env.py)) subclasses poke-env's `Player`, but uses
none of poke-env's battle model: raw protocol lines are intercepted in
`_handle_battle_message` and fed to the same tracker/beliefs/searcher stack
as self-play; orders go out as raw Showdown choice strings
(`SingleBattleOrder` accepts them verbatim). poke-env is kept for what it is
good at — websockets, auth, challenge handling — and bypassed for state,
so live play and self-play cannot disagree about what the bot believes.
Needs a local Showdown server; `--ladder` queues matchmaking instead of
accepting challenges.

### 8. `python play.py` — human vs bot

A thin orchestration layer over stage 7: it spawns the local server, prints
the client URL and your chosen replica team ([teams.py](teams.py)), plugs one of four
choosers (determinized DUCT / policy-only / max-damage / random — all behind the same
`.choose()` seam) into `make_live_player`, and serves a zero-dependency
dashboard (stdlib `http.server`) fed by the live player's `on_decision`
callback. The dashboard's "it expects YOU to…" bars are the searcher's
opponent priors — the same numbers that seed the opponent's bandit table in
the search, so the display is exactly what the bot acts on, not a separate
estimate.

### 9. Debug modes — finding slow code and model weaknesses

All gated behind flags; zero cost when off ([search/debug.py](search/debug.py)):

- **`--debug`** (scenarios.py, observe_game.py, play.py): per-decision
  printout with four parts. (a) A **phase profiler** — wall time bucketed
  into fork / sim step / forced switches / tracker copy / damage features /
  tokenize / net — chosen over generic flamegraphs because the search is
  RPC-bound and a Python flamegraph would mostly show pipe reads; the
  buckets map one-to-one onto tuning levers. (b) **Health counters**:
  invalid-action fallback rate (THE reconstruction-fidelity metric — rising
  means the rebuilt battles drift from real ones), forced-switch counts,
  terminal vs value-head leaves, tree depth. (c) **Per-determinization root
  tables** (N/Q/prior per action) — disagreement between determinizations is
  belief uncertainty showing up in the search. (d) The **root particle
  monitor**: per opponent mon, effective sample size, entropy, depletions,
  hard constraints, speed range and top particles; in self-play
  (`observe_game.py --debug`) each is also graded against the true set
  (in-prior? rank? mass?) since the process knows both teams.
- **`--cprofile out.prof`** (scenarios.py, observe_game.py): Python-level
  cProfile dump for snakeviz/speedscope, complementing the phase buckets.
  For sampling flamegraphs run py-spy externally:
  `py-spy record -- python scenarios.py`.
- **[profile_selfplay.py](profile_selfplay.py)**: aggregate throughput
  profiler for game playing — plays real games through the self-play
  skeleton with the phase profiler on and reports moves/min, sims/s,
  per-move latency percentiles, the phase time table summed over the whole
  run, and net batching stats (calls, avg batch size, positions/s). Where
  `--debug` answers "what did this move cost", this answers "what bounds
  data generation". Runs a random-init baseline-architecture net when no
  checkpoint is present. `python profile_selfplay.py --games 3` or
  `--max-decisions 20 --sims 100` for a quick read; `--cprofile` works
  here too.
- **`evaluate.py --worst N`**: decodes the N held-out positions where the
  model assigns the human's actual move the least probability — the fixed
  token layout decodes back into a readable position, so you see exactly
  which situations (Protect timing, switches, doubles targeting) the model
  systematically misreads.
- **`evaluate.py --aux`**: auxiliary-head accuracy against oracle sheets
  (item/ability top-1, moves hit@4) — the learned counterpart to
  `beliefs.py --audit`; comparing the two says whether hidden-set inference
  is bottlenecked by the filter's prior or by the net.
- **bridge cache stats** are printed with `--debug` (hit rate, size): the
  damage calculator is the usual prep/search bottleneck and the cache hit
  rate says whether it is amortizing.

## Cross-branch tournaments — the agent pile

`benchmark.py` compares checkpoints that share this checkout's code. The
*pile* compares agents that need not share any code: each contestant is an
exported bundle run as a black-box subprocess speaking a small JSON-lines
game protocol, so experiment branches with incompatible tokenizers, action
spaces, or no neural net at all can still play each other — no merge needed.

- [export_agent.py](export_agent.py) snapshots the working tree + behavior
  assets (checkpoint, vocab, usage, dex, spreads — none of which travel by
  git) into an immutable bundle under the pile (default `../vgc-pile`,
  shared by sibling worktrees; override with `--pile` / `$VGC_PILE`).
- [agent_server.py](agent_server.py) is the subprocess adapter: it wraps any
  chooser behind the protocol (hello / game_start / lines / request →
  choice / game_end). Experiment branches extend `build_chooser` with new
  kinds; defaults reproduce `benchmark.run_game` behavior exactly.
- [round_robin.py](round_robin.py) is the coordinator: it owns the one
  battle engine both sides play on, assigns the replica teams with the same
  pairing grid and side alternation as `benchmark.py`, and appends results
  to `<pile>/results.jsonl` (they travel with the pile). Every row records
  each side's seconds/move; `--move-budget` optionally enforces a per-move
  clock. Crashed or hung agents forfeit, with stderr kept in `<pile>/logs/`.

```bash
python export_agent.py rr-baseline --agent search --notes "frozen 1x"
python export_agent.py rr-random --agent random
python round_robin.py list
python round_robin.py play rr-random rr-baseline --quick 10
python round_robin.py star --anchor rr-baseline   # connected BT graph, N series
python round_robin.py standings
```

The star schedule (everyone vs one anchor) keeps the Bradley–Terry pairing
graph connected at N series instead of N(N−1)/2; save `all` for finalists.
See `EXPERIMENT_BRIEF.md` for the rules experiment branches must follow so
their bundles compose here.

### Experiment branch: team preview + forced switch-ins (exp/lead-switch)

The stock adapter answers team preview with `team 1234` and forced switches
(faints, Roar, U-turn/pivots) with a random legal pick — the two decisions
the frozen chooser never sees. This branch measures how much Elo those two
blind spots cost, **without touching the base model or any shared behavior
file**. [lead_switch_server.py](lead_switch_server.py) subclasses the stock
adapter and routes exactly those two request kinds through pluggable
selectors under [agents/lead_switch/](agents/lead_switch/):

- [agents/lead_switch/matchup.py](agents/lead_switch/matchup.py) — belief-aware
  pairwise damage/speed tables from the `@smogon/calc` bridge (type-chart
  fallback without one); the primitive every selector shares.
- [agents/lead_switch/expert.py](agents/lead_switch/expert.py) — hard-coded
  selectors: preview scored over all 90 bring/lead combos against a
  softmax-weighted model of the opponent's 15 lead pairs plus a bring-4
  coverage term; forced switches use the classic trainer-AI recipe (best
  damage out, danger-weighted damage in, speed tiebreak).
- [agents/lead_switch/value.py](agents/lead_switch/value.py) — the FROZEN
  baseline net's value head evaluates hypothetical post-decision positions
  (tokenized exactly as in training); preview becomes an expert-pruned
  payoff matrix scored with an opponent-model/maximin blend.
- [agents/lead_switch/leadnet.py](agents/lead_switch/leadnet.py) +
  [train_leads.py](train_leads.py) — LeadNet, a ~1M-param transformer trained
  on the dataset's human preview choices (leads are public in turn 1;
  brought mons are the ones that appeared), outcome- and rating-weighted.
- [agents/lead_switch/lscfg.py](agents/lead_switch/lscfg.py) — every
  experiment knob (kept out of the hashed `config.py` on purpose).
- [export_lead_switch.py](export_lead_switch.py) — presets the three bundle
  variants (`expert` / `value` / `nn`) and ships LeadNet inside the bundle.

Move requests still go through the untouched frozen chooser, so a round-robin
series between a variant bundle and the plain baseline bundle isolates the
value of positioning alone. [tests/test_lead_switch.py](tests/test_lead_switch.py)
gates the selectors without needing Node or a checkpoint.

### File-by-file summary

| file | what / why it exists |
|---|---|
| [contracts.py](contracts.py), [DATA_CONTRACTS.md](DATA_CONTRACTS.md) | Runtime-neutral `TypedDict`/aliases plus the exhaustive per-function input/output catalog used for manual review. |
| [agents/interfaces.py](agents/interfaces.py) | Small `MoveChooser`, belief, encoder, prior, evaluator, and search protocols. Gameplay depends on these seams instead of a monolithic concrete search class. |
| [agents/ids.py](agents/ids.py), [agents/registry.py](agents/registry.py) | Stable versioned implementation IDs and the fail-closed allow-list that constructs archived agents. Also records/verifies behavior-source identities. |
| [agents/spec.py](agents/spec.py) | Typed `AgentSpec`/`BrickSpec` manifest, safe archive-relative path resolution, and authoritative brick-config restoration. |
| [agents/determinized_duct/v1.py](agents/determinized_duct/v1.py) | Current Elo-rated full-search architecture. Its algorithmic, versioned name is what new archives record. |
| [agents/encoding/v1.py](agents/encoding/v1.py), [agents/priors/v1.py](agents/priors/v1.py), [agents/evaluators/v1.py](agents/evaluators/v1.py), [agents/search/v1.py](agents/search/v1.py), [agents/beliefs/v1.py](agents/beliefs/v1.py) | Frozen v1 bricks: exact token path, legal masking/top-k, batched policy/value evaluation, DUCT mechanics, and particle belief identity. New behavior gets v2 rather than changing an archived ID. |
| [agents/policy_only/v1.py](agents/policy_only/v1.py), [agents/max_damage/v1.py](agents/max_damage/v1.py), [agents/random/v1.py](agents/random/v1.py) | Alternative complete chooser architectures behind the same gameplay contract. |
| [agents/evaluation.py](agents/evaluation.py) | Append-only, comparable brick quality/latency evaluations stored as JSONL. |
| [config.py](config.py) | The one dataclass with every knob (format, paths, model dims, search budget). Exists so experiments are one-line edits and nothing reads hidden globals. |
| [env.py](env.py) | Node sidecar around the real Showdown sim (create/step/save/restore/reconstruct) + benchmark/selftest + the poke-env live backend. Exists because the sim must be the real engine, forkable, and rebuildable from public info. |
| [data.py](data.py) | Log download/parse/prep; `LogParser` doubles as the live battle tracker. Exists to produce CTS-honest training data with oracle labels, splits and weights done correctly once. |
| [actions.py](actions.py) | Doubles slot/joint action space, masks, position legality, and Showdown choice strings. Exists so model, search, data and env all agree on what an action *is*. |
| [damage.py](damage.py) | Cached JSON-lines bridge to `@smogon/calc`. Exists because damage math must match the games's exact formulas, forward (features) and backward (likelihoods). |
| [beliefs.py](beliefs.py) | Particle filter over opponent sets + `--audit`. Exists to turn the hidden-information problem into concrete sampled sets the search can determinize over. |
| [tokenizer.py](tokenizer.py) | Layout-3 position tokenizer (561 tokens) + vocab building. |
| [models/policy_value.py](models/policy_value.py) | Small joint-policy transformer with value and aux set prediction; `clean_state_dict` keeps torch.compile out of checkpoints. |
| [selfplay.py](selfplay.py) | AlphaZero-style self-play: batched-GPU generation, replay buffer, soft-target training, per-iteration gate. Exists to let the bot learn from its own play, not just human clones. |
| [benchmark.py](benchmark.py) | Frozen baseline plus immutable full-agent archives, model-vs-model series, and architecture-grouped Elo/BT standings. |
| [models/baselines.py](models/baselines.py) | Batched random and max-damage predictor baselines used by offline evaluation; gameplay chooser versions live under `agents/`. |
| [train.py](train.py) | Behavior cloning loop with AMP/cosine/resume. Exists to fit the network; nothing else trains anything. |
| [evaluate.py](evaluate.py) | Predictor metrics, headline pruned-set recall@k. Exists to certify that top-k pruning is safe before search trusts it. |
| [evaluation_common.py](evaluation_common.py) | Shared held-out loading and policy metric calculations for evaluation scripts. |
| [probe_policy.py](probe_policy.py) | Focused policy sanity and behavior probes used before expensive search benchmarks. |
| [agent_server.py](agent_server.py) | Stdio adapter serving one chooser behind the JSON-lines game protocol. Exists so a contestant is a black-box process, not a Python import. |
| [export_agent.py](export_agent.py) | Writes immutable, self-contained agent bundles (source snapshot + assets + manifest) into the shared pile. Exists because git carries code, never agents. |
| [round_robin.py](round_robin.py) | Cross-branch tournament coordinator: one fair engine, subprocess contestants, pile-local ledger + Bradley-Terry standings. Exists so incompatible experiment branches can still be compared by Elo. |
| [cli_help.py](cli_help.py) | Shared, dependency-free `-h`/`--help` text for every script entry point. |
| [search/node.py](search/node.py) | Decoupled-UCT node (two PUCT tables). Exists because simultaneous turns demand per-player statistics — the whole reason vanilla UCT is banned here. |
| [search/mcts.py](search/mcts.py) | Determinized DUCT reconstruction/orchestration; delegates encoding, priors, evaluation, and mechanics to versioned bricks. |
| [search/debug.py](search/debug.py) | Phase profiler, cProfile hook, root particle monitor, per-det root tables. Exists so "why is it slow / why did it do that" has a flag instead of an archaeology session. |
| [scenarios.py](scenarios.py) | Endgame gates (mixed-strategy assertion) + earlygame/midgame model diagnostics (switch-outs, weather wars, Contrary boost lines) + real-replay endgame mining. Exists so search regressions fail loudly instead of costing Elo silently. |
| [observe_game.py](observe_game.py) | Step-through self-play viewer of beliefs/predictions/strategy/value. Exists because a bot you can't watch thinking can't be debugged. |
| [replays.py](replays.py) | Terminal search over saved replays + local server that opens them in the Showdown replay player (VS Code port-forward friendly). Exists because hundreds of saved games are useless if you can't find the one you need. |
| [teams.py](teams.py) | 10 replica Reg M-B teams (real tournament archetypes) + export parser + sim-validator + dataset team miner + the mined/imported self-play team pool (`--build-pool`). Exists so human games start from realistic, legal, varied matchups — and so self-play sees enough distinct teams not to memorize pool artifacts. |
| [play.py](play.py) | Human-vs-bot orchestration: local server spawn, chooser menu, live "bot thoughts" dashboard. Exists so a human can actually fight — and read — the bot without any custom battle GUI. |
| [tests/test_agents.py](tests/test_agents.py), [tests/test_documentation.py](tests/test_documentation.py), [tests/conftest.py](tests/conftest.py) | Modular/archive contracts, documentation coverage/alignment gates, and pytest fixtures for real Showdown/calc integration tests. |
| [colab.ipynb](colab.ipynb) | End-to-end pipeline on any Jupyter GPU box (remote A100/L4, Colab Pro) — rootless Node bootstrap, skip-if-present data download. Exists so the whole pipeline runs on rented strong compute. |

### Scaling to strong hardware

The code is laptop-debuggable but sized for a big box; the knobs that matter:

- **Training** (`config.py`): raise `batch_size` (A100 takes 1024+ at d=256),
  `num_workers`, `epochs`. bf16 autocast is already on for CUDA.
- **Search**: `sims_per_move` and `n_determinizations` scale linearly with
  sim throughput — re-run `env.py --benchmark` on the target machine and set
  `sims_per_move` so `sims x (1 fork + ~2 steps)` fits the ladder turn timer.
  `top_k_actions` widens with recall@k. The damage-request cache warms up
  across a game, so the first turns are the slowest.
- **Prep**: `data.py prep` is embarrassingly parallel over battles if it ever
  becomes the bottleneck (it is bridge-bound; check `beliefs.py --audit`'s
  timing split first).
- Deliberately sequential in v1 (documented levers, not yet code):
  determinizations could run on parallel sidecars, and leaf evaluations could
  be batched across simulations with virtual loss.

### Known limitations (v1, accepted and documented)

- Demonstrators saw open sheets; the model sees CTS reconstructions. Bias is
  accepted for v1 (a non-OTS fine-tune pass is the documented follow-up).
- Team sheets redact stat training and nature. The normal prior therefore uses
  external objective spread/nature marginals; off-list or uncovered builds use
  feasible stat-point intervals and conservative defensive bounds rather than
  exact hidden stats. Authored scenario sets do carry exact spreads through
  `determinized()`; `beliefs.py --audit` measures the remaining prior gap.
- Transitions where a slot's choice is unobservable (flinch/sleep/KO'd before
  acting — ~24%) are dropped rather than partially labeled.
- Team preview (lead selection) is not modeled; bots bring the first four.
- Reconstruction restores the public Protect streak counter, but still drops
  choice lock, encore/taunt/substitute, spent PP, and exact weather/screen
  durations. The search is
  therefore slightly too pessimistic about what a choice-locked opponent can
  do.
- Forced switch-ins *inside* simulations are random (the following turn's
  search corrects); at the root they are handled by a trivial policy, not
  search.
- Within a determinization the simulated opponent "sees" our true sets —
  standard determinization paranoia; it overestimates the opponent.
- Opponent brought-4 under CTS is inferred (appeared mons + preview order
  fill), so the search may let the opponent switch to a mon they didn't
  actually bring.
- The pinned Showdown/`champions` mod sim lags the live game's movepool in
  places: real Reg M-B tournament teams pulled by `teams.py --fetch-pool`
  included legal current-game sets the pinned sim's `champions` learnsets
  reject (seen: Icy Wind on Archaludon/Hydreigon/Rotom-Frost, Moonblast on
  Ninetales-Alola — 378 of 1226 fetched pastes, 2026-07). Those teams are
  correctly dropped by the TeamValidator rather than force-included, since
  self-play must stay sim-legal, but it means the self-play team pool is
  missing some real, currently-legal sets until the sim pin is bumped.
