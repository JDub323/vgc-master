# vgc-bot

A Pokémon VGC (doubles) battling AI for the **Pokémon Champions ranked ladder**:
Regulation M-B (config-switchable to M-A), megas allowed, no Terastallization,
restricted legendaries banned, **closed team sheets** at play time.

## Architecture

Three cooperating pieces:

1. **Policy/value network** ([models/policy_value.py](models/policy_value.py)) — a small
   from-scratch transformer (~6 layers, d=256) trained by behavior cloning on
   ~89k human Champions replays. Given a tokenized position from one player's
   perspective it outputs (a) per-slot distributions over that player's actions
   (the doubles action space is factorized: two softmaxes, recombined into joint
   actions downstream), (b) a value in [-1, 1], (c) an auxiliary prediction of
   the opponent's hidden sets (representation shaping + a sanity check against
   the particle filter). Run from both perspectives it proposes "my candidate
   moves" and "opponent's likely moves"; its job in search is pruning to top-k.
2. **Belief tracker** ([beliefs.py](beliefs.py)) — a particle filter over opponent sets,
   no neural net. Priors come from train-split team sheets; reveals are hard
   constraints; speed-order and observed-damage evidence kill inconsistent
   particles (with one-sided slack because team sheets redact stat training).
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

Point `Config.node_dir` at the directory containing `node_modules` (default
`artifacts/node`; the Colab notebook symlinks it), and `Config.checkpoint_dir`
at Google Drive when on Colab.

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

## Phase 2 — search (this phase)

```bash
python env.py --selftest     # proves mid-battle state reconstruction — run first
python beliefs.py --audit    # particle-filter breakdown rates on held-out games
python scenarios.py          # endgame assertions incl. the mixed-strategy test
python scenarios.py --mine   # extract real-replay endgame candidates
python observe_game.py --step        # watch a search-vs-search game, turn by turn
python env.py --live [ckpt]          # play on a local Showdown server via poke-env
```

`scenarios.py` runs with or without a trained checkpoint (endgames are solved
to terminal); everything else in this phase needs the phase-1 artifacts.
Note that changing the tokenizer layout in this phase (belief and damage
tokens, see below) **invalidates phase-1 artifacts: re-run `data.py prep` and
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
   the probability it assigns to **your** likely actions, its belief about
   your items/speed (with sprites), and its win-confidence sparkline.

`python teams.py --mine` extracts the most common real high-rated team sheets
from the dataset — legal by construction — to swap in for any replica the
validator flags.

## Phase 3 — evaluation (next)

Bot-vs-bot round-robin Elo (random / max-damage / policy-only / policy+search)
and notebook polish. The gap between policy-only and policy+search is the
evidence that search adds value. (`Searcher.choose(policy_only=True)` and the
play.py choosers are the four contestants, already built.)

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
  a joint action is a pair. Factorizing per-slot keeps the policy head at
  2x39 outputs instead of one ~1.5k joint softmax, and the few illegal
  recombinations (double mega, same switch target) are masked downstream.
  Switch actions index **team-preview order**, which is stable all game,
  and are translated to Showdown's volatile party positions only when a
  choice string is built.
- [beliefs.py](beliefs.py) — `OpponentBelief`, a particle filter over each opponent
  mon's full set. Particles are the distinct sets seen in train-split sheets,
  weighted by frequency. Update rules: reveals are hard constraints;
  same-priority move order implies effective-speed inequalities (this is
  what concentrates mass on choice-scarf/fast variants); damage they deal to
  us is a tight likelihood (our defenses are known exactly); damage we deal
  to them loosely constrains HP x defense. Sheets redact EVs, so constraints
  get one-sided `investment_slack`. If evidence kills every particle the
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
  of 537 tokens** with a fixed layout (position i always means the same
  thing, so learned positional embeddings carry the structure and no
  schema/type embeddings are needed). Blocks: field flags, 6+6 mon blocks
  of 17, per-opponent-mon belief tokens (modal item + its probability +
  speed-range low/high + bulk), and a 6x4x6 damage matrix as **(min, max)
  roll-bucket pairs** — two bounds fully describe a Showdown damage roll
  because the game draws uniformly from 16 evenly spaced multipliers in
  [0.85, 1.00]. `encode()` asserts the layout on every call.

### 2. `python train.py` — behavior cloning

Standard supervised training: weighted cross-entropy on the two slot actions
(both perspectives of every game), MSE on final outcome for the value head,
and an auxiliary set-prediction loss (~0.2 weight) on the oracle sheets.
bf16 autocast on CUDA, AdamW, cosine LR with warmup, grad clipping,
resumable checkpoints (`ckpt_last.pt` / `ckpt_best.pt`), TensorBoard + a
plain terminal table. [models/policy_value.py](models/policy_value.py) is a vanilla ~6-layer
pre-norm transformer encoder (~17M params — deliberately small: it must run
2x per node expansion *inside* the search). The aux head reads the six
opponent-species token positions; it exists to shape representations toward
hidden-set inference and to cross-check the particle filter, not to feed
search. `predict_batch()` is the single inference entry point everything
downstream uses.

### 3. `python evaluate.py` — is pruning safe?

Rebuilds joint distributions from the per-slot factorization (outer product,
masked to legal recombinations, renormalized — exactly what search does) and
scores the checkpoint on held-out battles against [models/baselines.py](models/baselines.py)
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

The search itself lives in [search/](search/):

- [search/node.py](search/node.py) — the **decoupled UCT** node. Simultaneous turns break
  alternating-move UCT and alpha-beta: modeling the opponent as moving
  *after* you converges to pure strategies that a human immediately exploits
  (always Sucker Punch / never Sucker Punch). Instead each node keeps two
  independent PUCT bandit tables, one per player, each seeded with that
  player's policy prior; both players "select" simultaneously and the joint
  pair indexes the child. Visit distributions then converge toward mixed
  strategies at equilibrium points. Values are stored from the searcher's
  perspective; the opponent table accumulates the negation.
- [search/mcts.py](search/mcts.py) — the searcher. Per decision: sample K opponent teams
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
is **Metagross vs Kingambit 1v1**: Bullet Punch outprioritizes and blanks
Sucker Punch (target already moved), Hammer Arm OHKOs at 4x but eats Sucker
Punch first, Kowtow Cleave punishes the Bullet Punch line — matching-pennies
structure, so a correct simultaneous-move search **must** output a mixed
strategy (assertion: both Metagross options ≥ 20%). A pure answer here is
the signature failure of sequential-move search. Two more authored endgames
assert won positions are recognized (value ≥ threshold, top action attacks).
The runner prints the actual damage matrix via the calc bridge so you can
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
choosers (Searcher / policy-only / max-damage / random — all behind the same
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

### File-by-file summary

| file | what / why it exists |
|---|---|
| [config.py](config.py) | The one dataclass with every knob (format, paths, model dims, search budget). Exists so experiments are one-line edits and nothing reads hidden globals. |
| [env.py](env.py) | Node sidecar around the real Showdown sim (create/step/save/restore/reconstruct) + benchmark/selftest + the poke-env live backend. Exists because the sim must be the real engine, forkable, and rebuildable from public info. |
| [data.py](data.py) | Log download/parse/prep; `LogParser` doubles as the live battle tracker. Exists to produce CTS-honest training data with oracle labels, splits and weights done correctly once. |
| [actions.py](actions.py) | Factorized doubles action space + legality + Showdown choice strings. Exists so model, search, data and env all agree on what an action *is*. |
| [damage.py](damage.py) | Cached JSON-lines bridge to `@smogon/calc`. Exists because damage math must match the games's exact formulas, forward (features) and backward (likelihoods). |
| [beliefs.py](beliefs.py) | Particle filter over opponent sets + `--audit`. Exists to turn the hidden-information problem into concrete sampled sets the search can determinize over. |
| [tokenizer.py](tokenizer.py) | Fixed-layout position encoder (537 tokens) + vocab building. Exists as the single, swappable definition of "what the network sees". |
| [models/policy_value.py](models/policy_value.py) | Small transformer: per-slot policy, value, aux set prediction. Exists as the learned prior/evaluator; small because search calls it constantly. |
| [models/baselines.py](models/baselines.py) | Random and max-damage policies. Exist so every benchmark has a floor. |
| [train.py](train.py) | Behavior cloning loop with AMP/cosine/resume. Exists to fit the network; nothing else trains anything. |
| [evaluate.py](evaluate.py) | Predictor metrics, headline pruned-set recall@k. Exists to certify that top-k pruning is safe before search trusts it. |
| [search/node.py](search/node.py) | Decoupled-UCT node (two PUCT tables). Exists because simultaneous turns demand per-player statistics — the whole reason vanilla UCT is banned here. |
| [search/mcts.py](search/mcts.py) | Determinized DUCT searcher + solve-to-terminal endgames. Exists to convert net + beliefs + sim into an actual move decision (a mixed strategy). |
| [search/debug.py](search/debug.py) | Phase profiler, cProfile hook, root particle monitor, per-det root tables. Exists so "why is it slow / why did it do that" has a flag instead of an archaeology session. |
| [scenarios.py](scenarios.py) | Endgame assertions (mixed-strategy gate) + real-replay endgame mining. Exists so search regressions fail loudly instead of costing Elo silently. |
| [observe_game.py](observe_game.py) | Step-through self-play viewer of beliefs/predictions/strategy/value. Exists because a bot you can't watch thinking can't be debugged. |
| [teams.py](teams.py) | 10 replica Reg M-B teams (real tournament archetypes) + export parser + sim-validator + dataset team miner. Exists so human games start from realistic, legal, varied matchups. |
| [play.py](play.py) | Human-vs-bot orchestration: local server spawn, chooser menu, live "bot thoughts" dashboard. Exists so a human can actually fight — and read — the bot without any custom battle GUI. |
| [colab.ipynb](colab.ipynb) | End-to-end pipeline on Colab Pro (A100/L4), resumable from Drive artifacts. Exists so the whole pipeline runs on rented strong compute. |

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
- Team sheets redact EVs/stat training, so oracle sets and belief particles
  use 0 EVs with one-sided investment slack (`Config.investment_slack`);
  `beliefs.py --audit` measures what this costs.
- Transitions where a slot's choice is unobservable (flinch/sleep/KO'd before
  acting — ~24%) are dropped rather than partially labeled.
- Team preview (lead selection) is not modeled; bots bring the first four.
- Reconstruction drops volatiles: choice lock, Protect streaks, encore/taunt/
  substitute, spent PP, exact weather/screen durations. The search is
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
