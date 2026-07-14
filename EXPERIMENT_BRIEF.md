# Standing brief for experiment agents

You are running **one** experiment on the vgc-bot codebase, on your own branch,
in your own worktree, in parallel with other agents doing the same thing on
other experiments. You will never see their work. Everything below exists so
that your result and theirs can be **played against each other** at the end.

Read `README.md` for what the bot is, and the codebase primer for how the
algorithms work. This document is only about *how to land an experiment so it
composes with everyone else's*.

---

## 1. The one rule: be additive

**Add a new agent. Do not modify the existing one.**

The trunk carries a frozen contestant named `baseline` (the 1x joint-policy /
layout-3 model). It is the anchor of the whole rating system. Every experiment
is measured against it. If you change how `baseline` behaves, you have
destroyed the measuring stick, and every previously recorded result becomes
uninterpretable.

Concretely, these ten files are **shared behavior** and are hashed into every
archive ever made (`agents/registry.py BEHAVIOR_SOURCE_FILES`):

```
actions.py  beliefs.py  config.py  damage.py  data.py
env.py  models/policy_value.py  search/mcts.py  search/node.py  tokenizer.py
```

Any **logic** edit to one of these invalidates the source hash of *every*
existing archive, including `baseline`, and they will refuse to load
(fail-closed). Comments and docstrings are safe — hashing is ast-v1, i.e. the
docstring-stripped AST — but a new `if` branch is not.

- **If you do not need to touch them: don't.** Put everything under
  `agents/<your-name>/v1.py`.
- **If you genuinely must** (e.g. your experiment is a new tokenizer layout),
  that is allowed, but you must say so loudly in your final report, because it
  forces every contestant to be re-archived at integration time. Make the edit
  **purely additive** (a new layout branch, a new optional flag defaulting to
  today's behavior) so existing agents are bit-identical in behavior.
- **Never** change the default value of an existing `config.py` field. Add a
  new field, defaulted to today's behavior.

## 2. The contract

Your experiment is done when it is a **registered, archivable `MoveChooser`**.

1. **Implement the protocol.** `agents/interfaces.py` defines it. One method:

   ```python
   def choose(self, tracker, belief, my_id, request, brought,
              opp_brought=None, temperature=None, root_noise=None
              ) -> tuple[JointAction, ChoiceInfo]:
   ```

   That is the *entire* interface for playing games and earning Elo. Return a
   legal joint action and the diagnostics dict. If you cannot express your idea
   behind this signature, stop and report why — that is itself a finding.

2. **Take a stable ID.** Add a constant to `agents/ids.py`:

   ```python
   MY_THING_V1 = "agents.my_thing.v1.MyThingChooser"
   ```

   The ID is permanent and versioned. `v1` means "this exact behavior, forever."
   If you later change the behavior, that is `v2` — you do not edit `v1`.

3. **Register it.** One line in `agents/registry.py`:

   ```python
   REGISTRY.register_agent(MY_THING_V1, MyThingChooser)
   ```

   The registry is a fail-closed allow-list. There is deliberately no
   `importlib` fallback: an unregistered ID raises rather than silently running
   today's code.

4. **Make it playable over the game protocol.** Register your kind in
   `agent_server.build_chooser` on your branch (one `if` branch). If your
   agent cannot be driven by `agent_server.py`'s request/choice loop, stop
   and report why — that too is a finding.

5. **Freeze it into the pile.** `python export_agent.py <name> --agent
   <kind> --notes "..."` writes a self-contained bundle (source snapshot +
   assets + manifest). See §7.

If you are only swapping *part* of the search (a different leaf evaluator, a
different prior, a different tree policy), you probably want a **brick**, not a
whole chooser: implement the relevant protocol in `agents/interfaces.py`
(`PositionEncoder`, `PolicyPrior`, `LeafEvaluator`, `Searcher`, `BeliefModel`),
give it a `..._V2` ID, register with `register_brick`, and inject it. Bricks
compose; whole choosers don't.

## 3. Hard constraints you will otherwise discover the hard way

- **`artifacts/` is gitignored — in its entirety.** Checkpoints, vocab, data
  shards, and every benchmark bundle. **Your branch carries code, not your
  agent.** If your experiment trains anything, the weights exist only on the
  machine you trained on. Shipping the bundle is part of your job (§4), not an
  afterthought. A bundle is a self-contained directory; `rsync`/`scp -r` moves
  it fine.
- **Bundles are immutable.** `archive` refuses to overwrite an existing name.
  Pick a fresh, descriptive name. Use `benchmark.py rename` if you must.
- **Don't retrain the baseline's data.** `data.py prep` is hours long and the
  shards are shared. If your experiment needs a different tokenization, you are
  producing *new* shards — say so, and do not clobber the existing ones.
- **The test suite is a gate, not a suggestion.** Before you report done:
  ```
  python -m pytest -q
  python tests/test_documentation.py
  ```
  Two documentation rules will catch you specifically:
  - every production function needs a **docstring** (or a `DATA_CONTRACTS.md`
    entry) — a bare helper with no docstring fails CI;
  - every production **module** must be named in `README.md` or
    `DATA_CONTRACTS.md` — so a new `agents/my_thing/v1.py` requires a
    documentation line. Add it.
- **Scenario gates must still pass**: `python scenarios.py`. If your agent
  makes the Metagross/Kingambit mixed-strategy gate fail (both options ≥ 20%),
  that is a real finding about your architecture — report it, don't weaken the
  gate.

## 4. Definition of done

You are not done when the code works. You are done when **all** of these are
true:

1. Code is committed on your branch `exp/<short-name>`, and pushed.
2. `pytest -q`, `tests/test_documentation.py`, and `scenarios.py` pass.
3. Your agent is registered with a versioned ID and can be constructed.
4. If it plays: an exported pile bundle exists (`export_agent.py`), **and you
   have stated where it lives** (which machine, which pile path) so it can be
   collected for the round robin.
5. If it plays: you have run at least a quick series against the anchor and
   reported the number:
   ```
   python round_robin.py play <your-name> <anchor> --quick 20
   ```
   (or `benchmark.py play <name> baseline --quick 20` for DUCT-family
   agents). `--quick 20` is triage, not a verdict — a 100-game series has
   roughly a ±10% Wilson interval, so 20 games tells you only "not obviously
   broken."
6. You have appended a section to `EXPERIMENTS.md` (see §6).
7. You have explicitly listed **every shared file you touched** (§1), or stated
   "none."

## 5. Two freezing mechanisms — know which one you need

- **`benchmark.py archive`** freezes *DUCT-family* agents (checkpoint swaps,
  brick swaps) for the in-repo dev ladder. It hardcodes the five-brick DUCT
  manifest, so a genuinely new architecture cannot be archived faithfully
  through it — do not try.
- **`export_agent.py` + the pile** freezes *anything*. It snapshots your
  branch's source and assets into a self-contained bundle that runs as a
  subprocess behind the game protocol (`agent_server.py`), so your agent
  never needs to merge or share code to be compared. This is the tournament
  path, and the one your experiment must end on (§7).

Rule of thumb: iterating on the DUCT system → `benchmark.py play current
baseline` while you work; anything else, or anything final → export to the
pile.

## 6. What to write in `EXPERIMENTS.md`

The code may be deleted. The conclusion must survive. `EXPERIMENTS.md` is the
durable record — one-off training scripts and giant checkpoints deliberately
are not maintained.

Append a section with:

- **What you changed**, in one paragraph, including the shared files touched.
- **A table of numbers against the frozen baseline's fixed splits** — the
  existing sections use 871,433 train / 46,993 val / 46,751 test transitions.
  Use the same ones or explain why not.
- **Which metric moved and which didn't.** Validation loss and top-1 are not
  the goal; the goal is Elo. If you only have offline metrics, say so — the
  `EXPERIMENTS.md` damage-ablation and MLP rows are explicitly flagged as
  *never evaluated in search*, and that is the honest way to report it.
- **The negative result if it's negative.** The 100x scaling row is in there
  precisely because it *overfit and lost to the 1x baseline*. That is a
  valuable, expensive fact. Do not bury yours.

## 7. Exporting to the pile (the tournament handoff)

The *pile* is a shared directory of exported agent bundles (default
`../vgc-pile`, next to the repo so sibling worktrees share it; override with
`--pile` or `$VGC_PILE`). The round-robin coordinator plays bundles against
each other as **subprocesses**, so your bundle competes even if your branch's
code is incompatible with everyone else's.

Your handoff, from your worktree:

```bash
# 1. register your chooser kind in agent_server.build_chooser (your branch)
# 2. sanity-check the adapter starts:  python agent_server.py -h
python export_agent.py exp-<short-name> --agent <kind> \
    --notes "one line: what this is and what changed"
python round_robin.py list
python round_robin.py play exp-<short-name> <anchor> --quick 10
```

Facts to respect:

- **Bundles are immutable and self-contained.** The exporter snapshots your
  working tree (dirty is fine — the manifest records `dirty: true`) plus the
  behavior assets. Re-export under a new name after changes; never edit a
  bundle in place.
- **The pile does not travel by git either.** If you trained on a different
  machine than the tournament box, `rsync -a` your bundle directory into the
  tournament machine's pile. Stating where the bundle lives is part of your
  definition of done (§4).
- **Timing is recorded on every result row** (seconds/move per side), and the
  coordinator may enforce `--move-budget`. A slow agent is not disqualified —
  but its cost is public, so don't hide search-time regressions.
- **Crashes forfeit.** Your process dying or going silent past the hang
  timeout loses that game and leaves a stderr log under `<pile>/logs/`. Test
  your adapter with a `--quick` series before calling the experiment done.
- The coordinator assigns teams (the replica set) and owns the battle engine;
  your agent just answers requests. Team preview and forced-switch requests
  are forwarded to you too — the default adapter reproduces the standard
  behavior, and improving on it is a legitimate experiment.

## 8. Anti-goals

- Do not "clean up" v1 modules, the versioning, or the registry. The freezing
  *is* the design: it is what lets your agent and a six-month-old agent play a
  fair game.
- Do not tune the baseline to make your experiment look good.
- Do not widen your scope. One experiment. If you find a second idea, write it
  down in your report; someone else gets that branch.
- Do not use `--allow-source-drift` to make a red test go away. It exists for
  audited emergencies; it taints every result row and flags contestants
  `*drift` in standings.
- Do not report a win from a 20-game series.
