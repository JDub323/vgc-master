# Standing brief for experiment agents

You are running **one** experiment on the vgc-bot codebase, on your own branch,
in your own worktree, in parallel with other agents running *different*
experiments on theirs. You will never see their work; do not try to coordinate
with them or guess what they are doing.

The blades converge at the end in a **round robin**: every experiment exports a
self-contained agent bundle into a shared pile, and a coordinator plays them
against each other for Elo. An experiment that produces a brilliant result but
cannot be exported and played is worth nothing to this process.

Read `README.md` for what the bot is, and the codebase primer for how the
algorithms work. This document is only about *how to land an experiment so it
composes with everyone else's*.

---

## 0. The frame: the interface is the game protocol, not any class

**The tournament never imports your code.**

`round_robin.py` owns the one battle engine both sides play on. It runs each
contestant as a **black-box subprocess** and speaks a small JSON-lines protocol
to it: here are protocol lines, here is a sim request, give me a Showdown
choice string. What is inside — transformer, hand-coded search, seq2seq, a
Magic 8-Ball — is invisible to it and irrelevant.

That is the whole design, and it has one enormous consequence:

> **Nothing has to merge.** Two blades with incompatible tokenizers, different
> action spaces, or no neural network at all play each other on day one. No
> git discipline can reconcile "two definitions of what a position is" — so we
> don't try. The bundle snapshots your source; the protocol is the only thing
> both sides agree on.

This is the chess-engine (UCI) model. It is why you should feel free to make
deep, incompatible changes if your experiment calls for them — and why the
rules that *do* still bind you are narrower than they look.

## 1. Declare your lane, in your first report

Two lanes. Pick one deliberately; say which.

**Pile-only** — you are exploring something that cannot merge (new tokenizer
layout, new action space, no transformer, seq2seq). **Change anything you
want.** Your bundle carries its own source, so trunk compatibility is not your
problem. §3 does not apply to you.

**Mergeable** — you are improving the current DUCT system and want the change
to land on trunk and be comparable through `benchmark.py`'s in-repo ladder
(checkpoint swaps, brick swaps, knob changes). Then §3 binds.

Most structural rewrites are pile-only. Most tuning is mergeable. **When in
doubt, pick pile-only**: it costs nothing, and a pile-only experiment that
turns out to be worth keeping can be re-landed additively later, whereas a
blade that over-constrains itself protecting an archive path it never uses has
simply wasted effort.

## 2. Speak the protocol

Your experiment is done when it is a **bundle the coordinator can run**. There
are two paths to that, and you should decide which one you are on early.

### Path A — your agent fits the `MoveChooser` shape (most experiments)

If your idea can be expressed as "given the tracker, belief, and request,
return a joint action," implement `MoveChooser` from `agents/interfaces.py`:

```python
def choose(self, tracker, belief, my_id, request, brought,
           opp_brought=None, temperature=None, root_noise=None
           ) -> tuple[JointAction, ChoiceInfo]:
```

Then register a kind in `agent_server.build_chooser` (one `if` branch on your
branch) and export with `--agent <kind>`. `agent_server.py` handles the
protocol for you.

If you are only swapping *part* of the search (a different leaf evaluator,
prior, or tree policy), you want a **brick**, not a whole chooser: implement
the relevant protocol in `agents/interfaces.py` (`PositionEncoder`,
`PolicyPrior`, `LeafEvaluator`, `Searcher`, `BeliefModel`) and inject it.
Bricks compose; whole choosers don't.

### Path B — it doesn't fit

Then **do not contort it to fit.** `agent_server.py` is a *convenience
implementation* of the protocol for `MoveChooser`-shaped agents — it is not the
contract. The contract is the protocol itself.

Write your own server that speaks it and point the bundle at it:

```bash
python export_agent.py exp-my-thing --entrypoint "python my_server.py --flag" \
    --ckpt artifacts/checkpoints/my_weights.pt --architecture "MyThing"
```

The manifest records that command; the coordinator runs it from the bundle's
`src/` with `$VGC_NODE_DIR` set. Nothing else about your agent is inspected.

### The protocol (v1)

One JSON object per line. `agent_server.py`'s module docstring is the full
spec; the shape is:

```
-> {"type":"hello","protocol":1}                    <- {"type":"ready", ...}
-> {"type":"game_start","side":"p1","team":[...],
    "opp_preview":[...],"seed":N,"temperature":T}   <- {"type":"game_ready"}
-> {"type":"lines","lines":["|move|...", ...]}         (no reply)
-> {"type":"request","rqid":N,"request":{...},
    "deadline_s":null}                              <- {"type":"choice","rqid":N,
                                                        "choice":"move 1 2, move 3"}
-> {"type":"game_end","winner":"p1"}                   (no reply)
-> {"type":"quit"}
```

Four rules that will otherwise cost you a day:

- **stdout is the protocol channel.** A stray `print` corrupts the stream.
  `agent_server.py` redirects `sys.stdout` to stderr and keeps a private handle
  for protocol writes; do the same.
- **You answer every request kind**, including team preview and forced
  switches. The default adapter reproduces the standard behavior (`team 1234`,
  random legal switch) so `baseline` is unchanged — but improving on either is
  a legitimate experiment, and this is the only harness where you *can*.
- **`deadline_s` is advisory.** The coordinator enforces budgets; you cannot
  preempt your own search. Answer as fast as you can and let it judge.
- **Crashing or hanging forfeits the game.** Stderr is kept under
  `<pile>/logs/`.

## 3. Mergeable lane only: be additive

Skip this section if you declared pile-only.

**Add a new agent. Do not modify the existing one.** The trunk carries a frozen
contestant named `baseline` (the 1× joint-policy / layout-3 model). It is the
anchor of the rating system. If you change how it behaves, every previously
recorded result becomes uninterpretable.

These ten files are **shared behavior**, hashed into every archive ever made
(`agents/registry.py BEHAVIOR_SOURCE_FILES`):

```
actions.py  beliefs.py  config.py  damage.py  data.py
env.py  models/policy_value.py  search/mcts.py  search/node.py  tokenizer.py
```

Any **logic** edit to one invalidates the source hash of every existing
archive, including `baseline`, and they fail closed. Comments and docstrings
are safe (hashing is ast-v1 — the docstring-stripped AST); a new `if` branch is
not.

- Put everything under `agents/<your-name>/v1.py` if you can.
- Take a stable ID in `agents/ids.py` (`MY_THING_V1 = "agents.my_thing.v1.MyThingChooser"`)
  and register it in `agents/registry.py` (`REGISTRY.register_agent(...)`). The
  registry is a fail-closed allow-list — no `importlib` fallback, so an
  unregistered ID raises rather than silently running today's code.
- `v1` means "this exact behavior, forever." Changing behavior means `v2`; you
  do not edit `v1`.
- **Never** change the default of an existing `config.py` field. Add a new one,
  defaulted to today's behavior.

Note `benchmark.py archive` hardcodes the five-brick DUCT manifest, so it can
only freeze DUCT-family agents. Anything else goes to the pile — which is
everything, so this is a fallback ladder, not a restriction.

## 4. Hard constraints (both lanes)

- **`artifacts/` is gitignored — in its entirety.** Checkpoints, vocab, data
  shards, benchmark bundles. **Your branch carries code, not your agent.** If
  you train anything, the weights exist only on the machine you trained on.
  `export_agent.py` copies them into the bundle explicitly; shipping that
  bundle is part of your job (§5), not an afterthought.
- **Bundles are immutable.** The exporter refuses to overwrite a name. Pick a
  fresh, descriptive one; re-export after changes rather than editing in place.
- **Don't clobber the shared data.** `data.py prep` is hours long and the
  shards are shared. If your experiment needs a different tokenization you are
  producing *new* shards — say so, and leave the existing ones alone.
- **The gates are gates, not suggestions.** Before reporting done:
  ```
  python tests/test_documentation.py
  python tests/test_agents.py
  python scenarios.py
  ```
  Two documentation rules catch experiments specifically: every production
  function needs a **docstring** (or a `DATA_CONTRACTS.md` entry), and every
  production **module** must be named in `README.md` or `DATA_CONTRACTS.md` —
  so a new `agents/my_thing/v1.py` needs a documentation line. Add it.
  If your architecture makes a gate *meaningless* (e.g. `scenarios.py`'s
  mixed-strategy assertion against a deliberately pure-strategy agent), report
  that as a finding — do not weaken the gate.

## 5. Definition of done

You are not done when the code works. You are done when all of these are true:

1. Code committed and pushed on `exp/<short-name>`.
2. The gates in §4 pass, or you have explained precisely which one doesn't and
   why that is the experiment's result rather than its failure.
3. An exported bundle exists **and you have stated where it lives** (which
   machine, which pile path) so it can be collected for the round robin.
4. You have run a quick series against the anchor and reported it:
   ```
   python round_robin.py play exp-<short-name> <anchor> --quick 10
   ```
   `--quick 10` is triage, not a verdict — a full 100-game series carries a
   roughly ±10% Wilson interval, so 10 games tells you only "not obviously
   broken."
5. A section appended to `EXPERIMENTS.md` (§6).
6. Your lane (§1), and — if mergeable — **every shared file you touched**, or
   "none."

## 6. What to write in `EXPERIMENTS.md`

The code on the separate branch could get deleted. The conclusion must survive. `EXPERIMENTS.md` is the
durable record; one-off training scripts and giant checkpoints are deliberately
not maintained.

- **What you changed**, in one paragraph.
- **Numbers against the frozen baseline's fixed splits** — existing sections
  use 871,433 train / 46,993 val / 46,751 test transitions. Use those or
  explain why not.
- **Which metric moved and which didn't.** Validation loss and top-1 are not
  the goal; Elo is. If you only have offline metrics, say so — the existing
  damage-ablation and MLP rows are explicitly flagged as *never evaluated in
  search*, and that is the honest way to report it. Note also that
  `evaluate.py` prints policy metrics over two action sets (static mask vs
  position-legal) which are **not comparable to each other**; say which you
  are quoting.
- **The negative result, if it's negative.** The 100× scaling row exists
  precisely because it overfit and lost to the 1× baseline. That is a valuable,
  expensive fact. Do not bury yours.

## 7. Exporting to the pile

The *pile* is a plain shared directory of bundles — not a git thing. It
defaults to `../vgc-pile`, a **sibling of the repo**, so every sibling worktree
resolves to the same one with no configuration (`--pile` / `$VGC_PILE`
override). Each bundle holds a snapshot of your source (via `git ls-files`, so
gitignored paths are excluded), the behavior assets, any checkpoint (at
`artifacts/checkpoints/ckpt.pt`), and a manifest recording the entrypoint, git
provenance, and architecture label.

```bash
python export_agent.py exp-<short-name> --agent <kind> \
    --notes "one line: what this is and what changed"
python round_robin.py list
python round_robin.py play exp-<short-name> <anchor> --quick 10
```

- **The pile does not travel by git either.** If you trained on a different
  machine than the tournament box, `rsync -a` your bundle into that machine's
  pile.
- **Dirty working trees are fine** — the manifest records `dirty: true`. The
  snapshot is the truth, the commit is provenance.
- **Timing is public.** Every result row records each side's seconds/move, and
  the coordinator may enforce `--move-budget`. A slow agent is not
  disqualified, but don't hide a search-time regression.
- **The coordinator owns teams and the engine.** It assigns the replica teams
  and alternates sides; you just answer requests.

## 8. Anti-goals

- Do not tune the baseline to make your experiment look good.
- Do not widen your scope. One experiment. If you find a second idea, write it
  in your report; someone else gets that blade.
- Do not "clean up" the v1 modules, the versioning, or the registry. The
  freezing *is* the design: it is what lets your agent and a six-month-old
  agent play a fair game.
- Do not use `--allow-source-drift` to make a red test go away. It exists for
  audited emergencies; it taints every result row and flags contestants
  `*drift` in standings.
- Do not report a win from a 10-game series.
- Do not contort your idea to fit `MoveChooser` (§2, Path B). If the shape is
  wrong, that is a finding about the architecture — report it.
