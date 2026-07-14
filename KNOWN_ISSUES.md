# Known issues

Documented, deliberately **not** fixed on this branch — several sit inside
frozen v1 modules or hashed shared behavior files, so fixes should land
together with a versioned implementation bump (or at an archive-generation
boundary), not piecemeal. Found during the 2026-07-10 codebase review.

## Edge-case bugs

### 1. Simulations that hit the 120-ply cap back up a draw

`agents/search/v1.py DecoupledUCTSearcher.simulate` walks at most 120 plies.
If a trajectory neither reaches a terminal nor expands a new leaf within the
cap, `z` stays `None` and every node on the path is updated with `0.0` — a
silent draw. In solve mode (`solve_endgame_at`) expansion never breaks the
loop, so a long, stally endgame can genuinely hit the cap and bias root
values toward 0. No health counter records it.

**Suggested fix (v2):** count cap hits in `chooser.health` (e.g.
`h["ply_cap"]`), and either back up the leaf-evaluator value at the cap
position instead of 0.0, or make the cap configurable per solve mode. Backup
semantics change ⇒ new searcher implementation ID.

### 2. `MaxDamageChooser` joint repair can raise `StopIteration`

`agents/max_damage/v1.py`:

```python
picks[1] = next(action for action in slot_actions[1]
                if joint_ok(picks[0], action))
```

has no default. If slot 0's pick and every slot-1 legal action violate
`joint_ok` (reachable when both slots' only remaining option is switching to
the same target), the baseline crashes mid-series instead of degrading.

**Suggested fix (v2):** `next(..., SlotAction("pass"))`. Behavior-changing
for the frozen v1 baseline ⇒ `agents.max_damage.v2`.

### 3. Action labels encode targets the search can never propose

`data.py`'s `_event` infers a move's target from the protocol line, and that
inference does not agree with the action space `actions.legal_slot_actions`
enumerates from a sim request. Two cases, both measured at **8.7% of test
transitions** (random 6,000-row sample, `evaluate.py` self-check):

* **~62%** — a single-target (`normal`) move recorded as `T_AUTO`. `tcode`
  defaults to `T_AUTO` and is only overridden when the move line carries a
  target reference, so a move with no target ref (e.g. it fizzled) keeps the
  default. `legal_slot_actions` offers `normal` moves only
  `T_FOE_A`/`T_FOE_B`/`T_ALLY`, never `T_AUTO`.
* **~37%** — a spread move (`allAdjacentFoes`) recorded with a *specific*
  target. Showdown omits the `[spread]` tag when only one target is hit, so
  the parser reads the explicit target instead. `legal_slot_actions` offers
  spread moves only `T_AUTO`.

Consequence: for those rows the model is trained to put mass on a joint index
(e.g. Rock Slide at `T_FOE_B`, index 29) that search never reads, while search
queries the index it was taught is rare (Rock Slide at `T_AUTO`, index 25).
The prior therefore under-weights spread moves in exactly the positions where
one target remains, and under-weights single-target moves whose label got the
`T_AUTO` default. This is a live Elo cost, not only a metrics artifact.

`evaluate.py`'s position-legal table works around it for *measurement* by
projecting such a label onto the target codes the move admits and scoring the
resulting set (`evaluation_common.PositionLegality.label_mask`), which is why
its self-check reads 100%. That does not fix training.

**Suggested fix:** canonicalize in the parser — resolve a spread move to
`T_AUTO` regardless of the `[spread]` tag, and either resolve or drop a
single-target move with no target reference. Touches `data.py` (a hashed
behavior file) and requires re-parse + re-prep, so it wants an
archive-generation boundary. Consider pairing it with partial labels for the
~24% of transitions currently dropped for unobservable actions.

## Minor issues

### 4. Policy-only Q values always read 0.0

`DecoupledUCTSearcher.aggregate_root(..., policy_only=True)` zips
`root.my_p` (used as the count column) with `root.my_w` (all zeros at an
unsearched root), so the `q` entries in `ChoiceInfo` — and the play.py
dashboard — show 0.0 for the policy-only bot. Cosmetic, but it looks like a
value-head failure.

### 5. `sys.argv` parsing edges (all CLI scripts)

- The shared `opt()` helper does `args[args.index(flag) + 1]`; a value-taking
  flag passed as the last token crashes with `IndexError`.
- Zero is falsy in several conversions, so `--sims 0`, `--depth 0`, and
  `--quick 0` (benchmark.py) silently mean "unset".
- `env.py`'s flags are non-exclusive `if`s: `--benchmark --selftest` runs
  both, and `--benchmark`'s optional count consumes the next token whenever
  one exists (`env.py --benchmark --selftest` raises on `int("--selftest")`).

### 6. Bradley–Terry standings degenerate cases

`benchmark.py standings`: a contestant with zero wins is clamped to rating
`1e-9` and prints as roughly Elo −2100; disconnected pairing graphs within an
era still "converge" but cross-component ratings are meaningless. `elo_diff`
clips scores to [0.01, 0.99], capping any reported gap at ±798 Elo.

### 7. Benchmark series are nondeterministic run-to-run

Per-team win rates swing between runs of the same series: GPU/thread float
nondeterminism in search (acknowledged in the module docstring), RNG'd forced
switches, and workers racing for jobs. A single 100-game series carries a
roughly ±10% Wilson interval regardless — use `--repeat` before reading
per-team splits as signal.

### 8. The spreads prior ages silently

`build_spreads.py` discovers Pikalytics' format key and data date by probing;
both drift monthly. The filter degrades gracefully when
`artifacts/spreads.json` is *absent* (archetype fallback), but a stale file
degrades silently — only `beliefs.py --audit` would notice the prior aging.
Consider recording the scrape date in the JSON and warning when old.

## Resolved on this branch

- **Frozen-v1 archives died to cosmetic source churn, with no escape hatch** —
  source identity was raw SHA-256 file bytes and any mismatch was a hard
  failure. Now: (1) new manifests record `hash_scheme: "ast-v1"` — the hash of
  the docstring-stripped AST — so comments/docstrings/formatting no longer
  invalidate archives while logic edits still fail closed. (2)
  `benchmark.py play ... --allow-source-drift` explicitly tolerates a real
  mismatch: it warns loudly, runs the archive through current code, stamps
  `source_drift_a/b` (the drifted file list) onto every result row, and
  `report`/`standings` flag those games/contestants (`*drift`). Bit-faithful
  replay after a genuine behavior change remains covered by retained v1
  modules (with content-addressed source vendoring as a possible later
  upgrade).

- **Circular layering between the search brick and the chooser** —
  `agents/search/v1.py` deferred-imported `joint_choice` from
  `agents.determinized_duct.v1`, which re-exported it from `search.mcts`,
  which imports `agents.search.v1` at module level. Fixed by moving
  `_pos_maps`/`joint_choice` (with a local `_sid`) into `actions.py`, the
  leaf module; `search.mcts` re-exports both names for existing importers,
  and the search brick now imports `joint_choice` at module level with no
  cycle.
