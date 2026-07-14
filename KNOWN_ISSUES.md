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

## Minor issues

### 3. Policy-only Q values always read 0.0

`DecoupledUCTSearcher.aggregate_root(..., policy_only=True)` zips
`root.my_p` (used as the count column) with `root.my_w` (all zeros at an
unsearched root), so the `q` entries in `ChoiceInfo` — and the play.py
dashboard — show 0.0 for the policy-only bot. Cosmetic, but it looks like a
value-head failure.

### 4. `sys.argv` parsing edges (all CLI scripts)

- The shared `opt()` helper does `args[args.index(flag) + 1]`; a value-taking
  flag passed as the last token crashes with `IndexError`.
- Zero is falsy in several conversions, so `--sims 0`, `--depth 0`, and
  `--quick 0` (benchmark.py) silently mean "unset".
- `env.py`'s flags are non-exclusive `if`s: `--benchmark --selftest` runs
  both, and `--benchmark`'s optional count consumes the next token whenever
  one exists (`env.py --benchmark --selftest` raises on `int("--selftest")`).

### 5. Bradley–Terry standings degenerate cases

`benchmark.py standings`: a contestant with zero wins is clamped to rating
`1e-9` and prints as roughly Elo −2100; disconnected pairing graphs within an
era still "converge" but cross-component ratings are meaningless. `elo_diff`
clips scores to [0.01, 0.99], capping any reported gap at ±798 Elo.

### 6. Benchmark series are nondeterministic run-to-run

Per-team win rates swing between runs of the same series: GPU/thread float
nondeterminism in search (acknowledged in the module docstring), RNG'd forced
switches, and workers racing for jobs. A single 100-game series carries a
roughly ±10% Wilson interval regardless — use `--repeat` before reading
per-team splits as signal.

### 7. The spreads prior ages silently

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
