"""Full-agent benchmarking with immutable archived snapshots.

An *archive* is a frozen, behavior-asset-complete bundle — checkpoint +
vocab.json + config + metadata + behavior assets — in
artifacts/benchmarks/<name>/. It still requires the recorded retained Python
source and external runtime identities; loading verifies both rather than
silently substituting current behavior.
Bundles are never modified or deleted; each records the git commit, agent
implementation ID, and an "era" hash of the search/particle config, so results
across logic changes are flagged instead of silently mixed. Archive the current
run BEFORE landing changes that invalidate it (tokenizer layout, particle
logic).

A *series* between two contestants is every ordered pairing of the replica
teams (10 teams -> 100 games, mirrors included): contestant A plays team i,
B plays team j, for all (i, j). Both orders occur, so no team assignment
favors a side; p1/p2 alternates by game parity. Each archived contestant loads
its own model architecture, tokenizer layout, search/belief cfg, and archived
usage/dex/spread assets. Every bundle carries an AgentSpec whose versioned
chooser and brick IDs are resolved through an explicit registry.

CLI: python benchmark.py archive <name> [--ckpt path] [--notes "..."]
     python benchmark.py list
     python benchmark.py play <A> [B=baseline] [--sims N] [--workers W] [--temp T]
                                      [--repeat R] [--quick N]
                                      [--teams t1,t2,...] [--spectate]
                                      [--port P] [--no-save]
                                      [--allow-source-drift]
     python benchmark.py standings
"baseline" is the frozen 1x reference agent. "current" means the live
artifacts (ckpt_best.pt + vocab.json), so the normal experiment is
`python benchmark.py play current baseline`.
Every play run saves .log + browser-openable .html replays under
artifacts/replays/<A>_vs_<B>/ (--no-save to skip). --spectate also serves a
live dashboard (http://localhost:PORT) to flip between parallel games.
--teams restricts to games where team A is one of the named teams, replaying
those matchups with their original game seeds (the per-game seed index is
preserved; outcomes reproduce up to GPU/thread float nondeterminism in search).
Source identity: archives record AST-normalized ("ast-v1") hashes, so
comment/docstring churn does not invalidate them. A real mismatch still fails closed unless
--allow-source-drift is passed, which runs the archive through CURRENT code,
warns loudly, stamps every result row with the drifted files, and marks the
contestant in report/standings.
"""

import dataclasses
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import date

import torch
import numpy as np

import teams as teams_mod
from agents.ids import DETERMINIZED_DUCT_V1
from agents.registry import (HASH_SCHEME, build_agent,
                             implementation_source_hashes,
                             verify_implementation_sources)
from agents.spec import (AGENT_SPEC_FILENAME, AgentSpec,
                         config_from_agent_spec, default_duct_spec)
from config import (CFG, config_diff, config_from_snapshot, config_snapshot,
                    load_config_snapshot)
from env import Sidecar, SidecarBattle, pack_team, random_choice
from models.policy_value import MODEL_CFG_FIELDS, PolicyValueNet
from observe_game import Bot
from tokenizer import PositionTokenizer

# config fields that change what a game means; results across different era
# hashes are apples-to-oranges and standings segregates them
ERA_FIELDS = ("n_particles", "resample_floor", "damage_tolerance",
              "investment_slack", "belief_damage_hits_per_pair",
              "spread_archetypes", "strict_attack_ev", "strict_speed_ev",
              "strict_sp_step", "spreads_prior", "spreads_top_k",
              "spreads_any_weight", "factored_fallback", "top_k_actions",
              "n_determinizations", "solve_endgame_at", "c_puct",
              "rollout_depth", "format_id")
AGENT_IMPL = DETERMINIZED_DUCT_V1
BASELINE_NAME = "baseline"
ARCHIVE_ASSETS = ("usage_stats.json", "dex.json", "spreads.json")


def era_hash(cfg=CFG):
    """Return the ten-hex behavior-era hash for the configured Elo fields."""
    blob = json.dumps({f: str(getattr(cfg, f)) for f in ERA_FIELDS})
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def git_commit():
    """Return the current short Git commit, or an empty string on failure."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=str(CFG.artifacts_dir.parent)).stdout.strip()
    except OSError:
        return ""


def git_dirty():
    """Return whether tracked/untracked worktree changes are present."""
    try:
        return bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            timeout=10, cwd=str(CFG.artifacts_dir.parent)).stdout.strip())
    except OSError:
        return True


def runtime_identities(cfg=CFG):
    """External identities that can affect an archived battle decision."""
    package = cfg.node_dir / "package.json"
    lock = cfg.node_dir / "package-lock.json"
    deps, packages = {}, {}
    if package.exists():
        deps = json.loads(package.read_text()).get("dependencies", {})
    if lock.exists():
        packages = json.loads(lock.read_text()).get("packages", {})
    calc = packages.get("node_modules/@smogon/calc", {})
    showdown = packages.get("node_modules/pokemon-showdown", {})
    return {
        "format_id": cfg.format_id,
        "pokemon_showdown": {
            "requested": deps.get("pokemon-showdown", ""),
            "version": showdown.get("version", ""),
            "resolved": showdown.get("resolved", ""),
            "integrity": showdown.get("integrity", ""),
        },
        "smogon_calc": {
            "requested": deps.get("@smogon/calc", ""),
            "version": calc.get("version", ""),
            "resolved": calc.get("resolved", ""),
            "integrity": calc.get("integrity", ""),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def bench_dir(cfg=CFG):
    """Return ``cfg.artifacts_dir / 'benchmarks'``."""
    return cfg.artifacts_dir / "benchmarks"


def registry_path(cfg=CFG):
    """Return the benchmark result-registry JSON path."""
    return bench_dir(cfg) / "registry.json"


def load_registry(cfg=CFG):
    """Return decoded results registry or ``{'results': []}`` when absent."""
    p = registry_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {"results": []}


def save_registry(reg, cfg=CFG):
    """Serialize a result-registry mapping to its configured path."""
    registry_path(cfg).write_text(json.dumps(reg, indent=1))


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------

def archive(name, ckpt=None, notes="", cfg=CFG):
    """Create one immutable full-agent bundle; return ``None`` or raise."""
    dst = bench_dir(cfg) / name
    assert not dst.exists(), f"bundle '{name}' already exists — bundles are " \
        "immutable; pick a new name"
    src_ckpt = ckpt or cfg.checkpoint_dir / "ckpt_best.pt"
    required = [src_ckpt, cfg.artifacts_dir / "vocab.json"]
    required += [cfg.artifacts_dir / asset for asset in ARCHIVE_ASSETS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "cannot create a static agent archive; missing behavior assets: "
            + ", ".join(missing))
    dst.mkdir(parents=True)
    shutil.copy2(src_ckpt, dst / "ckpt.pt")
    shutil.copy2(cfg.artifacts_dir / "vocab.json", dst / "vocab.json")
    for asset in ARCHIVE_ASSETS:
        src = cfg.artifacts_dir / asset
        if src.exists():
            shutil.copy2(src, dst / asset)
    hp = torch.load(dst / "ckpt.pt", map_location="cpu",
                    weights_only=False)["hp"]
    ck = torch.load(dst / "ckpt.pt", map_location="cpu", weights_only=False)
    snap = ck.get("cfg") or config_snapshot(cfg)
    archive_cfg = config_from_snapshot(snap, base=cfg)
    (dst / "config.json").write_text(json.dumps(snap, indent=1))
    commit = git_commit()
    spec = default_duct_spec(
        archive_cfg,
        runtime=runtime_identities(cfg),
        source={"git_commit": commit, "git_dirty": git_dirty()},
        archive={"name": name, "created": date.today().isoformat(),
                 "notes": notes, "source_checkpoint": str(src_ckpt)},
    )
    spec = dataclasses.replace(
        spec, source=dict(spec.source) |
        {"hash_scheme": HASH_SCHEME,
         "files": implementation_source_hashes(spec)})
    spec.dump(dst / AGENT_SPEC_FILENAME)
    (dst / "meta.json").write_text(json.dumps({
        "name": name, "created": date.today().isoformat(),
        "source_ckpt": str(src_ckpt), "git": commit,
        "era": era_hash(archive_cfg), "notes": notes,
        "agent_impl": spec.agent_impl,
        "architecture": spec.architecture,
        "agent_spec": AGENT_SPEC_FILENAME,
        "policy_head": hp["policy_head"],
        "n_tokens": hp["n_tokens"],
        "layout": json.loads((dst / "vocab.json").read_text()).get("layout", 1),
    }, indent=1))
    print(f"archived '{name}': head={hp['policy_head']}, "
          f"n_tokens={hp['n_tokens']}, era={era_hash(archive_cfg)}")


def list_bundles(cfg=CFG):
    """Print every archive directory containing ``meta.json``."""
    cur_era = era_hash(cfg)
    rows = []
    if bench_dir(cfg).exists():
        for d in sorted(bench_dir(cfg).iterdir()):
            if (d / "meta.json").exists():
                rows.append(json.loads((d / "meta.json").read_text()))
    if not rows:
        print("no archived bundles — `python benchmark.py archive <name>` "
              "before changing layouts/logic")
        return
    print(f"{'name':16s} {'architecture':26s} {'created':11s} {'head':6s} "
          f"{'tokens':7s} {'era':11s} notes")
    for m in rows:
        era = m["era"] + ("" if m["era"] == cur_era else " (old!)")
        arch = m["architecture"]
        print(f"{m['name']:16s} {arch:26.26s} {m['created']:11s} "
              f"{m['policy_head']:6s} {m['n_tokens']:<7d} {era:11s} "
              f"{m.get('notes', '')}")


# ---------------------------------------------------------------------------
# contestants and games
# ---------------------------------------------------------------------------

class Contestant:
    """A complete frozen chooser specification plus its loaded neural assets."""

    def __init__(self, name, cfg=CFG, device="cpu", allow_source_drift=False):
        """Load ``current``, the frozen ``baseline``, or a named archive."""
        self.name = name
        self.source_drift = []
        if name == "current":
            ckpt, vocab = cfg.checkpoint_dir / "ckpt_best.pt", None
            self.agent_spec = default_duct_spec(
                cfg, runtime=runtime_identities(cfg),
                source={"git_commit": git_commit(), "git_dirty": git_dirty()})
            self.meta = {
                "era": era_hash(cfg), "git": git_commit(),
                "agent_impl": self.agent_spec.agent_impl,
                "architecture": self.agent_spec.architecture,
            }
            self.bundle_dir = None
            self.archive_cfg = cfg
            self.static_spec = False
        else:
            d = bench_dir(cfg) / name
            assert d.exists(), f"no bundle '{name}' (see benchmark.py list)"
            self.meta = json.loads((d / "meta.json").read_text())
            self.bundle_dir = d
            spec_path = d / self.meta.get("agent_spec", AGENT_SPEC_FILENAME)
            if not spec_path.exists():
                raise ValueError(f"{name}: unsupported pre-AgentSpec archive")
            self.static_spec = True
            self.agent_spec = AgentSpec.load(spec_path)
            missing = sorted(
                path for path in self.agent_spec.behavior_paths()
                if not self.agent_spec.resolve(d, path).exists())
            if missing:
                raise FileNotFoundError(
                    f"{name}: static archive is incomplete: {missing}")
            from agents.registry import REGISTRY
            REGISTRY.validate(self.agent_spec)
            self.source_drift = verify_implementation_sources(
                self.agent_spec, allow_drift=allow_source_drift)
            cfg_path = self.agent_spec.resolve(d, self.agent_spec.config)
            self.archive_cfg = load_config_snapshot(cfg_path, base=cfg)
            self.archive_cfg = config_from_agent_spec(
                self.archive_cfg, self.agent_spec)
            ckpt = self.agent_spec.resolve(
                d, self.agent_spec.assets["checkpoint"])
            vocab = self.agent_spec.resolve(
                d, self.agent_spec.assets["vocab"])
            _verify_runtime(self.agent_spec, cfg)
        self.model_cfg = _runtime_cfg(
            self.archive_cfg, cfg, self.bundle_dir,
            strict_archive=self.static_spec)
        self.model = PolicyValueNet.load(ckpt, self.model_cfg, device)
        self.tok = PositionTokenizer.load(self.model_cfg, path=vocab)
        self.search_cfg = self.model_cfg
        self.usage = _load_usage(
            self.search_cfg, cfg, strict_archive=self.static_spec)
        assert self.model.hp["n_tokens"] == self.tok.n_tokens, \
            f"{name}: checkpoint/vocab layout mismatch"


def _verify_runtime(spec, current_cfg):
    """Refuse to silently run a static archive on a different engine stack."""
    expected = spec.runtime
    if not expected:
        return
    current = runtime_identities(current_cfg)
    mismatches = []
    for key in ("format_id", "python", "torch", "numpy"):
        if expected.get(key) and expected[key] != current.get(key):
            mismatches.append(f"{key}: {expected[key]} != {current.get(key)}")
    for package in ("pokemon_showdown", "smogon_calc"):
        for key in ("requested", "version", "resolved", "integrity"):
            want = expected.get(package, {}).get(key)
            got = current.get(package, {}).get(key)
            if want and want != got:
                mismatches.append(f"{package}.{key}: {want} != {got}")
    if mismatches:
        raise RuntimeError(
            "archived runtime identity does not match this installation: "
            + "; ".join(mismatches))


def _runtime_cfg(saved_cfg, current_cfg, bundle_dir=None,
                 strict_archive=False):
    """Use saved behavior knobs with current machine paths where needed."""
    frozen_assets = bundle_dir and (bundle_dir / "usage_stats.json").exists() \
        and (bundle_dir / "dex.json").exists() \
        and (not getattr(saved_cfg, "spreads_prior", True)
             or (bundle_dir / "spreads.json").exists())
    if strict_archive and not frozen_assets:
        raise FileNotFoundError(
            f"static archive {bundle_dir} is missing required behavior assets")
    artifacts_dir = bundle_dir if frozen_assets else current_cfg.artifacts_dir
    return dataclasses.replace(
        saved_cfg,
        artifacts_dir=artifacts_dir,
        node_dir=current_cfg.node_dir,
        node_bin=current_cfg.node_bin,
        checkpoint_dir=current_cfg.checkpoint_dir,
        data_dir=current_cfg.data_dir,
        parsed_dir=current_cfg.parsed_dir,
        prepped_dir=current_cfg.prepped_dir,
    )


def _load_usage(search_cfg, current_cfg, strict_archive=False):
    """Return decoded usage stats, forbidding fallback for static archives."""
    p = search_cfg.artifacts_dir / "usage_stats.json"
    if not p.exists() and not strict_archive:
        p = current_cfg.artifacts_dir / "usage_stats.json"
    if not p.exists():
        raise FileNotFoundError(f"missing archived usage stats: {p}")
    return json.loads(p.read_text())


def _with_runtime_overrides(saved_cfg, current_cfg, sims=None, depth=None):
    """Return saved behavior config with explicit run/machine overrides."""
    return dataclasses.replace(
        saved_cfg,
        sims_per_move=sims or saved_cfg.sims_per_move,
        rollout_depth=depth or saved_cfg.rollout_depth,
        node_dir=current_cfg.node_dir,
        node_bin=current_cfg.node_bin,
    )


def _print_cfg_diffs(a, b, current_cfg):
    """Print implementation and meaningful behavior/model config differences."""
    fields = ERA_FIELDS + MODEL_CFG_FIELDS
    diffs = config_diff(a.search_cfg, b.search_cfg, fields=fields)
    cur_a = config_diff(current_cfg, a.search_cfg, fields=fields)
    cur_b = config_diff(current_cfg, b.search_cfg, fields=fields)
    print(f"  agent impl: {a.meta.get('agent_impl', a.meta.get('search_impl', 'unknown'))} "
          f"vs {b.meta.get('agent_impl', b.meta.get('search_impl', 'unknown'))}")
    if diffs:
        print("  archived cfg differences:")
        for name, va, vb in diffs:
            print(f"    {name}: {a.name}={va!r}  {b.name}={vb!r}")
    if cur_a or cur_b:
        print("  current-vs-archive cfg differences:")
        for label, rows in ((a.name, cur_a), (b.name, cur_b)):
            if rows:
                msg = ", ".join(f"{n}: current={va!r}, {label}={vb!r}"
                                for n, va, vb in rows[:8])
                extra = "" if len(rows) <= 8 else f" (+{len(rows) - 8} more)"
                print(f"    {label}: {msg}{extra}")


def _make_searcher(contestant, run_cfg):
    """Construct the exact registered chooser required by ``contestant``."""
    return build_agent(
        contestant.agent_spec, model=contestant.model,
        tokenizer=contestant.tok, cfg=run_cfg, apply_spec_config=False)


def run_game(sc, bots, sets_by_side, cfg, temperature, rng, max_turns=300,
             feed=None):
    """One quiet CTS-honest game. Returns (winner side id or None, turns).
    If `feed` (a spectate.GameFeed) is given, streams the protocol to it for
    live spectating + replay saving."""
    b = SidecarBattle.create(sc, cfg.format_id,
                             pack_team(sets_by_side["p1"]),
                             pack_team(sets_by_side["p2"]))
    if feed:
        feed.feed(b.log)
    for bot in bots.values():
        bot.feed(b.log)
    turns = 0
    while not b.ended:
        choices = {}
        for side in b.pending_sides():
            req, bot = b.requests[side], bots[side]
            if req.get("teamPreview"):
                n = min(req.get("maxChosenTeamSize") or 4,
                        len(sets_by_side[side]))
                bot.brought = list(range(n))
                choices[side] = "team " + "".join(str(i + 1) for i in range(n))
            elif req.get("forceSwitch"):
                choices[side] = random_choice(req, rng)
            else:
                choices[side], _ = bot.decide(req, temperature)
        resp = b.step(choices)
        if feed:
            feed.feed(resp["log"])
        for bot in bots.values():
            bot.feed(resp["log"])
        if resp["errors"]:
            resp = b.step({s: "default" for s in resp["errors"]})
            if feed:
                feed.feed(resp["log"])
            for bot in bots.values():
                bot.feed(resp["log"])
        turns += 1
        if turns >= max_turns:      # stall war: call it a tie, don't hang
            break
    winner = b.winner if b.ended else None
    if feed:
        feed.finish(winner)
    b.destroy()
    return winner, turns


def series_pairings(team_names, repeat=1, quick=None, seed=7):
    """Every ordered (team_a, team_b) pairing x repeat. quick=N: a random
    N-subset for cheap gating (self-play uses this between iterations)."""
    pairs = [(ta, tb) for ta in team_names for tb in team_names] * repeat
    if quick and quick < len(pairs):
        pairs = random.Random(seed).sample(pairs, quick)
    return pairs


def run_series(name_a, name_b, cfg=CFG, sims=None, temperature=0.0,
               workers=4, repeat=1, quick=None, record=True, verbose=True,
               only_teams=None, spectate=False, save_replays=True,
               port=8020, depth=None, label=None, allow_source_drift=False):
    """The 100-game (per repeat) series. Returns the result rows.

    only_teams: restrict to games where team_a is in this set. The game index g
    is preserved from the full pairing list, so each game keeps its original
    seed and side assignment (outcomes reproduce up to GPU/thread float
    nondeterminism in search).
    spectate: serve a live dashboard; save_replays: write .log/.html per game."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = Contestant(name_a, cfg, device, allow_source_drift=allow_source_drift)
    B = Contestant(name_b, cfg, device, allow_source_drift=allow_source_drift)
    run_cfg = dataclasses.replace(
        cfg, sims_per_move=sims or cfg.sims_per_move,
        rollout_depth=depth or cfg.rollout_depth)
    cfg_a = _with_runtime_overrides(A.search_cfg, cfg, sims=sims, depth=depth)
    cfg_b = _with_runtime_overrides(B.search_cfg, cfg, sims=sims, depth=depth)
    if cfg_a.format_id != cfg_b.format_id:
        print(f"  WARNING: contestants use different formats: "
              f"{A.name}={cfg_a.format_id}, {B.name}={cfg_b.format_id}; "
              f"the game engine will use current={run_cfg.format_id}")
    if verbose:
        _print_cfg_diffs(A, B, run_cfg)
    team_names = list(teams_mod.TEAMS)
    team_sets = {t: teams_mod.get(t) for t in team_names}
    jobs = [(g, ta, tb) for g, (ta, tb) in
            enumerate(series_pairings(team_names, repeat, quick))
            if not only_teams or ta in only_teams]

    spectator = None
    if spectate or save_replays:
        from spectate import Spectator
        run_tag = f"{name_a}_vs_{name_b}" + (f"_{label}" if label else "")
        spectator = Spectator(run_tag, cfg, live=spectate, port=port,
                              save=save_replays)
        if save_replays and not spectate:
            print(f"  saving replays under {spectator.dir}/")

    jobs_lock, results, t0 = threading.Lock(), [], time.time()

    def worker():
        # per-thread searchers: each owns a sidecar + damage bridge; the
        # models/tokenizers are shared (inference only)
        sa = _make_searcher(A, cfg_a)
        sb = _make_searcher(B, cfg_b)
        sc = Sidecar(run_cfg)
        try:                      # noqa: the finally closes all node procs
            while True:
                with jobs_lock:
                    if not jobs:
                        return
                    g, ta, tb = jobs.pop(0)
                # alternate engine sides so p1/p2 quirks can't favor a model
                side_of = {"a": "p1", "b": "p2"} if g % 2 == 0 else \
                          {"a": "p2", "b": "p1"}
                sets = {side_of["a"]: team_sets[ta], side_of["b"]: team_sets[tb]}
                rng = random.Random((g << 8) + 1)
                bots = {
                    side_of["a"]: Bot(side_of["a"], sets[side_of["a"]],
                                      sets[side_of["b"]], sa, A.usage, cfg_a),
                    side_of["b"]: Bot(side_of["b"], sets[side_of["b"]],
                                      sets[side_of["a"]], sb, B.usage, cfg_b)}
                feed = spectator.new_game(name_a, name_b, ta, tb, side_of,
                                          run_cfg.format_id) \
                    if spectator else None
                winner, turns = run_game(sc, bots, sets, run_cfg,
                                         temperature, rng, feed=feed)
                res = {"a": name_a, "b": name_b, "team_a": ta, "team_b": tb,
                       "winner": {side_of["a"]: "a", side_of["b"]: "b"}.get(
                           winner, "tie"),
                       "turns": turns, "sims": run_cfg.sims_per_move,
                       "sims_a": cfg_a.sims_per_move,
                       "sims_b": cfg_b.sims_per_move,
                       "rollout_depth": run_cfg.rollout_depth,
                       "rollout_depth_a": cfg_a.rollout_depth,
                       "rollout_depth_b": cfg_b.rollout_depth,
                       "temp": temperature, "date": date.today().isoformat(),
                       "era_a": A.meta.get("era", ""),
                       "era_b": B.meta.get("era", ""),
                       "era_run": era_hash(run_cfg),
                       "era_run_a": era_hash(cfg_a),
                       "era_run_b": era_hash(cfg_b),
                       "git": git_commit(),
                       "agent_impl_a": A.meta["agent_impl"],
                       "agent_impl_b": B.meta["agent_impl"],
                       "architecture_a": A.meta["architecture"],
                       "architecture_b": B.meta["architecture"]}
                if A.source_drift or B.source_drift:
                    res["source_drift_a"] = list(A.source_drift)
                    res["source_drift_b"] = list(B.source_drift)
                with jobs_lock:
                    results.append(res)
                    if verbose:
                        n = len(results)
                        wa = sum(r["winner"] == "a" for r in results)
                        print(f"  game {n:3d}: {ta} vs {tb} -> "
                              f"{res['winner']}   (A {wa}/{n}, "
                              f"{(time.time() - t0) / n:.0f}s/game)")
        finally:
            sc.close()
            sa.close()
            sb.close()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(max(1, workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if record:
        reg = load_registry(cfg)
        reg["results"] += results
        bench_dir(cfg).mkdir(parents=True, exist_ok=True)
        save_registry(reg, cfg)
    report(name_a, name_b, results)
    return results


def report(name_a, name_b, results):
    """Print aggregate score, confidence/Elo interval, and per-team split."""
    n = len(results)
    wa = sum(r["winner"] == "a" for r in results)
    wb = sum(r["winner"] == "b" for r in results)
    ties = n - wa - wb
    score = (wa + 0.5 * ties) / max(1, n)
    lo, hi = wilson(wa + 0.5 * ties, n)
    print(f"\n{name_a} vs {name_b}: {wa}-{wb}-{ties} "
          f"({score:.1%}, 95% CI {lo:.1%}-{hi:.1%})  "
          f"elo {elo_diff(score):+.0f} [{elo_diff(lo):+.0f}, {elo_diff(hi):+.0f}]")
    if {r["era_a"] for r in results} != {r["era_b"] for r in results}:
        print("  NOTE: contestants come from different search/particle eras — "
              "the gap includes logic changes, not just the model")
    drifted = {f for r in results
               for f in r.get("source_drift_a", []) + r.get("source_drift_b", [])}
    if drifted:
        print("  NOTE: games ran with --allow-source-drift; archived agents "
              "executed CURRENT code for: " + ", ".join(sorted(drifted)))
    by_team = {}
    for r in results:
        w = by_team.setdefault(r["team_a"], [0.0, 0])
        w[1] += 1
        w[0] += 1.0 if r["winner"] == "a" else \
            0.5 if r["winner"] == "tie" else 0.0
    print(f"  {name_a} by team: " + "  ".join(
        f"{t}:{w / max(1, c):.0%}" for t, (w, c) in sorted(by_team.items())))


def wilson(w, n, z=1.96):
    """Return a Wilson ``(lower, upper)`` interval for fractional successes."""
    if n == 0:
        return 0.0, 1.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (c - s) / d, (c + s) / d


def elo_diff(score):
    """Convert a score fraction to a clipped logistic Elo difference."""
    s = min(0.99, max(0.01, score))
    return 400 * math.log10(s / (1 - s))


def standings(cfg=CFG):
    """Bradley-Terry ratings over every recorded result, segregated by run
    era (games under different search/particle logic don't mix)."""
    reg = load_registry(cfg)
    by_era = {}
    for r in reg["results"]:
        by_era.setdefault(r.get("era_run", "?"), []).append(r)
    for era, rows in by_era.items():
        players = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
        wins = {p: {q: 0.0 for q in players} for p in players}
        for r in rows:
            pa, pb = r["a"], r["b"]
            if r["winner"] == "a":
                wins[pa][pb] += 1
            elif r["winner"] == "b":
                wins[pb][pa] += 1
            else:
                wins[pa][pb] += 0.5
                wins[pb][pa] += 0.5
        rating = {p: 1.0 for p in players}
        for _ in range(200):        # standard BT fixed-point iteration
            for p in players:
                num = sum(wins[p].values())
                den = sum((wins[p][q] + wins[q][p]) / (rating[p] + rating[q])
                          for q in players if q != p)
                if den > 0:
                    rating[p] = max(num / den, 1e-9)
            m = sum(rating.values()) / len(rating)
            rating = {p: v / m for p, v in rating.items()}
        family = {}
        tainted = set()
        for row in rows:
            family.setdefault(row["a"], row["architecture_a"])
            family.setdefault(row["b"], row["architecture_b"])
            if row.get("source_drift_a"):
                tainted.add(row["a"])
            if row.get("source_drift_b"):
                tainted.add(row["b"])
        print(f"\nera {era} ({len(rows)} games):")
        for architecture in sorted(set(family.values())):
            print(f"  {architecture}:")
            members = [p for p in players if family[p] == architecture]
            for p in sorted(members, key=lambda p: -rating[p]):
                games = sum(wins[p].values()) + sum(
                    wins[q][p] for q in players)
                mark = " *drift" if p in tainted else ""
                print(f"    {1500 + 400 * math.log10(rating[p]):7.0f}  {p}"
                      f"  ({games:.0f} games){mark}")
        if tainted:
            print("  *drift: includes games where the archived sources did "
                  "not match the running code (--allow-source-drift)")


# ---------------------------------------------------------------------------

def main(cfg=CFG):
    """Dispatch archive/list/rename/play/standings from ``sys.argv``."""
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    cmd = args[0]
    if cmd == "archive":
        archive(args[1], ckpt=opt("--ckpt"), notes=opt("--notes", ""), cfg=cfg)
    elif cmd == "list":
        list_bundles(cfg)
    elif cmd == "play":
        teams_opt = opt("--teams")
        opponent = args[2] if len(args) > 2 and not args[2].startswith("--") \
            else BASELINE_NAME
        run_series(args[1], opponent, cfg,
                   sims=int(opt("--sims", 0)) or None,
                   temperature=float(opt("--temp", 0.0)),
                   workers=int(opt("--workers", 4)),
                   repeat=int(opt("--repeat", 1)),
                   quick=int(opt("--quick", 0)) or None,
                   only_teams=set(teams_opt.split(",")) if teams_opt else None,
                   spectate="--spectate" in args,
                   save_replays="--no-save" not in args,
                   port=int(opt("--port", 8020)),
                   depth=int(opt("--depth", 0)) or None,
                   label=opt("--label"),
                   allow_source_drift="--allow-source-drift" in args)
    elif cmd == "standings":
        standings(cfg)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
