"""Model-vs-model benchmarking with immutable archived snapshots.

An *archive* is a frozen, self-contained bundle — checkpoint + vocab.json +
config + metadata — in artifacts/benchmarks/<name>/. Bundles are never
modified or deleted; each records the git commit and an "era" hash of the
search/particle config, so results across logic changes are flagged instead of
silently mixed. Archive the current run BEFORE landing changes that invalidate
it (tokenizer layout, particle logic).

A *series* between two contestants is every ordered pairing of the replica
teams (10 teams -> 100 games, mirrors included): contestant A plays team i,
B plays team j, for all (i, j). Both orders occur, so no team assignment
favors a side; p1/p2 alternates by game parity. Both contestants run under
the CURRENT search code and config (same sims, same beliefs) — the variable
under test is the model, each loaded with its own head architecture and
tokenizer layout via its bundle.

CLI: python benchmark.py archive <name> [--ckpt path] [--notes "..."]
     python benchmark.py rename <old> <new>
     python benchmark.py list
     python benchmark.py play <A> <B> [--sims N] [--workers W] [--temp T]
                                      [--repeat R] [--quick N]
                                      [--teams t1,t2,...] [--spectate]
                                      [--port P] [--no-save]
     python benchmark.py standings
"current" as a contestant name means the live artifacts (ckpt_best.pt +
vocab.json), so you can fight work-in-progress against any archive.
Every play run saves .log + browser-openable .html replays under
artifacts/replays/<A>_vs_<B>/ (--no-save to skip). --spectate also serves a
live dashboard (http://localhost:PORT) to flip between parallel games.
--teams restricts to games where team A is one of the named teams, replaying
those matchups with their original game seeds (the per-game seed index is
preserved; outcomes reproduce up to GPU/thread float nondeterminism in search).
"""

import dataclasses
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import date

import torch

import teams as teams_mod
from config import CFG
from env import Sidecar, SidecarBattle, pack_team, random_choice
from models.policy_value import PolicyValueNet
from observe_game import Bot
from search.mcts import Searcher
from tokenizer import PositionTokenizer

# config fields that change what a game means; results across different era
# hashes are apples-to-oranges and standings segregates them
ERA_FIELDS = ("n_particles", "resample_floor", "damage_tolerance",
              "investment_slack", "belief_damage_hits_per_pair",
              "spread_archetypes", "top_k_actions", "n_determinizations",
              "solve_endgame_at", "c_puct", "format_id")


def era_hash(cfg=CFG):
    blob = json.dumps({f: str(getattr(cfg, f)) for f in ERA_FIELDS})
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=str(CFG.artifacts_dir.parent)).stdout.strip()
    except OSError:
        return ""


def bench_dir(cfg=CFG):
    return cfg.artifacts_dir / "benchmarks"


def registry_path(cfg=CFG):
    return bench_dir(cfg) / "registry.json"


def load_registry(cfg=CFG):
    p = registry_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {"results": []}


def save_registry(reg, cfg=CFG):
    registry_path(cfg).write_text(json.dumps(reg, indent=1))


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------

def archive(name, ckpt=None, notes="", cfg=CFG):
    dst = bench_dir(cfg) / name
    assert not dst.exists(), f"bundle '{name}' already exists — bundles are " \
        "immutable; pick a new name"
    src_ckpt = ckpt or cfg.checkpoint_dir / "ckpt_best.pt"
    dst.mkdir(parents=True)
    shutil.copy2(src_ckpt, dst / "ckpt.pt")
    shutil.copy2(cfg.artifacts_dir / "vocab.json", dst / "vocab.json")
    hp = torch.load(dst / "ckpt.pt", map_location="cpu",
                    weights_only=False)["hp"]
    (dst / "config.json").write_text(json.dumps(
        {k: str(v) for k, v in dataclasses.asdict(cfg).items()}, indent=1))
    (dst / "meta.json").write_text(json.dumps({
        "name": name, "created": date.today().isoformat(),
        "source_ckpt": str(src_ckpt), "git": git_commit(),
        "era": era_hash(cfg), "notes": notes,
        "policy_head": hp.get("policy_head", "slot"),
        "n_tokens": hp["n_tokens"],
        "layout": json.loads((dst / "vocab.json").read_text()).get("layout", 1),
    }, indent=1))
    print(f"archived '{name}': head={hp.get('policy_head', 'slot')}, "
          f"n_tokens={hp['n_tokens']}, era={era_hash(cfg)}")


def rename(old, new, cfg=CFG):
    """Rename an archived bundle and migrate every registry result that
    references it, so standings/history stay consistent under the new name."""
    src, dst = bench_dir(cfg) / old, bench_dir(cfg) / new
    assert src.exists(), f"no bundle named '{old}'"
    assert not dst.exists(), f"'{new}' already exists"
    src.rename(dst)
    meta = json.loads((dst / "meta.json").read_text())
    meta["name"] = new
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    reg = load_registry(cfg)
    migrated = 0
    for r in reg["results"]:
        for side in ("a", "b"):
            if r.get(side) == old:
                r[side] = new
                migrated += 1
    save_registry(reg, cfg)
    print(f"renamed '{old}' -> '{new}'  ({migrated} registry entries migrated)")


def list_bundles(cfg=CFG):
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
    print(f"{'name':16s} {'created':11s} {'head':6s} {'tokens':7s} "
          f"{'era':11s} notes")
    for m in rows:
        era = m["era"] + ("" if m["era"] == cur_era else " (old!)")
        print(f"{m['name']:16s} {m['created']:11s} {m['policy_head']:6s} "
              f"{m['n_tokens']:<7d} {era:11s} {m.get('notes', '')}")


# ---------------------------------------------------------------------------
# contestants and games
# ---------------------------------------------------------------------------

class Contestant:
    """A frozen (model, tokenizer) pair. Search config is NOT frozen — both
    sides of a series play under the current one."""

    def __init__(self, name, cfg=CFG, device="cpu"):
        self.name = name
        if name == "current":
            ckpt, vocab = cfg.checkpoint_dir / "ckpt_best.pt", None
            self.meta = {"era": era_hash(cfg), "git": git_commit()}
        else:
            d = bench_dir(cfg) / name
            assert d.exists(), f"no bundle '{name}' (see benchmark.py list)"
            ckpt, vocab = d / "ckpt.pt", d / "vocab.json"
            self.meta = json.loads((d / "meta.json").read_text())
        self.model = PolicyValueNet.load(ckpt, cfg, device)
        self.tok = PositionTokenizer.load(cfg, path=vocab)
        assert self.model.hp["n_tokens"] == self.tok.n_tokens, \
            f"{name}: checkpoint/vocab layout mismatch"


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
               port=8020, depth=None, label=None):
    """The 100-game (per repeat) series. Returns the result rows.

    only_teams: restrict to games where team_a is in this set. The game index g
    is preserved from the full pairing list, so each game keeps its original
    seed and side assignment (outcomes reproduce up to GPU/thread float
    nondeterminism in search).
    spectate: serve a live dashboard; save_replays: write .log/.html per game."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_cfg = dataclasses.replace(
        cfg, sims_per_move=sims or cfg.sims_per_move,
        rollout_depth=depth or cfg.rollout_depth)
    A = Contestant(name_a, cfg, device)
    B = Contestant(name_b, cfg, device)
    usage = json.loads((cfg.artifacts_dir / "usage_stats.json").read_text())
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
        sa = Searcher(A.model, A.tok, run_cfg)
        sb = Searcher(B.model, B.tok, run_cfg)
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
                                      sets[side_of["b"]], sa, usage, run_cfg),
                    side_of["b"]: Bot(side_of["b"], sets[side_of["b"]],
                                      sets[side_of["a"]], sb, usage, run_cfg)}
                feed = spectator.new_game(name_a, name_b, ta, tb, side_of,
                                          run_cfg.format_id) \
                    if spectator else None
                winner, turns = run_game(sc, bots, sets, run_cfg,
                                         temperature, rng, feed=feed)
                res = {"a": name_a, "b": name_b, "team_a": ta, "team_b": tb,
                       "winner": {side_of["a"]: "a", side_of["b"]: "b"}.get(
                           winner, "tie"),
                       "turns": turns, "sims": run_cfg.sims_per_move,
                       "rollout_depth": run_cfg.rollout_depth,
                       "temp": temperature, "date": date.today().isoformat(),
                       "era_a": A.meta.get("era", ""),
                       "era_b": B.meta.get("era", ""),
                       "era_run": era_hash(run_cfg), "git": git_commit()}
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
    by_team = {}
    for r in results:
        w = by_team.setdefault(r["team_a"], [0.0, 0])
        w[1] += 1
        w[0] += 1.0 if r["winner"] == "a" else \
            0.5 if r["winner"] == "tie" else 0.0
    print(f"  {name_a} by team: " + "  ".join(
        f"{t}:{w / max(1, c):.0%}" for t, (w, c) in sorted(by_team.items())))


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (c - s) / d, (c + s) / d


def elo_diff(score):
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
        print(f"\nera {era} ({len(rows)} games):")
        for p in sorted(players, key=lambda p: -rating[p]):
            games = sum(wins[p].values()) + sum(wins[q][p] for q in players)
            print(f"  {1500 + 400 * math.log10(rating[p]):7.0f}  {p}"
                  f"  ({games:.0f} games)")


# ---------------------------------------------------------------------------

def main(cfg=CFG):
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
    elif cmd == "rename":
        rename(args[1], args[2], cfg=cfg)
    elif cmd == "play":
        teams_opt = opt("--teams")
        run_series(args[1], args[2], cfg,
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
                   label=opt("--label"))
    elif cmd == "standings":
        standings(cfg)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
