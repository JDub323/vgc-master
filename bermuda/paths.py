"""Path collection: simulate full battles, log (φ, turn, outcome) per side.

This is the Monte Carlo half of LSMC: an ensemble of paths under a behavior
measure. Gen 0 uses the type-chart heuristic at a sampling temperature (a
deliberately diffuse measure); gen ≥ 1 uses the current exercise policy with
exploration temperature against opponents drawn from the generation
reservoir (smoothed fictitious play, plan.md §2.4). Both sides' viewpoints
are recorded — a path is two filtrations over one battle.

CLI:
  python bermuda/paths.py --games N --out DIR
      [--behavior heuristic | --behavior CKPT.pt]
      [--opponents heuristic,CKPT_A.pt,...]   (default: mirror of behavior)
      [--teams replicas|pool] [--workers W] [--seed S]
      [--temp T] [--scenarios K] [--candidates A] [--shard-games M]
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import json
import random
import threading
import time
from pathlib import Path

import numpy as np

from actions import _pos_maps, joint_choice
from bermuda.config import BCFG, apply_runtime_env
from bermuda.features import featurize, load_dex
from bermuda.heuristic import (active_infos, forced_choice, mon_infos,
                               sample_joint)
from config import CFG
from data import LogParser, Side, sid
from env import Sidecar, SidecarBattle, pack_team
from observe_game import cts_placeholder


class PathBot:
    """One side of a path game: tracker + behavior policy + record buffer."""

    def __init__(self, side, my_sets, opp_sets, mode, dex, rng, temp,
                 cfg=CFG, chooser=None, usage=None, collect=True):
        self.side, self.cfg, self.dex = side, cfg, dex
        self.opp = "p2" if side == "p1" else "p1"
        self.mode, self.chooser = mode, chooser
        self.rng, self.temp, self.collect = rng, temp, collect
        self.tracker = LogParser("path-" + side, 0, "", cfg.format_id)
        self.tracker.sides = {
            side: Side(my_sets),
            self.opp: Side([cts_placeholder(s) for s in opp_sets])}
        self.name_to_idx = {s["name"]: k for k, s in enumerate(my_sets)}
        self.brought = list(range(min(4, len(my_sets))))
        self.records = []
        self.belief = None
        if mode == "chooser":
            from beliefs import OpponentBelief
            self.belief = OpponentBelief(
                [sid(s["species"]) for s in opp_sets], usage or {}, cfg,
                chooser.bridge, my_team=my_sets)

    def feed(self, lines):
        for line in lines:
            self.tracker.feed(line)

    def decide(self, request):
        if self.collect:
            self.records.append((featurize(self.tracker, self.side, self.dex),
                                 self.tracker.turn_no))
        if self.mode == "chooser":
            self.belief.update(self.tracker.drain_events(), viewer=self.side)
            joint, _ = self.chooser.choose(self.tracker, self.belief,
                                           self.side, request, self.brought,
                                           temperature=self.temp)
            return joint_choice(request, joint, self.name_to_idx)
        if self.mode == "random":
            from env import random_choice
            return random_choice(request, self.rng)
        idx_of_pos, _ = _pos_maps(request, self.name_to_idx)
        joint = sample_joint(request, idx_of_pos,
                             mon_infos(self.tracker, self.side, self.dex),
                             active_infos(self.tracker, self.opp, self.dex),
                             self.dex, self.rng, self.temp)
        return joint_choice(request, joint, self.name_to_idx)

    def forced(self, request):
        idx_of_pos, _ = _pos_maps(request, self.name_to_idx)
        return forced_choice(request, idx_of_pos,
                             mon_infos(self.tracker, self.side, self.dex),
                             active_infos(self.tracker, self.opp, self.dex),
                             self.dex, self.rng)


def play_path_game(sc, bots, sets_by_side, cfg=CFG, max_turns=None):
    """One battle between two PathBots; returns (winner, turns, wall_stats)."""
    max_turns = max_turns or BCFG.max_turns
    b = SidecarBattle.create(sc, cfg.format_id,
                             pack_team(sets_by_side["p1"]),
                             pack_team(sets_by_side["p2"]))
    for bot in bots.values():
        bot.feed(b.log)
    turns = 0
    stats = {s: {"moves": 0, "wall": 0.0} for s in bots}
    while not b.ended and turns < max_turns:
        choices = {}
        for side in b.pending_sides():
            req, bot = b.requests[side], bots[side]
            t0 = time.perf_counter()
            if req.get("teamPreview"):
                n = min(req.get("maxChosenTeamSize") or 4,
                        len(sets_by_side[side]))
                bot.brought = list(range(n))
                choices[side] = "team " + "".join(str(i + 1)
                                                  for i in range(n))
            elif req.get("forceSwitch"):
                choices[side] = bot.forced(req)
            else:
                choices[side] = bot.decide(req)
                stats[side]["moves"] += 1
            stats[side]["wall"] += time.perf_counter() - t0
        resp = b.step(choices)
        for bot in bots.values():
            bot.feed(resp["log"])
        if resp["errors"]:
            resp = b.step({s: "default" for s in resp["errors"]})
            for bot in bots.values():
                bot.feed(resp["log"])
        turns += 1
    winner = b.winner if b.ended else None
    b.destroy()
    return winner, turns, stats


class ShardWriter:
    """Thread-safe accumulator flushing npz shards + meta sidecars."""

    def __init__(self, out_dir, shard_games, meta):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_games, self.meta = shard_games, meta
        self.lock = threading.Lock()
        self.rows, self.games_in_shard, self.shard_no = [], 0, 0
        existing = sorted(self.dir.glob("shard_*.npz"))
        if existing:
            self.shard_no = int(existing[-1].stem.split("_")[1]) + 1

    def add_game(self, gid, per_side_records, z_by_side):
        with self.lock:
            for side_i, side in enumerate(("p1", "p2")):
                for step, (feat, turn) in enumerate(per_side_records[side]):
                    self.rows.append((feat, turn, z_by_side[side], gid,
                                      side_i, step))
            self.games_in_shard += 1
            if self.games_in_shard >= self.shard_games:
                self._flush()

    def _flush(self):
        if not self.rows:
            self.games_in_shard = 0
            return
        feats = np.stack([r[0] for r in self.rows])
        path = self.dir / f"shard_{self.shard_no:04d}.npz"
        np.savez_compressed(
            path, feats=feats,
            turns=np.array([r[1] for r in self.rows], dtype=np.int16),
            z=np.array([r[2] for r in self.rows], dtype=np.int8),
            gid=np.array([r[3] for r in self.rows], dtype=np.int64),
            side=np.array([r[4] for r in self.rows], dtype=np.int8),
            step=np.array([r[5] for r in self.rows], dtype=np.int32))
        (self.dir / f"shard_{self.shard_no:04d}.meta.json").write_text(
            json.dumps({**self.meta, "rows": len(self.rows)}))
        print(f"  wrote {path.name}: {len(self.rows)} rows")
        self.rows, self.games_in_shard = [], 0
        self.shard_no += 1

    def close(self):
        with self.lock:
            self._flush()


def build_policy(token, cfg, scenarios, candidates, seed):
    """'heuristic' | 'random' | ckpt path -> (mode, chooser|None)."""
    if token in ("heuristic", "random"):
        return token, None
    from bermuda.chooser import BermudaChooser
    return "chooser", BermudaChooser(token, cfg, seed=seed,
                                     n_scenarios=scenarios,
                                     n_candidates=candidates)


def team_source(kind, cfg):
    import teams as teams_mod
    if kind == "pool":
        return teams_mod.selfplay_pool(cfg)
    return {name: teams_mod.get(name) for name in teams_mod.TEAMS}


def collect(games, out_dir, behavior="heuristic", opponents=None,
            teams="replicas", workers=2, seed=0, temp=None, scenarios=None,
            candidates=None, shard_games=500, cfg=CFG, bcfg=BCFG,
            label=""):
    """Run ``games`` path games and shard the records under ``out_dir``."""
    apply_runtime_env(cfg)
    dex = load_dex(cfg)
    opp_tokens = opponents or [behavior]
    temp = temp if temp is not None else (
        bcfg.heuristic_temp if behavior == "heuristic" else bcfg.explore_temp)
    pool = team_source(teams, cfg)
    names = sorted(pool)
    usage_p = cfg.artifacts_dir / "usage_stats.json"
    usage = json.loads(usage_p.read_text()) if usage_p.exists() else {}
    writer = ShardWriter(out_dir, shard_games, {
        "behavior": str(behavior), "opponents": list(map(str, opp_tokens)),
        "teams": teams, "seed": seed, "temp": temp, "label": label,
        "feat_version": bcfg.feat_version})

    jobs = list(range(games))
    lock = threading.Lock()
    done = {"n": 0, "t0": time.time()}

    def worker(wid):
        sc = Sidecar(cfg)
        rng = random.Random(seed * 7919 + wid)
        cache = {}

        def policy(token, s):
            if token not in cache:
                cache[token] = build_policy(token, cfg, scenarios,
                                            candidates, seed=s)
            return cache[token]

        try:
            while True:
                with lock:
                    if not jobs:
                        return
                    g = jobs.pop(0)
                grng = random.Random(seed * 100003 + g)
                ta, tb = grng.sample(names, 2)
                opp_token = grng.choice(opp_tokens)
                # behavior side alternates by parity for side fairness
                side_of = {"beh": "p1", "opp": "p2"} if g % 2 == 0 else \
                          {"beh": "p2", "opp": "p1"}
                sets = {side_of["beh"]: pool[ta], side_of["opp"]: pool[tb]}
                bots = {}
                for tag, token in (("beh", behavior), ("opp", opp_token)):
                    side = side_of[tag]
                    mode, chooser = policy(token, seed + wid)
                    bots[side] = PathBot(
                        side, sets[side],
                        sets["p2" if side == "p1" else "p1"], mode, dex,
                        random.Random(grng.randrange(1 << 30)), temp,
                        cfg, chooser=chooser, usage=usage)
                winner, turns, _ = play_path_game(sc, bots, sets, cfg)
                z = {s: (0 if winner not in ("p1", "p2")
                         else (1 if winner == s else -1))
                     for s in ("p1", "p2")}
                writer.add_game((seed << 24) + g,
                                {s: bots[s].records for s in bots}, z)
                with lock:
                    done["n"] += 1
                    n = done["n"]
                    if n % 10 == 0 or n == games:
                        dt = time.time() - done["t0"]
                        print(f"  {n}/{games} games "
                              f"({dt / n:.1f}s/game, {turns} turns last)")
        finally:
            for mode, chooser in cache.values():
                if chooser is not None:
                    chooser.close()
            sc.close()

    threads = [threading.Thread(target=worker, args=(w,), daemon=True)
               for w in range(max(1, workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    writer.close()
    print(f"paths done: {games} games -> {out_dir}")


def main():
    import sys
    args = sys.argv[1:]

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    if not args or "--help" in args:
        print(__doc__)
        return
    opponents = opt("--opponents")
    collect(games=int(opt("--games", 20)),
            out_dir=opt("--out", str(BCFG.shards_dir / "gen0")),
            behavior=opt("--behavior", "heuristic"),
            opponents=opponents.split(",") if opponents else None,
            teams=opt("--teams", "replicas"),
            workers=int(opt("--workers", 2)),
            seed=int(opt("--seed", 0)),
            temp=float(opt("--temp")) if opt("--temp") else None,
            scenarios=int(opt("--scenarios")) if opt("--scenarios") else None,
            candidates=int(opt("--candidates")) if opt("--candidates")
            else None,
            shard_games=int(opt("--shard-games", 500)),
            label=opt("--label", ""))


if __name__ == "__main__":
    main()
