"""Local evaluation: arenas with Wilson intervals, and value diagnostics.

  python bermuda/eval.py arena --ckpt CKPT.pt [--vs heuristic|random|CKPT2]
      [--games N] [--workers W] [--teams replicas|pool] [--seed S]
      [--scenarios K] [--candidates A] [--opp-temp T]
  python bermuda/eval.py diag --ckpt CKPT.pt --shards DIR[,DIR...]

The arena is the per-generation gate (policy-iteration gain vs the gen-0
measure); the pile tournament (round_robin.py) is the real exam.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import json
import random
import threading
import time

import numpy as np

from benchmark import elo_diff, wilson
from bermuda.config import BCFG, apply_runtime_env
from bermuda.features import load_dex
from bermuda.paths import PathBot, build_policy, play_path_game, team_source
from config import CFG
from env import Sidecar


def arena(ckpt, vs="heuristic", games=50, workers=2, teams="replicas",
          seed=0, scenarios=None, candidates=None, opp_temp=0.25, cfg=CFG):
    """Head-to-head series: CKPT (greedy) vs a baseline policy."""
    apply_runtime_env(cfg)
    dex = load_dex(cfg)
    pool = team_source(teams, cfg)
    names = sorted(pool)
    usage_p = cfg.artifacts_dir / "usage_stats.json"
    usage = json.loads(usage_p.read_text()) if usage_p.exists() else {}
    jobs = list(range(games))
    lock = threading.Lock()
    results, t0 = [], time.time()

    def worker(wid):
        sc = Sidecar(cfg)
        cache = {}

        def policy(token):
            if token not in cache:
                cache[token] = build_policy(token, cfg, scenarios,
                                            candidates, seed + wid)
            return cache[token]

        try:
            while True:
                with lock:
                    if not jobs:
                        return
                    g = jobs.pop(0)
                grng = random.Random(seed * 100003 + g)
                ta, tb = grng.sample(names, 2)
                side_of = {"a": "p1", "b": "p2"} if g % 2 == 0 else \
                          {"a": "p2", "b": "p1"}
                sets = {side_of["a"]: pool[ta], side_of["b"]: pool[tb]}
                bots, walls = {}, {}
                for tag, token, temp in (("a", ckpt, 0.0),
                                         ("b", vs, opp_temp)):
                    side = side_of[tag]
                    mode, chooser = policy(token)
                    bots[side] = PathBot(
                        side, sets[side],
                        sets["p2" if side == "p1" else "p1"], mode, dex,
                        random.Random(grng.randrange(1 << 30)), temp,
                        cfg, chooser=chooser, usage=usage, collect=False)
                winner, turns, stats = play_path_game(sc, bots, sets, cfg)
                row = {"winner": {v: k for k, v in side_of.items()}.get(
                    winner, "tie"), "turns": turns}
                for tag in ("a", "b"):
                    st = stats[side_of[tag]]
                    row[f"spm_{tag}"] = st["wall"] / max(1, st["moves"])
                with lock:
                    results.append(row)
                    n = len(results)
                    wa = sum(r["winner"] == "a" for r in results)
                    print(f"  game {n:3d}/{games}: {ta} vs {tb} -> "
                          f"{row['winner']}  (A {wa}/{n}, "
                          f"{(time.time() - t0) / n:.0f}s/game)")
        finally:
            for _, chooser in cache.values():
                if chooser is not None:
                    chooser.close()
            sc.close()

    threads = [threading.Thread(target=worker, args=(w,), daemon=True)
               for w in range(max(1, workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    n = len(results)
    wa = sum(r["winner"] == "a" for r in results)
    wb = sum(r["winner"] == "b" for r in results)
    ties = n - wa - wb
    score = (wa + 0.5 * ties) / max(1, n)
    lo, hi = wilson(wa + 0.5 * ties, max(1, n))
    print(f"\n{ckpt} vs {vs}: {wa}-{wb}-{ties} ({score:.1%}, "
          f"95% CI {lo:.1%}-{hi:.1%})  elo {elo_diff(score):+.0f}")
    for tag, name in (("a", str(ckpt)), ("b", str(vs))):
        spm = np.mean([r[f"spm_{tag}"] for r in results]) if results else 0
        print(f"  {name}: {spm:.2f}s/move")
    return {"games": n, "wins": wa, "losses": wb, "ties": ties,
            "score": score, "ci": [lo, hi]}


def diag(ckpt, shard_dirs):
    """Calibration + residual diagnostics of a checkpoint on stored paths."""
    from bermuda.model import ValueMLP
    from bermuda.train import group_split, load_shards, order_index
    model = ValueMLP.load(ckpt)
    data = load_shards(shard_dirs)
    val = group_split(data["gid"], 1.0)      # every game: pure diagnosis
    order, nxt, back = order_index(data)
    from bermuda.train import diagnostics
    diagnostics(model, data, np.where(val)[0], order, nxt, back)


def main():
    import sys
    args = sys.argv[1:]

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    if not args or "--help" in args:
        print(__doc__)
        return
    if args[0] == "arena":
        arena(ckpt=opt("--ckpt", str(BCFG.default_ckpt)),
              vs=opt("--vs", "heuristic"),
              games=int(opt("--games", 50)),
              workers=int(opt("--workers", 2)),
              teams=opt("--teams", "replicas"),
              seed=int(opt("--seed", 0)),
              scenarios=int(opt("--scenarios")) if opt("--scenarios")
              else None,
              candidates=int(opt("--candidates")) if opt("--candidates")
              else None,
              opp_temp=float(opt("--opp-temp", 0.25)))
    elif args[0] == "diag":
        diag(opt("--ckpt", str(BCFG.default_ckpt)),
             opt("--shards", str(BCFG.shards_dir / "gen0")).split(","))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
