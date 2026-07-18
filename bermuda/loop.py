"""Generation driver: paths → LSMC fit → arena gate, with a reservoir.

Generation 0 collects heuristic-measure paths and fits V₀. Generation k
collects paths with the gen-(k−1) exercise policy against opponents drawn
uniformly from the reservoir {heuristic, V₀ … V_{k−1}} (smoothed fictitious
play), refits on the last ``--buffer`` generations' shards (LSMC path
reuse), and gates by arena score vs the heuristic. The last gated
checkpoint is copied to artifacts/checkpoints/bermuda.pt for export.

CLI:
  python bermuda/loop.py --gens 4 --games 2000 --arena-games 200
      [--workers 8] [--teams pool] [--buffer 3] [--epochs 6]
      [--scenarios K] [--candidates A] [--seed S] [--start-gen 0]
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from bermuda.config import BCFG


def run(cmd):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    args = sys.argv[1:]

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    if "--help" in args:
        print(__doc__)
        return
    gens = int(opt("--gens", 4))
    games = int(opt("--games", 2000))
    arena_games = int(opt("--arena-games", 200))
    workers = opt("--workers", "4")
    teams = opt("--teams", "pool")
    buffer_gens = int(opt("--buffer", 3))
    epochs = opt("--epochs", "6")
    seed = int(opt("--seed", 0))
    start = int(opt("--start-gen", 0))
    extra = []
    for flag in ("--scenarios", "--candidates"):
        if opt(flag):
            extra += [flag, opt(flag)]

    py = sys.executable
    root = BCFG.shards_dir.parent          # artifacts/bermuda
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "loop_summary.jsonl"
    ckpt_of = lambda g: BCFG.ckpt_dir / f"bermuda_g{g}.pt"

    for g in range(start, gens):
        shard_dir = BCFG.shards_dir / f"gen{g}"
        if g == 0:
            run([py, "bermuda/paths.py", "--games", games, "--out",
                 shard_dir, "--behavior", "heuristic", "--teams", teams,
                 "--workers", workers, "--seed", seed, "--label", "gen0"])
        else:
            reservoir = ["heuristic"] + [str(ckpt_of(i)) for i in range(g)]
            run([py, "bermuda/paths.py", "--games", games, "--out",
                 shard_dir, "--behavior", ckpt_of(g - 1), "--opponents",
                 ",".join(reservoir), "--teams", teams, "--workers",
                 workers, "--seed", seed + g, "--label", f"gen{g}", *extra])
        train_dirs = [str(BCFG.shards_dir / f"gen{i}")
                      for i in range(max(0, g - buffer_gens + 1), g + 1)
                      if (BCFG.shards_dir / f"gen{i}").exists()]
        run([py, "bermuda/train.py", "--shards", ",".join(train_dirs),
             "--out", ckpt_of(g), "--epochs", epochs, "--seed", seed + g])
        run([py, "bermuda/eval.py", "arena", "--ckpt", ckpt_of(g), "--vs",
             "heuristic", "--games", arena_games, "--workers", workers,
             "--teams", "replicas", "--seed", seed + g, *extra])
        with open(ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "gen": g, "shards": train_dirs,
                "ckpt": str(ckpt_of(g))}) + "\n")

    final = BCFG.default_ckpt
    shutil.copy2(ckpt_of(gens - 1), final)
    print(f"\nloop done — final checkpoint: {final}")
    print("export for the pile with:")
    print(f"  python export_agent.py bermuda --agent bermuda "
          f"--ckpt {final} --architecture 'BERMUDA LSMC-CE'")


if __name__ == "__main__":
    main()
