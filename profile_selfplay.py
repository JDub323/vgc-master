"""Throughput profiler for game playing / self-play generation.

Plays real games through the exact self-play skeleton
(selfplay.play_selfplay_game: shared sidecar, tracker + particle filter per
side, DeterminizedDUCTChooser per decision) and aggregates the search's
built-in phase profiler (search.debug.SearchDebug) across every move instead
of printing it per move like observe_game --debug does.

The headline numbers are throughput — moves/min and games/hour — because
that is what bounds self-play data generation. Per-move latency percentiles
are reported too. The phase table answers "what costs the most": time spent
in the node sidecar (restore/step/destroy), net inference, tokenization,
tracker deep-copies, determinization builds, and belief updates.

Works with or without a trained checkpoint: with none available it profiles
a randomly initialized net of the baseline architecture — weights change
where the search goes, not what a simulation costs, so throughput numbers
remain representative (the endgame solver may engage at different times, so
prefer a real checkpoint when one exists).

CLI: python profile_selfplay.py [--games N] [--max-decisions M]
                                [--sims S] [--dets K] [--policy-only]
                                [--ckpt PATH] [--seed N]
                                [--cprofile out.prof]
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("profile_selfplay.py"):
        raise SystemExit(0)

import dataclasses
import json
import random
import statistics
import sys
import time
from collections import Counter

import numpy as np
import torch

import teams as teams_mod
from agents.determinized_duct.v1 import DeterminizedDUCTChooser
from config import CFG
from env import Sidecar
from models.policy_value import PolicyValueNet
from observe_game import Bot
from search.debug import SearchDebug, maybe_cprofile
from selfplay import play_selfplay_game
from tokenizer import PositionTokenizer


class BudgetExhausted(Exception):
    """Raised mid-game once --max-decisions search decisions were profiled."""


class CountingModel:
    """predict_batch proxy recording call count, row count, and wall time.
    The chooser only ever calls predict_batch (same contract the self-play
    BatchedEvaluator relies on), so this sees every net evaluation."""

    def __init__(self, model):
        """Wrap `model`, whose predict_batch calls will be counted/timed."""
        self.model = model
        self.calls = 0
        self.rows = 0
        self.time = 0.0

    def predict_batch(self, tokens):
        t0 = time.perf_counter()
        out = self.model.predict_batch(tokens)
        self.time += time.perf_counter() - t0
        self.calls += 1
        self.rows += len(tokens)
        return out


class ProfiledBot(Bot):
    """Bot whose decide() splits wall time into belief update vs search and
    accumulates the chooser's per-move health counters."""

    def __init__(self, *args, policy_only=False, stats=None, max_dec=None,
                 **kw):
        """Extend Bot with a shared stats dict and an optional decision cap."""
        super().__init__(*args, **kw)
        self.policy_only = policy_only
        self.stats = stats
        self.max_dec = max_dec

    def decide(self, request, temperature):
        """Timed Bot.decide: belief update and search walls recorded apart."""
        from search.mcts import joint_choice
        st = self.stats
        if self.max_dec and st["decisions"] >= self.max_dec:
            raise BudgetExhausted
        t0 = time.perf_counter()
        self.belief.update(self.tracker.drain_events(), viewer=self.side)
        t1 = time.perf_counter()
        joint, info = self.searcher.choose(
            self.tracker, self.belief, self.side, request, self.brought,
            temperature=temperature, policy_only=self.policy_only)
        t2 = time.perf_counter()
        st["belief_s"] += t1 - t0
        st["move_walls"].append(t2 - t1)
        for k, v in info["health"].items():
            if k != "wall_s":
                st["health"][k] += v
        if info["solve"]:
            st["solve_moves"] += 1
        st["decisions"] += 1
        return joint_choice(request, joint, self.name_to_idx), info


def load_model(ckpt, cfg, tok, device):
    """Return (model, description). Falls back to a random-init baseline-arch
    net when no checkpoint is given or found."""
    if ckpt:
        return PolicyValueNet.load(ckpt, cfg, device), str(ckpt)
    for cand in (cfg.checkpoint_dir / "ckpt_best.pt",
                 cfg.checkpoint_dir / "selfplay" / "sp_last.pt"):
        if cand.exists():
            return PolicyValueNet.load(cand, cfg, device), str(cand)
    m = PolicyValueNet(tok.vocab_size(), tok.n_tokens,
                       tok.opp_species_positions(), len(tok.move_list),
                       len(tok.item_list), len(tok.ability_list), cfg,
                       policy_head="joint").to(device)
    return m, "RANDOM INIT (no checkpoint found — throughput-representative)"


def opt(args, flag, default=None, cast=str):
    """Return the value following `flag` in args, or default."""
    if flag in args:
        return cast(args[args.index(flag) + 1])
    return default


def main():
    args = sys.argv[1:]
    n_games = opt(args, "--games", 1, int)
    max_dec = opt(args, "--max-decisions", None, int)
    seed = opt(args, "--seed", 0, int)
    ckpt = opt(args, "--ckpt")
    cprof = opt(args, "--cprofile")
    policy_only = "--policy-only" in args
    cfg = dataclasses.replace(
        CFG,
        sims_per_move=opt(args, "--sims", CFG.sims_per_move, int),
        n_determinizations=opt(args, "--dets", CFG.n_determinizations, int))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = PositionTokenizer.load(cfg)
    model, desc = load_model(ckpt, cfg, tok, device)
    model.eval()
    net = CountingModel(model)
    usage = json.loads((cfg.artifacts_dir / "usage_stats.json").read_text())
    rng = random.Random(seed)

    sc = Sidecar(cfg)
    chooser = DeterminizedDUCTChooser(net, tok, cfg, seed=seed, sidecar=sc)
    # one persistent phase profiler across every move of every game; the
    # chooser's per-move debug printing is not wanted here
    chooser.dbg = SearchDebug(True)
    chooser._debug_print = lambda *a, **k: None

    stats = {"belief_s": 0.0, "move_walls": [], "health": Counter(),
             "decisions": 0, "solve_moves": 0}
    team_names = list(teams_mod.TEAMS)
    team_sets = {t: teams_mod.get(t) for t in team_names}

    print(f"device {device} | model {desc}")
    print(f"games {n_games} | sims/move {cfg.sims_per_move} | "
          f"determinizations {cfg.n_determinizations} | "
          f"policy_only {policy_only}")

    games_done, turns, results = 0, 0, Counter()
    t_run = time.perf_counter()
    try:
        with maybe_cprofile(cprof):
            for g in range(n_games):
                ta, tb = rng.choice(team_names), rng.choice(team_names)
                sets = {"p1": team_sets[ta], "p2": team_sets[tb]}
                bots = {s: ProfiledBot(s, sets[s], sets[o], chooser, usage,
                                       cfg, policy_only=policy_only,
                                       stats=stats, max_dec=max_dec)
                        for s, o in (("p1", "p2"), ("p2", "p1"))}
                t_g = time.perf_counter()
                try:
                    winner = play_selfplay_game(sc, bots, sets, cfg, rng)
                except BudgetExhausted:
                    print(f"  stopping mid-game: --max-decisions {max_dec} "
                          f"reached", flush=True)
                    break
                games_done += 1
                turns += bots["p1"].tracker.turn_no
                results[winner or "tie/cap"] += 1
                print(f"  game {g + 1}/{n_games}: {ta} vs {tb} -> "
                      f"{winner or 'tie/cap'}, "
                      f"{bots['p1'].tracker.turn_no} turns, "
                      f"{time.perf_counter() - t_g:.1f}s", flush=True)
    finally:
        wall = time.perf_counter() - t_run
        chooser.close()
        sc.close()
        report(stats, chooser, net, wall, games_done, turns)


def report(stats, chooser, net, wall, games, turns):
    """Print throughput, latency, phase breakdown, and net batching stats."""
    walls = stats["move_walls"]
    if not walls:
        print("no decisions made — nothing to report")
        return
    search_s = sum(walls)
    h = stats["health"]
    walls_sorted = sorted(walls)

    def pct(p):
        return walls_sorted[min(len(walls_sorted) - 1,
                                int(p * len(walls_sorted)))]

    print(f"\n=== throughput ({wall:.1f}s total wall) ===")
    print(f"games      {games}  ({3600 * games / wall:.1f} games/hour, "
          f"avg {turns / max(1, games):.0f} turns)")
    print(f"decisions  {stats['decisions']}  "
          f"({60 * stats['decisions'] / wall:.1f} moves/min)")
    print(f"sims       {int(h['sims'])}  ({h['sims'] / wall:.0f} sims/s "
          f"overall, {h['sims'] / max(1e-9, search_s):.0f} sims/s in-search)")
    print(f"sim steps  {int(h['steps'])}  ({h['steps'] / wall:.0f} steps/s)")
    print(f"\n=== per-move latency ===")
    print(f"mean {statistics.mean(walls):.2f}s  p50 {pct(0.50):.2f}s  "
          f"p95 {pct(0.95):.2f}s  max {walls_sorted[-1]:.2f}s"
          + (f"  ({stats['solve_moves']} endgame-solve moves)"
             if stats["solve_moves"] else ""))
    print(f"\n=== where the time goes (share of total wall) ===")
    print(chooser.dbg.report(wall))
    print(f"{'belief_upd':12s} {stats['belief_s']:7.2f}s "
          f"{stats['belief_s'] / wall:5.0%}  (outside search, incl. above "
          f"'(python)' row)")
    print(f"\n=== net inference ===")
    print(f"{net.calls} calls, {net.rows} rows "
          f"(avg batch {net.rows / max(1, net.calls):.2f}), "
          f"{net.time:.2f}s ({net.time / wall:.0%} of wall, "
          f"{1000 * net.time / max(1, net.calls):.2f} ms/call, "
          f"{net.rows / max(1e-9, net.time):.0f} positions/s)")
    if chooser.bridge:
        tot = max(1, chooser.bridge.hits + chooser.bridge.misses)
        print(f"\ndamage bridge cache: {chooser.bridge.hits}/{tot} hits "
              f"({chooser.bridge.hits / tot:.0%})")
    print("\nnotes: 'restore'/'step'/'destroy' are node-sidecar RPC "
          "(simulator), 'net' is model forward, 'views'/'encode' are "
          "tokenization, 'copy' is tracker deepcopy, 'det_build' is "
          "belief-sample reconstruction (its root expansions nest the "
          "net/views/encode timers, so phase %s can sum past 100).")


if __name__ == "__main__":
    main()
