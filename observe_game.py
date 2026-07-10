"""Self-play viewer: two search bots on one sidecar battle, with everything
the search knows printed each turn — belief posterior per opponent Pokemon,
the model's predicted opponent actions, the bot's own mixed strategy and the
value estimate. `--step` pauses for Enter before each turn resolves.

Both bots play under closed team sheets: each sees only its own team plus the
opponent's preview species, and learns the rest through its tracker and
particle filter — exactly the live-ladder information state, even though this
process holds both true teams.

CLI: python observe_game.py [--step] [--games N] [--p2 random] [--temp T]
                            [--ckpt path] [--teams a.txt b.txt]
     --debug           search profiler + root tables + belief-vs-ORACLE report
                       (self-play knows the true sets, so the particle filter
                       is graded against the truth every turn)
     --cprofile out.prof
"""

import json
import random
import sys

import torch

from beliefs import OpponentBelief
from config import CFG
from data import LogParser, Side, parse_packed_team, sid
from env import (TEAM_A, TEAM_B, Sidecar, SidecarBattle, pack_team,
                 random_choice)
from models.policy_value import PolicyValueNet
from agents.determinized_duct.v1 import DeterminizedDUCTChooser
from search.mcts import joint_choice
from tokenizer import PositionTokenizer


def cts_placeholder(s):
    """What team preview reveals about an opponent mon: species, gender,
    level — nothing else."""
    return {"name": s["name"], "species": s["species"], "item": "",
            "ability": "", "moves": [], "nature": "serious", "evs": [0] * 6,
            "gender": s["gender"], "level": s["level"]}


class Bot:
    """One CTS-honest side: tracker, external belief, chooser, and true oracle."""

    def __init__(self, side, my_sets, opp_sets, searcher, usage, cfg,
                 debug=False):
        """Initialize from side id, full teams, chooser, usage prior, and config."""
        self.side, self.cfg = side, cfg
        self.opp = "p2" if side == "p1" else "p1"
        self.searcher = searcher
        self.debug = debug
        self.oracle = opp_sets       # true sets, used ONLY for --debug grading
        self.tracker = LogParser("obs-" + side, 0, "", cfg.format_id)
        self.tracker.sides = {
            side: Side(my_sets),
            self.opp: Side([cts_placeholder(s) for s in opp_sets])}
        belief_cls = getattr(searcher, "belief_model_cls", OpponentBelief)
        self.belief = belief_cls([sid(s["species"]) for s in opp_sets],
                                 usage, cfg, searcher.bridge,
                                 my_team=my_sets)
        self.name_to_idx = {s["name"]: k for k, s in enumerate(my_sets)}
        self.brought = list(range(len(my_sets)))   # narrowed at team preview

    def feed(self, lines):
        """Feed protocol-line strings into the mutable ``LogParser``."""
        for line in lines:
            self.tracker.feed(line)

    def decide(self, request, temperature):
        """Update belief and return ``(Showdown choice str, ChoiceInfo)``."""
        self.belief.update(self.tracker.drain_events(), viewer=self.side)
        joint, info = self.searcher.choose(
            self.tracker, self.belief, self.side, request, self.brought,
            temperature=temperature)
        return joint_choice(request, joint, self.name_to_idx), info

    def show(self, info):
        """Print public state, beliefs, value, opponent prior, and strategy."""
        t = self.tracker
        print(f"\n=== turn {t.turn_no} — {self.side} thinking ===")
        for pid in ("p1", "p2"):
            active = "  ".join(
                f"{sid(m.species_cur)} {m.hp:.0%}"
                + (f" {m.status}" if m.status else "")
                for m in t.sides[pid].mons
                if m.active_slot is not None and not m.fainted)
            print(f"  {pid}: {active}")
        print(f"  value {info['value']:+.2f}"
              + ("  [endgame solve]" if info["solve"] else ""))
        print("  beliefs about opponent:")
        for k in range(len(self.belief.species)):
            m = t.sides[self.opp].mons[k]
            if m.fainted or not m.appeared:
                continue
            s = self.belief.summary()[k]
            items = ", ".join(f"{it or 'no item'} {p:.0%}"
                              for it, p in self.belief.item_posterior(k)[:3])
            flag = " [depleted]" if self.belief.soft_depletions[k] else ""
            print(f"    {self.belief.species[k]:18s} "
                  f"spe {s['spe_lo']:.0f}-{s['spe_hi']:.0f}  {items}{flag}")
        print("  opponent predicted:")
        for d, p in info["opp_pred"][:3]:
            print(f"    {p:6.1%}  {d}")
        print("  my mixed strategy:")
        for d, p in info["strategy"][:5]:
            print(f"    {p:6.1%}  {d}")
        if self.debug:
            from search.debug import belief_report
            print(belief_report(self.belief, oracle=self.oracle))


def play_game(sc, bots, teams, cfg, step_mode, temperature, p2_random, rng):
    """Play one observed battle and return winner ``SideID|None``."""
    b = SidecarBattle.create(sc, cfg.format_id,
                             pack_team(teams["p1"]), pack_team(teams["p2"]))
    for bot in bots.values():
        bot.feed(b.log)
    while not b.ended:
        choices = {}
        for side in b.pending_sides():
            req, bot = b.requests[side], bots[side]
            if req.get("teamPreview"):
                n = min(req.get("maxChosenTeamSize") or 4,
                        len(teams[side]))       # lead choice unmodeled (v1)
                bot.brought = list(range(n))
                choices[side] = "team " + "".join(str(i + 1) for i in range(n))
            elif req.get("forceSwitch"):
                choices[side] = random_choice(req, rng)
            elif side == "p2" and p2_random:
                choices[side] = random_choice(req, rng)
            else:
                choices[side], info = bot.decide(req, temperature)
                bot.show(info)
                print(f"  -> {choices[side]}")
        if step_mode:
            input("[enter] resolves the turn ")
        resp = b.step(choices)
        for bot in bots.values():
            bot.feed(resp["log"])
        if resp["errors"]:
            resp = b.step({s: "default" for s in resp["errors"]})
            for bot in bots.values():
                bot.feed(resp["log"])
    print(f"\nwinner: {b.winner}")
    b.destroy()
    return b.winner


def main(cfg=CFG):
    """Load the versioned chooser and run one observed game from CLI flags."""
    args = sys.argv[1:]

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    from search.debug import maybe_cprofile
    ckpt = opt("--ckpt", cfg.checkpoint_dir / "ckpt_best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PolicyValueNet.load(ckpt, cfg, device)
    tok = PositionTokenizer.load(cfg)
    debug = "--debug" in args
    searcher = DeterminizedDUCTChooser(model, tok, cfg, debug=debug)
    usage = json.loads((cfg.artifacts_dir / "usage_stats.json").read_text())

    teams = {"p1": parse_packed_team(TEAM_A), "p2": parse_packed_team(TEAM_B)}
    if "--teams" in args:
        i = args.index("--teams")
        teams = {"p1": parse_packed_team(open(args[i + 1]).read().strip()),
                 "p2": parse_packed_team(open(args[i + 2]).read().strip())}

    rng = random.Random(0)
    sc = Sidecar(cfg)      # the "real" battle, separate from the search's sim
    wins = {"p1": 0, "p2": 0, None: 0}
    with maybe_cprofile(opt("--cprofile")):
        for _ in range(int(opt("--games", 1))):
            bots = {"p1": Bot("p1", teams["p1"], teams["p2"], searcher, usage,
                              cfg, debug),
                    "p2": Bot("p2", teams["p2"], teams["p1"], searcher, usage,
                              cfg, debug)}
            w = play_game(sc, bots, teams, cfg, "--step" in args,
                          float(opt("--temp", cfg.play_temperature)),
                          "--p2" in args and opt("--p2") == "random", rng)
            wins[w] = wins.get(w, 0) + 1
    print(f"\nscore: {wins}")
    sc.close()


if __name__ == "__main__":
    main()
