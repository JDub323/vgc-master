"""Scripted endgame scenarios with assertions about search behavior.

The headline test is the Metagross/Kingambit 1v1: Bullet Punch outprioritizes
and blanks Sucker Punch (the target already moved), Hammer Arm OHKOs at 4x but
eats Sucker Punch first, Kowtow Cleave punishes the Bullet Punch line. That is
a matching-pennies structure, so a correct simultaneous-move search MUST
return a mixed strategy — a pure answer here means the search is broken
(that is exactly the failure mode of alternating-move UCT). The assertion:
both Metagross options carry >= 20% probability.

Scenarios run in solve-to-terminal mode (small endgames), so they work with
or without a trained checkpoint — priors just speed convergence. Opponent
sets are given to the search as a collapsed belief: these are known-sets
equilibrium checks, not hidden-information tests.

CLI:
  python scenarios.py [--uniform]   # run all scenario assertions
  python scenarios.py --mine        # dump real-replay endgame candidates
  python scenarios.py --replay N    # run the search on mined candidate N
  --debug                           # search phase profiler + root tables
  --cprofile out.prof               # python-level profile (snakeviz out.prof)
"""

import json
import sys
from dataclasses import replace

from beliefs import determinized, load_dex
from config import CFG
from data import LogParser, Side, sid
from search.mcts import Searcher


def mon(species, moves, item="", ability="", nature="serious", evs=None,
        gender="", level=50):
    return {"name": species, "species": species, "item": item,
            "ability": ability, "moves": list(moves), "nature": nature,
            "evs": evs or [0] * 6, "gender": gender, "level": level}


def filler():
    """Pre-fainted teammate. '1v1' scenarios are really 2v2 with a dead slot:
    a doubles side whose team never had a second mon leaves a null active
    slot the sim's choice machinery does not expect, and real endgames always
    have fainted teammates anyway. Rough Skin Garchomp: no switch-in effect
    to perturb the position before the reconstruct faints it."""
    return mon("Garchomp", ["protect"], ability="roughskin")


ATK = [0, 252, 0, 0, 0, 252]

SCENARIOS = [
    {
        "name": "metagross-kingambit",
        "doc": "1v1 mixed-strategy equilibrium. Assert: both Metagross "
               "options (Bullet Punch / Hammer Arm) >= 20% probability.",
        "p1": [mon("Metagross", ["bulletpunch", "hammerarm"],
                   item="metagrossite", ability="clearbody", nature="jolly",
                   evs=ATK), filler()],
        "p2": [mon("Kingambit", ["suckerpunch", "kowtowcleave"],
                   item="blackglasses", ability="defiant", nature="adamant",
                   evs=[252, 252, 0, 0, 0, 0]), filler()],
        "hp": {},
        "fainted": [("p1", 1), ("p2", 1)],
        "check": lambda info: [
            f"expected a mixed strategy, both options >= 20%, got "
            + ", ".join(f"{m}={p:.0%}" for m, p in _move_marginals(
                info, ("bulletpunch", "hammerarm")).items())
        ] if min(_move_marginals(info, ("bulletpunch", "hammerarm")).values(),
                 default=0) < 0.20 else [],
    },
    {
        "name": "closeout-2v1",
        "doc": "Garchomp + Gyarados (full) vs 30% Kingambit: a won position. "
               "Assert: value >= +0.4 and the top action attacks the foe.",
        "p1": [mon("Garchomp", ["earthquake", "dragonclaw"], item="lifeorb",
                   ability="roughskin", nature="jolly", evs=ATK),
               mon("Gyarados", ["waterfall", "protect"], item="leftovers",
                   ability="intimidate", nature="jolly", evs=ATK)],
        "p2": [mon("Kingambit", ["suckerpunch", "kowtowcleave"],
                   item="blackglasses", ability="defiant", nature="adamant",
                   evs=[252, 252, 0, 0, 0, 0]), filler()],
        "hp": {("p2", 0): 0.30},
        "fainted": [("p2", 1)],
        "check": lambda info: (
            [f"value {info['value']:+.2f} < +0.4 in a won position"]
            if info["value"] < 0.4 else []) + (
            [f"top action does not attack: {info['strategy'][0][0]}"]
            if not any(a in info["strategy"][0][0] for a in
                       ("earthquake", "dragonclaw>1", "waterfall>1")) else []),
    },
    {
        "name": "aero-race",
        "doc": "Full Mega Aerodactyl vs 20% Sneasler: faster side kills "
               "first. Assert: value >= +0.5.",
        "p1": [mon("Aerodactyl", ["rockslide", "dualwingbeat"],
                   item="aerodactylite", ability="unnerve", nature="jolly",
                   evs=ATK), filler()],
        "p2": [mon("Sneasler", ["closecombat", "gunkshot"], item="whiteherb",
                   ability="unburden", nature="adamant", evs=ATK), filler()],
        "hp": {("p2", 0): 0.20},
        "fainted": [("p1", 1), ("p2", 1)],
        "check": lambda info: (
            [f"value {info['value']:+.2f} < +0.5 in a winning race"]
            if info["value"] < 0.5 else []),
    },
]


def _move_marginals(info, moves):
    out = dict.fromkeys(moves, 0.0)
    for desc, p in info["strategy"]:
        for m in moves:
            if m in desc:
                out[m] += p
    return out


def build_tracker(p1_sets, p2_sets, hp, fainted, cfg):
    t = LogParser("scenario", 0, "", cfg.format_id)
    t.sides = {"p1": Side(p1_sets), "p2": Side(p2_sets)}
    for pid, k in fainted:
        m = t.sides[pid].mons[k]
        m.fainted, m.hp, m.appeared = True, 0.0, True
    for side in t.sides.values():
        slot = 0
        for m in side.mons:
            if slot > 1 or m.fainted:
                continue
            # mid-game endgame: on the field a while (no Fake Out artifacts)
            m.active_slot, m.appeared, m.turns_active = slot, True, 2
            slot += 1
    for (pid, k), frac in hp.items():
        t.sides[pid].mons[k].hp = frac
    t.turn_no = 9
    return t


def print_damage_matrix(searcher, p1_sets, p2_sets):
    """The scenario docs claim OHKO/2HKO patterns; print the actual numbers so
    a reviewer can see whether the position is what the assertion assumes."""
    if not searcher.bridge:
        return
    from damage import request
    dex = load_dex(searcher.cfg) or {"items": {}}
    for atk_sets, dfd_sets, tag in ((p1_sets, p2_sets, "p1"),
                                    (p2_sets, p1_sets, "p2")):
        for a in atk_sets:
            stone = dex["items"].get(a["item"], {}).get("megaStone") or {}
            a_sp = stone.get(sid(a["species"]), sid(a["species"]))
            for d in dfd_sets:
                cells = searcher.bridge.calc_batch(
                    [request({**a, "species": a_sp},
                             {**d, "species": sid(d["species"])}, mv)
                     for mv in a["moves"]])
                print(f"  {tag} {a_sp} -> {sid(d['species'])}: " + "  ".join(
                    f"{mv} {c[0]:.0%}-{c[1]:.0%}" if c else f"{mv} ?"
                    for mv, c in zip(a["moves"], cells)))


def run_scenarios(searcher, cfg):
    failures = []
    for scn in SCENARIOS:
        print(f"\n--- {scn['name']} ---\n{scn['doc']}")
        print_damage_matrix(searcher, scn["p1"], scn["p2"])
        tracker = build_tracker(scn["p1"], scn["p2"], scn["hp"],
                                scn.get("fainted", []), cfg)
        belief = determinized(scn["p2"], cfg)
        joint, info = searcher.choose(tracker, belief, "p1", None,
                                      my_brought=list(range(len(scn["p1"]))),
                                      opp_brought=list(range(len(scn["p2"]))))
        print(f"value {info['value']:+.2f}"
              + (" [solve-to-terminal]" if info["solve"] else ""))
        for desc, p in info["strategy"][:6]:
            print(f"  {p:6.1%}  {desc}")
        errs = scn["check"](info)
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        failures += [(scn["name"], e) for e in errs]
    print(f"\n{len(SCENARIOS) - len(set(n for n, _ in failures))}/"
          f"{len(SCENARIOS)} scenarios passed")
    return failures


# ---------------------------------------------------------------------------
# real-replay endgames: `--mine` scans held-out battles for 2v2-or-smaller
# turn-start positions and dumps them to artifacts/endgames.json; `--replay N`
# runs the search on one so its behavior can be documented and, once vetted,
# promoted into SCENARIOS with an assertion.
# ---------------------------------------------------------------------------

def mine(cfg, max_out=50):
    import pickle
    out = []
    for fn in cfg.dataset_files:
        fmt = fn[len("logs_"):-len(".json")]
        with open(cfg.parsed_dir / f"{fmt}.pkl", "rb") as f:
            recs = [r for r in pickle.load(f) if r["split"] == "test"]
        for rec in recs:
            for turn in rec["turns"]:
                if turn["states"] is None:
                    continue
                s = turn["states"]["p1"]
                sides = (s["my"]["team"], s["opp"]["team"])
                alive = [sum(m["appeared"] and not m["fainted"] for m in t)
                         for t in sides]
                down = [sum(m["fainted"] for m in t) for t in sides]
                if max(alive) <= 2 and min(down) >= 2:
                    out.append({"tag": rec["tag"], "format": fmt,
                                "turn": turn["n"], "winner": rec["winner"],
                                "teams": rec["teams"], "state": s})
                    break   # one position per battle is plenty
            if len(out) >= max_out:
                break
        if len(out) >= max_out:
            break
    path = cfg.artifacts_dir / "endgames.json"
    path.write_text(json.dumps(out))
    print(f"{len(out)} endgame candidates -> {path}")
    for i, e in enumerate(out):
        print(f"  [{i}] {e['tag']} turn {e['turn']} winner {e['winner']}")


def _apply_view(side, views):
    for m, v in zip(side.mons, views):
        m.species_cur, m.hp, m.status = v["species_cur"], v["hp"], v["status"]
        m.boosts = dict(v["boosts"])
        m.fainted, m.active_slot = v["fainted"], v["active_slot"]
        m.appeared, m.mega_done = v["appeared"], v["mega_done"]
        m.item_consumed = v["item_consumed"]
        m.turns_active = v.get("turns_active", 2)


def _infer_brought(mons):
    n = min(4, len(mons))
    brought = [m.team_idx for m in mons if m.appeared]
    return brought + [m.team_idx for m in mons if not m.appeared][:n - len(brought)]


def replay(searcher, cfg, i):
    e = json.loads((cfg.artifacts_dir / "endgames.json").read_text())[i]
    t = LogParser(e["tag"], 0, "", cfg.format_id)
    t.sides = {"p1": Side(e["teams"]["p1"]), "p2": Side(e["teams"]["p2"])}
    s = e["state"]
    _apply_view(t.sides["p1"], s["my"]["team"])
    _apply_view(t.sides["p2"], s["opp"]["team"])
    t.sides["p1"].mega_used = not s["my"]["mega_available"]
    t.sides["p2"].mega_used = not s["opp"]["mega_available"]
    t.sides["p1"].conditions = dict(s["my"]["conditions"])
    t.sides["p2"].conditions = dict(s["opp"]["conditions"])
    t.weather, t.terrain = s["weather"], s["terrain"]
    t.trickroom, t.turn_no = s["trickroom"], s["turn"]
    belief = determinized(e["teams"]["p2"], cfg)
    joint, info = searcher.choose(t, belief, "p1", None,
                                  my_brought=_infer_brought(t.sides["p1"].mons),
                                  opp_brought=_infer_brought(t.sides["p2"].mons))
    print(f"{e['tag']} turn {e['turn']} — value {info['value']:+.2f}, "
          f"actual winner {e['winner']}")
    for desc, p in info["strategy"][:8]:
        print(f"  {p:6.1%}  {desc}")


def main():
    from search.debug import maybe_cprofile
    cfg = replace(CFG, n_determinizations=1)   # sets are known here
    if "--mine" in sys.argv:
        mine(cfg)
        return
    model = tok = None
    ckpt = cfg.checkpoint_dir / "ckpt_best.pt"
    if ckpt.exists() and "--uniform" not in sys.argv:
        import torch

        from models.policy_value import PolicyValueNet
        from tokenizer import PositionTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PolicyValueNet.load(ckpt, cfg, device)
        tok = PositionTokenizer.load(cfg)
        print(f"priors/values from {ckpt}")
    else:
        print("no checkpoint: uniform priors, solve-to-terminal only")
    searcher = Searcher(model, tok, cfg, debug="--debug" in sys.argv)
    cprof = sys.argv[sys.argv.index("--cprofile") + 1] \
        if "--cprofile" in sys.argv else None
    with maybe_cprofile(cprof):
        if "--replay" in sys.argv:
            replay(searcher, cfg, int(sys.argv[sys.argv.index("--replay") + 1]))
            return
        failures = run_scenarios(searcher, cfg)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
