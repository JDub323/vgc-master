"""Tests for the objective (nature, SP-spread) prior + nature-as-inferred-dim
(beliefs.py _expand_spreads, config.spreads_prior), backed by artifacts/
spreads.json (build_spreads.py).

The dataset redacts nature+SP, so there is no opponent EV/nature ground truth to
validate against on real battles. Instead we CONSTRUCT the ground truth: author a
true (nature, spread) build, generate the observation it produces through the
exact calc / speed formula, feed it to the filter, and assert the true build
survives and inconsistent natures are pruned -- i.e. the range of survivors
contains the truth. This is the "very strict unit test against Showdown ground
truth" for the range predictions, with the ground truth synthesized.

Run:  python tests/test_spreads_nature.py            (expansion + speed: no node)
      python tests/test_spreads_nature.py --bridge   (also the damage calc)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG
from beliefs import (OpponentBelief, load_spreads, load_dex, calc_stat,
                     STAT_KEYS, NATURES, MAX_SP)

DEX = load_dex(CFG)
SPREADS = load_spreads(CFG)
assert CFG.spreads_prior, "these tests exercise the spreads path (default on)"
assert SPREADS, "needs artifacts/spreads.json (python build_spreads.py)"

GARCHOMP_MOVES = ["earthquake", "dragonclaw", "protect", "swordsdance"]


def cov_belief(species, moves, item, ability, my_team, bridge=None):
    """A covered mon with ONE redacted usage set (nature='serious', no evs);
    _expand_spreads pulls the real (spread,nature) builds from spreads.json."""
    usage = {species: [[10, list(moves), item, ability, "serious"]]}
    return OpponentBelief([species], usage, CFG, bridge, my_team=my_team)


def base_spe(species):
    return DEX["species"][species]["baseStats"]["spe"]


def alive(bel, j):
    return bel.weights[0][j] > 1e-9


# --------------------------------------------------------------------------- #
# expansion: real natures + concrete spreads replace the neutral archetype grid
# --------------------------------------------------------------------------- #

def test_expansion_uses_real_natures_and_spreads():
    bel = cov_belief("garchomp", GARCHOMP_MOVES, "lifeorb", "roughskin", [])
    ps = bel.particles[0]
    concrete = [p for p in ps if p["arch"] is None]
    assert concrete, "covered mon must produce concrete builds"
    nats = {p["nature"] for p in concrete}
    # Garchomp is ~60% Jolly / 39% Adamant -- both must appear, and NOT 'serious'
    assert "jolly" in nats and "adamant" in nats, nats
    assert "serious" not in nats, "concrete builds must carry the real nature"
    # every concrete build has a real 66-pt spread from the page
    page = {tuple(ev) for ev, _ in SPREADS["garchomp"]["spreads"]}
    assert all(tuple(p["evs"]) in page for p in concrete), "spreads must be real rows"
    # exactly one 'any' slack cushion per base set
    assert sum(p["arch"] == "any" for p in ps) == 1
    assert abs(sum(bel.weights[0]) - 1.0) < 1e-9
    print(f"ok  expansion: {len(concrete)} real builds ({sorted(nats)}) + any cushion")


def test_authored_set_passes_through():
    """A determinized/authored set already carrying evs is NOT re-expanded even
    for a covered species (search determinizations must stay fixed)."""
    usage = {"garchomp": [[1, GARCHOMP_MOVES, "lifeorb", "roughskin", "jolly",
                           [0, 32, 0, 0, 0, 34 - 2]]]}   # concrete spread present
    bel = OpponentBelief(["garchomp"], usage, CFG, None, my_team=[])
    assert len(bel.particles[0]) == 1, "authored set must pass through untouched"
    assert bel.particles[0][0]["nature"] == "jolly"
    print("ok  authored set (evs present) passes through, no re-expansion")


def test_uncovered_species_falls_back_to_archetypes():
    """A species absent from spreads.json still expands via the neutral-nature
    archetype grid, so the fallback path is intact."""
    assert "furret" not in SPREADS, "pick a species not on the page"
    usage = {"furret": [[10, ["doubleedge", "protect"], "sitrusberry", "runaway",
                         "serious"]]}
    bel = OpponentBelief(["furret"], usage, CFG, None, my_team=[])
    archs = {p["arch"] for p in bel.particles[0]}
    assert "fast-strong" in archs or "bulky-phys" in archs, archs
    print("ok  uncovered species falls back to archetype grid")


# --------------------------------------------------------------------------- #
# nature inferred from a move-order fact (no node)
# --------------------------------------------------------------------------- #

def test_speed_prunes_incompatible_nature():
    """Opponent Garchomp acted before my mon, at a speed only a +Spe (Jolly)
    build can reach -- an Adamant build (neutral Spe) at the same 32-Spe spread
    is too slow. The filter must keep Jolly builds and kill Adamant builds:
    nature inferred purely from move order."""
    gbase = base_spe("garchomp")
    jolly_max = calc_stat(gbase, "spe", "jolly", MAX_SP)      # 169
    adamant_max = calc_stat(gbase, "spe", "adamant", MAX_SP)  # 154 (Spe neutral)
    # my mon whose SLOWEST speed sits strictly between the two
    my_species = "dragapult"                                   # base 142
    mine = calc_stat(base_spe(my_species), "spe", "serious", 0)
    assert adamant_max < mine <= jolly_max, (adamant_max, mine, jolly_max)

    my_team = [{"species": my_species, "nature": "serious", "item": "",
                "evs": [0] * 6, "level": 50}]
    bel = cov_belief("garchomp", GARCHOMP_MOVES, "lifeorb", "roughskin", my_team)
    # opponent (p2) moved before my mon (p1), same-priority damaging moves
    order = [("p2", 0, "earthquake", {"spe": 0, "tw": False, "par": False}),
             ("p1", 0, "dragondarts", {"spe": 0, "tw": False, "par": False})]
    bel.update([("move_order", order, {"tr": False})], viewer="p1")

    for j, p in enumerate(bel.particles[0]):
        if p["arch"] is not None:                # skip the 'any' cushion
            continue
        their_spe = calc_stat(gbase, "spe", p["nature"], p["evs"][5])
        assert alive(bel, j) == (their_spe >= mine), (
            f"nat={p['nature']} spe={their_spe} mine={mine} alive={alive(bel,j)}")
    jolly_alive = any(alive(bel, j) and p["nature"] == "jolly"
                      for j, p in enumerate(bel.particles[0]))
    adamant_alive = any(alive(bel, j) and p["nature"] == "adamant" and p["arch"] is None
                        for j, p in enumerate(bel.particles[0]))
    assert jolly_alive and not adamant_alive, (jolly_alive, adamant_alive)
    print("ok  speed fact infers nature: Jolly builds kept, Adamant builds killed")


# --------------------------------------------------------------------------- #
# damage -> nature, and the Adamant-truth-survives fix (needs the node calc)
# --------------------------------------------------------------------------- #

def _dmg_event(move, frac, def_hp_before=1.0, allies_fainted=0):
    ctx = {"crit": False, "spread": False, "multi": False, "burn": False,
           "weather": None, "terrain": None, "atk_boosts": {}, "def_boosts": {},
           "screens": [], "def_hp_before": def_hp_before, "def_transformed": False,
           "allies_fainted": allies_fainted}
    return ("dmg", "p2", 0, move, "p1", 0, frac, ctx)


def test_supreme_overlord_allies_fainted(bridge):
    """A Supreme-Overlord hit with fainted allies deals MORE than the calc's
    unboosted max, so without the alliesFainted context the observed damage
    kills every build. Passing the real allies_fainted through the ctx lets the
    true build reproduce it and survive -- the plumbing data.py -> ctx ->
    request -> @smogon/calc must be intact."""
    from damage import request
    move = "ironhead"
    ev = next(e for e, _ in SPREADS["kingambit"]["spreads"] if e[1] == 32)  # max Atk
    defender = {"species": "hippowdon", "level": 50, "item": "", "ability": "",
                "nature": "impish", "evs": [0] * 6, "boosts": {}}
    atk = {"species": "kingambit", "level": 50, "item": "leftovers",
           "ability": "supremeoverlord", "nature": "adamant", "evs": list(ev),
           "boosts": {}, "alliesFainted": 3}
    frac = bridge.calc_batch([request(atk, defender, move, {})])[0][1]  # 3-ally max

    moves = ["ironhead", "suckerpunch", "kowtowcleave", "protect"]
    my_team = [{"species": "hippowdon", "nature": "impish", "item": "",
                "ability": "", "evs": [0] * 6, "level": 50}]

    def run(allies):
        bel = cov_belief("kingambit", moves, "leftovers", "supremeoverlord",
                         my_team, bridge)
        bel.update([_dmg_event(move, frac, allies_fainted=allies)], viewer="p1")
        ad = sum(w for w, p in zip(bel.weights[0], bel.particles[0])
                 if p["nature"] == "adamant" and p["arch"] is None)
        return bel.soft_depletions[0], ad

    # with the real context the boosted hit is reproducible: no depletion, and
    # the true (Adamant) build is the survivor.
    dep3, ad3 = run(3)
    assert dep3 == 0 and ad3 > 0, (dep3, ad3)
    # without it the calc's unboosted max can't reach the observed damage for
    # ANY build, so evidence wipes every particle -> soft depletion fires. That
    # was the un-modeled-boost true-set kill; the ctx plumbing is what avoids it.
    dep0, _ = run(0)
    assert dep0 > 0, "unboosted, the Supreme Overlord hit should be unreachable"
    print("ok  Supreme Overlord alliesFainted plumbed (reproducible with, "
          "depletes without)")


def test_adamant_truth_survives_its_own_damage(bridge):
    """THE fix. Author a true Adamant Garchomp on its real max-Atk spread, get
    the damage the calc says its Earthquake deals to a known defender, feed it,
    and assert (a) the true Adamant build survives, (b) a -Atk (Modest) build at
    the same spread is killed (its max roll can't reach the observed damage),
    (c) the neutral 'serious' assumption the OLD filter used could NOT have
    reproduced it -- serious max roll < observed. That (c) is exactly why the
    neutral-nature prior was killing true sets."""
    from damage import request
    move, key = "earthquake", "atk"
    true_ev = next(ev for ev, _ in SPREADS["garchomp"]["spreads"]
                   if ev[1] == 32)                       # a 32-Atk spread
    defender = {"species": "hippowdon", "level": 50, "item": "", "ability": "",
                "nature": "impish", "evs": [0] * 6, "boosts": {}}

    def roll(nature):
        atk = {"species": "garchomp", "level": 50, "item": "lifeorb",
               "ability": "roughskin", "nature": nature, "evs": list(true_ev),
               "boosts": {}, "status": ""}
        return bridge.calc_batch([request(atk, defender, move, {})])[0]

    r_ad = roll("adamant")
    assert r_ad and r_ad[1] < 1.0, f"defender must survive: {r_ad}"
    frac = r_ad[1]                                       # observed = adamant max roll
    r_se = roll("serious")
    assert r_se[1] < frac - CFG.damage_tolerance, \
        f"neutral must undershoot the observed adamant hit: {r_se[1]} vs {frac}"

    my_team = [{"species": "hippowdon", "nature": "impish", "item": "",
                "ability": "", "evs": [0] * 6, "level": 50}]
    bel = cov_belief("garchomp", GARCHOMP_MOVES, "lifeorb", "roughskin", my_team, bridge)
    assert bel.strict_atk
    bel.update([_dmg_event(move, frac)], viewer="p1")

    ad_mass = sum(w for w, p in zip(bel.weights[0], bel.particles[0])
                  if p["nature"] == "adamant" and p["arch"] is None)
    modest_alive = any(alive(bel, j) for j, p in enumerate(bel.particles[0])
                       if p["nature"] == "modest" and p["arch"] is None)
    assert ad_mass > 0, "the TRUE Adamant build must survive its own damage"
    assert not modest_alive, "a -Atk (Modest) build cannot reach the observed damage"
    print(f"ok  Adamant truth survives (mass {ad_mass:.3f}); neutral would undershoot "
          f"({r_se[1]:.3f} < {frac:.3f}) -- the kill the old prior caused is gone")


PURE = [test_expansion_uses_real_natures_and_spreads,
        test_authored_set_passes_through,
        test_uncovered_species_falls_back_to_archetypes,
        test_speed_prunes_incompatible_nature]
BRIDGE = [test_adamant_truth_survives_its_own_damage,
          test_supreme_overlord_allies_fainted]


if __name__ == "__main__":
    assert DEX, "needs artifacts/dex.json"
    for t in PURE:
        t()
    if "--bridge" in sys.argv:
        from damage import DamageBridge
        bridge = DamageBridge(CFG)
        try:
            for t in BRIDGE:
                t(bridge)
        finally:
            bridge.close()
    else:
        print("(skipping damage-calc tests; pass --bridge to run them)")
    print("all spreads/nature tests passed")
