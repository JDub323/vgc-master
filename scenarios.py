"""Scripted scenarios with assertions about search behavior — endgame gates
plus earlygame/midgame diagnostics.

The headline test is the Metagross/Kingambit 1v1: Bullet Punch outprioritizes
and blanks Sucker Punch (the target already moved), Hammer Arm OHKOs at 4x but
eats Sucker Punch first, Kowtow Cleave punishes the Bullet Punch line. That is
a matching-pennies structure, so a correct simultaneous-move search MUST
return a mixed strategy — a pure answer here means the search is broken
(that is exactly the failure mode of alternating-move UCT). The assertion:
both Metagross options carry >= 20% probability.

Two scenario classes:

  endgame gates    <= solve_endgame_at mons per side, run in solve-to-terminal
                   mode, so they work with or without a trained checkpoint —
                   priors just speed convergence. FAILs gate the suite.
  early/midgame    full or near-full teams (back mons, weather wars, megas,
  diagnostics      trick room). These need a checkpoint (value-head leaves)
                   and probe model understanding rather than search
                   correctness: switching an endangered mon out, weather
                   control, predicted opponent switch-ins, Contrary boost
                   lines. They print NOTEs, never gate — track them across
                   checkpoints.

Opponent sets are given to the search as a collapsed belief: these are
known-sets checks, not hidden-information tests. Every position prints its
real damage matrix (through @smogon/calc, with the scenario's weather and
fainted-ally state) so a reviewer can verify each doc claim is what the
engine actually computes.

CLI:
  python scenarios.py [--uniform]   # run all scenario assertions
  python scenarios.py --mine        # dump real-replay endgame candidates
  python scenarios.py --replay N    # run the search on mined candidate N
  --debug                           # search phase profiler + root tables
  --cprofile out.prof               # python-level profile (snakeviz out.prof)
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("scenarios.py"):
        raise SystemExit(0)

import json
import sys
from dataclasses import replace

from beliefs import determinized, load_dex
from config import CFG
from data import LogParser, Side, sid
from agents.determinized_duct.v1 import DeterminizedDUCTChooser


def mon(species, moves, item="", ability="", nature="serious", evs=None,
        gender="", level=50):
    """Return one normalized authored scenario ``PokemonSet`` mapping."""
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


# Champions stat points (max 32/stat, 66 total), not EVs
ATK = [2, 32, 0, 0, 0, 32]

SCENARIOS = [
    {
        "name": "metagross-kingambit",
        "doc": "1v1 mixed-strategy equilibrium. Metagross at 70% so Sucker "
               "Punch's min roll (~74%) always KOs — at full HP it survives "
               "and Hammer Arm just dominates. Assert: both Metagross "
               "options (Bullet Punch / Hammer Arm) >= 20% probability.",
        "p1": [mon("Metagross", ["bulletpunch", "hammerarm"],
                   item="metagrossite", ability="clearbody", nature="jolly",
                   evs=ATK), filler()],
        "p2": [mon("Kingambit", ["suckerpunch", "kowtowcleave"],
                   item="blackglasses", ability="defiant", nature="adamant",
                   evs=[32, 32, 0, 0, 2, 0]), filler()],
        "hp": {("p1", 0): 0.70},
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
                   evs=[32, 32, 0, 0, 2, 0]), filler()],
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
    {
        # DIAGNOSTIC, not a gate: probes the suspected underswitching bias
        # after search (evaluate.py --switches probes it in the prior).
        "name": "peli-weather-war",
        "doc": "Weather war, mid-game (needs a checkpoint; value-head mode). "
               "My Pelipper + Archaludon vs Torkoal + Venusaur with SUN up "
               "(Torkoal entered after Pelipper). Full-power sun Eruption "
               "threatens both my mons; Wide Guard blanks it entirely, and "
               "switching Pelipper out preserves a later Drizzle re-entry "
               "that flips the weather back. Diagnostic: the defensive "
               "lines (Wide Guard / switch Pelipper / Protect) should carry "
               "real mass; ~zero mass here is the underswitching signature.",
        "needs_model": True,
        "diagnostic": True,
        "weather": "sunnyday",
        "p1": [mon("Pelipper", ["hurricane", "weatherball", "wideguard",
                                "protect"],
                   item="focussash", ability="drizzle", nature="modest",
                   evs=[32, 0, 0, 32, 2, 0], gender="F"),
               mon("Archaludon", ["electroshot", "dracometeor", "bodypress",
                                  "protect"],
                   item="assaultvest", ability="stamina", nature="modest",
                   evs=[32, 0, 0, 32, 2, 0]),
               mon("Barraskewda", ["liquidation", "closecombat", "protect"],
                   item="focussash", ability="swiftswim", nature="adamant",
                   evs=ATK, gender="F")],
        "p2": [mon("Torkoal", ["eruption", "heatwave", "protect"],
                   item="charcoal", ability="drought", nature="quiet",
                   evs=[32, 0, 0, 32, 2, 0], gender="F"),
               mon("Venusaur", ["sludgebomb", "gigadrain", "sleeppowder",
                                "protect"],
                   item="lifeorb", ability="chlorophyll", nature="modest",
                   evs=[2, 0, 0, 32, 0, 32], gender="F"),
               mon("Flareon", ["flareblitz", "protect"], item="leftovers",
                   ability="flashfire", nature="adamant", evs=ATK,
                   gender="F")],
        "hp": {("p1", 0): 0.75},
        "fainted": [],
        "check": lambda info: (lambda defensive: [
            f"defensive lines carry only {defensive:.0%} "
            "(wide guard + pelipper switch + pelipper protect) — "
            "underswitching signature if this stays ~0 across checkpoints"
        ] if defensive < 0.15 else [])(
            _mass(info, "wideguard") + _slot_mass(info, 0, "sw ")
            + _slot_mass(info, 0, "protect")),
    },
    {
        # DIAGNOSTIC: the joint-context case the factorized head could not
        # express — a frail attacker's best action depends on whether its
        # partner redirects.
        "name": "chomp-redirect",
        "doc": "Joint-action context (needs a checkpoint). 25% Garchomp + "
               "Amoonguss vs faster Dragapult that KOs Garchomp on any hit. "
               "Attacking with Garchomp only makes sense alongside partner "
               "Rage Powder; otherwise it should protect or switch. "
               "Diagnostic: P(chomp attacks | Amoonguss rage powders) should "
               "exceed P(chomp attacks | it doesn't) — a factorized policy "
               "is structurally unable to show that gap.",
        "needs_model": True,
        "diagnostic": True,
        "p1": [mon("Garchomp", ["dragonclaw", "rockslide", "protect"],
                   item="lifeorb", ability="roughskin", nature="jolly",
                   evs=ATK),
               mon("Amoonguss", ["ragepowder", "sludgebomb", "protect"],
                   item="sitrusberry", ability="regenerator", nature="sassy",
                   evs=[32, 0, 17, 0, 17, 0], gender="F"),
               mon("Gyarados", ["waterfall", "protect"], item="leftovers",
                   ability="intimidate", nature="jolly", evs=ATK, gender="F")],
        "p2": [mon("Dragapult", ["dragondarts", "phantomforce", "protect"],
                   item="choiceband", ability="clearbody", nature="jolly",
                   evs=ATK),
               mon("Primarina", ["moonblast", "protect"], item="sitrusberry",
                   ability="torrent", nature="modest",
                   evs=[32, 0, 0, 32, 2, 0], gender="F"),
               mon("Clefairy", ["followme", "protect"], item="eviolite",
                   ability="friendguard", nature="sassy",
                   evs=[32, 0, 17, 0, 17, 0], gender="F")],
        "hp": {("p1", 0): 0.25},
        "fainted": [],
        "check": lambda info: (lambda with_rp, without_rp: [
            f"P(chomp attacks & rage powder)={with_rp:.0%} <= "
            f"P(chomp attacks & no rage powder)={without_rp:.0%} — the "
            "attack is not conditioned on the redirect"
        ] if with_rp <= without_rp else [])(
            _joint_mass(info, 0, ("dragonclaw", "rockslide"), 1, ("ragepowder",)),
            _joint_mass(info, 0, ("dragonclaw", "rockslide"), 1,
                        ("sludgebomb", "protect", "sw "))),
    },
    {
        # ENDGAME GATE (solve mode): priority chip changes an HP-scaled nuke.
        "name": "torkoal-room-eruption",
        "doc": "Trick Room, sun. My Kingambit (85%) + Garchomp (42%) vs full "
               "Torkoal. Full-HP sun Eruption KOs Garchomp (min 44% vs 42) "
               "and always KOs Kingambit; Eruption's power scales with "
               "Torkoal's HP, so a Sucker Punch chip (~35-41%) drops it below "
               "Garchomp's bar (max ~34%). Under Trick Room Torkoal moves "
               "before Garchomp, so the winning line is one turn: Sucker "
               "Punch chips first (priority ignores TR), the weakened "
               "Eruption still fells Kingambit but not Garchomp, and Life "
               "Orb Earthquake (min 72%) finishes the chipped Torkoal from "
               "at most 65.5%. Kingambit is dead in every line, so spending "
               "it on the chip is right (both its attacks are all-attack "
               "sets, so Sucker Punch always connects). Assert: P(Sucker "
               "Punch) >= 50%, the Sucker Punch + Earthquake joint >= 35%, "
               "value >= +0.5.",
        "stage": "endgame",
        "weather": "sunnyday",
        "trickroom": True,
        "p1": [mon("Kingambit", ["suckerpunch", "kowtowcleave", "ironhead",
                                 "protect"],
                   item="blackglasses", ability="defiant", nature="adamant",
                   evs=[32, 32, 0, 0, 2, 0]),
               mon("Garchomp", ["earthquake", "dragonclaw", "protect"],
                   item="lifeorb", ability="roughskin", nature="jolly",
                   evs=ATK)],
        "p2": [mon("Torkoal", ["eruption", "heatwave"],
                   item="charcoal", ability="drought", nature="quiet",
                   evs=[32, 0, 0, 32, 2, 0], gender="F"), filler()],
        "hp": {("p1", 0): 0.85, ("p1", 1): 0.42},
        "fainted": [("p2", 1)],
        "check": lambda info: (
            [f"P(sucker punch)={_slot_mass(info, 0, 'suckerpunch'):.0%} < 50% "
             "— the priority chip that saves Garchomp is being missed"]
            if _slot_mass(info, 0, "suckerpunch") < 0.50 else []) + (
            [f"sucker+earthquake joint={_joint_mass(info, 0, ('suckerpunch',), 1, ('earthquake',)):.0%} < 35%"]
            if _joint_mass(info, 0, ("suckerpunch",), 1,
                           ("earthquake",)) < 0.35 else []) + (
            [f"value {info['value']:+.2f} < +0.5 in a won position"]
            if info["value"] < 0.5 else []),
    },
    {
        # DIAGNOSTIC: weather war + priority Tailwind race with real
        # counterplay on both sides ("just a complex position all-around").
        "name": "whimsi-chomp-snow",
        "doc": "Midgame weather war (needs a checkpoint). My Whimsicott + "
               "Garchomp vs Ninetales-Alola (45%) + Kingambit (65%) in snow, "
               "Pelipper in their back. Snow Blizzard KOs both my mons "
               "(96-114% / 125-151%) and Ninetales (178 Spe) outruns "
               "Garchomp (169) — but Prankster Tailwind flips the race "
               "mid-turn: doubled Garchomp Earthquakes first and its Life "
               "Orb EQ kills BOTH (min 47% vs 45, min 69% vs 65). Their "
               "counterplay: Iron Head OHKOs Whimsicott before it matters "
               "(Tailwind already resolved, so maybe that trade is fine), "
               "Ninetales can Protect the EQ turn, Sucker Punch chips "
               "Garchomp (45-54%), and even a won exchange hands Pelipper a "
               "Drizzle re-entry that flips the weather war again. Encore "
               "(Whimsicott) and Protect (Garchomp) give p1 the same stall "
               "tools. Diagnostic: the Tailwind + Earthquake race should "
               "carry real mass; ~zero means the model is not seeing the "
               "mid-turn speed flip.",
        "stage": "midgame",
        "needs_model": True,
        "diagnostic": True,
        "weather": "snowscape",
        "turn": 3,
        "p1": [mon("Whimsicott", ["moonblast", "tailwind", "encore",
                                  "protect"],
                   item="mentalherb", ability="prankster", nature="timid",
                   evs=[2, 0, 0, 32, 0, 32], gender="F"),
               mon("Garchomp", ["earthquake", "dragonclaw", "protect"],
                   item="lifeorb", ability="roughskin", nature="jolly",
                   evs=ATK)],
        "p2": [mon("Ninetales-Alola", ["blizzard", "moonblast", "auroraveil",
                                       "protect"],
                   item="lightclay", ability="snowwarning", nature="timid",
                   evs=[2, 0, 0, 32, 0, 32], gender="F"),
               mon("Kingambit", ["suckerpunch", "kowtowcleave", "ironhead",
                                 "protect"],
                   item="blackglasses", ability="defiant", nature="adamant",
                   evs=[32, 32, 0, 0, 2, 0]),
               mon("Pelipper", ["hurricane", "weatherball", "tailwind",
                                "protect"],
                   item="focussash", ability="drizzle", nature="timid",
                   evs=[2, 0, 0, 32, 0, 32], gender="F")],
        "hp": {("p2", 0): 0.45, ("p2", 1): 0.65},
        "fainted": [],
        "check": lambda info: (lambda race: [
            f"tailwind+earthquake race carries only {race:.0%} — the "
            "mid-turn Tailwind speed flip is not being credited"
        ] if race < 0.10 else [])(
            _joint_mass(info, 0, ("tailwind",), 1, ("earthquake",))),
    },
    {
        # DIAGNOSTIC: opponent modeling — a threatened side's best line is a
        # weather-flipping switch, and the model should PREDICT it.
        "name": "zardy-sun-peli-switch",
        "doc": "Midgame opponent switch prediction (needs a checkpoint). My "
               "Mega Charizard Y + Garchomp vs Sinistcha + Kingambit in sun, "
               "Pelipper in their back. Sun Heat Wave OHKOs both actives "
               "(129-153% / 114-136%), but a Pelipper switch-in flips sun to "
               "rain on entry and eats Heat Wave at 13-16% instead of the "
               "41-48% it would take in sun — a third of the damage, and it "
               "shields whichever mon it replaces. Switching out the "
               "endangered mon is the opponent's best line, so the model's "
               "opponent prior should put real mass on 'sw pelipper'. "
               "Diagnostic: predicted P(pelipper switches in) >= 10%.",
        "stage": "midgame",
        "needs_model": True,
        "diagnostic": True,
        "weather": "sunnyday",
        "turn": 5,
        "mega": [("p1", 0)],
        "p1": [mon("Charizard", ["heatwave", "solarbeam", "weatherball",
                                 "protect"],
                   item="charizarditey", ability="solarpower", nature="timid",
                   evs=[2, 0, 0, 32, 0, 32]),
               mon("Garchomp", ["earthquake", "dragonclaw", "protect"],
                   item="lifeorb", ability="roughskin", nature="jolly",
                   evs=ATK)],
        "p2": [mon("Sinistcha", ["matchagotcha", "ragepowder", "lifedew",
                                 "protect"],
                   item="kasibberry", ability="hospitality", nature="calm",
                   evs=[32, 0, 32, 2, 0, 0]),
               mon("Kingambit", ["kowtowcleave", "suckerpunch", "ironhead",
                                 "protect"],
                   item="blackglasses", ability="defiant", nature="adamant",
                   evs=[32, 32, 0, 0, 2, 0]),
               mon("Pelipper", ["hurricane", "weatherball", "tailwind",
                                "protect"],
                   item="focussash", ability="drizzle", nature="timid",
                   evs=[2, 0, 0, 32, 0, 32], gender="F")],
        "hp": {},
        "fainted": [],
        "check": lambda info: (lambda p_sw: [
            f"predicted P(sw pelipper)={p_sw:.0%} < 10% — the model does not "
            "expect the weather-flipping protective switch"
        ] if p_sw < 0.10 else [])(_opp_mass(info, "sw pelipper")),
    },
    {
        # DIAGNOSTIC: Contrary boost line, unpunished. Mega Staraptor's
        # ability is Contrary, so an ally Tickle is +1 Atk/+1 Def and an
        # ally Charm is +2 Atk.
        "name": "self-tickle-free",
        "doc": "Boost-line discovery (needs a checkpoint). My Whimsicott + "
               "Mega Staraptor (Contrary) vs slow, low-pressure Snorlax + "
               "Milotic. Tickle on my own Staraptor is +1 Atk / +1 Def and "
               "Charm is +2 Atk (Contrary inverts the drops); nothing on "
               "the field meaningfully threatens Staraptor, so stacking "
               "boosts before Brave Bird / Close Combat (64-76% unboosted "
               "vs Milotic) is a legitimate line. Diagnostic: the ally-"
               "targeted Tickle/Charm should carry >= 10% mass.",
        "stage": "midgame",
        "needs_model": True,
        "diagnostic": True,
        "turn": 4,
        "mega": [("p1", 1)],
        "p1": [mon("Whimsicott", ["tickle", "charm", "moonblast", "encore"],
                   item="focussash", ability="prankster", nature="timid",
                   evs=[2, 0, 0, 32, 0, 32], gender="F"),
               mon("Staraptor", ["bravebird", "closecombat", "doubleedge",
                                 "protect"],
                   item="staraptite", ability="intimidate", nature="jolly",
                   evs=ATK)],
        "p2": [mon("Snorlax", ["bodyslam", "highhorsepower", "yawn",
                               "protect"],
                   item="sitrusberry", ability="thickfat", nature="brave",
                   evs=[32, 32, 0, 0, 0, 0]),
               mon("Milotic", ["muddywater", "icebeam", "recover", "protect"],
                   item="leftovers", ability="competitive", nature="calm",
                   evs=[32, 0, 2, 0, 32, 0], gender="F")],
        "hp": {},
        "fainted": [],
        "check": lambda info: (lambda boost: [
            f"self-Tickle/Charm carries only {boost:.0%} — the Contrary "
            "boost line is invisible to the model"
        ] if boost < 0.10 else [])(
            _mass(info, "tickle>ally") + _mass(info, "charm>ally")),
    },
    {
        # DIAGNOSTIC: the same boost line under a real threat. NOTE the
        # actual redirection mechanics, verified against the sim: Rage
        # Powder is a powder move, and Whimsicott is Grass-type, so
        # Sinistcha CANNOT redirect the ally-targeted Tickle (powder
        # immunity). The danger here is not the redirect — it is Zen
        # Headbutt (120-142%) deleting Mega Staraptor before any boost
        # pays off.
        "name": "self-tickle-threatened",
        "doc": "Boost line vs pressure (needs a checkpoint). Same Whimsicott "
               "+ Mega Staraptor (Contrary), now into Rage Powder Sinistcha "
               "+ Zen Headbutt Mega Metagross, Incineroar in my back. Rage "
               "Powder does NOT redirect the self-Tickle — Whimsicott is "
               "Grass-type and Rage Powder is a powder move — but greeding "
               "boosts is still wrong when Zen Headbutt OHKOs Staraptor "
               "(120-142%): the payoff dies with the bird. Protecting "
               "Staraptor or switching it out (Incineroar) has to carry "
               "mass. Diagnostic: defensive Staraptor lines >= 15%.",
        "stage": "midgame",
        "needs_model": True,
        "diagnostic": True,
        "turn": 4,
        "mega": [("p1", 1), ("p2", 1)],
        "p1": [mon("Whimsicott", ["tickle", "charm", "moonblast", "encore"],
                   item="focussash", ability="prankster", nature="timid",
                   evs=[2, 0, 0, 32, 0, 32], gender="F"),
               mon("Staraptor", ["bravebird", "closecombat", "doubleedge",
                                 "protect"],
                   item="staraptite", ability="intimidate", nature="jolly",
                   evs=ATK),
               mon("Incineroar", ["fakeout", "throatchop", "flareblitz",
                                  "partingshot"],
                   item="sitrusberry", ability="intimidate", nature="careful",
                   evs=[32, 2, 0, 0, 32, 0])],
        "p2": [mon("Sinistcha", ["matchagotcha", "ragepowder", "lifedew",
                                 "protect"],
                   item="kasibberry", ability="hospitality", nature="calm",
                   evs=[32, 0, 32, 2, 0, 0]),
               mon("Metagross", ["zenheadbutt", "bulletpunch", "ironhead",
                                 "protect"],
                   item="metagrossite", ability="clearbody", nature="jolly",
                   evs=ATK)],
        "hp": {},
        "fainted": [],
        "check": lambda info: (lambda defensive: [
            f"defensive Staraptor lines carry only {defensive:.0%} "
            "(protect + switch out) under a clean OHKO threat"
        ] if defensive < 0.15 else [])(
            _slot_mass(info, 1, "protect") + _slot_mass(info, 1, "sw ")),
    },
    {
        # DIAGNOSTIC: the redirect that DOES steal the boost. Follow Me has
        # no powder flag, so unlike Rage Powder it redirects Grass-type
        # Whimsicott's ally-targeted Tickle onto Clefairy (the Pollen Puff
        # interaction) — the drop lands on a non-Contrary mon and the turn
        # is wasted.
        "name": "self-tickle-follow-me",
        "doc": "Boost line vs redirection (needs a checkpoint). Whimsicott + "
               "Mega Staraptor into Follow Me Clefairy + Mega Metagross. "
               "Follow Me (unlike Rage Powder) redirects the ally-targeted "
               "Tickle onto Clefairy — no Contrary there, so the self-boost "
               "line is a wasted turn whenever Clefairy commits to Follow "
               "Me, on top of the standing Zen Headbutt threat. Diagnostic: "
               "self-Tickle/Charm should NOT dominate here (< 25%); compare "
               "with self-tickle-free.",
        "stage": "midgame",
        "needs_model": True,
        "diagnostic": True,
        "turn": 4,
        "mega": [("p1", 1), ("p2", 1)],
        "p1": [mon("Whimsicott", ["tickle", "charm", "moonblast", "encore"],
                   item="focussash", ability="prankster", nature="timid",
                   evs=[2, 0, 0, 32, 0, 32], gender="F"),
               mon("Staraptor", ["bravebird", "closecombat", "doubleedge",
                                 "protect"],
                   item="staraptite", ability="intimidate", nature="jolly",
                   evs=ATK),
               mon("Incineroar", ["fakeout", "throatchop", "flareblitz",
                                  "partingshot"],
                   item="sitrusberry", ability="intimidate", nature="careful",
                   evs=[32, 2, 0, 0, 32, 0])],
        "p2": [mon("Clefairy", ["followme", "helpinghand", "moonblast",
                                "protect"],
                   item="eviolite", ability="friendguard", nature="sassy",
                   evs=[32, 0, 17, 0, 17, 0], gender="F"),
               mon("Metagross", ["zenheadbutt", "bulletpunch", "ironhead",
                                 "protect"],
                   item="metagrossite", ability="clearbody", nature="jolly",
                   evs=ATK)],
        "hp": {},
        "fainted": [],
        "check": lambda info: (lambda boost: [
            f"self-Tickle/Charm carries {boost:.0%} >= 25% into Follow Me — "
            "the redirect stealing the boost is not priced in"
        ] if boost >= 0.25 else [])(
            _mass(info, "tickle>ally") + _mass(info, "charm>ally")),
    },
]


def _move_marginals(info, moves):
    """Return strategy probability by requested move substring."""
    out = dict.fromkeys(moves, 0.0)
    for desc, p in info["strategy"]:
        for m in moves:
            if m in desc:
                out[m] += p
    return out


def _mass(info, sub):
    """Total probability of joint actions whose description contains sub."""
    return sum(p for desc, p in info["strategy"] if sub in desc)


def _slot_mass(info, slot, prefix):
    """Total probability where the given slot's action starts with prefix
    (descriptions are 'slot_a_action, slot_b_action')."""
    return sum(p for desc, p in info["strategy"]
               if desc.split(", ")[slot].startswith(prefix))


def _joint_mass(info, slot_a, prefixes_a, slot_b, prefixes_b):
    """Probability that slot_a plays one of prefixes_a AND slot_b one of
    prefixes_b — the joint-context marginal a factorized policy can't shape."""
    total = 0.0
    for desc, p in info["strategy"]:
        parts = desc.split(", ")
        if (any(parts[slot_a].startswith(x) for x in prefixes_a)
                and any(parts[slot_b].startswith(x) for x in prefixes_b)):
            total += p
    return total


def _opp_mass(info, sub):
    """Predicted-opponent prior mass on joint actions whose description
    contains ``sub`` (from the truncated top of ``info['opp_pred']``, so this
    is a lower bound — good enough for 'is this line prominent')."""
    return sum(p for desc, p in info["opp_pred"] if sub in desc)


def build_tracker(p1_sets, p2_sets, hp, fainted, cfg, weather="",
                  trickroom=False, mega=(), turn=9):
    """Return a ``LogParser`` seeded to the authored public scenario state.

    ``mega`` lists (side_id, team_idx) pairs that have already mega evolved:
    the mon's public forme flips to its stone's mega and the side's mega is
    spent, exactly what a real tracker holds after |detailschange|."""
    t = LogParser("scenario", 0, "", cfg.format_id)
    t.sides = {"p1": Side(p1_sets), "p2": Side(p2_sets)}
    t.weather = weather
    t.trickroom = trickroom
    dex = load_dex(cfg) or {"items": {}}
    for pid, k in mega:
        m = t.sides[pid].mons[k]
        stone = dex["items"].get(m.set["item"], {}).get("megaStone") or {}
        m.species_cur = stone.get(sid(m.set["species"]), m.species_cur)
        m.mega_done = True
        t.sides[pid].mega_used = True
    for pid, k in fainted:
        m = t.sides[pid].mons[k]
        m.fainted, m.hp, m.appeared = True, 0.0, True
    for side in t.sides.values():
        slot = 0
        for m in side.mons:
            if slot > 1 or m.fainted:
                continue
            # mid-game: on the field a while (no Fake Out artifacts)
            m.active_slot, m.appeared, m.turns_active = slot, True, 2
            slot += 1
    for (pid, k), frac in hp.items():
        t.sides[pid].mons[k].hp = frac
    t.turn_no = turn
    return t


def print_damage_matrix(searcher, p1_sets, p2_sets, weather="", fainted=()):
    """The scenario docs claim OHKO/2HKO patterns; print the actual numbers so
    a reviewer can see whether the position is what the assertion assumes.

    Matches the engine's turn state: scenario weather is applied, stone
    holders are calculated as their mega forme with the forme's own default
    ability (megas resolve before moves), and fainted teammates feed Supreme
    Overlord."""
    if not searcher.bridge:
        return
    from damage import request
    dex = load_dex(searcher.cfg) or {"items": {}}
    field = {"weather": weather} if weather else None
    down = {pid: sum(1 for p, _ in fainted if p == pid) for pid in ("p1", "p2")}

    def effective(s):
        """Mega-forme (default-ability) calc set for stone holders."""
        stone = dex["items"].get(s["item"], {}).get("megaStone") or {}
        mega_sp = stone.get(sid(s["species"]))
        if mega_sp:
            return {**s, "species": mega_sp, "ability": ""}
        return {**s, "species": sid(s["species"])}

    for atk_sets, dfd_sets, tag in ((p1_sets, p2_sets, "p1"),
                                    (p2_sets, p1_sets, "p2")):
        for a in atk_sets:
            ea = {**effective(a), "alliesFainted": down[tag]}
            for d in dfd_sets:
                ed = effective(d)
                cells = searcher.bridge.calc_batch(
                    [request(ea, ed, mv, field) for mv in a["moves"]])
                print(f"  {tag} {ea['species']} -> {ed['species']}: "
                      + "  ".join(
                          f"{mv} {c[0]:.0%}-{c[1]:.0%}" if c else f"{mv} ?"
                          for mv, c in zip(a["moves"], cells)))


def run_scenarios(searcher, cfg):
    """Run authored gates and return the integer failure count."""
    failures, ran = [], 0
    for scn in SCENARIOS:
        stage = scn.get("stage", "endgame")
        print(f"\n--- {scn['name']} [{stage}"
              + (", diagnostic" if scn.get("diagnostic") else "")
              + f"] ---\n{scn['doc']}")
        if scn.get("needs_model") and searcher.model is None:
            print("  SKIP (needs a trained checkpoint)")
            continue
        ran += 1
        print_damage_matrix(searcher, scn["p1"], scn["p2"],
                            weather=scn.get("weather", ""),
                            fainted=scn.get("fainted", []))
        tracker = build_tracker(scn["p1"], scn["p2"], scn["hp"],
                                scn.get("fainted", []), cfg,
                                weather=scn.get("weather", ""),
                                trickroom=scn.get("trickroom", False),
                                mega=scn.get("mega", ()),
                                turn=scn.get("turn", 9))
        belief = determinized(scn["p2"], cfg)
        joint, info = searcher.choose(tracker, belief, "p1", None,
                                      my_brought=list(range(len(scn["p1"]))),
                                      opp_brought=list(range(len(scn["p2"]))))
        print(f"value {info['value']:+.2f}"
              + (" [solve-to-terminal]" if info["solve"] else ""))
        for desc, p in info["strategy"][:6]:
            print(f"  {p:6.1%}  {desc}")
        errs = scn["check"](info)
        # diagnostic scenarios inform (underswitching / joint-context bias);
        # they never gate the suite
        tag = "NOTE" if scn.get("diagnostic") else "FAIL"
        for e in errs:
            print(f"  {tag}: {e}")
        if not errs:
            print("  PASS")
        if not scn.get("diagnostic"):
            failures += [(scn["name"], e) for e in errs]
    print(f"\n{ran - len(set(n for n, _ in failures))}/{ran} scenarios passed")
    return failures


# ---------------------------------------------------------------------------
# real-replay endgames: `--mine` scans held-out battles for 2v2-or-smaller
# turn-start positions and dumps them to artifacts/endgames.json; `--replay N`
# runs the search on one so its behavior can be documented and, once vetted,
# promoted into SCENARIOS with an assertion.
# ---------------------------------------------------------------------------

def mine(cfg, max_out=50):
    """Write up to ``max_out`` held-out endgame candidates to JSON."""
    from data import iter_battles
    out = []
    for fn in cfg.dataset_files:
        fmt = fn[len("logs_"):-len(".json")]
        for rec in iter_battles(cfg.parsed_dir / f"{fmt}.pkl"):
            if rec["split"] != "test":
                continue
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
    """Reconstruct candidate ``i`` and print one chooser decision."""
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
    """Load optional checkpoint and dispatch scenarios/mine/replay CLI modes."""
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
    searcher = DeterminizedDUCTChooser(
        model, tok, cfg, debug="--debug" in sys.argv)
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
