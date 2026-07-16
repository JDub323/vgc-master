"""Replica Regulation M-B teams for human-vs-bot play, plus the mined
self-play team pool.

Rosters mirror real tournament/community teams from the current Reg M-B meta
(championsmeta.io tournament results + Pikalytics usage: the Garchomp/
Whimsicott and Archaludon/Pelipper/Mega-Swampert cores, Charizard-Y balance,
sand, sun, snow, Trick Room, tailwind and Froslass hyper offense). Movesets
are standard replicas, not the original players' exact hidden spreads.

Teams are stored in Showdown export format because that is what a human
pastes into the client teambuilder; parse_export() converts to the set dicts
the rest of the repo uses.

Self-play team pool: ten replica teams are too few for self-play — the model
can memorize pairwise interactions that are artifacts of the fixed pool
(which spreads/items every Garchomp always has) rather than the metagame.
``--build-pool N`` mines the N most common real high-rated Reg M-B team
sheets from the parsed dataset (legal by construction, one per distinct
species combination), fills their redacted natures/stat points from the
Pikalytics objective prior (``spreads.json``; a base-stat heuristic when a
species is uncovered), validates each through the sim's TeamValidator, and
writes ``artifacts/selfplay_teams.json``. ``--import-pool FILE`` adds teams
from any Showdown export/backup dump (the format every teams database
exports). ``selfplay_pool()`` — used by selfplay.py and profile_selfplay.py —
returns replicas plus the pool; benchmark.py and round_robin.py keep the
fixed replica grid so ratings stay comparable.

CLI:
  python teams.py --list          # names + archetypes
  python teams.py --show NAME     # print the export text (paste into client)
  python teams.py --validate      # run every team through the sim's TeamValidator
  python teams.py --mine [N]      # top-N real teams from the parsed dataset
  python teams.py --build-pool [N]     # mine+fill+validate the self-play pool
  python teams.py --import-pool FILE   # add a Showdown export dump to the pool
  python teams.py --pool          # list the current self-play pool
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("teams.py"):
        raise SystemExit(0)

import json
import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from config import CFG
from data import sid

STATS = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}

TEAMS = {
    "rain-archaludon": ("Rain", """
Pelipper @ Focus Sash
Ability: Drizzle
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Hurricane
- Weather Ball
- Tailwind
- Protect

Archaludon @ Leftovers
Ability: Stamina
Level: 50
EVs: 32 HP / 32 SpA / 2 SpD
Modest Nature
- Electro Shot
- Draco Meteor
- Flash Cannon
- Protect

Swampert @ Swampertite
Ability: Damp
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Liquidation
- Earthquake
- Ice Punch
- Protect

Basculegion (M) @ Choice Scarf
Ability: Adaptability
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Wave Crash
- Last Respects
- Aqua Jet
- Flip Turn

Sneasler @ White Herb
Ability: Unburden
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Fake Out
- Close Combat
- Gunk Shot
- Protect

Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
EVs: 32 HP / 2 Atk / 32 SpD
Careful Nature
- Fake Out
- Throat Chop
- Flare Blitz
- Parting Shot
"""),
    "sand-hydreigon": ("Sand", """
Tyranitar @ Smooth Rock
Ability: Sand Stream
Level: 50
EVs: 32 HP / 32 Atk / 2 SpD
Adamant Nature
- Rock Slide
- Knock Off
- Low Kick
- Protect

Excadrill @ Focus Sash
Ability: Sand Rush
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- High Horsepower
- Iron Head
- Rock Slide
- Protect

Gyarados @ Gyaradosite
Ability: Intimidate
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Waterfall
- Crunch
- Taunt
- Protect

Hydreigon @ Choice Scarf
Ability: Levitate
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Modest Nature
- Draco Meteor
- Dark Pulse
- Flamethrower
- Earth Power

Sinistcha @ Kasib Berry
Ability: Hospitality
Level: 50
EVs: 32 HP / 32 Def / 2 SpA
Calm Nature
- Matcha Gotcha
- Rage Powder
- Life Dew
- Protect

Rotom-Heat @ Sitrus Berry
Ability: Levitate
Level: 50
EVs: 32 HP / 32 SpA / 2 SpD
Modest Nature
- Overheat
- Thunderbolt
- Will-O-Wisp
- Protect
"""),
    "charizard-balance": ("Balance (tournament winner core)", """
Charizard @ Charizardite Y
Ability: Solar Power
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Heat Wave
- Solar Beam
- Weather Ball
- Protect

Aerodactyl @ Focus Sash
Ability: Unnerve
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Rock Slide
- Dual Wingbeat
- Taunt
- Protect

Farigiraf @ Sitrus Berry
Ability: Armor Tail
Level: 50
EVs: 32 HP / 32 Def / 2 SpD
Bold Nature
- Trick Room
- Foul Play
- Psychic Noise
- Helping Hand

Garchomp @ Life Orb
Ability: Rough Skin
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Earthquake
- Stomping Tantrum
- Dragon Claw
- Protect

Sylveon @ Fairy Feather
Ability: Pixilate
Level: 50
EVs: 32 HP / 32 SpA / 2 SpD
Modest Nature
- Hyper Voice
- Moonblast
- Mystical Fire
- Shadow Ball

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
EVs: 32 HP / 32 Atk / 2 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Low Kick
- Iron Head
"""),
    "floette-balance": ("Balance (Garchomp/Whimsicott core)", """
Floette-Eternal @ Floettite
Ability: Flower Veil
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Moonblast
- Dazzling Gleam
- Calm Mind
- Protect

Garchomp @ Choice Scarf
Ability: Rough Skin
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Stomping Tantrum

Whimsicott @ Focus Sash
Ability: Prankster
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Moonblast
- Tailwind
- Encore
- Taunt

Sneasler @ White Herb
Ability: Unburden
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Fake Out
- Coaching
- Close Combat
- Gunk Shot

Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
EVs: 32 HP / 2 Atk / 32 SpD
Careful Nature
- Fake Out
- Throat Chop
- Flare Blitz
- Parting Shot

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
EVs: 32 HP / 32 Atk / 2 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Low Kick
- Iron Head
"""),
    "delphox-room": ("Trick Room", """
Delphox @ Delphoxite
Ability: Blaze
Level: 50
EVs: 32 HP / 32 SpA / 2 SpD
Quiet Nature
- Heat Wave
- Psychic
- Trick Room
- Protect

Floette-Eternal @ Leftovers
Ability: Flower Veil
Level: 50
EVs: 32 HP / 32 SpA / 2 SpD
Modest Nature
- Moonblast
- Dazzling Gleam
- Calm Mind
- Protect

Sinistcha @ Kasib Berry
Ability: Hospitality
Level: 50
EVs: 32 HP / 32 Def / 2 SpA
Sassy Nature
- Matcha Gotcha
- Rage Powder
- Life Dew
- Trick Room

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
EVs: 32 HP / 32 Atk / 2 Def
Brave Nature
- Kowtow Cleave
- Sucker Punch
- Low Kick
- Iron Head

Sneasler @ Focus Sash
Ability: Unburden
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Fake Out
- Close Combat
- Gunk Shot
- Protect

Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
EVs: 32 HP / 2 Atk / 32 SpD
Careful Nature
- Fake Out
- Throat Chop
- Flare Blitz
- Parting Shot
"""),
    "gholdengo-tailwind": ("Tailwind", """
Gholdengo @ Life Orb
Ability: Good as Gold
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Make It Rain
- Shadow Ball
- Nasty Plot
- Protect

Raichu @ Focus Sash
Ability: Lightning Rod
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Fake Out
- Thunderbolt
- Electroweb
- Protect

Basculegion (M) @ Choice Scarf
Ability: Adaptability
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Wave Crash
- Last Respects
- Aqua Jet
- Flip Turn

Whimsicott @ Mental Herb
Ability: Prankster
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Moonblast
- Tailwind
- Encore
- Taunt

Garchomp @ Sitrus Berry
Ability: Rough Skin
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Floette-Eternal @ Floettite
Ability: Flower Veil
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Moonblast
- Dazzling Gleam
- Calm Mind
- Protect
"""),
    "snow-gengar": ("Snow", """
Ninetales-Alola @ Light Clay
Ability: Snow Warning
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Aurora Veil
- Blizzard
- Moonblast
- Protect

Gengar @ Gengarite
Ability: Cursed Body
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Shadow Ball
- Sludge Bomb
- Icy Wind
- Protect

Snorlax @ Sitrus Berry
Ability: Thick Fat
Level: 50
EVs: 32 HP / 32 Atk / 2 Def
Brave Nature
- Body Slam
- High Horsepower
- Yawn
- Protect

Dragonite @ Dragon Fang
Ability: Multiscale
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Extreme Speed
- Ice Spinner
- Low Kick
- Protect

Scrafty @ Leftovers
Ability: Intimidate
Level: 50
EVs: 32 HP / 32 Atk / 2 SpD
Adamant Nature
- Fake Out
- Knock Off
- Drain Punch
- Ice Punch

Incineroar @ Chople Berry
Ability: Intimidate
Level: 50
EVs: 32 HP / 2 Atk / 32 SpD
Careful Nature
- Fake Out
- Throat Chop
- Flare Blitz
- Parting Shot
"""),
    "sun-venusaur": ("Sun", """
Venusaur @ Venusaurite
Ability: Chlorophyll
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Modest Nature
- Sludge Bomb
- Giga Drain
- Sleep Powder
- Protect

Ninetales @ Heat Rock
Ability: Drought
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Heat Wave
- Solar Beam
- Will-O-Wisp
- Protect

Typhlosion-Hisui @ Choice Scarf
Ability: Blaze
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Eruption
- Shadow Ball
- Heat Wave
- Infernal Parade

Dragonite @ Dragon Fang
Ability: Multiscale
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Extreme Speed
- Fire Punch
- Ice Spinner
- Aerial Ace

Gardevoir @ Gardevoirite
Ability: Trace
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Moonblast
- Psychic
- Hypnosis
- Protect

Arcanine-Hisui @ Focus Sash
Ability: Intimidate
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Rock Slide
- Flare Blitz
- Extreme Speed
- Protect
"""),
    "scizor-balance": ("Balance (Mega Scizor)", """
Scizor @ Scizorite
Ability: Technician
Level: 50
EVs: 32 HP / 32 Atk / 2 SpD
Adamant Nature
- Bullet Punch
- U-turn
- Swords Dance
- Protect

Sneasler @ White Herb
Ability: Unburden
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Fake Out
- Close Combat
- Gunk Shot
- Coaching

Garchomp @ Life Orb
Ability: Rough Skin
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Earthquake
- Stomping Tantrum
- Dragon Claw
- Protect

Sinistcha @ Kasib Berry
Ability: Hospitality
Level: 50
EVs: 32 HP / 32 Def / 2 SpA
Calm Nature
- Matcha Gotcha
- Rage Powder
- Life Dew
- Protect

Aerodactyl @ Focus Sash
Ability: Unnerve
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Rock Slide
- Dual Wingbeat
- Tailwind
- Protect

Milotic @ Leftovers
Ability: Competitive
Level: 50
EVs: 32 HP / 2 SpA / 32 SpD
Calm Nature
- Muddy Water
- Ice Beam
- Recover
- Protect
"""),
    "froslass-offense": ("Hyper offense", """
Froslass @ Froslassite
Ability: Cursed Body
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Shadow Ball
- Icy Wind
- Destiny Bond
- Protect

Basculegion (M) @ Choice Scarf
Ability: Adaptability
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Wave Crash
- Last Respects
- Flip Turn
- Aqua Jet

Sneasler @ White Herb
Ability: Unburden
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Fake Out
- Coaching
- Close Combat
- Gunk Shot

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
EVs: 32 HP / 32 Atk / 2 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Low Kick
- Iron Head

Garchomp @ Focus Sash
Ability: Rough Skin
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Floette-Eternal @ Leftovers
Ability: Flower Veil
Level: 50
EVs: 32 HP / 32 SpA / 2 SpD
Modest Nature
- Moonblast
- Dazzling Gleam
- Calm Mind
- Protect
"""),
}


def parse_export(text) -> list:
    """Showdown export text -> set dicts (the repo-wide team format)."""
    sets = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        head = lines[0]
        name, _, item = (x.strip() for x in head.partition(" @ "))
        gender = ""
        gm = re.search(r"\((M|F)\)\s*$", name)
        if gm:
            gender = gm.group(1)
            name = name[:gm.start()].strip()
        s = {"name": name, "species": name, "item": sid(item), "ability": "",
             "moves": [], "nature": "serious", "evs": [0] * 6,
             "gender": gender, "level": 50}
        for ln in lines[1:]:
            if ln.startswith("Ability:"):
                s["ability"] = sid(ln[8:])
            elif ln.startswith("Level:"):
                s["level"] = int(ln[6:])
            elif ln.startswith("EVs:"):
                for part in ln[4:].split("/"):
                    n, stat = part.split()
                    s["evs"][STATS[stat.lower()]] = int(n)
            elif ln.endswith("Nature"):
                s["nature"] = sid(ln[:-6])
            elif ln.startswith("- "):
                s["moves"].append(sid(ln[2:]))
        sets.append(s)
    return sets


def export_text(sets) -> str:
    """set dicts -> Showdown export text (for --mine output and pasting)."""
    inv = {v: k for k, v in STATS.items()}
    blocks = []
    for s in sets:
        head = s["name"] + (f" ({s['gender']})" if s["gender"] else "")
        if s["item"]:
            head += f" @ {s['item']}"
        lines = [head, f"Ability: {s['ability']}", f"Level: {s['level']}"]
        if any(s["evs"]):
            lines.append("EVs: " + " / ".join(
                f"{v} {inv[i].capitalize()}" for i, v in enumerate(s["evs"]) if v))
        lines.append(f"{s['nature'].capitalize()} Nature")
        lines += [f"- {m}" for m in s["moves"]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def get(name) -> list:
    """Parse and return a fresh full-set list for registered team ``name``."""
    return parse_export(TEAMS[name][1])


def menu() -> list:
    """Return ``[(team_name, archetype_description), ...]``."""
    return [(name, arch) for name, (arch, _) in TEAMS.items()]


def validate(cfg=CFG):
    """Every replica team through the sim's own TeamValidator (needs node)."""
    from env import Sidecar, pack_team
    sc = Sidecar(cfg)
    bad = 0
    for name in TEAMS:
        resp = sc.rpc({"op": "validate", "format": cfg.format_id,
                       "team": pack_team(get(name))})
        problems = resp.get("problems") or []
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        print(f"{name:24s} {status}")
        for p in problems:
            print(f"    {p}")
        bad += bool(problems)
    sc.close()
    print(f"\n{len(TEAMS) - bad}/{len(TEAMS)} teams valid"
          + ("" if not bad else " — fix sets or swap in --mine output"))
    sys.exit(1 if bad else 0)


def mine(n, cfg=CFG):
    """The n most common real team sheets among higher-rated dataset games,
    printed as export text — the ground truth to swap in for any replica the
    validator rejects (dataset teams are legal by construction)."""
    from data import iter_battles
    counts, samples = Counter(), {}
    for fn in cfg.dataset_files:
        fmt = fn[len("logs_"):-len(".json")]
        for rec in iter_battles(cfg.parsed_dir / f"{fmt}.pkl"):
            for p, team in rec["teams"].items():
                r = rec["ratings"].get(p)
                if r is not None and r < 1200:
                    continue
                key = tuple(sorted(
                    (s["species"], s["item"], tuple(sorted(s["moves"])),
                     s["ability"], s["nature"]) for s in team))
                counts[key] += 1
                samples[key] = team
    for key, c in counts.most_common(n):
        print(f"\n=== seen {c}x: "
              + ", ".join(sorted(s["species"] for s in samples[key])) + " ===")
        print(export_text(samples[key]))


# ---------------------------------------------------------------------------
# self-play team pool: many real teams so the model can't memorize the fixed
# replica pool's pairwise artifacts
# ---------------------------------------------------------------------------

def pool_path(cfg=CFG):
    """Return the self-play pool file ``Path``."""
    return cfg.artifacts_dir / "selfplay_teams.json"


def selfplay_pool(cfg=CFG):
    """Return {name: sets} for self-play generation: the replica teams plus
    every mined/imported pool team when ``selfplay_teams.json`` exists.
    Benchmarks and tournaments deliberately do NOT use this — their fixed
    replica grid is what keeps ratings comparable across runs."""
    out = {name: get(name) for name in TEAMS}
    p = pool_path(cfg)
    if p.exists():
        for name, entry in json.loads(p.read_text())["teams"].items():
            out[name] = entry["sets"]
    return out


def _weighted(pairs, rng):
    """Weighted choice over ``[value, weight]`` pairs."""
    vals = [p[0] for p in pairs]
    wts = [max(1e-9, float(p[1])) for p in pairs]
    return rng.choices(vals, weights=wts)[0]


def _fill_spread(s, spreads, dex, rng):
    """Assign a real stat-point spread (and nature if redacted) in place.

    Dataset team sheets carry real natures but redact stat points (SP fields
    arrive empty -> all-zero evs). Sets with explicit SP (imported teams)
    are kept as-is. Covered species draw a spread from the top of the
    Pikalytics objective prior (and a nature only when the sheet's is the
    redaction default 'serious'); uncovered species get a base-stat
    heuristic (physical/special attacker by higher attacking stat, speed
    investment unless clearly a Trick Room statline). Returns a source tag."""
    if any(s["evs"]):
        return "kept"                        # explicit SP: an imported team
    redacted_nature = s["nature"] == "serious"
    entry = spreads.get(sid(s["species"]))
    if entry:
        s["evs"] = list(_weighted(entry["spreads"][:3], rng))
        if redacted_nature:
            top_n = sorted(entry["natures"].items(), key=lambda kv: -kv[1])[:3]
            s["nature"] = _weighted([[n, w] for n, w in top_n], rng)
        return "pikalytics"
    base = dex.get("species", {}).get(sid(s["species"]), {}).get("baseStats")
    if not base:
        s["evs"] = [32, 0, 0, 0, 2, 32]
        if redacted_nature:
            s["nature"] = "timid"
        return "blind"
    physical = base["atk"] >= base["spa"]
    slow = base["spe"] <= 50                 # Trick-Room-ish statline
    atk_i = 1 if physical else 3
    s["evs"] = [0] * 6
    if slow:
        s["evs"][0], s["evs"][atk_i], s["evs"][2] = 32, 32, 2
        if redacted_nature:
            s["nature"] = "brave" if physical else "quiet"
    else:
        s["evs"][0], s["evs"][atk_i], s["evs"][5] = 2, 32, 32
        if redacted_nature:
            s["nature"] = "jolly" if physical else "timid"
    return "heuristic"


def _load_priors(cfg):
    """Return (pikalytics spread prior mons, dex) — empty dicts if unbuilt."""
    sp = cfg.artifacts_dir / "spreads.json"
    dx = cfg.artifacts_dir / "dex.json"
    spreads = json.loads(sp.read_text()).get("mons", {}) if sp.exists() else {}
    dex = json.loads(dx.read_text()) if dx.exists() else {}
    return spreads, dex


def _finish_pool_teams(candidates, cfg, seed, source, existing=None):
    """Fill spreads, validate through the sim, and return pool entries.

    candidates: iterable of (team_sets, seen_count). One team per distinct
    species combination is kept (variety beats duplicates in self-play)."""
    from env import Sidecar, pack_team
    spreads, dex = _load_priors(cfg)
    rng = random.Random(seed)
    sc = Sidecar(cfg)
    out = dict(existing or {})
    used = {frozenset(sid(s["species"]) for s in e["sets"])
            for e in out.values()}
    fills, dropped = Counter(), 0
    try:
        for team, seen in candidates:
            key = frozenset(sid(s["species"]) for s in team)
            if key in used or len(team) < 4:
                continue
            team = [dict(s, name=s["species"], moves=list(s["moves"]),
                         evs=list(s["evs"])) for s in team]
            for s in team:
                fills[_fill_spread(s, spreads, dex, rng)] += 1
            resp = sc.rpc({"op": "validate", "format": cfg.format_id,
                           "team": pack_team(team)})
            if resp.get("problems"):
                dropped += 1
                print(f"  drop {', '.join(sorted(s['species'] for s in team))}"
                      f": {resp['problems'][0]}")
                continue
            used.add(key)
            name = (f"{source}{len(out):03d}-"
                    + "-".join(sid(s["species"])[:10] for s in team[:2]))
            out[name] = {"sets": team, "seen": seen}
    finally:
        sc.close()
    print(f"  spread fill: {dict(fills)}; {dropped} dropped by the validator")
    return out


def build_pool(n=30, cfg=CFG, seed=0, min_rating=1200):
    """Mine, fill, validate, and write the self-play pool; return None.

    Scans the parsed Reg M-B dataset for the most common team sheets among
    higher-rated games (mirrors --mine), keeps one per species combination,
    and writes artifacts/selfplay_teams.json. Reproducible for a given
    (dataset, n, seed)."""
    from data import iter_battles
    counts, samples = Counter(), {}
    for fn in cfg.dataset_files:
        if "regmb" not in fn:
            continue                          # the pool is for the play format
        fmt = fn[len("logs_"):-len(".json")]
        for rec in iter_battles(cfg.parsed_dir / f"{fmt}.pkl"):
            for p, team in rec["teams"].items():
                r = rec["ratings"].get(p)
                if (r is not None and r < min_rating) or len(team) < 6:
                    continue
                key = tuple(sorted(
                    (s["species"], s["item"], tuple(sorted(s["moves"])),
                     s["ability"]) for s in team))
                counts[key] += 1
                samples.setdefault(key, team)
    print(f"{len(counts)} distinct high-rated Reg M-B sheets in the dataset")
    candidates = [(samples[k], c) for k, c in counts.most_common(n * 3)]
    entries = _finish_pool_teams(candidates, cfg, seed, "mined")   # dedupes
    entries = dict(list(entries.items())[:n])
    payload = {"_meta": {"built": date.today().isoformat(), "seed": seed,
                         "min_rating": min_rating, "format": cfg.format_id,
                         "source": "dataset mine + spreads.json fill"},
               "teams": entries}
    pool_path(cfg).write_text(json.dumps(payload, indent=1))
    print(f"{len(entries)} pool teams -> {pool_path(cfg)} "
          f"(self-play now samples {len(entries) + len(TEAMS)} teams)")


def import_pool(path, cfg=CFG, seed=0):
    """Append teams from a Showdown export/backup dump to the pool.

    Accepts the teambuilder backup format (``=== [format] name ===``
    headers) or a plain concatenation of six-mon exports — the formats any
    'good teams' database exports. Redacted spreads (no EVs/serious) are
    filled like build_pool; explicit EVs and natures are kept as-is."""
    text = Path(path).read_text()
    if not text.strip():
        print(f"{path} is empty — nothing to import")
        return
    chunks = []
    if re.search(r"^===", text, re.M):
        for block in re.split(r"^===.*===\s*$", text, flags=re.M):
            if block.strip():
                chunks.append(parse_export(block))
    else:
        mons = parse_export(text)
        chunks = [mons[i:i + 6] for i in range(0, len(mons), 6)]
    p = pool_path(cfg)
    payload = json.loads(p.read_text()) if p.exists() else \
        {"_meta": {"built": date.today().isoformat(), "seed": seed,
                   "format": cfg.format_id, "source": "import"},
         "teams": {}}
    payload["teams"] = _finish_pool_teams(
        [(t, 0) for t in chunks if len(t) >= 4], cfg, seed, "import",
        existing=payload["teams"])
    p.write_text(json.dumps(payload, indent=1))
    print(f"pool now holds {len(payload['teams'])} teams -> {p}")


def show_pool(cfg=CFG):
    """Print the current self-play pool with provenance."""
    p = pool_path(cfg)
    if not p.exists():
        print(f"no pool at {p} — build one with --build-pool N")
        return
    data = json.loads(p.read_text())
    print(f"{len(data['teams'])} pool teams ({data['_meta']})")
    for name, e in data["teams"].items():
        mons = ", ".join(sorted(s["species"] for s in e["sets"]))
        seen = f"  (seen {e['seen']}x)" if e.get("seen") else ""
        print(f"  {name:34s} {mons}{seen}")


if __name__ == "__main__":
    if "--list" in sys.argv:
        for name, arch in menu():
            print(f"{name:24s} {arch}")
    elif "--show" in sys.argv:
        print(TEAMS[sys.argv[sys.argv.index("--show") + 1]][1].strip())
    elif "--validate" in sys.argv:
        validate()
    elif "--mine" in sys.argv:
        i = sys.argv.index("--mine")
        mine(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 10)
    elif "--build-pool" in sys.argv:
        i = sys.argv.index("--build-pool")
        build_pool(int(sys.argv[i + 1]) if len(sys.argv) > i + 1
                   and sys.argv[i + 1].isdigit() else 30)
    elif "--import-pool" in sys.argv:
        import_pool(sys.argv[sys.argv.index("--import-pool") + 1])
    elif "--pool" in sys.argv:
        show_pool()
    else:
        print(__doc__)
