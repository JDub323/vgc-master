"""Replica Regulation M-B teams for human-vs-bot play.

Rosters mirror real tournament/community teams from the current Reg M-B meta
(championsmeta.io tournament results + Pikalytics usage: the Garchomp/
Whimsicott and Archaludon/Pelipper/Mega-Swampert cores, Charizard-Y balance,
sand, sun, snow, Trick Room, tailwind and Froslass hyper offense). Movesets
are standard replicas, not the original players' exact hidden spreads.

Teams are stored in Showdown export format because that is what a human
pastes into the client teambuilder; parse_export() converts to the set dicts
the rest of the repo uses.

CLI:
  python teams.py --list          # names + archetypes
  python teams.py --show NAME     # print the export text (paste into client)
  python teams.py --validate      # run every team through the sim's TeamValidator
  python teams.py --mine [N]      # top-N real teams from the parsed dataset
"""

import re
import sys
from collections import Counter

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


if __name__ == "__main__":
    from cli_help import show_help
    if show_help("teams.py"):
        raise SystemExit(0)
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
    else:
        print(__doc__)
