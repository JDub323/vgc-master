"""The F_t-measurable feature map φ.

Everything here is either public battle state (species, HP fractions, status,
boosts, field, faints, mega flags) or the viewer's own sheet (their moves'
types feed the matchup grid). Opponent EVs, items, and movesets are
deliberately absent: the LSMC regression must marginalize hidden information
under the path measure, and it can only do that if hidden state never enters
the basis. Keep it that way — adding a "revealed item" feature here would be
a filtration leak between training (viewer-honest paths) and play-time
scenario evaluation (where sampled opponent sheets are known to the sim).
"""

import json
from functools import lru_cache

import numpy as np

from bermuda.typechart import TYPES, best_offense
from config import CFG
from data import sid

STATUSES = ("brn", "psn", "tox", "par", "slp", "frz")
WEATHERS = ("", "sandstorm", "raindance", "sunnyday", "snowscape")
TERRAINS = ("", "electricterrain", "grassyterrain", "mistyterrain",
            "psychicterrain")
CONDS = ("tailwind", "reflect", "lightscreen", "auroraveil")
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

N_MON_FEATS = (6 + len(STATUSES) + len(BOOSTS) + 4 + len(STAT_KEYS)
               + len(TYPES))                      # 47
N_GLOBAL = (1 + len(WEATHERS) + len(TERRAINS) + 3   # turn/weather/terrain/tr/megas
            + 2 * len(CONDS) + 5)                   # side conds + faint/hp tallies
FEAT_DIM = N_GLOBAL + 12 * N_MON_FEATS + 2 * 36     # 663


@lru_cache(maxsize=4)
def _load_dex(path):
    return json.loads(open(path, encoding="utf-8").read())


def load_dex(cfg=CFG):
    """Cached artifacts/dex.json (species base stats/types, move data)."""
    return _load_dex(str(cfg.artifacts_dir / "dex.json"))


def species_entry(dex, species_name):
    """Dex row for a display species name; None when unknown to the pin."""
    return dex["species"].get(sid(species_name))


def own_offense_types(mon_set, dex):
    """Types of the viewer's own damaging moves (their sheet is theirs to
    see). Falls back to the species' own types for all-status sets."""
    out = []
    for mv in mon_set.get("moves", ()):
        row = dex["moves"].get(sid(mv))
        if row and row.get("category") != "Status" and row.get("basePower", 0):
            out.append(row["type"])
    return tuple(dict.fromkeys(out))


def _mon_block(mon, dex, own_side):
    """One mon's 47 public features (own_side only widens nothing — the
    block is identical either way; move types live in the matchup grid)."""
    f = np.zeros(N_MON_FEATS, dtype=np.float32)
    f[0] = mon.hp
    f[1] = float(mon.fainted)
    f[2] = float(mon.active_slot == 0)
    f[3] = float(mon.active_slot == 1)
    f[4] = float(mon.active_slot is None and not mon.fainted)
    f[5] = float(mon.appeared)
    o = 6
    if mon.status in STATUSES:
        f[o + STATUSES.index(mon.status)] = 1.0
    o += len(STATUSES)
    for i, k in enumerate(BOOSTS):
        f[o + i] = mon.boosts.get(k, 0) / 6.0
    o += len(BOOSTS)
    f[o] = min(mon.turns_active, 5) / 5.0
    f[o + 1] = float(mon.item_consumed)
    f[o + 2] = float(mon.mega_done)
    f[o + 3] = mon.set.get("level", 50) / 100.0
    o += 4
    entry = species_entry(dex, mon.species_cur)
    if entry:
        for i, k in enumerate(STAT_KEYS):
            f[o + i] = entry["baseStats"].get(k, 0) / 255.0
        for t in entry["types"]:
            f[o + len(STAT_KEYS) + TYPES.index(t)] = 1.0
    return f


def _types_of(mon, dex):
    entry = species_entry(dex, mon.species_cur)
    return tuple(entry["types"]) if entry else ()


def featurize(tracker, my_id, dex=None):
    """φ(public state ∪ own sheet) from ``my_id``'s viewpoint -> float32[D]."""
    dex = dex or load_dex()
    opp_id = "p2" if my_id == "p1" else "p1"
    me, opp = tracker.sides[my_id], tracker.sides[opp_id]
    f = np.zeros(FEAT_DIM, dtype=np.float32)

    f[0] = min(tracker.turn_no, 25) / 25.0
    o = 1
    f[o + (WEATHERS.index(tracker.weather)
           if tracker.weather in WEATHERS else 0)] = 1.0
    o += len(WEATHERS)
    f[o + (TERRAINS.index(tracker.terrain)
           if tracker.terrain in TERRAINS else 0)] = 1.0
    o += len(TERRAINS)
    f[o] = float(tracker.trickroom)
    f[o + 1] = float(not me.mega_used)
    f[o + 2] = float(not opp.mega_used)
    o += 3
    for side in (me, opp):
        for i, c in enumerate(CONDS):
            f[o + i] = float(side.conditions.get(c, False))
        o += len(CONDS)
    my_faint = sum(m.fainted for m in me.mons)
    opp_faint = sum(m.fainted for m in opp.mons)
    f[o] = my_faint / 4.0
    f[o + 1] = opp_faint / 4.0
    f[o + 2] = (opp_faint - my_faint) / 4.0
    f[o + 3] = sum(m.hp for m in me.mons if not m.fainted) / 6.0
    f[o + 4] = sum(m.hp for m in opp.mons if not m.fainted) / 6.0
    o += 5

    for side in (me, opp):
        own = side is me
        for k in range(6):
            if k < len(side.mons):
                f[o:o + N_MON_FEATS] = _mon_block(side.mons[k], dex, own)
            o += N_MON_FEATS

    # matchup grids: my->opp uses my sheet's damaging-move types (measurable:
    # it is my own sheet); opp->my uses the opponent's species types only.
    for i in range(6):
        for j in range(6):
            if i < len(me.mons) and j < len(opp.mons):
                a, d = me.mons[i], opp.mons[j]
                if not a.fainted and not d.fainted:
                    off = own_offense_types(a.set, dex) or _types_of(a, dex)
                    f[o + i * 6 + j] = best_offense(
                        off, _types_of(d, dex), _types_of(a, dex)) / 6.0
    o += 36
    for i in range(6):
        for j in range(6):
            if i < len(opp.mons) and j < len(me.mons):
                a, d = opp.mons[i], me.mons[j]
                if not a.fainted and not d.fainted:
                    ts = _types_of(a, dex)
                    f[o + i * 6 + j] = best_offense(
                        ts, _types_of(d, dex), ts) / 6.0
    return f
