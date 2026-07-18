"""Type-chart softmax policy: gen-0 path measure, candidate screen, and the
play-time scenario opponent model.

Deliberately cheap (no Node calls): it runs inside every scenario at play
time and for every decision of gen-0 path collection. It only needs a
Showdown request, per-mon public info, and the dex.
"""

import math

from actions import (T_ALLY, T_AUTO, T_FOE_A, T_FOE_B, SlotAction, joint_ok,
                     legal_slot_actions)
from bermuda.typechart import best_offense, effectiveness
from data import sid

PROTECTS = {"protect", "detect", "wideguard", "quickguard", "spikyshield",
            "banefulbunker", "burningbulwark", "kingsshield", "silktrap",
            "obstruct"}
SETUP_FIELD = {"tailwind", "trickroom", "reflect", "lightscreen",
               "auroraveil", "ragepowder", "followme"}


def mon_infos(tracker, side_id, dex):
    """{team_idx: {"types","base","hp","alive"}} public info for one side."""
    from bermuda.features import species_entry
    out = {}
    for m in tracker.sides[side_id].mons:
        entry = species_entry(dex, m.species_cur) or {}
        out[m.team_idx] = {
            "types": tuple(entry.get("types", ())),
            "base": entry.get("baseStats", {}),
            "hp": 0.0 if m.fainted else m.hp,
            "alive": not m.fainted,
        }
    return out


def active_infos(tracker, side_id, dex):
    """[slot0, slot1] info dicts (or None) for one side's active mons."""
    infos = mon_infos(tracker, side_id, dex)
    out = [None, None]
    for m in tracker.sides[side_id].mons:
        if m.active_slot is not None and not m.fainted:
            out[m.active_slot] = infos[m.team_idx]
    return out


def request_infos(request, idx_of_pos, sets, dex):
    """Info dict for a side we only know through its request + set list
    (the scenario opponent: sampled sheets, live HP from the sim request)."""
    from bermuda.features import species_entry
    out = {}
    for pos, p in enumerate(request["side"]["pokemon"], start=1):
        try:
            idx = idx_of_pos(pos)
        except KeyError:
            continue
        species = sets[idx]["species"] if idx < len(sets) else ""
        entry = species_entry(dex, species) or {}
        cond = p.get("condition", "")
        if cond.endswith("fnt") or cond.startswith("0"):
            hp = 0.0
        else:
            try:
                cur, mx = cond.split(" ")[0].split("/")
                hp = int(cur) / max(1, int(mx))
            except ValueError:
                hp = 1.0
        out[idx] = {"types": tuple(entry.get("types", ())),
                    "base": entry.get("baseStats", {}),
                    "hp": hp, "alive": hp > 0}
    return out


def _matchup(info, foes):
    """Offense-minus-defense type edge of one mon against the present foes."""
    foes = [x for x in foes if x]
    if not info or not foes:
        return 0.0
    edge = 0.0
    for foe in foes:
        off = best_offense(info["types"], foe["types"], info["types"])
        threat = best_offense(foe["types"], info["types"], foe["types"])
        edge += (off - threat) / 4.0
    return edge / len(foes)


def score_action(act, slot, request, idx_of_pos, minfo, foes, dex):
    """Heuristic value of one SlotAction for the acting side."""
    if act.kind == "pass":
        return 0.0
    try:
        my_idx = idx_of_pos(slot + 1)     # actives sit at party pos 1/2
    except KeyError:
        return 0.0
    mine = minfo.get(my_idx)
    if mine is None:
        return 0.0

    if act.kind == "switch":
        cand = minfo.get(act.switch_to)
        if not cand or not cand["alive"]:
            return -1.0
        return (0.15 + 0.5 * (_matchup(cand, foes) - _matchup(mine, foes))
                + 0.2 * (1.0 - mine["hp"]))

    slot_req = (request.get("active") or [None, None])[slot]
    if not slot_req or act.move_slot >= len(slot_req["moves"]):
        return 0.0
    mid = sid(slot_req["moves"][act.move_slot].get("id")
              or slot_req["moves"][act.move_slot].get("move", ""))
    mv = dex["moves"].get(mid)
    score = 0.03 if act.mega else 0.0
    if mv is None or mv.get("category") == "Status" or not mv.get("basePower"):
        if mid in PROTECTS:
            return score + 0.25 + 0.45 * (1.0 - mine["hp"])
        if mid in SETUP_FIELD:
            return score + 0.35
        return score + 0.15

    cat = mv.get("category")
    a = mine["base"].get("atk" if cat == "Physical" else "spa", 80)
    stab = 1.5 if mv["type"] in mine["types"] else 1.0

    def hit(foe):
        d = foe["base"].get("def" if cat == "Physical" else "spd", 80)
        ratio = min(2.5, max(0.4, a / max(1, d)))
        frac = (mv["basePower"] / 100.0) * 0.33 * ratio * stab \
            * effectiveness(mv["type"], foe["types"])
        return frac + (0.35 if frac >= foe["hp"] else 0.0)

    tgt = mv.get("target", "normal")
    if act.target in (T_FOE_A, T_FOE_B):
        foe = foes[0 if act.target == T_FOE_A else 1]
        return score + (hit(foe) if foe else 0.01)
    if act.target == T_ALLY:
        return score + 0.02
    if act.target == T_AUTO and tgt in ("allAdjacentFoes", "allAdjacent"):
        return score + sum(0.75 * hit(f) for f in foes if f)
    return score + 0.15


def scored_joints(request, idx_of_pos, minfo, foes, dex):
    """Every legal joint action with its additive heuristic score."""
    acts = [legal_slot_actions(request, s, idx_of_pos) for s in (0, 1)]
    scores = [{a: score_action(a, s, request, idx_of_pos, minfo, foes, dex)
               for a in acts[s]} for s in (0, 1)]
    return [((a, b), scores[0][a] + scores[1][b])
            for a in acts[0] for b in acts[1] if joint_ok(a, b)]


def sample_joint(request, idx_of_pos, minfo, foes, dex, rng, temp,
                 uniform_mix=0.0):
    """Softmax-sample one legal joint (temp<=0: argmax); the opponent model."""
    pool = scored_joints(request, idx_of_pos, minfo, foes, dex)
    if not pool:
        return (SlotAction("pass"), SlotAction("pass"))
    if uniform_mix and rng.random() < uniform_mix:
        return rng.choice(pool)[0]
    if temp is None or temp <= 0:
        return max(pool, key=lambda js: js[1])[0]
    mx = max(s for _, s in pool)
    ws = [math.exp((s - mx) / temp) for _, s in pool]
    return rng.choices([j for j, _ in pool], weights=ws)[0]


def forced_choice(request, idx_of_pos, minfo, foes, dex, rng):
    """Matchup-greedy forced-switch choice string (replaces random_choice)."""
    picks, out = set(), []
    for slot, force in enumerate(request.get("forceSwitch") or []):
        if not force:
            out.append("pass")
            continue
        options = []
        for pos, p in enumerate(request["side"]["pokemon"], start=1):
            if p["active"] or p["condition"] == "0 fnt" or pos in picks:
                continue
            try:
                info = minfo.get(idx_of_pos(pos))
            except KeyError:
                info = None
            options.append((pos, _matchup(info, foes) if info else 0.0))
        if options:
            pos = max(options, key=lambda pv: pv[1])[0]
            picks.add(pos)
            out.append(f"switch {pos}")
        else:
            out.append("pass")
    return ", ".join(out) if out else "pass"
