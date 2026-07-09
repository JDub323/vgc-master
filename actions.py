"""Doubles action space.

Per-slot action index (N_SLOT_ACTIONS = 39):
  0          PASS (slot empty / nothing to choose)
  1..32      move: 1 + move_slot*8 + target*2 + mega
             move_slot in 0..3, target in TARGETS, mega in {0,1}
  33..38     switch: 33 + team_idx, where team_idx is the mon's position in the
             player's team-preview order (0..5). Identity-based, so the label is
             stable across the battle; converted to Showdown's party position
             only when building a choice string.

A joint action is a pair (slot_a_index, slot_b_index), flattened to
slot_a_index * 39 + slot_b_index when a single index is needed (the joint
policy head, evaluation, search priors). The 39x39 static mask drops combos
that are illegal in every position (double mega, both switching to the same
mon); position-specific legality always comes from the sim request via
legal_joint_actions.
"""

from dataclasses import dataclass

import numpy as np

# target codes
T_AUTO, T_FOE_A, T_FOE_B, T_ALLY = 0, 1, 2, 3  # AUTO = spread/self/field moves
N_MOVES, N_TARGETS, N_TEAM = 4, 4, 6
N_SLOT_ACTIONS = 1 + N_MOVES * N_TARGETS * 2 + N_TEAM  # 39
PASS = 0


@dataclass(frozen=True)
class SlotAction:
    kind: str            # "pass" | "move" | "switch"
    move_slot: int = 0   # 0..3
    target: int = T_AUTO
    mega: bool = False
    switch_to: int = 0   # team-preview index 0..5


def to_index(a: SlotAction) -> int:
    if a.kind == "pass":
        return PASS
    if a.kind == "move":
        return 1 + a.move_slot * 8 + a.target * 2 + int(a.mega)
    return 1 + N_MOVES * N_TARGETS * 2 + a.switch_to


def from_index(i: int) -> SlotAction:
    if i == PASS:
        return SlotAction("pass")
    if i <= N_MOVES * N_TARGETS * 2:
        i -= 1
        return SlotAction("move", move_slot=i // 8, target=(i % 8) // 2, mega=bool(i % 2))
    return SlotAction("switch", switch_to=i - 1 - N_MOVES * N_TARGETS * 2)


def joint_ok(a: SlotAction, b: SlotAction) -> bool:
    """Combos a factorized policy can propose but the game forbids."""
    if a.kind == "move" and b.kind == "move" and a.mega and b.mega:
        return False  # one mega per side per battle
    if a.kind == "switch" and b.kind == "switch" and a.switch_to == b.switch_to:
        return False
    return True


N_JOINT_ACTIONS = N_SLOT_ACTIONS * N_SLOT_ACTIONS  # 1521


def joint_index(a: SlotAction, b: SlotAction) -> int:
    return to_index(a) * N_SLOT_ACTIONS + to_index(b)


_static_mask = None


def static_joint_mask() -> np.ndarray:
    """[39, 39] bool, False where joint_ok can never hold."""
    global _static_mask
    if _static_mask is None:
        m = np.ones((N_SLOT_ACTIONS, N_SLOT_ACTIONS), dtype=bool)
        for a in range(N_SLOT_ACTIONS):
            for b in range(N_SLOT_ACTIONS):
                m[a, b] = joint_ok(from_index(a), from_index(b))
        _static_mask = m
    return _static_mask


def _choice(a: SlotAction, slot: int, party_pos_of_team_idx) -> str:
    """One slot's Showdown choice. party_pos_of_team_idx maps team-preview index
    -> current 1-based party position (from the sim request's side.pokemon order)."""
    if a.kind == "pass":
        return "pass"
    if a.kind == "switch":
        return f"switch {party_pos_of_team_idx(a.switch_to)}"
    s = f"move {a.move_slot + 1}"
    if a.target == T_FOE_A:
        s += " 1"
    elif a.target == T_FOE_B:
        s += " 2"
    elif a.target == T_ALLY:
        s += " -2" if slot == 0 else " -1"
    if a.mega:
        s += " mega"
    return s


def to_choice_string(joint, party_pos_of_team_idx) -> str:
    a, b = joint
    return f"{_choice(a, 0, party_pos_of_team_idx)}, {_choice(b, 1, party_pos_of_team_idx)}"


def legal_slot_actions(request: dict, slot: int, team_idx_of_party_pos) -> list:
    """Legal SlotActions for one slot, from a Showdown sim request JSON.
    team_idx_of_party_pos maps 1-based party position -> team-preview index."""
    active = request.get("active") or []
    side = request["side"]
    if request.get("wait"):
        return [SlotAction("pass")]
    if request.get("teamPreview") or slot >= len(active) or active[slot] is None:
        return [SlotAction("pass")]
    if side["pokemon"][slot]["condition"] == "0 fnt":
        return [SlotAction("pass")]   # fainted mon holding the slot, nothing left to send

    acts = []
    slot_req = active[slot]
    if not slot_req.get("trapped"):
        for pos, mon in enumerate(side["pokemon"], start=1):
            if mon["condition"] != "0 fnt" and not mon["active"]:
                try:
                    idx = team_idx_of_party_pos(pos)
                except KeyError:
                    continue   # party slot we can't map to a team-preview idx
                               # (forme/transform divergence in a simulated
                               # playout) — skip as a switch target, don't crash
                acts.append(SlotAction("switch", switch_to=idx))
    for mslot, mv in enumerate(slot_req["moves"]):
        if mv.get("disabled") or mv.get("pp") == 0:
            continue
        tgt = mv.get("target", "normal")
        if tgt in ("normal", "any", "adjacentFoe"):
            targets = [T_FOE_A, T_FOE_B, T_ALLY]
        elif tgt == "adjacentAlly":
            targets = [T_ALLY]
        else:  # self, allAdjacentFoes, allAdjacent, allySide, foeSide, all, ...
            targets = [T_AUTO]
        for t in targets:
            acts.append(SlotAction("move", move_slot=mslot, target=t))
            if slot_req.get("canMegaEvo"):
                acts.append(SlotAction("move", move_slot=mslot, target=t, mega=True))
    return acts or [SlotAction("pass")]


def legal_joint_actions(request: dict, team_idx_of_party_pos) -> list:
    a_acts = legal_slot_actions(request, 0, team_idx_of_party_pos)
    b_acts = legal_slot_actions(request, 1, team_idx_of_party_pos)
    return [(a, b) for a in a_acts for b in b_acts if joint_ok(a, b)]
