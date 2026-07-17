"""Structured entity features for the JEPA world model.

Converts one CTS ``PositionState`` (``LogParser._view``) plus external belief
state into the fixed 16-entity layout the model consumes (1 global + 6 ally + 6
foe mons + 2 opponent-intent + 1 CLS). One extractor serves both the live
chooser and the offline transition prep, so training and play see identical
features.

Layout constants (token order): 0 global, 1..6 my mons (preview order), 7..12
opponent mons (preview order), 13..14 opponent-intent, 15 CLS. Mon action arrays
run over the 12 mon tokens in the same order (0..5 mine, 6..11 opponent).
"""

from dataclasses import dataclass

import numpy as np

from actions import (N_MOVES, T_ALLY, T_AUTO, T_FOE_A, T_FOE_B, SlotAction,
                     from_index, joint_ok)
from beliefs import calc_stat
from data import sid

N_TOKENS = 16
MON_TOKENS = 12
N_MON = 6
ROLE_GLOBAL, ROLE_ALLY, ROLE_FOE, ROLE_INTENT, ROLE_CLS = range(5)
ROLES = np.array([ROLE_GLOBAL] + [ROLE_ALLY] * N_MON + [ROLE_FOE] * N_MON
                 + [ROLE_INTENT] * 2 + [ROLE_CLS], dtype=np.int64)

# mon categorical columns
MC_SPECIES, MC_ITEM, MC_ABILITY = 0, 1, 2
MC_MV0 = 3                                   # moves occupy 3..6
MC_STATUS, MC_TYPE0, MC_TYPE1 = 7, 8, 9
N_MON_CAT = 10

# mon scalar columns (see module design notes / JEPA_DESIGN.md)
MS_HP, MS_FAINTED, MS_ACTIVE = 0, 1, 2
MS_SLOT_A, MS_SLOT_B, MS_BENCH, MS_UNSEEN = 3, 4, 5, 6
MS_BROUGHT, MS_APPEARED, MS_TURNS = 7, 8, 9
MS_FAKEOUT, MS_CAN_PROTECT, MS_PROT_SUCC = 10, 11, 12
MS_MEGA_AVAIL, MS_MEGA_DONE, MS_ITEM_CONSUMED = 13, 14, 15
MS_BOOST0 = 16                               # 7 boosts 16..22
MS_BASE0 = 23                                # 6 base stats 23..28
MS_BP_ITEM, MS_SPE_LO, MS_SPE_HI, MS_BULK, MS_BP_NAT = 29, 30, 31, 32, 33
MS_NREV, MS_IS_ALLY = 34, 35
N_MON_SCALAR = 36

# global scalar columns
GS_TR, GS_TURN = 0, 1
GS_MY_TW, GS_MY_REF, GS_MY_LS, GS_MY_AV = 2, 3, 4, 5
GS_OPP_TW, GS_OPP_REF, GS_OPP_LS, GS_OPP_AV = 6, 7, 8, 9
GS_MY_MEGA, GS_OPP_MEGA = 10, 11
GS_GRAVITY, GS_W_DUR, GS_T_DUR, GS_TR_DUR, GS_MYTW_DUR, GS_OPPTW_DUR = range(12, 18)
N_GLOBAL_SCALAR = 18

BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


@dataclass
class Position:
    """Model tensors + action-resolution metadata for one encoded position."""

    global_cat: np.ndarray      # int64 [2]      weather, terrain
    global_scalar: np.ndarray   # float32 [18]
    mon_cat: np.ndarray         # int64 [12, 10]
    mon_scalar: np.ndarray      # float32 [12, 36]
    dmg_edge: np.ndarray        # float32 [6, 6] ally i -> foe k best avg damage
    my_active: dict             # slot -> team_idx
    opp_active: dict            # slot -> team_idx
    my_movesets: list           # 6 x list[str] move sids
    opp_movesets: list          # 6 x list[str] move sids


class FeatureExtractor:
    """Turn a CTS view + belief into a :class:`Position` (model-ready arrays)."""

    def __init__(self, vocab):
        """Bind the id/dex ``vocab`` used to encode every categorical field."""
        self.vocab = vocab

    def extract(self, view, belief_summary, brought=None, opp_brought=None,
                opp_movesets=None, dmg=None):
        """Build a :class:`Position` from a view, belief summary and optionals.

        ``opp_movesets`` (6 x up-to-4 sids) fixes each foe's assumed moves for
        this determinization; without it, revealed moves are used. ``dmg`` is a
        ``damage.damage_features`` dict (ally->foe); absent means zero edges."""
        v = self.vocab
        gcat = np.array([v.weather_id(view["weather"]),
                         v.terrain_id(view["terrain"])], dtype=np.int64)
        gscal = np.zeros(N_GLOBAL_SCALAR, dtype=np.float32)
        gscal[GS_TR] = float(view["trickroom"])
        gscal[GS_TURN] = min(view["turn"], 30) / 30.0
        myc, oppc = view["my"]["conditions"], view["opp"]["conditions"]
        gscal[GS_MY_TW], gscal[GS_MY_REF] = myc["tailwind"], myc["reflect"]
        gscal[GS_MY_LS], gscal[GS_MY_AV] = myc["lightscreen"], myc["auroraveil"]
        gscal[GS_OPP_TW], gscal[GS_OPP_REF] = oppc["tailwind"], oppc["reflect"]
        gscal[GS_OPP_LS], gscal[GS_OPP_AV] = oppc["lightscreen"], oppc["auroraveil"]
        gscal[GS_MY_MEGA] = float(view["my"]["mega_available"])
        gscal[GS_OPP_MEGA] = float(view["opp"]["mega_available"])
        # GS_GRAVITY and the *_DUR slots stay 0: the shared tracker does not
        # track gravity or condition durations (documented repo limitation).

        mon_cat = np.zeros((MON_TOKENS, N_MON_CAT), dtype=np.int64)
        mon_scalar = np.zeros((MON_TOKENS, N_MON_SCALAR), dtype=np.float32)
        my_movesets = [[] for _ in range(N_MON)]
        opp_ms = [[] for _ in range(N_MON)]
        my_active, opp_active = {}, {}

        brought = set(brought if brought is not None else range(N_MON))
        for m in view["my"]["team"]:
            k = m["team_idx"]
            ms = list(m["set"]["moves"])[:N_MOVES]
            my_movesets[k] = ms
            self._own_mon(mon_cat[k], mon_scalar[k], m, ms, k in brought)
            mon_scalar[k, MS_IS_ALLY] = 1.0
            if m["active_slot"] is not None and not m["fainted"]:
                my_active[m["active_slot"]] = k

        opp_brought = set(opp_brought) if opp_brought is not None else None
        for m in view["opp"]["team"]:
            k = m["team_idx"]
            assumed = (list(opp_movesets[k])[:N_MOVES] if opp_movesets
                       else list(m["revealed_moves"])[:N_MOVES])
            opp_ms[k] = assumed
            b = (belief_summary or {}).get(k)
            was_brought = (k in opp_brought if opp_brought is not None
                           else m["appeared"])
            self._opp_mon(mon_cat[N_MON + k], mon_scalar[N_MON + k], m, assumed,
                          b, was_brought)
            if m["active_slot"] is not None and not m["fainted"]:
                opp_active[m["active_slot"]] = k

        dmg_edge = np.zeros((N_MON, N_MON), dtype=np.float32)
        if dmg:
            for (i, _j, k), cell in dmg.items():
                if cell:
                    dmg_edge[i, k] = max(dmg_edge[i, k], (cell[0] + cell[1]) / 2)
        return Position(gcat, gscal, mon_cat, mon_scalar, dmg_edge,
                        my_active, opp_active, my_movesets, opp_ms)

    # -- per-mon feature writers --------------------------------------------
    def _common_mon(self, cat, scal, m, moves, species_cur):
        """Fill fields shared by ally and foe mons (position/boost/species)."""
        v = self.vocab
        cat[MC_SPECIES] = v.species_id(species_cur)
        for i in range(N_MOVES):
            cat[MC_MV0 + i] = v.move_id(moves[i]) if i < len(moves) else 0
        cat[MC_STATUS] = v.status_id(m["status"])
        t0, t1 = v.type_ids(species_cur)
        cat[MC_TYPE0], cat[MC_TYPE1] = t0, t1
        scal[MS_HP] = 0.0 if m["fainted"] else m["hp"]
        scal[MS_FAINTED] = float(m["fainted"])
        act = m["active_slot"]
        scal[MS_ACTIVE] = float(act is not None and not m["fainted"])
        scal[MS_SLOT_A] = float(act == 0)
        scal[MS_SLOT_B] = float(act == 1)
        scal[MS_BENCH] = float(act is None and m["appeared"] and not m["fainted"])
        scal[MS_UNSEEN] = float(not m["appeared"])
        scal[MS_APPEARED] = float(m["appeared"])
        scal[MS_TURNS] = min(m.get("turns_active", 0), 8) / 8.0
        scal[MS_FAKEOUT] = float("fakeout" in moves and m.get("turns_active", 0) == 0
                                 and act is not None and not m["fainted"])
        prot_likes = {"protect", "detect", "spikyshield", "kingsshield",
                      "banefulbunker", "silktrap", "burningbulwark", "wideguard",
                      "quickguard", "obstruct", "maxguard"}
        scal[MS_CAN_PROTECT] = float(any(mv in prot_likes for mv in moves))
        pc = min(2, m.get("protect_ct", 0))
        scal[MS_PROT_SUCC] = (1.0, 1 / 3, 1 / 9)[pc]
        scal[MS_MEGA_DONE] = float(m.get("mega_done", False))
        for i, bk in enumerate(BOOST_KEYS):
            scal[MS_BOOST0 + i] = m["boosts"].get(bk, 0) / 6.0
        for i, bs in enumerate(v.base_stats(species_cur)):
            scal[MS_BASE0 + i] = bs / 200.0

    def _own_mon(self, cat, scal, m, moves, brought):
        """Write an ally mon: full set is known, so belief scalars are exact."""
        v = self.vocab
        s = m["set"]
        self._common_mon(cat, scal, m, moves, m["species_cur"])
        cat[MC_ITEM] = 0 if m["item_consumed"] else v.item_id(s["item"])
        cat[MC_ABILITY] = v.ability_id(s["ability"])
        scal[MS_ITEM_CONSUMED] = float(m["item_consumed"])
        scal[MS_BROUGHT] = float(brought)
        scal[MS_MEGA_AVAIL] = float(v.has_mega(m["species_cur"], s["item"])
                                    and not m.get("mega_done", False))
        base_spe = v.base_stats(m["species_cur"])[5]
        evs = s.get("evs") or [0] * 6
        spe = calc_stat(int(base_spe), "spe", s["nature"], evs[5]) if base_spe else 0
        scal[MS_SPE_LO] = scal[MS_SPE_HI] = spe / 300.0
        scal[MS_BP_ITEM] = 1.0
        scal[MS_BP_NAT] = 1.0
        scal[MS_NREV] = 1.0

    def _opp_mon(self, cat, scal, m, moves, b, brought):
        """Write a foe mon: reveals where known, belief summary otherwise."""
        v = self.vocab
        self._common_mon(cat, scal, m, moves, m["species_cur"])
        if m["item_consumed"]:
            cat[MC_ITEM] = 0
        elif m["revealed_item"]:
            cat[MC_ITEM] = v.item_id(m["revealed_item"])
        elif b and b.get("item"):
            cat[MC_ITEM] = v.item_id(b["item"])
        cat[MC_ABILITY] = v.ability_id(m["revealed_ability"] or "")
        scal[MS_ITEM_CONSUMED] = float(m["item_consumed"])
        scal[MS_BROUGHT] = float(brought)
        scal[MS_MEGA_AVAIL] = 0.0        # opponent mega availability is per-side
        if b:
            scal[MS_BP_ITEM] = b.get("p_item", 0.0)
            scal[MS_SPE_LO] = min(b.get("spe_lo", 0.0), 300) / 300.0
            scal[MS_SPE_HI] = min(b.get("spe_hi", 0.0), 300) / 300.0
            scal[MS_BULK] = min(b.get("bulk", 0.0), 25000) / 25000.0
            scal[MS_BP_NAT] = b.get("p_nature", 0.0)
        scal[MS_NREV] = len(m["revealed_moves"]) / 4.0


# ---- action arrays ---------------------------------------------------------
# action-kind ids and target ids reused across the codebase (actions.py).
AK_PASS, AK_MOVE, AK_SWITCH = 0, 1, 2
# fields: kind, move_id, target, mega, switch_idx (incoming preview index+1),
# basePower, priority(+6). switch_idx 0 = not a switch.
N_ACT_FIELDS = 7


def action_arrays(pos, my_joint, opp_joint, vocab):
    """Per-mon-token action descriptors for a joint ``(mine, opp)`` action.

    ``my_joint``/``opp_joint`` are ``(slot0_idx, slot1_idx)`` slot-action index
    pairs (``actions.to_index`` space). Returns int64 ``[12, N_ACT_FIELDS]``
    where each mon token carries the action it takes (PASS for inactive mons);
    move ids are resolved through the token's own assumed moveset so identity is
    preserved regardless of slot order.
    """
    arr = np.zeros((MON_TOKENS, N_ACT_FIELDS), dtype=np.int64)
    _fill_side(arr, 0, pos.my_active, pos.my_movesets, my_joint, vocab)
    _fill_side(arr, N_MON, pos.opp_active, pos.opp_movesets, opp_joint, vocab)
    return arr


def legal_my_joints(view, max_cand=64):
    """Enumerate plausible own joint actions from a view (no sim request).

    Approximate doubles legality from public state: each active own mon's move
    slots x foe/spread targets, plus switches to live bench mons. Used to build
    behavior-cloning candidate sets during prep and, at play time, as a fallback
    when the real request's legal set is unavailable. PP/disable are unknown
    from a view, so this is a superset the policy head learns to rank."""
    team = view["my"]["team"]
    active = {m["active_slot"]: m for m in team
              if m["active_slot"] is not None and not m["fainted"]}
    bench = [m["team_idx"] for m in team
             if m["active_slot"] is None and not m["fainted"]]

    def slot_acts(m):
        """Own slot actions for one (possibly empty) active mon."""
        if m is None:
            return [SlotAction("pass")]
        acts = []
        for j in range(min(N_MOVES, len(m["set"]["moves"]))):
            for tgt in (T_FOE_A, T_FOE_B, T_AUTO):
                acts.append(SlotAction("move", move_slot=j, target=tgt))
        acts += [SlotAction("switch", switch_to=k) for k in bench]
        return acts or [SlotAction("pass")]

    s0, s1 = slot_acts(active.get(0)), slot_acts(active.get(1))
    joints = [(a, b) for a in s0 for b in s1 if joint_ok(a, b)]
    return joints[:max_cand]


PASS_JOINT = (SlotAction("pass"), SlotAction("pass"))


def my_action_arrays(pos, my_joint, vocab):
    """Action arrays with only the own side filled (opponent tokens = no-op).

    The consequence model conditions on MY move alone; the opponent's response
    is integrated into the predicted consequence, never given as an input."""
    return action_arrays(pos, my_joint, PASS_JOINT, vocab)


def _fill_side(arr, base, active, movesets, joint, vocab):
    """Place one side's two slot actions onto its active mon tokens."""
    for slot in (0, 1):
        k = active.get(slot)
        if k is None:
            continue
        a = from_index(int(joint[slot])) if not isinstance(joint[slot], SlotAction) \
            else joint[slot]
        row = arr[base + k]
        if a.kind == "move":
            moves = movesets[k]
            mv = moves[a.move_slot] if a.move_slot < len(moves) else ""
            cat, bp, prio, _ = vocab.move_meta(mv)
            row[0] = AK_MOVE
            row[1] = vocab.move_id(mv)
            row[2] = a.target
            row[3] = int(a.mega)
            row[5] = int(min(bp, 250))
            row[6] = int(prio) + 6           # shift priority into a small range
        elif a.kind == "switch":
            row[0] = AK_SWITCH
            row[4] = a.switch_to + 1         # incoming preview index (+1; 0 = none)
        # kind == "pass" leaves the zero row
