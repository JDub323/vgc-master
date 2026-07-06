"""PositionTokenizer: CTS-observable state -> fixed-length token sequence.

Fixed layout (one position always means the same thing, so learned positional
embeddings carry the structure; encode() asserts the layout every call):

  [0]        CLS
  [1..4]     turn bucket, weather, terrain, trick room flag
  [5..9]     my side:  mega available, tailwind, reflect, light screen, aurora veil
  [10..14]   opp side: same five
  [15..116]  my 6 mons x 17: slot, species, item, ability, hp, status,
             4 moves, 7 boosts   (team-preview order — switch action k in
             actions.py refers to the mon at block k)
  [117..218] opp 6 mons x 17: same shape, revealed info only
  [219..248] opp 6 mons x 5 belief tokens: modal item, P(that item) bucket,
             speed-range low, speed-range high, bulk (the item token reuses the
             item vocab, so this stays general — no item-specific features;
             a choice scarf shows up as the modal item AND as a stretched
             speed-range high end)
  [249..536] damage matrix: my mon i x move j x opp mon k -> (min, max) roll
             bucket pair. Two bounds fully describe the roll distribution:
             Showdown damage is a uniform pick from 16 evenly spaced
             multipliers in [0.85, 1.00], so "uniform over [lo, hi]" is exact
             up to integer rounding, no empirical check needed.

Designed to be swapped out wholesale: everything downstream only calls
encode() / vocab_size() / the aux-label index helpers.
"""

import json
import re

import numpy as np

from config import CFG

WEATHERS = ["", "sandstorm", "raindance", "sunnyday", "snowscape", "hail",
            "primordialsea", "desolateland", "deltastream"]
TERRAIN_LIST = ["", "electricterrain", "grassyterrain", "mistyterrain",
                "psychicterrain"]
STATUS_LIST = ["", "brn", "par", "slp", "frz", "psn", "tox"]
TURN_EDGES = [3, 6, 9, 12, 16, 20, 25]      # 8 buckets
MON_BLOCK = 17   # slot, species, item, ability, hp, status, 4 moves, 7 boosts
BELIEF_BLOCK = 5  # modal item, P(item), speed lo, speed hi, bulk
N_MONS = 6


class PositionTokenizer:
    def __init__(self, vocab: dict, lists: dict, cfg=CFG):
        self.vocab = vocab
        self.cfg = cfg
        self.move_list = lists["move"]
        self.item_list = lists["item"]
        self.ability_list = lists["ability"]
        self._move_map = {m: i + 1 for i, m in enumerate(self.move_list)}
        self._item_map = {m: i + 1 for i, m in enumerate(self.item_list)}
        self._abil_map = {m: i + 1 for i, m in enumerate(self.ability_list)}
        self.my_base = 15
        self.opp_base = self.my_base + N_MONS * MON_BLOCK              # 117
        self.belief_base = self.opp_base + N_MONS * MON_BLOCK          # 219
        self.dmg_base = self.belief_base + N_MONS * BELIEF_BLOCK       # 249
        self.n_tokens = self.dmg_base + N_MONS * 4 * N_MONS * 2        # 537

    # -- construction --------------------------------------------------------
    @classmethod
    def build(cls, cfg=CFG):
        """Deterministic vocab from artifacts/vocab_names.json; saves vocab.json."""
        names = json.loads((cfg.artifacts_dir / "vocab_names.json").read_text())
        toks = ["PAD", "CLS", "UNK",
                "SLOT_A", "SLOT_B", "BENCH", "FAINTED", "UNSEEN",
                "ON", "OFF", "MEGA_AVAIL", "MEGA_USED",
                "NO_MOVE", "UNK_MOVE", "NO_ITEM", "UNK_ITEM", "UNK_ABILITY",
                "UNK_SPECIES", "DMG_UNK", "DMG_OHKO", "DMG_2HKO"]
        toks += [f"TURN_{i}" for i in range(len(TURN_EDGES) + 1)]
        toks += [f"WEATHER_{w or 'none'}" for w in WEATHERS]
        toks += [f"TERRAIN_{t or 'none'}" for t in TERRAIN_LIST]
        toks += [f"ST_{s or 'none'}" for s in STATUS_LIST]
        toks += [f"HP_{i}" for i in range(cfg.n_hp_buckets + 1)]
        toks += [f"BOOST_{i}" for i in range(-6, 7)]
        toks += [f"DMG_B{i}" for i in range(cfg.n_dmg_buckets)]
        toks += [f"PROB_{i}" for i in range(cfg.n_prob_buckets)]
        toks += [f"SPD_{i}" for i in range(cfg.n_speed_buckets)]
        toks += [f"BULK_{i}" for i in range(cfg.n_bulk_buckets)]
        for ns in ("species", "move", "item", "ability", "nature"):
            toks += [f"{ns}:{n}" for n in names[ns]]
        vocab = {t: i for i, t in enumerate(toks)}
        lists = {k: names[k] for k in ("move", "item", "ability")}
        (cfg.artifacts_dir / "vocab.json").write_text(
            json.dumps({"vocab": vocab, "lists": lists}))
        return cls(vocab, lists, cfg)

    @classmethod
    def load(cls, cfg=CFG):
        d = json.loads((cfg.artifacts_dir / "vocab.json").read_text())
        return cls(d["vocab"], d["lists"], cfg)

    def vocab_size(self):
        return len(self.vocab)

    def decode(self, ids):
        """Token ids -> names; the layout is fixed, so a decoded sequence is
        a readable position (evaluate.py --worst pretty-prints it)."""
        if not hasattr(self, "_inv"):
            self._inv = {i: t for t, i in self.vocab.items()}
        return [self._inv[int(i)] for i in ids]

    # -- aux-label helpers (indices into namespace lists, 0 = unknown) -------
    def move_idx(self, m):
        return self._move_map.get(m, 0)

    def item_idx(self, it):
        return self._item_map.get(it, 0)

    def ability_idx(self, ab):
        return self._abil_map.get(ab, 0)

    def opp_species_positions(self):
        return [self.opp_base + k * MON_BLOCK + 1 for k in range(N_MONS)]

    # -- encoding -------------------------------------------------------------
    def _t(self, name):
        return self.vocab.get(name, self.vocab["UNK"])

    def _slot_tok(self, m):
        if m["fainted"]:
            return "FAINTED"
        if m["active_slot"] == 0:
            return "SLOT_A"
        if m["active_slot"] == 1:
            return "SLOT_B"
        return "BENCH" if m["appeared"] else "UNSEEN"

    def _hp_tok(self, m):
        if m["fainted"] or m["hp"] <= 0:
            return "HP_0"
        return f"HP_{max(1, min(self.cfg.n_hp_buckets, int(np.ceil(m['hp'] * self.cfg.n_hp_buckets))))}"

    def _boost_toks(self, m):
        return [f"BOOST_{m['boosts'][k]}" for k in
                ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")]

    def _dmg_toks(self, cell):
        """(min, max) roll fractions -> a token per bound. The low token says
        what is guaranteed (OHKO/2HKO markers), the high token what is
        possible; rolls are uniform between them (see module docstring)."""
        if cell is None:
            return ["DMG_UNK", "DMG_UNK"]
        return [self._dmg_bound_tok(v) for v in cell]

    def _dmg_bound_tok(self, v):
        if v >= 1.0:
            return "DMG_OHKO"
        if v >= 0.5:
            return "DMG_2HKO"
        return f"DMG_B{min(self.cfg.n_dmg_buckets - 1, int(v * self.cfg.n_dmg_buckets))}"

    def encode(self, state, belief_summary, dmg) -> np.ndarray:
        cfg = self.cfg
        out = ["CLS",
               f"TURN_{sum(state['turn'] > e for e in TURN_EDGES)}",
               f"WEATHER_{state['weather'] or 'none'}" if state["weather"] in WEATHERS
               else "UNK",
               f"TERRAIN_{state['terrain'] or 'none'}",
               "ON" if state["trickroom"] else "OFF"]
        for side in ("my", "opp"):
            s = state[side]
            out.append("MEGA_AVAIL" if s["mega_available"] else "MEGA_USED")
            out += ["ON" if s["conditions"][c] else "OFF"
                    for c in ("tailwind", "reflect", "lightscreen", "auroraveil")]

        for m in state["my"]["team"]:
            item = ("NO_ITEM" if m["item_consumed"]
                    else f"item:{m['set']['item']}" if m["set"]["item"] else "NO_ITEM")
            moves = [f"move:{mv}" for mv in m["set"]["moves"]]
            out += [self._slot_tok(m), f"species:{sid_of(m['species_cur'])}", item,
                    f"ability:{m['set']['ability']}", self._hp_tok(m),
                    f"ST_{m['status'] or 'none'}"]
            out += moves + ["NO_MOVE"] * (4 - len(moves)) + self._boost_toks(m)
        # scenario/endgame teams can be short; training teams are always 6
        out += ["PAD"] * (MON_BLOCK * (N_MONS - len(state["my"]["team"])))

        for m in state["opp"]["team"]:
            if m["item_consumed"]:
                item = "NO_ITEM"
            elif m["revealed_item"]:
                item = f"item:{m['revealed_item']}"
            else:
                item = "UNK_ITEM"
            abil = (f"ability:{m['revealed_ability']}" if m["revealed_ability"]
                    else "UNK_ABILITY")
            moves = [f"move:{mv}" for mv in m["revealed_moves"][:4]]
            out += [self._slot_tok(m), f"species:{sid_of(m['species_cur'])}", item,
                    abil, self._hp_tok(m), f"ST_{m['status'] or 'none'}"]
            out += moves + ["UNK_MOVE"] * (4 - len(moves)) + self._boost_toks(m)
        out += ["PAD"] * (MON_BLOCK * (N_MONS - len(state["opp"]["team"])))

        for k in range(N_MONS):
            b = belief_summary.get(k)
            if b is None:
                out += ["UNK"] * BELIEF_BLOCK
            else:
                out += [f"item:{b['item']}" if b["item"] else "UNK_ITEM",
                        f"PROB_{min(cfg.n_prob_buckets - 1, int(b['p_item'] * cfg.n_prob_buckets))}",
                        f"SPD_{min(cfg.n_speed_buckets - 1, int(b['spe_lo'] / cfg.speed_bucket_width))}",
                        f"SPD_{min(cfg.n_speed_buckets - 1, int(b['spe_hi'] / cfg.speed_bucket_width))}",
                        f"BULK_{min(cfg.n_bulk_buckets - 1, int(b['bulk'] / cfg.bulk_bucket_width))}"]

        for i in range(N_MONS):
            for j in range(4):
                for k in range(N_MONS):
                    out += self._dmg_toks(dmg.get((i, j, k)))

        assert len(out) == self.n_tokens, (len(out), self.n_tokens)
        return np.array([self._t(t) for t in out], dtype=np.uint16)

    def active_dmg_grid(self, state, dmg) -> np.ndarray:
        """uint8 [my_slot, move, opp_slot] avg damage% for the max-damage baseline."""
        grid = np.zeros((2, 4, 2), dtype=np.uint8)
        my_at = {m["active_slot"]: m["team_idx"] for m in state["my"]["team"]
                 if m["active_slot"] is not None and not m["fainted"]}
        opp_at = {m["active_slot"]: m["team_idx"] for m in state["opp"]["team"]
                  if m["active_slot"] is not None and not m["fainted"]}
        for s, i in my_at.items():
            for j in range(4):
                for t, k in opp_at.items():
                    cell = dmg.get((i, j, k))
                    if cell:
                        grid[s, j, t] = min(255, int((cell[0] + cell[1]) / 2 * 100))
        return grid


def sid_of(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())
