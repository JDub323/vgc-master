"""Dataset pipeline: download HF battle logs -> parse into per-turn transitions
from both players' perspectives -> tokenize into npz shards.

The logs are Open Team Sheet games but the bot plays Closed Team Sheets, so the
input state for a perspective contains only what that player could see: their
own full team, the opponent's preview species, and whatever the opponent has
revealed (moves used, items consumed/shown, abilities triggered, HP%, field).
The opponent's `showteam` is kept ONLY as oracle labels for the auxiliary
set-prediction head and for building the belief prior from the train split.

CLI:  python data.py download | parse | prep | all
"""

import hashlib
import json
import pickle
import re
import sys
from collections import Counter, defaultdict

from actions import SlotAction, T_ALLY, T_AUTO, T_FOE_A, T_FOE_B, to_index
from config import CFG

STATUSES = {"brn", "par", "slp", "frz", "psn", "tox"}
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
SIDE_CONDS = {"tailwind": "tailwind", "reflect": "reflect",
              "lightscreen": "lightscreen", "auroraveil": "auroraveil"}
TERRAINS = {"electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"}


def sid(name: str) -> str:
    """Showdown id: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parse_packed_team(packed: str) -> list:
    """Showdown packed team -> list of set dicts (team-preview order)."""
    team = []
    for entry in packed.split("]"):
        f = entry.split("|")
        evs = [int(x) if x else 0 for x in f[6].split(",")] if f[6] else [0] * 6
        team.append({
            "name": f[0], "species": f[1] or f[0], "item": sid(f[2]),
            "ability": sid(f[3]), "moves": [sid(m) for m in f[4].split(",")],
            "nature": sid(f[5]) or "serious", "evs": evs, "gender": f[7],
            "level": int(f[10]) if len(f) > 10 and f[10] else 50,
        })
    return team


def base_species(species: str) -> str:
    """Strip a Showdown ``-Mega``, ``-Mega-X``, or ``-Mega-Y`` suffix."""
    return re.sub(r"-Mega(-[XY])?$", "", species)


def _forme_base(species: str) -> str:
    """Base-species sid of a display-form name ('Floette-Eternal' -> 'floette').
    The forme suffix follows a hyphen, so this is safe against distinct species
    that merely share a prefix ('Mewtwo' -> 'mewtwo', never 'mew'). A sid-form
    name ('floetteeternal') has no hyphen and yields itself unchanged."""
    return sid(species.split("-")[0])


# moves whose SUCCESS starts/refreshes the sim's stall counter (announced as
# |-singleturn|; verified against pinned pokemon-showdown e440c4a: the ten
# protect-likes both check and add the counter, Wide/Quick Guard add it
# without ever checking it, Mat Block checks without adding)
STALL_ADDERS = {"protect", "detect", "endure", "kingsshield", "spikyshield",
                "banefulbunker", "obstruct", "silktrap", "burningbulwark",
                "maxguard", "wideguard", "quickguard"}


class Mon:
    """Mutable public state for one team-preview-indexed Pokémon."""

    def __init__(self, team_idx, set_):
        """Initialize from ``team_idx:int`` and a full ``PokemonSet`` mapping."""
        self.team_idx = team_idx
        self.set = set_
        self.species_cur = set_["species"]
        self.hp = 1.0
        self.status = ""
        self.boosts = dict.fromkeys(BOOST_KEYS, 0)
        self.fainted = False
        self.active_slot = None   # 0/1/None
        self.appeared = False
        self.mega_done = False
        self.transformed = False
        self.turns_active = 0     # turn starts spent on the field (Fake Out legality)
        # consecutive successful stall-adder uses (Protect family + Wide/Quick
        # Guard). 0 = the next protect-like succeeds for sure; n>=1 = 1/3^n.
        # Public info for both sides. Verified against the pinned sim:
        # the stall counter triples per consecutive use, Wide/Quick Guard
        # never fail from it but DO increment it, Mat Block checks it
        # without incrementing.
        self.protect_ct = 0
        self.stall_refreshed = False   # a stall-adder succeeded this turn
        # what the opponent has seen
        self.revealed_moves = []
        self.revealed_item = None
        self.item_consumed = False
        self.revealed_ability = None

    def view_own(self):
        """Return a full-information ``MonOwnView`` mapping."""
        return {"team_idx": self.team_idx, "species_cur": self.species_cur,
                "hp": self.hp, "status": self.status, "boosts": dict(self.boosts),
                "fainted": self.fainted, "active_slot": self.active_slot,
                "appeared": self.appeared, "mega_done": self.mega_done,
                "turns_active": self.turns_active, "protect_ct": self.protect_ct,
                "item_consumed": self.item_consumed, "set": self.set}

    def view_opp(self):
        """Return a redacted CTS-safe ``MonOpponentView`` mapping."""
        return {"team_idx": self.team_idx, "species_cur": self.species_cur,
                "level": self.set["level"], "gender": self.set["gender"],
                "hp": self.hp, "status": self.status, "boosts": dict(self.boosts),
                "fainted": self.fainted, "active_slot": self.active_slot,
                "appeared": self.appeared, "mega_done": self.mega_done,
                "turns_active": self.turns_active, "protect_ct": self.protect_ct,
                "revealed_moves": list(self.revealed_moves),
                "revealed_item": self.revealed_item,
                "item_consumed": self.item_consumed,
                "revealed_ability": self.revealed_ability}


class Side:
    """Mutable team/side state and nickname-to-mon resolution."""

    def __init__(self, team):
        """Build preview-ordered ``Mon`` objects from ``list[PokemonSet]``."""
        self.mons = [Mon(i, s) for i, s in enumerate(team)]
        self.mega_used = False
        self.conditions = dict.fromkeys(SIDE_CONDS.values(), False)
        self.by_name = {m.set["name"]: m for m in self.mons}

    def mon(self, nickname, details=""):
        """Resolve protocol identity strings to a tracked ``Mon``."""
        if nickname in self.by_name:
            return self.by_name[nickname]
        # nicknames are often the bare species while details carry the forme
        # (p1b: Floette / Floette-Eternal), or vice versa — match on full sid,
        # then on the part before the forme dash (Species Clause makes the
        # base name unique within a team)
        want = sid(details.split(",")[0] if details else nickname)
        m = next((m for m in self.mons
                  if want == sid(m.set["species"]) or want == sid(m.species_cur)
                  # regional/forme base-name match. The sim emits the BASE
                  # display name ('Floette' for Floette-Eternal). species_cur is
                  # display-form ('Floette-Eternal'), so its hyphen-split base
                  # sids to 'floette' and matches; the old code split
                  # set["species"] instead, which for a belief-sampled set is
                  # sid-form ('floetteeternal', no hyphen) and never matched.
                  or want == _forme_base(m.species_cur)
                  or want == _forme_base(m.set["species"])), None)
        if m is None:
            # never crash a whole rollout/game over one unresolved ident: a
            # StopIteration here silently discarded every sample from the game
            m = self.active(0) or self.active(1) or self.mons[0]
        self.by_name[nickname] = m
        return m

    def active(self, slot):
        """Return the live/tracked ``Mon`` in slot ``0|1``, else ``None``."""
        return next((m for m in self.mons if m.active_slot == slot), None)


class LogParser:
    """One battle log -> record with both perspectives' states/actions/events."""

    def __init__(self, tag, ts, log, fmt):
        """Initialize batch or streaming parsing state for one battle."""
        self.tag, self.ts, self.log, self.fmt = tag, ts, log, fmt
        self.seen_species = set()   # formes that appeared (incl. megas), for vocab
        self.sides = {}
        self.teams = {}
        self.players, self.ratings = {}, {}
        self.weather, self.terrain, self.trickroom = "", "", False
        self.winner = None
        self.match_id = tag
        self.turns = []          # finished turn dicts
        self.events = []         # events of the turn being read
        self.turn_no = 0
        self.in_turn = False     # between |turn| and |upkeep|
        self._reset_turn_track()

    def _reset_turn_track(self):
        self.chosen = {"p1": [None, None], "p2": [None, None]}   # SlotAction/None
        self.unknown = {"p1": [False, False], "p2": [False, False]}
        self.moved = {"p1": [False, False], "p2": [False, False]}
        self.pending_mega = {"p1": [False, False], "p2": [False, False]}
        self.move_order = []
        self.last_move = None

    # -- helpers ---------------------------------------------------------
    def _pos(self, ref, details=""):
        """'p2a: Sneasler' -> (side_id, slot, Mon)"""
        head, _, nick = ref.partition(": ")
        side_id, slot = head[:2], {"a": 0, "b": 1}.get(head[2:3])
        return side_id, slot, self.sides[side_id].mon(nick, details)

    def _hp(self, s):
        """Parse a condition string to ``(hp_fraction, status_id)``."""
        cur = s.split(" ")[0]
        m = re.match(r"(\d+)(?:/(\d+))?", cur)   # '56/100y' has an hp-color suffix
        status = next((t for t in s.split(" ")[1:] if t in STATUSES), "")
        return int(m.group(1)) / int(m.group(2) or 100), status

    def _reveal(self, side_id, mon, kind, name):
        name = sid(name)
        if kind == "move" and name and name != "struggle":
            if name not in mon.revealed_moves:
                mon.revealed_moves.append(name)
        elif kind == "item" and name:
            mon.revealed_item = name
        elif kind == "ability" and name:
            mon.revealed_ability = name
        else:
            return
        self.events.append(("reveal", side_id, mon.team_idx, kind, name))

    def _reveal_from_tags(self, tags, side_id, mon):
        """[from] item:/ability: tags reveal things, attributed to [of] if given."""
        src = next((t[6:].strip() for t in tags if t.startswith("[from]")), None)
        of = next((t[4:].strip() for t in tags if t.startswith("[of]")), None)
        if not src or ":" not in src:
            return
        kind, _, name = src.partition(":")
        if kind.strip() in ("item", "ability", "move"):
            if of:
                side_id, _, mon = self._pos(of)
            self._reveal(side_id, mon, kind.strip(), name.strip())

    def _spe_ctx(self, side_id, mon):
        """Return ``{'spe':stage,'par':bool,'tw':bool}`` speed context."""
        return {"spe": mon.boosts["spe"], "par": mon.status == "par",
                "tw": self.sides[side_id].conditions["tailwind"]}

    # -- snapshots ---------------------------------------------------------
    def _view(self, p):
        """Return one side's complete CTS ``PositionState`` snapshot."""
        me, opp = self.sides[p], self.sides["p2" if p == "p1" else "p1"]
        return {
            "turn": self.turn_no, "weather": self.weather,
            "terrain": self.terrain, "trickroom": self.trickroom,
            "my": {"team": [m.view_own() for m in me.mons],
                   "mega_available": not me.mega_used,
                   "conditions": dict(me.conditions)},
            "opp": {"team": [m.view_opp() for m in opp.mons],
                    "mega_available": not opp.mega_used,
                    "conditions": dict(opp.conditions)},
        }

    def _close_turn(self):
        if self.move_order:
            self.events.append(("move_order", self.move_order, {"tr": self.trickroom}))
        actions = {}
        for p in ("p1", "p2"):
            pair = []
            for slot in (0, 1):
                a = self.chosen[p][slot]
                if a is None and not self.unknown[p][slot]:
                    a = SlotAction("pass")   # slot was empty all turn
                pair.append(None if self.unknown[p][slot] else to_index(a))
            actions[p] = None if None in pair else tuple(pair)
        self.turns[-1]["actions"] = actions
        self.turns[-1]["events"] = self.events
        self.events = []
        self._reset_turn_track()

    def _open_turn(self, n):
        if self.turns:
            self._close_turn()
        else:
            self.turns.append({"n": 0, "states": None, "actions": None,
                               "events": self.events})
            self.events = []
        self.turn_no = n
        self.in_turn = True
        for p in ("p1", "p2"):
            for m in self.sides[p].mons:
                # stall volatile survives exactly one turn without a refresh:
                # no successful protect-like last turn -> counter is gone
                if not m.stall_refreshed:
                    m.protect_ct = 0
                m.stall_refreshed = False
        for p in ("p1", "p2"):   # slots holding a live mon must produce a choice
            for slot in (0, 1):
                m = self.sides[p].active(slot)
                if m is not None and not m.fainted:
                    self.unknown[p][slot] = True
                    m.turns_active += 1
        self.turns.append({"n": n, "states": {"p1": self._view("p1"),
                                              "p2": self._view("p2")}})

    # -- main loop ---------------------------------------------------------
    def feed(self, line) -> bool:
        """One protocol line; returns True once the battle is decided.
        Streaming entry point: the live/self-play trackers construct a
        LogParser with an empty log, set .sides themselves, and feed lines as
        they arrive from the server or the sim sidecar."""
        if not line.startswith("|"):
            return False
        parts = line.split("|")
        cmd = parts[1]
        if cmd == "player" and len(parts) > 3 and parts[3]:
            self.players.setdefault(parts[2], parts[3])
            if len(parts) > 5 and parts[5].isdigit():
                self.ratings.setdefault(parts[2], int(parts[5]))
        elif cmd == "showteam":
            self.teams[parts[2]] = parse_packed_team("|".join(parts[3:]))
        elif cmd == "start":
            if not self.sides:
                if len(self.teams) < 2:
                    return True   # no team sheets -> parse() rejects the log
                self.sides = {p: Side(self.teams[p]) for p in ("p1", "p2")}
        elif cmd == "uhtml" and "bestof" in line:
            m = re.search(r"bestof\d*-[a-z0-9]+-(\d+)", line)
            if m:
                self.match_id = m.group(1)
        elif cmd == "win":
            self.winner = next((p for p, n in self.players.items()
                                if n == parts[2]), None)
            return True
        elif self.sides:
            self._event(cmd, parts, line)
        return False

    def drain_events(self):
        """Everything observed since the last drain, for live belief updates."""
        if self.move_order:
            self.events.append(("move_order", self.move_order, {"tr": self.trickroom}))
            self.move_order = []
        evs, self.events = self.events, []
        return evs

    def parse(self):
        """Parse the stored log and return a battle record mapping or ``None``."""
        for line in self.log.split("\n"):
            if self.feed(line):
                break
        if self.winner is None or not self.turns:
            return None
        self._close_turn()
        return {"tag": self.tag, "format": self.fmt, "ts": self.ts,
                "match_id": self.match_id, "players": dict(self.players),
                "ratings": {p: self.ratings.get(p) for p in ("p1", "p2")},
                "teams": self.teams, "winner": self.winner, "turns": self.turns}

    def _event(self, cmd, parts, line):
        tags = [p for p in parts[4:] if p.startswith("[")]
        if cmd == "turn":
            self._open_turn(int(parts[2]))
        elif cmd == "upkeep":
            self.in_turn = False
        elif cmd in ("switch", "drag", "replace"):
            side_id, slot, mon = self._pos(parts[2], parts[3])
            prev = self.sides[side_id].active(slot)
            if prev is not None:
                prev.active_slot = None
                prev.protect_ct = 0          # volatiles clear on switch-out
                prev.stall_refreshed = False
            voluntary = (cmd == "switch" and self.in_turn
                         and not self.moved[side_id][slot]
                         and not any(t.startswith("[from]") for t in tags)
                         and self.unknown[side_id][slot])
            if voluntary:
                self.chosen[side_id][slot] = SlotAction("switch", switch_to=mon.team_idx)
                self.unknown[side_id][slot] = False
            mon.active_slot = slot
            mon.appeared = True
            mon.hp, mon.status = self._hp(parts[4])
            mon.species_cur = parts[3].split(",")[0]
            self.seen_species.add(sid(mon.species_cur))
            mon.boosts = dict.fromkeys(BOOST_KEYS, 0)
            mon.transformed = False
        elif cmd == "swap":
            side_id, slot, mon = self._pos(parts[2])
            other = self.sides[side_id].active(1 - slot)
            mon.active_slot = 1 - slot
            if other is not None:
                other.active_slot = slot
        elif cmd == "move":
            side_id, slot, mon = self._pos(parts[2])
            move = sid(parts[3])
            from_tag = next((t for t in tags if t.startswith("[from]")), None)
            called = from_tag is not None and "lockedmove" not in from_tag
            target = next((p for p in parts[4:] if re.match(r"p[12][ab]: ", p)), None)
            if not called:
                self._reveal(side_id, mon, "move", parts[3])
                self.move_order.append((side_id, mon.team_idx, move,
                                        self._spe_ctx(side_id, mon)))
            if (not called and not from_tag and slot is not None
                    and self.unknown[side_id][slot] and not self.moved[side_id][slot]):
                if move in mon.set["moves"] and not mon.transformed:
                    tcode = T_AUTO
                    if target and not any(t.startswith("[spread]") for t in tags):
                        t_side, t_slot, _ = self._pos(target)
                        if t_side != side_id:
                            tcode = T_FOE_A if t_slot == 0 else T_FOE_B
                        elif t_slot != slot:
                            tcode = T_ALLY
                    self.chosen[side_id][slot] = SlotAction(
                        "move", move_slot=mon.set["moves"].index(move),
                        target=tcode, mega=self.pending_mega[side_id][slot])
                    self.unknown[side_id][slot] = False
            if slot is not None:
                self.moved[side_id][slot] = True
            spread = any(t.startswith("[spread]") for t in tags)
            self.last_move = {"side": side_id, "idx": mon.team_idx, "move": move,
                              "spread": spread, "crit": False, "hits": 0,
                              "burn": mon.status == "brn",
                              "boosts": dict(mon.boosts)} if not called else None
        elif cmd == "cant":
            side_id, slot, mon = self._pos(parts[2])
            if len(parts) > 4 and parts[4]:
                self._reveal(side_id, mon, "move", parts[4])
            if slot is not None:
                self.moved[side_id][slot] = True
        elif cmd == "faint":
            _, _, mon = self._pos(parts[2])
            mon.hp, mon.fainted, mon.status = 0.0, True, ""
            mon.protect_ct, mon.stall_refreshed = 0, False
        elif cmd in ("-damage", "-heal", "-sethp"):
            side_id, slot, mon = self._pos(parts[2])
            before = mon.hp
            mon.hp, mon.status = self._hp(parts[3])
            self._reveal_from_tags(tags, side_id, mon)
            lm = self.last_move
            if (cmd == "-damage" and lm and not tags and side_id != lm["side"]):
                lm["hits"] += 1
                dside = self.sides[side_id]
                self.events.append(("dmg", lm["side"], lm["idx"], lm["move"],
                                    side_id, mon.team_idx, before - mon.hp, {
                    "crit": lm["crit"], "spread": lm["spread"],
                    "multi": lm["hits"] > 1, "burn": lm["burn"],
                    "weather": self.weather, "terrain": self.terrain,
                    "atk_boosts": lm["boosts"], "def_boosts": dict(mon.boosts),
                    "screens": [c for c in ("reflect", "lightscreen", "auroraveil")
                                if dside.conditions[c]],
                    # attacker's already-fainted teammates at hit time, for
                    # Supreme Overlord (+10% atk & spa per fainted ally, cap 5) --
                    # the calc applies it exactly when passed alliesFainted
                    "allies_fainted": sum(m.fainted for m in self.sides[lm["side"]].mons),
                    "def_hp_before": before, "def_transformed": mon.transformed}))
        elif cmd == "-crit":
            if self.last_move:
                self.last_move["crit"] = True
        elif cmd == "-status":
            side_id, _, mon = self._pos(parts[2])
            mon.status = parts[3]
            self._reveal_from_tags(tags, side_id, mon)
        elif cmd == "-curestatus":
            side_id, _, mon = self._pos(parts[2])
            mon.status = ""
            self._reveal_from_tags(tags, side_id, mon)
        elif cmd in ("-boost", "-unboost", "-setboost"):
            _, _, mon = self._pos(parts[2])
            amt = int(parts[4])
            cur = mon.boosts[parts[3]]
            mon.boosts[parts[3]] = max(-6, min(6, {"-boost": cur + amt,
                                                   "-unboost": cur - amt,
                                                   "-setboost": amt}[cmd]))
        elif cmd == "-clearboost" or cmd == "-clearnegativeboost":
            _, _, mon = self._pos(parts[2])
            for k, v in mon.boosts.items():
                if cmd == "-clearboost" or v < 0:
                    mon.boosts[k] = 0
        elif cmd == "-clearallboost":
            for side in self.sides.values():
                for m in side.mons:
                    m.boosts = dict.fromkeys(BOOST_KEYS, 0)
        elif cmd == "-invertboost":
            _, _, mon = self._pos(parts[2])
            mon.boosts = {k: -v for k, v in mon.boosts.items()}
        elif cmd == "-copyboost":
            _, _, mon = self._pos(parts[2])
            _, _, src = self._pos(parts[3])
            mon.boosts = dict(src.boosts)
        elif cmd == "-swapboost":
            _, _, a = self._pos(parts[2])
            _, _, b = self._pos(parts[3])
            a.boosts, b.boosts = dict(b.boosts), dict(a.boosts)
        elif cmd == "-item":
            side_id, _, mon = self._pos(parts[2])
            self._reveal(side_id, mon, "item", parts[3])
        elif cmd == "-enditem":
            side_id, _, mon = self._pos(parts[2])
            self._reveal(side_id, mon, "item", parts[3])
            mon.item_consumed = True
            self.events.append(("consumed", side_id, mon.team_idx, sid(parts[3])))
        elif cmd == "-ability":
            side_id, _, mon = self._pos(parts[2])
            self._reveal(side_id, mon, "ability", parts[3])
        elif cmd == "-activate":
            side_id, _, mon = self._pos(parts[2])
            if len(parts) > 3 and ":" in parts[3]:
                kind, _, name = parts[3].partition(":")
                # NB: "move:" on -activate names the EFFECT, not the mon's set --
                # protect-likes all announce generic "move: Protect" when they
                # block, and Poltergeist announces "move: Poltergeist" on the
                # TARGET whose item it reveals. Treating those as move reveals
                # invented a move the true set never had and killed every
                # particle (soft depletion + a false hard constraint). The real
                # move a mon used is already revealed by its |move| line, so we
                # only take item/ability reveals here.
                if kind in ("item", "ability"):
                    self._reveal(side_id, mon, kind, name.strip())
        elif cmd == "-singleturn":
            # announced on SUCCESS of protect-likes / Wide / Quick Guard —
            # the only reliable public signal that the stall counter grew
            side_id, _, mon = self._pos(parts[2])
            if sid(parts[3].replace("move: ", "")) in STALL_ADDERS:
                mon.protect_ct += 1
                mon.stall_refreshed = True
        elif cmd == "-mega":
            side_id, slot, mon = self._pos(parts[2])
            self.sides[side_id].mega_used = True
            mon.mega_done = True
            self._reveal(side_id, mon, "item", parts[4])
            self.events.append(("mega", side_id, mon.team_idx))
            if slot is not None:
                self.pending_mega[side_id][slot] = True
        elif cmd in ("detailschange", "-formechange"):
            _, _, mon = self._pos(parts[2])
            mon.species_cur = parts[3].split(",")[0]
            self.seen_species.add(sid(mon.species_cur))
        elif cmd == "-transform":
            _, _, mon = self._pos(parts[2])
            mon.transformed = True
        elif cmd == "-weather":
            self.weather = "" if parts[2] == "none" else sid(parts[2])
            rest = [p for p in parts[3:] if p.startswith("[")]
            of = next((t[4:].strip() for t in rest if t.startswith("[of]")), None)
            if of:   # e.g. |-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: T-tar
                side_id, _, mon = self._pos(of)
                self._reveal_from_tags(rest, side_id, mon)
        elif cmd == "-fieldstart":
            what = sid(parts[2])
            if what in TERRAINS:
                self.terrain = what
            elif what == "movetrickroom":
                self.trickroom = True
        elif cmd == "-fieldend":
            what = sid(parts[2])
            if what in TERRAINS:
                self.terrain = ""
            elif what == "movetrickroom":
                self.trickroom = False
        elif cmd == "-sidestart" or cmd == "-sideend":
            side_id = parts[2][:2]
            cond = SIDE_CONDS.get(sid(parts[3].split(":")[-1]))
            if cond:
                self.sides[side_id].conditions[cond] = cmd == "-sidestart"


# ---------------------------------------------------------------------------
# pipeline steps
# ---------------------------------------------------------------------------

def download(cfg=CFG):
    from huggingface_hub import hf_hub_download
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    for fn in cfg.dataset_files:
        hf_hub_download(cfg.dataset_name, fn, repo_type="dataset",
                        local_dir=cfg.data_dir)
        print("downloaded", fn)


def split_of(match_id, cfg=CFG):
    h = int(hashlib.md5(f"{match_id}:{cfg.split_seed}".encode()).hexdigest(), 16)
    u = (h % 10_000) / 10_000
    if u < cfg.test_frac:
        return "test"
    if u < cfg.test_frac + cfg.val_frac:
        return "val"
    return "train"


def iter_battles(*paths):
    """Yield parsed records one at a time from streamed-pickle files (parse
    writes one pickle per record). Never holds a whole file in memory, so
    readers scale to the 1.15GB bo3 pickle without the OOM that
    pickle.load(whole_list) caused. Tolerates a truncated tail (a parse killed
    mid-file) by stopping at the last complete record."""
    import pickle
    for path in paths:
        with open(path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break


def parse(cfg=CFG):
    """Raw logs -> parsed battle pickles + vocab.json + usage_stats.json."""
    cfg.parsed_dir.mkdir(parents=True, exist_ok=True)
    vocab_names = {"species": set(), "move": set(), "item": set(),
                   "ability": set(), "nature": set()}
    usage = defaultdict(Counter)
    n_ok = n_bad = 0
    for fn in cfg.dataset_files:
        fmt = fn[len("logs_"):-len(".json")]
        with open(cfg.data_dir / fn, encoding="utf-8") as f:
            logs = json.load(f)
        # Stream one pickle per record straight to disk (read back the same way
        # via iter_battles). The whole-format list never lives in memory, so
        # peak RAM is one record + the shrinking raw-log dict, not the multi-GB
        # bo3 battle list that used to OOM here. vocab/usage accumulators are
        # bounded by metagame diversity, not battle count, so they stay small.
        n_fmt = 0
        with open(cfg.parsed_dir / f"{fmt}.pkl", "wb") as out:
            for tag in list(logs):
                ts, log = logs.pop(tag)   # free each raw log as we parse it
                parser = LogParser(tag, ts, log, fmt)
                try:
                    rec = parser.parse()
                except Exception:
                    rec = None
                if rec is None:
                    n_bad += 1
                    continue
                rec["split"] = split_of(rec["match_id"], cfg)
                pickle.dump(rec, out)
                n_ok += 1
                n_fmt += 1
                vocab_names["species"].update(parser.seen_species)
                for team in rec["teams"].values():
                    for s in team:
                        vocab_names["species"].add(sid(s["species"]))
                        vocab_names["item"].add(s["item"])
                        vocab_names["ability"].add(s["ability"])
                        vocab_names["nature"].add(s["nature"])
                        vocab_names["move"].update(s["moves"])
                if rec["split"] == "train":
                    for team in rec["teams"].values():
                        for s in team:
                            usage[sid(s["species"])][
                                (tuple(sorted(s["moves"])), s["item"],
                                 s["ability"], s["nature"])] += 1
        print(f"{fmt}: {n_fmt} battles parsed")
    print(f"total parsed={n_ok} skipped={n_bad}")

    with open(cfg.artifacts_dir / "vocab_names.json", "w") as f:
        json.dump({k: sorted(v - {""}) for k, v in vocab_names.items()}, f)
    stats = {sp: [[c, list(mv), it, ab, na] for (mv, it, ab, na), c
                  in sorted(cnt.items(), key=lambda x: -x[1])]
             for sp, cnt in usage.items()}
    with open(cfg.artifacts_dir / "usage_stats.json", "w") as f:
        json.dump(stats, f)
    print(f"usage stats for {len(stats)} species")


def battle_weight(rec, p, max_ts, cfg=CFG):
    r = rec["ratings"].get(p)
    rating_w = cfg.unrated_weight if r is None else min(max(r / cfg.rating_pivot, 0.5), 1.5)
    age_days = (max_ts - rec["ts"]) / 86_400
    return (cfg.format_weights.get(rec["format"], 1.0) * rating_w
            * 0.5 ** (age_days / cfg.recency_halflife_days))


def prep(cfg=CFG, resume=False):
    """Parsed battles -> tokenized npz shards, running beliefs + damage calc.

    resume: keep the shards already on disk and only regenerate what's missing.
        Battles are assigned per-split (rec["split"]) and processed in a fixed
        deterministic order, so we replay that order, cheaply counting how many
        transitions each battle would contribute to its split, and SKIP any
        battle that already lies fully within an on-disk shard for that split.
        The remaining battles (all of a split that never got flushed, e.g. val/
        test, plus the tail of train) are regenerated and appended after the
        existing shards. Note: a split resumes only on a whole-shard boundary,
        so any split with a partial final buffer (never flushed) restarts from
        scratch for that split -- which is exactly the val/test case.
    """
    import numpy as np
    from beliefs import OpponentBelief
    from damage import DamageBridge, damage_features
    from tokenizer import PositionTokenizer

    tok = PositionTokenizer.build(cfg)
    bridge = DamageBridge(cfg) if cfg.use_damage_features else None
    cfg.prepped_dir.mkdir(parents=True, exist_ok=True)
    usage = json.loads((cfg.artifacts_dir / "usage_stats.json").read_text())

    files = [cfg.parsed_dir / f"{fn[len('logs_'):-len('.json')]}.pkl"
             for fn in cfg.dataset_files]
    # pass 1: recency weighting needs the global max timestamp, and the
    # progress line needs a total. Streaming keeps memory flat (one record at a
    # time) at the cost of unpickling twice -- cheap next to the belief+damage
    # work in pass 2, and the only way to avoid holding every battle in RAM.
    max_ts = n_battles = 0
    for rec in iter_battles(*files):
        max_ts = max(max_ts, rec["ts"])
        n_battles += 1

    bufs = {s: defaultdict(list) for s in ("train", "val", "test")}
    shard_n = {s: 0 for s in bufs}

    # On resume, count how many transitions are already safely on disk per split.
    # A battle is skipped only if its split's cumulative transition count stays
    # within `done[split]` -- i.e. it lies fully inside an existing shard.
    done = {s: 0 for s in bufs}
    seen = {s: 0 for s in bufs}
    if resume:
        for f in sorted(cfg.prepped_dir.glob("*.npz")):
            split = f.stem.rsplit("_", 1)[0]
            if split in shard_n:
                done[split] += len(np.load(f)["tokens"])
                shard_n[split] += 1
        print(f"resume: on disk -> " +
              ", ".join(f"{s}: {shard_n[s]} shards / {done[s]} transitions"
                        for s in bufs))

    def flush(split, force=False):
        buf = bufs[split]
        if not buf["tokens"] or (not force and len(buf["tokens"]) < cfg.shard_size):
            return
        arrs = {"tokens": np.array(buf["tokens"], dtype=np.uint16),
                "acts": np.array(buf["acts"], dtype=np.int8),
                "value": np.array(buf["value"], dtype=np.int8),
                "weight": np.array(buf["weight"], dtype=np.float32),
                "opp_items": np.array(buf["opp_items"], dtype=np.int16),
                "opp_abils": np.array(buf["opp_abils"], dtype=np.int16),
                "opp_moves": np.array(buf["opp_moves"], dtype=np.int16),
                "dmg_active": np.array(buf["dmg_active"], dtype=np.uint8)}
        np.savez_compressed(cfg.prepped_dir / f"{split}_{shard_n[split]:03d}.npz", **arrs)
        print(f"wrote {split}_{shard_n[split]:03d}.npz ({len(buf['tokens'])} transitions)")
        shard_n[split] += 1
        bufs[split] = defaultdict(list)

    def n_transitions(rec):
        return sum(1 for p in ("p1", "p2") for turn in rec["turns"]
                   if turn["states"] is not None and turn["actions"][p] is not None)

    for bi, rec in enumerate(iter_battles(*files)):
        split = rec["split"]
        if resume:
            # Skip battles already fully captured in on-disk shards for their split.
            nt = n_transitions(rec)
            if seen[split] + nt <= done[split]:
                seen[split] += nt
                continue
        for p in ("p1", "p2"):
            opp = "p2" if p == "p1" else "p1"
            belief = OpponentBelief([sid(s["species"]) for s in rec["teams"][opp]],
                                    usage, cfg,
                                    bridge if cfg.use_belief_damage_updates else None,
                                    my_team=rec["teams"][p])
            w = battle_weight(rec, p, max_ts, cfg)
            outcome = 1 if rec["winner"] == p else -1
            oracle = rec["teams"][opp]
            for turn in rec["turns"]:
                if turn["states"] is not None and turn["actions"][p] is not None:
                    state = turn["states"][p]
                    dmg = damage_features(state, belief, bridge) if bridge else {}
                    buf = bufs[rec["split"]]
                    buf["tokens"].append(tok.encode(state, belief.summary(), dmg))
                    buf["acts"].append(turn["actions"][p])
                    buf["value"].append(outcome)
                    buf["weight"].append(w)
                    buf["opp_items"].append([tok.item_idx(s["item"]) for s in oracle])
                    buf["opp_abils"].append([tok.ability_idx(s["ability"]) for s in oracle])
                    buf["opp_moves"].append([[tok.move_idx(m) for m in s["moves"]]
                                             + [0] * (4 - len(s["moves"])) for s in oracle])
                    buf["dmg_active"].append(tok.active_dmg_grid(state, dmg))
                belief.update(turn["events"], viewer=p)
        for s in bufs:
            flush(s)
        if (bi + 1) % 1000 == 0:
            print(f"{bi + 1}/{n_battles} battles prepped")
    for s in bufs:
        flush(s, force=True)
    if bridge:
        bridge.close()


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("download", "all"):
        download()
    if step in ("parse", "all"):
        parse()
    if step in ("prep", "all"):
        prep(resume="resume" in sys.argv[2:])
