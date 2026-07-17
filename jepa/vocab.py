"""Stable id maps and dex lookups for the JEPA feature extractor.

Built from ``artifacts/dex.json`` (species base stats + types, move
type/category/priority/basePower, mega stones) and the move/item/ability name
lists inside ``artifacts/vocab.json`` — both of which travel inside an exported
bundle (``export_agent.ASSET_FILES``). The resulting maps are also serialized
into the checkpoint, so a trained agent never depends on rebuilding them.

Every categorical id reserves 0 for PAD/absent and 1 for UNK, so an unseen
species/move/item at play time is embedded as UNK rather than crashing.
"""

import json

# Fixed enums (mirror tokenizer.py / data.py so ids are reproducible).
TYPES = ["none", "normal", "fire", "water", "electric", "grass", "ice",
         "fighting", "poison", "ground", "flying", "psychic", "bug", "rock",
         "ghost", "dragon", "dark", "steel", "fairy", "stellar"]
WEATHERS = ["", "sandstorm", "raindance", "sunnyday", "snowscape", "hail",
            "primordialsea", "desolateland", "deltastream"]
TERRAINS = ["", "electricterrain", "grassyterrain", "mistyterrain",
            "psychicterrain"]
STATUSES = ["", "brn", "par", "slp", "frz", "psn", "tox"]
MOVE_CATEGORIES = ["Status", "Physical", "Special"]

PAD, UNK = 0, 1


def _idmap(names):
    """Return ``{name: id}`` with 0=PAD, 1=UNK, then the given names from 2."""
    m = {"__pad__": PAD, "__unk__": UNK}
    for n in names:
        if n not in m:
            m[n] = len(m)
    return m


class JEPAVocab:
    """Deterministic id spaces + numeric dex lookups shared by all features."""

    def __init__(self, species, moves, items, abilities, dex):
        """Build id maps from name lists and keep the raw dex for stat lookups."""
        self.species = _idmap(species)
        self.moves = _idmap(moves)
        self.items = _idmap(items)
        self.abilities = _idmap(abilities)
        self.types = {t: i for i, t in enumerate(TYPES)}
        self.weathers = {w: i for i, w in enumerate(WEATHERS)}
        self.terrains = {t: i for i, t in enumerate(TERRAINS)}
        self.statuses = {s: i for i, s in enumerate(STATUSES)}
        self.dex = dex or {"species": {}, "moves": {}, "items": {}}

    # -- construction --------------------------------------------------------
    @classmethod
    def build(cls, cfg):
        """Build from ``dex.json`` (species) and ``vocab.json`` lists (moves/…)."""
        art = cfg.artifacts_dir
        dex = _load_json(art / "dex.json") or {"species": {}, "moves": {},
                                               "items": {}}
        vjson = _load_json(art / "vocab.json") or {}
        lists = vjson.get("lists", {})
        species = sorted(dex.get("species", {}).keys())
        moves = lists.get("move") or sorted(dex.get("moves", {}).keys())
        items = lists.get("item") or sorted(dex.get("items", {}).keys())
        abilities = lists.get("ability") or []
        return cls(species, moves, items, abilities, dex)

    def state(self):
        """Serializable id-space snapshot (stored in the checkpoint)."""
        names = lambda m: [k for k, _ in sorted(m.items(), key=lambda kv: kv[1])
                           if k not in ("__pad__", "__unk__")]
        return {"species": names(self.species), "moves": names(self.moves),
                "items": names(self.items), "abilities": names(self.abilities)}

    @classmethod
    def from_state(cls, st, dex):
        """Rebuild from a checkpoint snapshot plus a dex for stat lookups."""
        return cls(st["species"], st["moves"], st["items"], st["abilities"], dex)

    # -- sizes (for embedding tables) ----------------------------------------
    def sizes(self):
        """Return the embedding-table sizes keyed by categorical field."""
        return {"species": len(self.species), "move": len(self.moves),
                "item": len(self.items), "ability": len(self.abilities),
                "type": len(TYPES), "status": len(STATUSES),
                "weather": len(WEATHERS), "terrain": len(TERRAINS)}

    # -- id lookups (UNK on miss) --------------------------------------------
    def species_id(self, s):
        """Map a species sid to its id, UNK if unseen."""
        return self.species.get(s, UNK)

    def move_id(self, m):
        """Map a move sid to its id, PAD for empty, UNK if unseen."""
        return PAD if not m else self.moves.get(m, UNK)

    def item_id(self, it):
        """Map an item sid to its id, PAD for empty, UNK if unseen."""
        return PAD if not it else self.items.get(it, UNK)

    def ability_id(self, a):
        """Map an ability sid to its id, PAD for empty, UNK if unseen."""
        return PAD if not a else self.abilities.get(a, UNK)

    def status_id(self, s):
        """Map a status id to its slot, 0 for healthy/unknown."""
        return self.statuses.get(s or "", 0)

    def weather_id(self, w):
        """Map a weather id to its slot, 0/UNK-safe."""
        return self.weathers.get(w or "", 0)

    def terrain_id(self, t):
        """Map a terrain id to its slot, 0/UNK-safe."""
        return self.terrains.get(t or "", 0)

    # -- numeric dex lookups -------------------------------------------------
    def base_stats(self, species):
        """Return ``[hp,atk,def,spa,spd,spe]`` base stats, zeros if unknown."""
        sp = self.dex.get("species", {}).get(species)
        if not sp:
            return [0.0] * 6
        bs = sp["baseStats"]
        return [float(bs[k]) for k in ("hp", "atk", "def", "spa", "spd", "spe")]

    def type_ids(self, species):
        """Return two type ids for a species (second = 'none' if monotype)."""
        sp = self.dex.get("species", {}).get(species)
        ts = (sp or {}).get("types", [])
        out = [self.types.get(t.lower(), 0) for t in ts[:2]]
        while len(out) < 2:
            out.append(0)
        return out

    def move_meta(self, m):
        """Return ``(category_id, base_power, priority, is_status)`` for a move."""
        md = self.dex.get("moves", {}).get(m)
        if not md:
            return 0, 0.0, 0.0, 1
        cat = MOVE_CATEGORIES.index(md.get("category", "Status")) \
            if md.get("category") in MOVE_CATEGORIES else 0
        return (cat, float(md.get("basePower") or 0),
                float(md.get("priority") or 0), int(md.get("category") == "Status"))

    def has_mega(self, species, item):
        """True when ``item`` is a mega stone that evolves ``species``."""
        it = self.dex.get("items", {}).get(item or "")
        stone = (it or {}).get("megaStone")
        return bool(stone and species in stone)


def _load_json(path):
    """Read a JSON file, returning ``None`` when it is absent."""
    return json.loads(path.read_text()) if path.exists() else None
