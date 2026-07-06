"""Damage calc via a persistent Node subprocess wrapping @smogon/calc
(JSON lines over stdin/stdout). There is no maintained Python port.

Used forward by the tokenizer (damage% feature matrix, precomputed during data
prep so the bridge is never in the training loop) and in reverse by beliefs.py
as a likelihood function. Requests are cached by their canonical JSON, which is
what makes per-particle likelihood evaluation affordable.
"""

import json
import re
import subprocess

from config import CFG

_BRIDGE_JS = r"""
const readline = require('readline');
const {calculate, Generations, Pokemon, Move, Field} = require('@smogon/calc');
const gen = Generations.get(9);
const WEATHER = {sandstorm: 'Sand', raindance: 'Rain', sunnyday: 'Sun',
  snowscape: 'Snow', hail: 'Hail', primordialsea: 'Heavy Rain',
  desolateland: 'Harsh Sunshine', deltastream: 'Strong Winds'};
const TERRAIN = {electricterrain: 'Electric', grassyterrain: 'Grassy',
  mistyterrain: 'Misty', psychicterrain: 'Psychic'};

function mon(q) {
  const item = q.item ? gen.items.get(q.item) : undefined;
  const abil = q.ability ? gen.abilities.get(q.ability) : undefined;
  const nat = q.nature ? gen.natures.get(q.nature) : undefined;
  return new Pokemon(gen, q.species, {
    level: q.level || 50,
    item: item ? item.name : undefined,
    ability: abil ? abil.name : undefined,
    nature: nat ? nat.name : 'Serious',
    evs: q.evs ? {hp: q.evs[0], atk: q.evs[1], def: q.evs[2],
                  spa: q.evs[3], spd: q.evs[4], spe: q.evs[5]} : undefined,
    boosts: q.boosts || undefined,
    status: q.status || undefined,
  });
}

const rl = readline.createInterface({input: process.stdin, terminal: false});
rl.on('line', (line) => {
  const q = JSON.parse(line);
  let out;
  try {
    const atk = mon(q.attacker);
    const def = mon(q.defender);
    const mv = gen.moves.get(q.move);
    const move = new Move(gen, mv.name, {isCrit: !!q.crit});
    const f = q.field || {};
    const field = new Field({
      gameType: 'Doubles',
      weather: WEATHER[f.weather] || undefined,
      terrain: TERRAIN[f.terrain] || undefined,
      defenderSide: {
        isReflect: (f.screens || []).includes('reflect'),
        isLightScreen: (f.screens || []).includes('lightscreen'),
        isAuroraVeil: (f.screens || []).includes('auroraveil'),
      },
    });
    const r = calculate(gen, atk, def, move, field);
    const d = r.range();
    out = {id: q.id, min: d[0], max: d[1], maxhp: def.maxHP()};
  } catch (e) {
    out = {id: q.id, err: String(e && e.message || e)};
  }
  process.stdout.write(JSON.stringify(out) + '\n');
});
"""


class DamageBridge:
    def __init__(self, cfg=CFG):
        cfg.node_dir.mkdir(parents=True, exist_ok=True)
        js = cfg.node_dir / "dmg_bridge.js"
        js.write_text(_BRIDGE_JS)
        self.proc = subprocess.Popen(
            [cfg.node_bin, str(js.resolve())], cwd=cfg.node_dir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            encoding="utf-8", bufsize=1)
        self.cache = {}
        self.hits = self.misses = 0   # read by the search debug report

    def calc_batch(self, reqs: list) -> list:
        """reqs: canonical dicts (see request()). Returns per-request
        (min_frac, max_frac) of defender max HP, or None on calc failure.
        Duplicate requests (e.g. particles sharing item/nature) go over the
        wire once."""
        keys = [json.dumps(r, sort_keys=True) for r in reqs]
        misses = {}   # key -> representative index
        for i, key in enumerate(keys):
            if key not in self.cache:
                misses.setdefault(key, i)
        self.misses += len(misses)
        self.hits += len(keys) - len(misses)
        if len(self.cache) > 2_000_000:
            self.cache.clear()
        # write-then-read in small chunks: both stdio pipes are only a few KB
        # on Windows and node's piped stdout writes are synchronous there, so
        # flooding all requests before reading any responses deadlocks
        items = list(misses.items())
        for c in range(0, len(items), 32):
            chunk = items[c:c + 32]
            for key, i in chunk:
                self.proc.stdin.write(json.dumps({**reqs[i], "id": i}) + "\n")
            self.proc.stdin.flush()
            for _ in chunk:
                out = json.loads(self.proc.stdout.readline())
                val = None if "err" in out or not out["maxhp"] else (
                    out["min"] / out["maxhp"], out["max"] / out["maxhp"])
                self.cache[keys[out["id"]]] = val
        return [self.cache[k] for k in keys]

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def request(attacker, defender, move, field=None, crit=False) -> dict:
    """Canonical calc request. attacker/defender: dicts with species (sid),
    level, item, ability, nature, evs, boosts, status — missing keys ok."""
    def side(d):
        boosts = {k: v for k, v in (d.get("boosts") or {}).items()
                  if k in ("atk", "def", "spa", "spd", "spe") and v}
        return {"species": d["species"], "level": d.get("level", 50),
                "item": d.get("item") or "", "ability": d.get("ability") or "",
                "nature": d.get("nature") or "", "evs": d.get("evs"),
                "boosts": boosts, "status": d.get("status") or ""}
    return {"attacker": side(attacker), "defender": side(defender),
            "move": move, "field": field or {}, "crit": crit}


def damage_features(state, beliefs, bridge) -> dict:
    """(my_team_idx, move_slot, opp_team_idx) -> (min_frac, max_frac).
    Opponent unknowns come from the belief filter's highest-weight particle."""
    field = {"weather": state["weather"], "terrain": state["terrain"],
             "screens": [c for c in ("reflect", "lightscreen", "auroraveil")
                         if state["opp"]["conditions"][c]]}
    keys, reqs = [], []
    for m in state["my"]["team"]:
        if m["fainted"]:
            continue
        s = m["set"]
        atk = {"species": _sid(m["species_cur"]), "level": s["level"],
               "item": None if m["item_consumed"] else s["item"],
               "ability": s["ability"], "nature": s["nature"], "evs": s["evs"],
               "boosts": {k: v for k, v in m["boosts"].items()
                          if k in ("atk", "def", "spa", "spd", "spe") and v},
               "status": m["status"] if m["status"] == "brn" else ""}
        for o in state["opp"]["team"]:
            if o["fainted"]:
                continue
            p = beliefs.top_particle(o["team_idx"])
            dfd = {"species": _sid(o["species_cur"]),
                   "level": o["level"],
                   "item": None if o["item_consumed"] else (o["revealed_item"] or p["item"]),
                   "ability": o["revealed_ability"] or p["ability"],
                   "nature": p["nature"],
                   "boosts": {k: v for k, v in o["boosts"].items()
                              if k in ("atk", "def", "spa", "spd", "spe") and v}}
            for j, mv in enumerate(s["moves"]):
                keys.append((m["team_idx"], j, o["team_idx"]))
                reqs.append(request(atk, dfd, mv, field))
    res = bridge.calc_batch(reqs)
    return {k: v for k, v in zip(keys, res) if v is not None}


def _sid(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())
