"""Battle environment, two backends behind one interface.

1. Search backend (this file's core): a Node sidecar wrapping the
   pokemon-showdown sim, exposing create / step / save-state / restore-state /
   reconstruct over JSON lines so the search can fork battle states cheaply
   and rebuild them from CTS-observable information plus sampled opponent
   sets. `python env.py --benchmark` measures steps/sec (sets the search
   budget); `python env.py --selftest` proves out reconstruction — run both
   on a new box before trusting search results.

2. Live play (phase 2): `python env.py --live` runs a poke-env player against
   a local Showdown server with the full tracker + beliefs + search stack.

Also: `python env.py --dump-dex` writes artifacts/dex.json (base stats, move
priority/category/target, mega stones) from the sim's own data, which
beliefs.py uses for stat and priority math.

The Champions formats only exist on recent pokemon-showdown git commits (see
requirements.txt). If the pinned install lacks the named format, the sidecar
falls back to gen9doublescustomgame, which allows megas and has no tera.
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("env.py"):
        raise SystemExit(0)

import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque

from config import CFG
from data import LogParser, Side, parse_packed_team, sid


def _write_atomic(path, content):
    """Write `content` to `path` atomically. Many worker processes race to
    materialize the same node script; a plain write_text can hand `node` a
    half-written file (-> SyntaxError -> instant exit -> broken pipe). Writing
    to a temp file and os.replace-ing is atomic, and identical content makes
    concurrent replaces harmless."""
    try:
        if path.exists() and path.read_text() == content:
            return
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def spawn_node(cfg, js_name, js_source):
    """Materialize `js_source` as cfg.node_dir/js_name and launch node on it,
    draining stderr on a background thread so a crash reason survives (node's
    stderr would otherwise be lost, leaving only cryptic broken pipes on the
    Python side). Returns (proc, stderr_tail) where stderr_tail() -> str."""
    cfg.node_dir.mkdir(parents=True, exist_ok=True)
    js = cfg.node_dir / js_name
    _write_atomic(js, js_source)
    proc = subprocess.Popen(
        [cfg.node_bin, str(js.resolve())], cwd=cfg.node_dir,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", bufsize=1)
    tail = deque(maxlen=50)

    def drain():
        for line in proc.stderr:
            tail.append(line)

    threading.Thread(target=drain, daemon=True).start()
    return proc, lambda: "".join(tail).strip()

_SIDECAR_JS = r"""
const sim = require('pokemon-showdown');
const {Battle, Dex, Teams} = sim;
let State = sim.State;
if (!State) {
  for (const p of ['pokemon-showdown/sim/state', 'pokemon-showdown/dist/sim/state']) {
    try { State = require(p).State; break; } catch (e) {}
  }
}

const battles = new Map();
let nextId = 1;

function resolveFormat(id) {
  const f = Dex.formats.get(id);
  if (f.exists) return {id: f.id, fallback: false};
  return {id: 'gen9doublescustomgame', fallback: true};
}

function requests(b) {
  const out = {};
  for (const side of b.sides) out[side.id] = side.activeRequest || null;
  return out;
}

function handle(q) {
  if (q.op === 'create') {
    const fmt = resolveFormat(q.format);
    const b = new Battle({formatid: fmt.id,
                          p1: {name: 'p1', team: q.p1team},
                          p2: {name: 'p2', team: q.p2team}});
    const id = nextId++;
    battles.set(id, {b, logPos: b.log.length});
    return {id, format: fmt.id, fallback: fmt.fallback,
            requests: requests(b), log: b.log};
  }
  if (q.op === 'step') {
    const e = battles.get(q.id);
    const errors = {};
    for (const [sideid, str] of Object.entries(q.choices)) {
      if (!e.b.choose(sideid, str)) {
        errors[sideid] = e.b.sides[sideid === 'p1' ? 0 : 1].choice.error || 'invalid';
      }
    }
    const log = e.b.log.slice(e.logPos);
    e.logPos = e.b.log.length;
    return {requests: requests(e.b), log, errors,
            ended: e.b.ended, winner: e.b.winner || null, turn: e.b.turn};
  }
  if (q.op === 'save') {
    return {state: State.serializeBattle(battles.get(q.id).b)};
  }
  if (q.op === 'restore') {
    const b = State.deserializeBattle(q.state);
    const id = nextId++;
    battles.set(id, {b, logPos: b.log.length});
    return {id, requests: requests(b), ended: b.ended, turn: b.turn};
  }
  if (q.op === 'destroy') {
    const e = battles.get(q.id);
    if (e) { e.b.destroy(); battles.delete(q.id); }
    return {};
  }
  if (q.op === 'reconstruct') {
    // Rebuild a mid-battle public state from scratch: fresh battle, leads
    // brought in via a real team-preview choice (Python pre-orders each team
    // actives-first), then direct engine mutations for everything visible,
    // incl. the consecutive-protect stall counter. Other volatiles (choice
    // lock, encore, subs) are NOT rebuilt — documented approximation.
    const fmt = resolveFormat(q.format);
    const b = new Battle({formatid: fmt.id,
                          p1: {name: 'p1', team: q.p1team},
                          p2: {name: 'p2', team: q.p2team}});
    for (const side of b.sides) {
      if (side.activeRequest && side.activeRequest.teamPreview) {
        const n = Math.min(side.pokemon.length, q.sides[side.id].n_brought,
                           b.ruleTable.pickedTeamSize || side.pokemon.length);
        b.choose(side.id, 'team ' + Array.from({length: n}, (_, i) => i + 1).join(''));
      }
    }
    for (const side of b.sides) {
      const spec = q.sides[side.id];
      spec.mons.forEach((ms, i) => {         // megas first: forme + stat changes
        const p = side.pokemon[i];
        if (p && ms.mega && p.canMegaEvo) b.actions.runMegaEvo(p);
      });
      if (spec.mega_used) for (const p of side.pokemon) p.canMegaEvo = false;
      spec.mons.forEach((ms, i) => {
        const p = side.pokemon[i];           // i >= brought: not in battle
        if (!p) return;
        Object.assign(p.boosts, ms.boosts || {});
        if (ms.status) p.setStatus(ms.status, null, null, true);
        if (ms.item_off && !p.clearItem()) {
          // setItem/clearItem refuse benched mons (!isActive) — mutate directly
          p.lastItem = p.item;
          p.item = '';
          p.itemState = b.initEffectState({id: '', target: p});
        }
        if (p.isActive) {                    // Fake Out / First Impression legality
          p.activeTurns = ms.turns || 1;
          p.activeMoveActions = Math.max(0, (ms.turns || 1) - 1);
        }
        if (p.isActive && ms.stall) {
          // consecutive-protect state: counter 3^n = the denominator of the
          // next protect-like's success chance; duration 1 = expires after
          // this turn unless refreshed, matching a live battle at turn start
          p.addVolatile('stall');
          if (p.volatiles['stall']) {
            p.volatiles['stall'].counter = Math.pow(3, ms.stall);
            p.volatiles['stall'].duration = 1;
          }
        }
        if (ms.fainted) p.faint();
        else if (ms.hp < 1) p.sethp(Math.round(ms.hp * p.maxhp));
      });
      for (const c of spec.conditions || []) side.addSideCondition(c, 'debug');
    }
    b.faintMessages();
    const f = q.field || {};
    if (f.weather) b.field.setWeather(f.weather, 'debug');
    if (f.terrain) b.field.setTerrain(f.terrain, 'debug');
    if (f.trickroom) b.field.addPseudoWeather('trickroom', 'debug');
    if (f.turn) b.turn = f.turn;
    b.clearRequest();
    b.makeRequest('move');
    const id = nextId++;
    battles.set(id, {b, logPos: b.log.length});
    return {id, format: fmt.id, fallback: fmt.fallback,
            requests: requests(b), ended: b.ended, turn: b.turn, log: b.log.slice()};
  }
  if (q.op === 'validate') {
    let TV = sim.TeamValidator;
    if (!TV) {
      for (const p of ['pokemon-showdown/sim/team-validator',
                       'pokemon-showdown/dist/sim/team-validator']) {
        try { TV = require(p).TeamValidator; break; } catch (e) {}
      }
    }
    const fmt = resolveFormat(q.format);
    const problems = new TV(fmt.id).validateTeam(Teams.unpack(q.team));
    return {format: fmt.id, fallback: fmt.fallback, problems: problems || []};
  }
  if (q.op === 'dumpdex') {
    const fmt = resolveFormat(q.format);
    const d = Dex.forFormat(Dex.formats.get(fmt.id));
    const species = {}, moves = {}, items = {};
    for (const s of d.species.all()) {
      if (s.exists) species[s.id] = {baseStats: s.baseStats, types: s.types};
    }
    for (const m of d.moves.all()) {
      if (m.exists) moves[m.id] = {priority: m.priority, category: m.category,
        basePower: m.basePower, type: m.type, target: m.target,
        multihit: !!m.multihit};
    }
    for (const it of d.items.all()) {
      if (!it.exists) continue;
      let stone = null;
      if (it.megaStone && typeof it.megaStone === 'string') {   // gen 6/7 form
        stone = {[Dex.toID(it.megaEvolves || '')]: Dex.toID(it.megaStone)};
      } else if (it.megaStone) {                                // champions map form
        stone = {};
        for (const [k, v] of Object.entries(it.megaStone)) stone[Dex.toID(k)] = Dex.toID(v);
      }
      items[it.id] = {megaStone: stone};
    }
    return {format: fmt.id, fallback: fmt.fallback,
            species, moves, items};
  }
  return {err: 'unknown op ' + q.op};
}

const rl = require('readline').createInterface({input: process.stdin, terminal: false});
rl.on('line', (line) => {
  const q = JSON.parse(line);
  let out;
  try { out = handle(q); } catch (e) { out = {err: String(e && e.stack || e)}; }
  out.rid = q.rid;
  process.stdout.write(JSON.stringify(out) + '\n');
});
"""


class Sidecar:
    """Owner of one JSON-lines Pokémon Showdown Node subprocess."""

    def __init__(self, cfg=CFG):
        """Start the sidecar using Node paths and format fallback from ``cfg``."""
        self.proc, self._stderr_tail = spawn_node(cfg, "sidecar.js", _SIDECAR_JS)
        self._rid = 0

    def rpc(self, obj):
        """Send one JSON-safe mapping and return its decoded response mapping."""
        self._rid += 1
        try:
            self.proc.stdin.write(json.dumps({**obj, "rid": self._rid}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise RuntimeError(
                f"sidecar died before op {obj.get('op')!r} "
                f"(code {self.proc.poll()}); stderr:\n{self._stderr_tail()}")
        line = self.proc.stdout.readline()
        if not line:                       # node exited: stdout closed
            raise RuntimeError(
                f"sidecar exited on op {obj.get('op')!r} "
                f"(code {self.proc.poll()}); stderr:\n{self._stderr_tail()}")
        out = json.loads(line)
        assert out.get("rid") == self._rid and "err" not in out, out.get("err", out)
        return out

    def close(self):
        """Close stdin and wait for the owned Node subprocess; return ``None``."""
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass                           # already dead; don't mask the real error
        self.proc.wait()


class SidecarBattle:
    """One forkable battle in the sidecar."""

    def __init__(self, sidecar, resp):
        """Wrap a sidecar create/restore response and cache battle state fields."""
        self.sc = sidecar
        self.id = resp["id"]
        self.requests = resp["requests"]
        self.log = resp.get("log", [])
        self.ended = resp.get("ended", False)
        self.winner = None
        self.turn = resp.get("turn", 0)

    @classmethod
    def create(cls, sidecar, format_id, p1team, p2team):
        """Create and return a battle from format id and two packed-team strings."""
        resp = sidecar.rpc({"op": "create", "format": format_id,
                            "p1team": p1team, "p2team": p2team})
        b = cls(sidecar, resp)
        b.format, b.fallback = resp["format"], resp["fallback"]
        return b

    def step(self, choices: dict):
        """choices: {'p1': 'move 1 2, switch 3', ...} for sides that must act."""
        resp = self.sc.rpc({"op": "step", "id": self.id, "choices": choices})
        self.requests = resp["requests"]
        self.log = resp["log"]
        self.ended = resp["ended"]
        self.winner = resp["winner"]
        self.turn = resp["turn"]
        return resp

    def save(self):
        """Return an opaque JSON-serializable simulator snapshot."""
        return self.sc.rpc({"op": "save", "id": self.id})["state"]

    @classmethod
    def restore(cls, sidecar, state):
        """Create an independent battle fork from a saved simulator state."""
        return cls(sidecar, sidecar.rpc({"op": "restore", "state": state}))

    def destroy(self):
        """Free this battle id inside the sidecar; return ``None``."""
        self.sc.rpc({"op": "destroy", "id": self.id})

    def pending_sides(self):
        """Return side ids whose non-wait requests need choices."""
        return [s for s, r in self.requests.items() if r and not r.get("wait")]


def pack_team(team) -> str:
    """Our set dicts (data.parse_packed_team output) -> Showdown packed team."""
    return "]".join(
        f"{s['name']}|{s['species'] if s['species'] != s['name'] else ''}|"
        f"{s['item']}|{s['ability']}|{','.join(s['moves'])}|{s['nature']}|"
        f"{','.join(str(e) for e in s['evs']) if any(s['evs']) else ''}|"
        f"{s['gender']}|||{s['level']}|" for s in team)


def full_set(s) -> dict:
    """Normalize a belief-sampled set (species/moves/item/ability/nature only)
    into the full set-dict shape pack_team and Side expect."""
    return {"name": s.get("name", s["species"]), "species": s["species"],
            "item": s["item"], "ability": s["ability"], "moves": list(s["moves"]),
            "nature": s.get("nature", "serious"), "evs": s.get("evs", [0] * 6),
            "gender": s.get("gender", ""), "level": s.get("level", 50)}


def reconstruct(sidecar, format_id, tracker, teams, brought):
    """Rebuild the tracker's public battle state as a fresh sidecar battle,
    with `teams` as ground truth for hidden information.

    teams / brought: {'p1': ..., 'p2': ...}; teams are set dicts in
    team-preview order (the opponent's sampled from the belief filter),
    brought lists the team-preview indices actually brought. Each side is
    packed actives-first so a plain `team 1..n` preview choice puts the right
    mons in the right slots; a fainted 'filler' lead keeps a lone survivor in
    its real slot. Returns (battle, orders) with orders[side][party_pos] =
    team-preview index.

    Rebuilt exactly: species/formes, HP%, status, boosts, fainted, consumed
    items, used megas, side conditions, weather/terrain/trick room, turns on
    the field, consecutive-protect (stall) counters. NOT rebuilt: choice
    lock, encore/taunt/sub volatiles, PP spent, exact residual durations.
    """
    q = {"op": "reconstruct", "format": format_id, "sides": {},
         "field": {"weather": tracker.weather, "terrain": tracker.terrain,
                   "trickroom": tracker.trickroom, "turn": tracker.turn_no}}
    orders = {}
    for side_id in ("p1", "p2"):
        side = tracker.sides[side_id]
        ms = side.mons
        alive_at = {m.active_slot: m.team_idx for m in ms
                    if m.active_slot is not None and not m.fainted}
        order = []
        for slot in (0, 1):
            if slot in alive_at:
                order.append(alive_at[slot])
            else:
                filler = next((k for k in brought[side_id]
                               if ms[k].fainted and k not in order), None)
                if filler is not None:
                    order.append(filler)
        order += sorted((k for k in brought[side_id] if k not in order),
                        key=lambda k: ms[k].fainted)      # live bench first
        order += [k for k in range(len(ms)) if k not in order]
        q["sides"][side_id] = {
            "n_brought": len(brought[side_id]), "mega_used": side.mega_used,
            "conditions": [c for c, v in side.conditions.items() if v],
            "mons": [{"hp": ms[k].hp, "status": ms[k].status,
                      "fainted": ms[k].fainted, "boosts": ms[k].boosts,
                      "item_off": ms[k].item_consumed, "mega": ms[k].mega_done,
                      "turns": ms[k].turns_active,
                      "stall": getattr(ms[k], "protect_ct", 0)} for k in order]}
        q[f"{side_id}team"] = pack_team([full_set(teams[side_id][k]) for k in order])
        orders[side_id] = order
    resp = sidecar.rpc(q)
    b = SidecarBattle(sidecar, resp)
    b.format, b.fallback = resp["format"], resp["fallback"]
    return b, orders


def random_choice(request, rng=random) -> str:
    """A uniformly random legal-ish choice for one side's request."""
    if request.get("teamPreview"):
        n = request.get("maxChosenTeamSize") or 4
        order = rng.sample(range(1, len(request["side"]["pokemon"]) + 1), n)
        return "team " + "".join(str(i) for i in order)
    if request.get("forceSwitch"):
        picks, out = set(), []
        for slot, force in enumerate(request["forceSwitch"]):
            if not force:
                out.append("pass")
                continue
            options = [i for i, p in enumerate(request["side"]["pokemon"], 1)
                       if not p["active"] and p["condition"] != "0 fnt"
                       and i not in picks]
            if options:
                pick = rng.choice(options)
                picks.add(pick)
                out.append(f"switch {pick}")
            else:
                out.append("pass")
        return ", ".join(out)
    out = []
    for slot, act in enumerate(request.get("active") or []):
        if act is None:
            out.append("pass")
            continue
        moves = [(i, m) for i, m in enumerate(act["moves"], 1)
                 if not m.get("disabled") and m.get("pp", 1) != 0]
        if not moves:
            out.append("move 1")
            continue
        i, m = rng.choice(moves)
        c = f"move {i}"
        tgt = m.get("target", "normal")
        if tgt in ("normal", "any", "adjacentFoe"):
            c += f" {rng.choice([1, 2])}"
        elif tgt == "adjacentAlly":
            c += " -2" if slot == 0 else " -1"
        if act.get("canMegaEvo") and rng.random() < 0.5:
            c += " mega"
        out.append(c)
    return ", ".join(out)


def _step_random(battle, rng):
    """One random step; falls back to 'default' for choices the sim rejects."""
    choices = {s: random_choice(battle.requests[s], rng)
               for s in battle.pending_sides()}
    resp = battle.step(choices)
    if resp["errors"]:
        battle.step({s: "default" for s in resp["errors"]})


# two Reg M-B teams from the dataset, used by the benchmark
TEAM_A = ("Blaziken||focussash|speedboost|heatwave,aurasphere,coaching,protect|Timid||F|||50|]"
          "Metagross||metagrossite|clearbody|ironhead,psychicfangs,icepunch,protect|Jolly|||||50|]"
          "Garchomp||sitrusberry|roughskin|dragonclaw,stompingtantrum,rockslide,protect|Jolly||M|||50|]"
          "Sinistcha||kasibberry|hospitality|protect,ragepowder,matchagotcha,lifedew|Calm|||||50|]"
          "Floette-Eternal||floettite|flowerveil|protect,drainingkiss,calmmind,dazzlinggleam|Timid||F|||50|]"
          "Gyarados||leftovers|intimidate|waterfall,thunderwave,taunt,protect|Jolly||M|||50|")
TEAM_B = ("Metagross||metagrossite|clearbody|hammerarm,ironhead,psychicfangs,protect|Jolly|||||50|]"
          "Garchomp||lifeorb|roughskin|earthquake,stompingtantrum,protect,dragonclaw|Jolly||F|||50|]"
          "Aerodactyl||aerodactylite|unnerve|dualwingbeat,tailwind,rockslide,taunt|Jolly||M|||50|]"
          "Whimsicott||focussash|prankster|moonblast,tailwind,encore,taunt|Timid||F|||50|]"
          "Sneasler||whiteherb|unburden|fakeout,coaching,closecombat,gunkshot|Adamant||F|||50|]"
          "Kingambit||blackglasses|defiant|kowtowcleave,lowkick,suckerpunch,ironhead|Brave||F|||50|")


def dump_dex(cfg=CFG):
    """Query the pinned simulator and write ``artifacts/dex.json``."""
    sc = Sidecar(cfg)
    d = sc.rpc({"op": "dumpdex", "format": cfg.format_id})
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts_dir / "dex.json").write_text(json.dumps(
        {k: d[k] for k in ("species", "moves", "items")}))
    print(f"dex.json written from format {d['format']} "
          f"({len(d['species'])} species, {len(d['moves'])} moves)"
          + (" [FALLBACK FORMAT — pinned sim lacks the named Champions format]"
             if d["fallback"] else ""))
    sc.close()


def benchmark(cfg=CFG, n_steps=2000, seed=7):
    """Assert save/restore replay identity and print step/fork throughput."""
    rng = random.Random(seed)
    sc = Sidecar(cfg)
    b = SidecarBattle.create(sc, cfg.format_id, TEAM_A, TEAM_B)
    print(f"format: {b.format}" + (" (FALLBACK — named format missing at pinned "
                                   "version; megas/no-tera must be checked)" if b.fallback else ""))

    # --- save/restore correctness: same state + same choices => same log ---
    for _ in range(3):
        _step_random(b, rng)
    state = b.save()
    fork1 = SidecarBattle.restore(sc, state)
    fork2 = SidecarBattle.restore(sc, state)
    assert json.dumps(fork1.save(), sort_keys=True) == json.dumps(state, sort_keys=True), \
        "save -> restore -> save is not identical"
    choices = {s: random_choice(fork1.requests[s], random.Random(123))
               for s in fork1.pending_sides()}
    r1 = fork1.step(dict(choices))
    r2 = fork2.step(dict(choices))
    assert r1["log"] == r2["log"], "restored forks diverged on identical choices"
    print("save/restore: OK (round-trip identical, forks replay identically)")

    # --- throughput ---
    steps = games = 0
    t0 = time.time()
    b = SidecarBattle.create(sc, cfg.format_id, TEAM_A, TEAM_B)
    while steps < n_steps:
        if b.ended:
            b.destroy()
            games += 1
            b = SidecarBattle.create(sc, cfg.format_id, TEAM_A, TEAM_B)
        _step_random(b, rng)
        steps += 1
    dt = time.time() - t0
    print(f"{steps} steps in {dt:.1f}s = {steps / dt:.0f} steps/s "
          f"({games} games completed)")

    # --- fork cost (save+restore per node expansion) ---
    state = b.save() if not b.ended else state
    t0 = time.time()
    n_forks = 200
    for _ in range(n_forks):
        SidecarBattle.restore(sc, state).destroy()
    dt = time.time() - t0
    print(f"{n_forks} save/restore forks in {dt:.2f}s = {n_forks / dt:.0f} forks/s")
    print("=> set cfg.sims_per_move so sims_per_move * (1 fork + ~turns steps) "
          "fits the ladder turn timer at these rates.")
    sc.close()


def selftest(cfg=CFG):
    """Reconstruction proof on a hand-built midgame state: every public
    override must show up in the rebuilt battle, and the battle must play to
    the end. Run once per box/sim pin before trusting search output."""
    sc = Sidecar(cfg)
    teams = {"p1": parse_packed_team(TEAM_A), "p2": parse_packed_team(TEAM_B)}
    t = LogParser("selftest", 0, "", cfg.format_id)
    t.sides = {p: Side(teams[p]) for p in ("p1", "p2")}
    t.weather, t.trickroom, t.turn_no = "sandstorm", True, 7

    p1 = t.sides["p1"].mons                 # Metagross (mega, +2) / burnt Sinistcha
    p1[1].active_slot, p1[1].appeared, p1[1].turns_active = 0, True, 3
    p1[1].mega_done, p1[1].boosts["atk"] = True, 2
    t.sides["p1"].mega_used = True
    p1[3].active_slot, p1[3].appeared, p1[3].turns_active = 1, True, 1
    p1[3].hp, p1[3].status = 0.42, "brn"
    p1[3].protect_ct = 2                    # protected twice in a row
    p1[0].appeared, p1[0].item_consumed = True, True     # Blaziken, sash gone

    p2 = t.sides["p2"].mons                 # Whimsicott + Kingambit, Metagross down
    p2[3].active_slot, p2[3].appeared, p2[3].turns_active = 0, True, 2
    p2[5].active_slot, p2[5].appeared, p2[5].turns_active = 1, True, 2
    p2[0].appeared, p2[0].fainted, p2[0].hp = True, True, 0.0
    t.sides["p2"].conditions["tailwind"] = True

    brought = {"p1": [0, 1, 2, 3], "p2": [0, 3, 4, 5]}
    b, orders = reconstruct(sc, cfg.format_id, t, teams, brought)
    assert not b.ended
    assert orders["p1"][:2] == [1, 3] and orders["p2"][:2] == [3, 5]

    state = b.save()
    by_name = [{p["set"]["name"]: p for p in s["pokemon"]} for s in state["sides"]]
    gross, tea, chick = by_name[0]["Metagross"], by_name[0]["Sinistcha"], by_name[0]["Blaziken"]
    assert "mega" in gross["details"].lower(), gross["details"]
    assert gross["boosts"]["atk"] == 2
    assert not gross["canMegaEvo"] and not by_name[0]["Garchomp"]["canMegaEvo"]
    assert tea["status"] == "brn" and tea["hp"] == round(0.42 * tea["maxhp"])
    assert chick["item"] == ""
    assert by_name[1]["Metagross"]["fainted"] and by_name[1]["Metagross"]["hp"] == 0
    assert "tailwind" in state["sides"][1]["sideConditions"]
    assert state["field"]["weather"] == "sandstorm"
    assert "trickroom" in state["field"]["pseudoWeather"]
    assert by_name[0]["Metagross"]["activeTurns"] == 3
    stall = tea.get("volatiles", {}).get("stall")
    assert stall and stall.get("counter") == 9, \
        f"stall volatile not rebuilt: {stall} (protect_ct=2 -> counter 3^2)"
    assert "stall" not in gross.get("volatiles", {})
    print("reconstruct: OK (megas, boosts, hp/status, faints, items, field, "
          "side conditions, turn + protect counters all match)")

    # forks of a reconstructed battle must round-trip like created ones
    assert json.dumps(SidecarBattle.restore(sc, state).save(), sort_keys=True) \
        == json.dumps(state, sort_keys=True)

    rng, steps = random.Random(3), 0
    while not b.ended and steps < 300:
        _step_random(b, rng)
        steps += 1
    assert b.ended, "reconstructed battle did not finish under random play"
    print(f"selftest: OK (played to terminal in {steps} steps, winner {b.winner})")
    sc.close()


# ---------------------------------------------------------------------------
# live play (phase 2): poke-env against a local Showdown server
# ---------------------------------------------------------------------------

def make_live_player(sets, searcher, usage, cfg=CFG, on_decision=None,
                     **player_kwargs):
    """Build the poke-env player (class created in here so importing env.py
    never needs poke-env unless live play is used). It feeds raw protocol
    lines into the same LogParser the dataset was parsed with, keeps a
    particle filter per battle, and asks the searcher for a mixed strategy at
    every move request; orders go out as raw Showdown choice strings.

    `searcher` is any `MoveChooser`: `.choose(tracker, belief, my_id, request,
    brought) -> (JointAction, ChoiceInfo)` plus a `.bridge` attribute. play.py
    plugs policy-only / max-damage / random choosers through the same seam.
    `on_decision(battle, game, info)` is called after every decision (the
    play.py dashboard hook)."""
    from poke_env.player import Player
    from poke_env.player.battle_order import SingleBattleOrder

    from beliefs import OpponentBelief
    from search.mcts import joint_choice

    class LivePlayer(Player):
        def __init__(self):
            super().__init__(battle_format=cfg.format_id,
                             team=pack_team([full_set(s) for s in sets]),
                             **player_kwargs)
            self.raw = {}      # battle_tag -> protocol lines seen so far
            self.games = {}    # battle_tag -> tracker / belief / progress

        async def _handle_battle_message(self, split_messages):
            tag = split_messages[0][0].lstrip(">").strip()
            lines = self.raw.setdefault(tag, [])
            for sm in split_messages[1:]:
                if len(sm) > 1:
                    lines.append("|".join(sm))
            await super()._handle_battle_message(split_messages)

        def teampreview(self, battle):
            n = min(4, len(sets))            # lead selection unmodeled (v1)
            self._game(battle)["brought"] = list(range(n))
            return "/team " + "".join(str(i + 1) for i in range(n))

        def _game(self, battle):
            g = self.games.get(battle.battle_tag)
            if g is None:
                me = battle.player_role
                opp = "p2" if me == "p1" else "p1"
                # poke-env squashes forme ids ('typhlosionhisui'); rebuild the
                # dashed form ('typhlosion-hisui') from base + remainder so
                # sprite urls and display stay valid before the first |switch|
                # line rewrites species_cur (sid() of both forms is identical,
                # so nothing downstream changes)
                def _dashed(p):
                    s, b = p.species, p.base_species
                    return f"{b}-{s[len(b):]}" if len(s) > len(b) else s
                opp_sets = [full_set({"species": _dashed(p), "item": "",
                                      "ability": "", "moves": ()})
                            for p in battle.teampreview_opponent_team]
                tracker = LogParser(battle.battle_tag, 0, "", cfg.format_id)
                tracker.sides = {me: Side(sets), opp: Side(opp_sets)}
                belief_cls = getattr(
                    searcher, "belief_model_cls", OpponentBelief)
                belief = belief_cls(
                    [sid(s["species"]) for s in opp_sets], usage, cfg,
                    searcher.bridge, my_team=sets)
                g = {"tracker": tracker, "belief": belief, "fed": 0,
                     "brought": list(range(min(4, len(sets))))}
                self.games[battle.battle_tag] = g
            return g

        def choose_move(self, battle):
            g = self._game(battle)
            lines = self.raw[battle.battle_tag]
            while g["fed"] < len(lines):
                g["tracker"].feed(lines[g["fed"]])
                g["fed"] += 1
            req = battle.last_request
            if battle.force_switch and any(battle.force_switch):
                return SingleBattleOrder("/choose " + random_choice(req))
            g["belief"].update(g["tracker"].drain_events(),
                               viewer=battle.player_role)
            joint, info = searcher.choose(g["tracker"], g["belief"],
                                          battle.player_role, req, g["brought"])
            print(f"[{battle.battle_tag}] value {info['value']:+.2f}  "
                  + "  ".join(f"{d} {p:.0%}" for d, p in info["strategy"][:3]))
            if on_decision:
                on_decision(battle, g, info)
            name_to_idx = {full_set(s)["name"]: i for i, s in enumerate(sets)}
            return SingleBattleOrder(
                "/choose " + joint_choice(req, joint, name_to_idx))

    return LivePlayer()


def run_live(ckpt, team_packed, n_games=1, ladder=False, cfg=CFG):
    """Load the versioned chooser and play ladder or accepted live games."""
    import asyncio

    import torch

    from agents.determinized_duct.v1 import DeterminizedDUCTChooser
    from models.policy_value import PolicyValueNet
    from tokenizer import PositionTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    searcher = DeterminizedDUCTChooser(
        PolicyValueNet.load(ckpt, cfg, device), PositionTokenizer.load(cfg), cfg)
    usage = json.loads((cfg.artifacts_dir / "usage_stats.json").read_text())
    player = make_live_player(parse_packed_team(team_packed), searcher, usage, cfg)
    if ladder:
        asyncio.run(player.ladder(n_games))
    else:
        print(f"waiting to accept {n_games} challenge(s) as {player.username}...")
        asyncio.run(player.accept_challenges(None, n_games))
    print(f"{player.n_won_battles}/{player.n_finished_battles} games won")


if __name__ == "__main__":
    if "--dump-dex" in sys.argv:
        dump_dex()
    if "--benchmark" in sys.argv:
        n = int(sys.argv[sys.argv.index("--benchmark") + 1]) \
            if sys.argv.index("--benchmark") + 1 < len(sys.argv) else 2000
        benchmark(n_steps=n)
    if "--selftest" in sys.argv:
        selftest()
    if "--live" in sys.argv:
        args = sys.argv[sys.argv.index("--live") + 1:]
        ckpt = args[0] if args and not args[0].startswith("--") \
            else str(CFG.checkpoint_dir / "ckpt_best.pt")
        team = TEAM_A
        if "--team" in args:
            team = open(args[args.index("--team") + 1]).read().strip()
        n = int(args[args.index("--n") + 1]) if "--n" in args else 1
        run_live(ckpt, team, n_games=n, ladder="--ladder" in args)
