"""OpponentBelief: a particle filter over each opponent Pokemon's hidden set.

Particles are the distinct full sets (moves, item, ability, nature) seen in the
train-split team sheets for that species, weighted by frequency. Evidence from
the battle stream reweights/kills them:
  - revealed move / item / ability / mega  -> hard constraint
  - speed order (they acted before/after us in the same priority bracket)
    -> effective-speed inequality per particle
  - damage they deal to us -> our defenses are exactly known, so a particle's
    roll range must contain the observed damage (constrains their attack set)
  - damage we deal to them -> constrains their HP x defense (looser)
If every particle dies, weights are rebuilt from the prior subject to the hard
reveals only.

Outputs: summary() buckets for the tokenizer, sample_sets(k) for search
determinization, top_particle(k) for the damage feature matrix.

Known blind spot: particles only cover sets seen in the train split, so a
genuinely novel set (off-meta mon, custom spread) can kill every particle and
force the prior fallback. `python beliefs.py --audit` measures how often that
happens on held-out battles (depletion rate, whether the oracle set was even
in the prior, and how much posterior mass it ends with). If the audit shows
frequent depletion the fix is widening the prior — e.g. per-species EV-spread
variants inferred from damage-roll residuals — not patching the filter.
"""

import json
import random
import re
import sys
import time

from config import CFG
from damage import request

NATURES = {  # nature -> (boosted stat, lowered stat); neutral natures omitted
    "adamant": ("atk", "spa"), "lonely": ("atk", "def"), "brave": ("atk", "spe"),
    "naughty": ("atk", "spd"), "bold": ("def", "atk"), "impish": ("def", "spa"),
    "relaxed": ("def", "spe"), "lax": ("def", "spd"), "modest": ("spa", "atk"),
    "mild": ("spa", "def"), "quiet": ("spa", "spe"), "rash": ("spa", "spd"),
    "calm": ("spd", "atk"), "gentle": ("spd", "def"), "sassy": ("spd", "spe"),
    "careful": ("spd", "spa"), "timid": ("spe", "atk"), "hasty": ("spe", "def"),
    "jolly": ("spe", "spa"), "naive": ("spe", "spd"),
}
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


_DEX_CACHE = {}


def load_dex(cfg=CFG):
    p = cfg.artifacts_dir / "dex.json"
    if p not in _DEX_CACHE:
        _DEX_CACHE[p] = json.loads(p.read_text()) if p.exists() else None
    return _DEX_CACHE[p]


def calc_stat(base, stat, nature, ev=0, iv=31, level=50):
    if stat == "hp":
        return (2 * base + iv + ev // 4) * level // 100 + level + 10
    v = (2 * base + iv + ev // 4) * level // 100 + 5
    up, down = NATURES.get(nature, ("", ""))
    return int(v * (1.1 if stat == up else 0.9 if stat == down else 1.0))


def boost_mult(b):
    return (2 + b) / 2 if b >= 0 else 2 / (2 - b)


class OpponentBelief:
    def __init__(self, opp_species, usage, cfg=CFG, bridge=None, my_team=None):
        self.cfg, self.bridge = cfg, bridge
        self.dex = load_dex(cfg)
        self.species = opp_species          # sids, team-preview order
        self.my_team = my_team or []
        self.particles = []                 # per mon: list of set dicts
        self.weights = []                   # per mon: list of floats
        self.priors = []
        for sp in opp_species:
            sets = [{"moves": tuple(mv), "item": it, "ability": ab, "nature": na,
                     "n": c} for c, mv, it, ab, na in usage.get(sp, [])]
            if not sets:
                sets = [{"moves": (), "item": "", "ability": "", "nature": "serious", "n": 1}]
            sets = sets[:cfg.n_particles]
            total = sum(s["n"] for s in sets)
            self.particles.append(sets)
            self.priors.append([s["n"] / total for s in sets])
            self.weights.append([s["n"] / total for s in sets])
        self.constraints = [{"moves": set(), "item": None, "consumed": False,
                             "ability": None, "mega": False} for _ in opp_species]
        self._dmg_hits = {}                 # (atk_idx, def_idx) -> count
        # per-mon fallback counters, read by `--audit` and observe_game:
        # soft = evidence killed all weighted particles, rebuilt from prior
        # hard = even the prior is inconsistent with the hard reveals
        self.soft_depletions = [0] * len(opp_species)
        self.hard_depletions = [0] * len(opp_species)

    # -- stats -----------------------------------------------------------
    def _base(self, species, stat):
        if not self.dex or species not in self.dex["species"]:
            return None
        return self.dex["species"][species]["baseStats"][stat]

    def _particle_speed(self, k, p, ctx=None):
        base = self._base(self.species[k], "spe")
        if base is None:
            return None
        spe = calc_stat(base, "spe", p["nature"])
        if p["item"] == "choicescarf" and not self.constraints[k]["consumed"]:
            spe = int(spe * 1.5)
        if ctx:
            spe *= boost_mult(ctx["spe"])
            if ctx["tw"]:
                spe *= 2
            if ctx["par"]:
                spe *= 0.5
        return spe

    def _my_speed(self, idx, ctx):
        s = self.my_team[idx]
        base = self._base(_sid(s["species"]), "spe")
        if base is None:
            return None
        spe = calc_stat(base, "spe", s["nature"], ev=s["evs"][5])
        if s["item"] == "choicescarf":
            spe = int(spe * 1.5)
        spe *= boost_mult(ctx["spe"])
        if ctx["tw"]:
            spe *= 2
        if ctx["par"]:
            spe *= 0.5
        return spe

    # -- constraint machinery ---------------------------------------------
    def _apply(self, k, keep):
        """keep(particle) -> bool | float multiplier. A pure-Python pass over
        <= n_particles (200) floats — microseconds. The filter's real cost is
        calc_batch in _damage_evidence; `--audit` prints both to confirm."""
        self._apply_list(k, [float(keep(p)) for p in self.particles[k]])

    def _apply_list(self, k, mults):
        new = [wi * m for wi, m in zip(self.weights[k], mults)]
        if sum(new) <= 0:
            self.soft_depletions[k] += 1
            new = [pr * self._hard_ok(k, p) for pr, p in
                   zip(self.priors[k], self.particles[k])]
        if sum(new) <= 0:
            self.hard_depletions[k] += 1
            new = list(self.priors[k])
        total = sum(new)
        self.weights[k] = [x / total for x in new]

    def _hard_ok(self, k, p):
        c = self.constraints[k]
        return float(c["moves"] <= set(p["moves"])
                     and (c["item"] is None or p["item"] == c["item"])
                     and (c["ability"] is None or p["ability"] == c["ability"]))

    def _resample_check(self, k):
        alive = sum(1 for w in self.weights[k] if w > 1e-9)
        if alive / len(self.weights[k]) < self.cfg.resample_floor:
            mixed = [0.9 * w + 0.1 * pr * self._hard_ok(k, p) for w, pr, p in
                     zip(self.weights[k], self.priors[k], self.particles[k])]
            total = sum(mixed) or 1.0
            self.weights[k] = [x / total for x in mixed]

    # -- update from one turn's events -------------------------------------
    def update(self, events, viewer):
        opp = "p2" if viewer == "p1" else "p1"
        for ev in events:
            if ev[0] == "reveal" and ev[1] == opp:
                _, _, k, kind, name = ev
                c = self.constraints[k]
                if kind == "move":
                    c["moves"].add(name)
                    self._apply(k, lambda p: name in p["moves"])
                elif kind == "item":
                    c["item"] = name
                    self._apply(k, lambda p: p["item"] == name)
                elif kind == "ability":
                    c["ability"] = name
                    self._apply(k, lambda p: p["ability"] == name)
            elif ev[0] == "consumed" and ev[1] == opp:
                self.constraints[ev[2]]["consumed"] = True
            elif ev[0] == "mega" and ev[1] == opp:
                self.constraints[ev[2]]["mega"] = True
            elif ev[0] == "move_order":
                self._speed_evidence(ev[1], ev[2], viewer, opp)
            elif ev[0] == "dmg":
                self._damage_evidence(ev, viewer, opp)
        for k in range(len(self.species)):
            self._resample_check(k)

    def _speed_evidence(self, order, glob, viewer, opp):
        if not self.dex:
            return
        pri = {m: self.dex["moves"].get(m, {}).get("priority", 0)
               for _, _, m, _ in order}
        for oi, (o_side, o_idx, o_move, o_ctx) in enumerate(order):
            if o_side != opp:
                continue
            for mi, (m_side, m_idx, m_move, m_ctx) in enumerate(order):
                if m_side != viewer or pri[o_move] != pri[m_move]:
                    continue
                mine = self._my_speed(m_idx, m_ctx)
                if mine is None:
                    continue
                first = oi < mi          # opponent acted first
                if glob["tr"]:
                    first = not first    # trick room: slower acts first
                k = o_idx
                slack = self.cfg.investment_slack  # hidden training, both sides

                def ok(p, mine=mine, first=first, k=k, ctx=o_ctx, slack=slack):
                    theirs = self._particle_speed(k, p, ctx)
                    if theirs is None:
                        return True
                    return theirs * slack >= mine if first else mine * slack >= theirs

                self._apply(k, ok)

    def _damage_evidence(self, ev, viewer, opp):
        _, atk_side, atk_idx, move, def_side, def_idx, frac, ctx = ev
        if self.bridge is None or frac <= 0 or ctx["crit"] or ctx["spread"] or ctx["multi"]:
            return
        if ctx.get("def_transformed"):
            return
        if self.dex and self.dex["moves"].get(move, {}).get("multihit"):
            return
        pair = (atk_side, atk_idx, def_idx)
        self._dmg_hits[pair] = self._dmg_hits.get(pair, 0) + 1
        if self._dmg_hits[pair] > self.cfg.belief_damage_hits_per_pair:
            return
        truncated = frac >= ctx["def_hp_before"] - 1e-6
        tol = self.cfg.damage_tolerance
        inv = self.cfg.investment_slack   # hidden attack investment raises damage,
        their_attack = atk_side == opp    # hidden bulk investment lowers it
        field = {"weather": ctx["weather"], "terrain": ctx["terrain"],
                 "screens": ctx["screens"]}

        if atk_side == opp and self.my_team:      # their attack, our known defender
            k = atk_idx
            d = self.my_team[def_idx]
            dfd = {"species": _sid(d["species"]), "level": d["level"],
                   "item": d["item"], "ability": d["ability"], "nature": d["nature"],
                   "evs": d["evs"], "boosts": ctx["def_boosts"]}
            reqs = [request(self._as_attacker(k, p, ctx), dfd, move, field)
                    for p in self.particles[k]]
        elif atk_side == viewer and self.my_team:  # our attack, their unknown defender
            k = def_idx
            a = self.my_team[atk_idx]
            atk = {"species": _sid(a["species"]), "level": a["level"],
                   "item": a["item"], "ability": a["ability"], "nature": a["nature"],
                   "evs": a["evs"], "boosts": ctx["atk_boosts"],
                   "status": "brn" if ctx["burn"] else ""}
            reqs = [request(atk, self._as_defender(k, p, ctx), move, field)
                    for p in self.particles[k]]
        else:
            return
        res = self.bridge.calc_batch(reqs)

        def ok(r):
            if r is None:
                return 1.0
            lo, hi = r
            if their_attack:      # their unknown atk investment can only add
                lo, hi = lo, hi * inv
            else:                 # their unknown bulk investment can only subtract
                lo, hi = lo / inv, hi
            if truncated:
                return float(hi >= frac - tol)
            return float(lo - tol <= frac <= hi + tol)

        self._apply_list(k, [ok(r) for r in res])

    def _as_attacker(self, k, p, ctx):
        return {"species": self._species_cur(k), "level": 50,
                "item": None if self.constraints[k]["consumed"] else p["item"],
                "ability": p["ability"], "nature": p["nature"],
                "boosts": ctx["atk_boosts"], "status": "brn" if ctx["burn"] else ""}

    def _as_defender(self, k, p, ctx):
        return {"species": self._species_cur(k), "level": 50,
                "item": None if self.constraints[k]["consumed"] else p["item"],
                "ability": p["ability"], "nature": p["nature"],
                "boosts": ctx["def_boosts"]}

    def _species_cur(self, k):
        sp = self.species[k]
        c = self.constraints[k]
        if not c["mega"]:
            return sp
        stone = (self.dex or {}).get("items", {}).get(c["item"] or "", {}).get("megaStone")
        return (stone or {}).get(sp) or sp + "mega"

    # -- outputs ------------------------------------------------------------
    def top_particle(self, k):
        i = max(range(len(self.weights[k])), key=lambda i: self.weights[k][i])
        return self.particles[k][i]

    def summary(self):
        """Per mon: modal item + its posterior mass, effective-speed range and
        expected bulk. Item-general on purpose: a scarf shows up as the modal
        item and as a stretched speed-range high end, a sash/berry/band the
        same way through the item token — no item gets bespoke features."""
        out = {}
        for k, sp in enumerate(self.species):
            ws, ps = self.weights[k], self.particles[k]
            item, p_item = self.item_posterior(k)[0]
            speeds = [(self._particle_speed(k, p) or 0, w) for p, w in zip(ps, ws)]
            speeds.sort()
            lo = _quantile(speeds, 0.05)
            hi = _quantile(speeds, 0.95)
            bulk = 0.0
            hp_b, d_b, sd_b = (self._base(sp, s) for s in ("hp", "def", "spd"))
            if hp_b is not None:
                for p, w in zip(ps, ws):
                    hp = calc_stat(hp_b, "hp", p["nature"])
                    df = calc_stat(d_b, "def", p["nature"])
                    sd = calc_stat(sd_b, "spd", p["nature"])
                    bulk += w * hp * (df + sd) / 2
            out[k] = {"item": item, "p_item": p_item,
                      "spe_lo": lo, "spe_hi": hi, "bulk": bulk}
        return out

    def item_posterior(self, k):
        """[(item, prob)] sorted by prob, for summary() and the game viewer."""
        acc = {}
        for w, p in zip(self.weights[k], self.particles[k]):
            acc[p["item"]] = acc.get(p["item"], 0.0) + w
        return sorted(acc.items(), key=lambda kv: -kv[1])

    def sample_sets(self, n, rng=random):
        """n determinizations: each a full team of sampled sets (preview order)."""
        teams = []
        for _ in range(n):
            team = []
            for k, sp in enumerate(self.species):
                p = rng.choices(self.particles[k], weights=self.weights[k])[0]
                team.append({"species": sp, "moves": list(p["moves"]),
                             "item": p["item"], "ability": p["ability"],
                             "nature": p["nature"]})
            teams.append(team)
        return teams


def determinized(sets, cfg=CFG):
    """A collapsed belief: one particle per mon, weight 1. The search uses it
    inside a determinization, where the opponent's sets are fixed, so
    summary()/top_particle() feed the tokenizer the sampled 'truth' through
    the same interface the real filter uses."""
    usage = {_sid(s["species"]): [(1, list(s["moves"]), s["item"],
                                   s["ability"], s["nature"])] for s in sets}
    return OpponentBelief([_sid(s["species"]) for s in sets], usage, cfg)


def _quantile(sorted_pairs, q):
    acc = 0.0
    for v, w in sorted_pairs:
        acc += w
        if acc >= q:
            return v
    return sorted_pairs[-1][0] if sorted_pairs else 0


def _sid(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------------------
# `python beliefs.py --audit [max_battles]` — replay held-out battles through
# the filter and measure how often it breaks down (the module-docstring
# question). Run after `data.py parse`, on the training box.
# ---------------------------------------------------------------------------

def _set_key(s):
    # team sheets redact EVs, so identity is (moves, item, ability, nature)
    return (tuple(sorted(s["moves"])), s["item"], s["ability"], s["nature"])


def audit(max_battles, cfg=CFG):
    import pickle
    from collections import Counter

    usage = json.loads((cfg.artifacts_dir / "usage_stats.json").read_text())
    bridge, bridge_time = None, [0.0]
    if cfg.use_belief_damage_updates:
        from damage import DamageBridge
        bridge = DamageBridge(cfg)
        orig = bridge.calc_batch

        def timed(reqs):
            t0 = time.perf_counter()
            out = orig(reqs)
            bridge_time[0] += time.perf_counter() - t0
            return out

        bridge.calc_batch = timed

    battles = []
    for fn in cfg.dataset_files:
        fmt = fn[len("logs_"):-len(".json")]
        with open(cfg.parsed_dir / f"{fmt}.pkl", "rb") as f:
            battles += [b for b in pickle.load(f) if b["split"] == "test"]
    battles = battles[:max_battles]

    mons = soft = hard = in_prior = top1 = depleted_battles = 0
    oracle_mass = 0.0
    by_species = Counter()
    t0 = time.perf_counter()
    for rec in battles:
        battle_depleted = False
        for p in ("p1", "p2"):
            opp = "p2" if p == "p1" else "p1"
            oracle = rec["teams"][opp]
            bel = OpponentBelief([_sid(s["species"]) for s in oracle], usage,
                                 cfg, bridge, my_team=rec["teams"][p])
            for turn in rec["turns"]:
                bel.update(turn["events"], viewer=p)
            for k, s in enumerate(oracle):
                mons += 1
                keys = [_set_key(pt) for pt in bel.particles[k]]
                i = keys.index(_set_key(s)) if _set_key(s) in keys else None
                in_prior += i is not None
                if i is not None:
                    oracle_mass += bel.weights[k][i]
                    top1 += bel.weights[k][i] == max(bel.weights[k])
                if bel.soft_depletions[k]:
                    soft += 1
                    by_species[bel.species[k]] += 1
                    battle_depleted = True
                hard += bel.hard_depletions[k] > 0
        depleted_battles += battle_depleted
    dt = time.perf_counter() - t0

    print(f"{len(battles)} test battles, {mons} opponent mons, {dt:.0f}s "
          f"({len(battles) / dt:.1f} battles/s"
          + (f", {bridge_time[0]:.0f}s inside the damage bridge)" if bridge else ")"))
    print(f"oracle set in prior:        {in_prior / mons:.1%}  (ceiling: the filter can never converge past this)")
    print(f"oracle posterior mass:      {oracle_mass / mons:.3f} (avg over mons where present)")
    print(f"oracle is top particle:     {top1 / mons:.1%}")
    print(f"mons with soft depletion:   {soft / mons:.1%}  (evidence killed every particle at least once)")
    print(f"mons with hard depletion:   {hard / mons:.1%}  (prior inconsistent even with reveals)")
    print(f"battles with any depletion: {depleted_battles / len(battles):.1%}")
    print("\nworst species (mons depleted | distinct train sets):")
    for sp, n in by_species.most_common(10):
        print(f"  {sp:24s} {n:4d} | {len(usage.get(sp, [])):4d}")
    if bridge:
        bridge.close()


if __name__ == "__main__":
    if "--audit" in sys.argv:
        i = sys.argv.index("--audit")
        audit(int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 500)
