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

Stat-point spreads: team sheets redact nature and stat training. For a species
covered by ``artifacts/spreads.json``, each train-split set is crossed with the
top objective (nature, spread) builds plus an off-list ``any`` cushion. Concrete
objective builds are tested at their exact Champions stats. The cushion, and
the hand-built archetype fallback used for uncovered species, instead keep
feasible attack/speed intervals that move-order and damage evidence narrow;
defensive evidence stays conservative. Archived tokenizer layout 2 represented
the archetype posterior directly; current layout 3 represents inferred nature
from the objective spread prior. Exact hidden SP values never become tokens.

Known blind spot: particles only cover sets seen in the train split, so a
genuinely novel set (off-meta mon) can kill every particle and force the
prior fallback. `python beliefs.py --audit` measures how often that happens
on held-out battles (depletion rate, whether the oracle set was even in the
prior, and how much posterior mass it ends with).
"""

import json
import random
import re
import sys
import time
from collections import Counter

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
    """Return cached decoded ``dex.json`` mapping, or ``None`` when absent."""
    p = cfg.artifacts_dir / "dex.json"
    if p not in _DEX_CACHE:
        _DEX_CACHE[p] = json.loads(p.read_text()) if p.exists() else None
    return _DEX_CACHE[p]


_SPREADS_CACHE = {}


def load_spreads(cfg=CFG):
    """artifacts/spreads.json (built by build_spreads.py): per-species objective
    (nature%, 66-pt SP spread%) marginals from Pikalytics. None when absent, so
    the filter cleanly falls back to the hand-built archetype prior."""
    p = cfg.artifacts_dir / "spreads.json"
    if p not in _SPREADS_CACHE:
        _SPREADS_CACHE[p] = (json.loads(p.read_text()).get("mons")
                             if p.exists() else None)
    return _SPREADS_CACHE[p]


MAX_SP = 32   # Champions stat-point cap per stat (66 total, IVs forced to 31)
TOTAL_SP = 66

# Spread archetypes: named allocations of the 66 stat points ("att" resolves
# to atk or spa per set — nature intent first, then base stats). "any" is the
# catch-all particle with no concrete spread; it keeps the conservative
# one-sided bounds, so the filter can never do worse than the pre-archetype
# version on an off-archetype custom spread.
ARCHETYPES = {
    "fast-strong":  {"spe": 32, "att": 32, "hp": 2},
    "fast-bulky":   {"spe": 32, "hp": 32, "att": 2},
    "strong-bulky": {"att": 32, "hp": 32, "spe": 2},
    "bulky-phys":   {"hp": 32, "def": 32, "spd": 2},
    "bulky-spec":   {"hp": 32, "spd": 32, "def": 2},
    "mixed":        {"hp": 22, "att": 22, "spe": 22},
    "any":          None,
}


def _att_key(species, nature, dex):
    """Which attack stat this set invests: the nature's intent if it has one,
    else the higher base stat."""
    up, down = NATURES.get(nature, ("", ""))
    if up in ("atk", "spa"):
        return up
    if down in ("atk", "spa"):                # dumped stat -> the other one
        return "spa" if down == "atk" else "atk"
    if dex and species in dex.get("species", {}):
        bs = dex["species"][species]["baseStats"]
        return "atk" if bs["atk"] >= bs["spa"] else "spa"
    return "atk"


def archetype_spread(arch, species, nature, dex):
    """Archetype name -> concrete [hp, atk, def, spa, spd, spe] SP list."""
    alloc = ARCHETYPES[arch]
    if alloc is None:
        return None
    att = _att_key(species, nature, dex)
    evs = [0] * 6
    for stat, sp in alloc.items():
        evs[STAT_KEYS.index(att if stat == "att" else stat)] = sp
    return evs


def _arch_prior_mult(arch, nature):
    """Nature is public on no one's sheet either, but the particle carries it;
    a Timid set is far likelier to run speed points than a Relaxed one."""
    up, down = NATURES.get(nature, ("", ""))
    m = 1.0
    if up == "spe":
        m *= 2.5 if arch.startswith("fast") else 0.5
    if down == "spe":
        m *= 0.2 if arch.startswith("fast") else 1.5
    if up in ("def", "spd") and arch.startswith("bulky"):
        m *= 2.0
    if up in ("atk", "spa") and "strong" in arch:
        m *= 1.5
    return m


def calc_stat(base, stat, nature, sp=0):
    """Champions flat formula (the mod's statModify, level-independent):
    HP = base + SP + 75, others = base + SP + 20, nature +/-10% after
    (integer math matches the mod's tr(tr(stat*110,16)/100))."""
    if stat == "hp":
        return base + sp + 75
    v = base + sp + 20
    up, down = NATURES.get(nature, ("", ""))
    return v * 110 // 100 if stat == up else v * 90 // 100 if stat == down else v


def boost_mult(b):
    """Return the numeric stat multiplier for boost stage ``b``."""
    return (2 + b) / 2 if b >= 0 else 2 / (2 - b)


class OpponentBelief:
    """Externally owned per-battle particle posteriors for opponent sets."""

    def __init__(self, opp_species, usage, cfg=CFG, bridge=None, my_team=None):
        """Construct preview-ordered particles from species ids and usage rows."""
        self.cfg, self.bridge = cfg, bridge
        self.dex = load_dex(cfg)
        self.spreads = load_spreads(cfg) if getattr(cfg, "spreads_prior", True) else None
        self.species = opp_species          # sids, team-preview order
        self.my_team = my_team or []
        self.particles = []                 # per mon: list of set dicts
        self.weights = []                   # per mon: list of floats
        self.priors = []
        for sp in opp_species:
            # usage rows may carry a 6th element: a known SP spread (authored
            # scenarios / determinized own team); real sheets redact it
            sets = [{"moves": tuple(mv), "item": it, "ability": ab, "nature": na,
                     "evs": rest[0] if rest else None, "arch": None, "n": c}
                    for c, mv, it, ab, na, *rest in usage.get(sp, [])]
            if not sets:
                sets = [{"moves": (), "item": "", "ability": "", "nature": "serious",
                     "evs": None, "arch": None, "n": 1}]
            if self.spreads and sp in self.spreads:
                sets = self._expand_spreads(sp, sets, cfg)
            elif getattr(cfg, "spread_archetypes", True):
                sets = self._expand_archetypes(sp, sets, cfg)
            else:
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
        # --- diagnostics (behavior-preserving; zero cost when unused) --------
        # cause-tagged depletion tallies, so `--audit` can say WHICH evidence
        # channel (reveal_move/item/ability vs speed vs damage) is doing the
        # killing -- i.e. whether the lever is prior coverage or archetype
        # sharpness. Populated only inside _apply_list.
        self.soft_by_cause = Counter()
        self.hard_by_cause = Counter()
        # oracle-mass attribution: set self.oracle_keys = {k: _set_key(true_set)}
        # (audit only) to have _apply_list record what collapses the TRUE set's
        # mass. A hard reveal killing the oracle set is a BUG signal (the true
        # set contains every revealed move); speed/damage loss is the
        # archetype-sharpness lever. None => no tracking, no overhead.
        self.oracle_keys = None
        self.oracle_keyfn = _set_key            # audit overrides to _id_key (nature-free)
        self.oracle_loss_by_cause = Counter()   # summed relative mass lost
        self.oracle_kills_by_cause = Counter()  # oracle set driven to ~0
        # --- phase-3.1 strict SP inversion ----------------------------------
        # Per particle, a feasible stat-point INTERVAL for the stats we infer
        # exactly (attack + speed). Evidence narrows [lo,hi]; a particle dies
        # only when its interval empties, so a coarse archetype spread that
        # misses the observed value is narrowed, not annihilated. Defensive
        # stats stay on the old slack, so no interval for def/spd/hp here.
        self.strict_spe = getattr(cfg, "strict_speed_ev", True)
        self.strict_atk = getattr(cfg, "strict_attack_ev", True) and bridge is not None
        self.sp_bounds = [[{"atk": [0, MAX_SP], "spa": [0, MAX_SP],
                            "spe": [0, MAX_SP]} for _ in ps]
                          for ps in self.particles]
        # --- factored depletion fallback -----------------------------------
        # two independent marginal buckets per mon, from the SAME train sets:
        # bucket 1 = (moveset, nature), bucket 2 = (item, ability). On hard
        # depletion we cross them (filtered by the hard reveals) to synthesize
        # combinations the joint prior never contained. Backoff only.
        self.factored_fallback = getattr(cfg, "factored_fallback", True)
        self._bucket_mn, self._bucket_ia = [], []
        for sp in opp_species:
            b_mn, b_ia = Counter(), Counter()
            for row in usage.get(sp, []):
                c, mv, it, ab, na, *_ = row
                b_mn[(tuple(mv), na)] += c
                b_ia[(it, ab)] += c
            self._bucket_mn.append(sorted(b_mn.items(), key=lambda kv: -kv[1]))
            self._bucket_ia.append(sorted(b_ia.items(), key=lambda kv: -kv[1]))

    def _free_sp(self, p):
        """True when this particle's stat points are UNKNOWN (redacted) and so
        eligible for interval inversion: every archetype particle, incl. the
        'any' catch-all. Authored / determinized sets carry a concrete, known
        spread (arch is None and evs is set) and are tested at their exact
        stats instead, so search determinizations and scenario sets are
        unaffected."""
        return p.get("arch") is not None or p.get("evs") is None

    def _spread_nature_combos(self, sp, k):
        """Top-k concrete (weight, ev-spread, nature) builds for a covered mon,
        from its Pikalytics marginals. Nature and spread are independent
        marginals on the page, so we cross them (weight = spread% x nature%) and
        down-weight combos where the nature LOWERS a heavily-invested stat (a
        32-SpA Adamant build is near-nonsense); the tail is culled by the top-k
        cut anyway. This is the objective replacement for the flat archetype
        grid + neutral-nature assumption."""
        data = self.spreads[sp]
        nats = data["natures"]
        combos = []
        for ev, spct in data["spreads"]:
            for nat, npct in nats.items():
                w = spct * npct
                down = NATURES.get(nat, ("", ""))[1]
                if down and ev[STAT_KEYS.index(down)] >= 16:
                    w *= 0.1
                combos.append((w, ev, nat))
        combos.sort(key=lambda x: -x[0])
        return combos[:k]

    def _expand_spreads(self, sp, sets, cfg):
        """Covered-mon expansion: each redacted usage set -> the top-K real
        (spread,nature) builds (concrete evs + real nature, arch=None so the
        existing concrete-particle path tests them EXACTLY) plus one 'any' slack
        cushion (arch='any') for spreads off the Pikalytics list. SP is fixed to
        the real values; NATURE is the inferred latent (which build survives the
        speed/damage facts). Authored/determinized sets (evs already set) pass
        through untouched, exactly like _expand_archetypes."""
        k = cfg.spreads_top_k
        combos = self._spread_nature_combos(sp, k)
        tot = sum(w for w, _, _ in combos) or 1.0
        any_frac = cfg.spreads_any_weight
        base = sets[:max(1, cfg.n_particles // (k + 1))]
        out = []
        for s in base:
            if s["evs"] is not None:
                out.append(s)
                continue
            for w, ev, nat in combos:
                out.append({**s, "nature": nat, "evs": list(ev), "arch": None,
                            "n": s["n"] * (1 - any_frac) * w / tot})
            out.append({**s, "nature": "serious", "evs": None, "arch": "any",
                        "n": s["n"] * any_frac})
        return out

    def _expand_archetypes(self, sp, sets, cfg):
        """Each redacted-spread set -> one particle per archetype, nature-
        weighted. Sets that already carry a concrete spread (authored
        scenarios, determinized teams) pass through untouched. The base-set
        cap keeps the total particle count (and so the damage-bridge cost)
        at ~n_particles, same as before the expansion."""
        out = []
        for s in sets[:max(1, cfg.n_particles // len(ARCHETYPES))]:
            if s["evs"] is not None:
                out.append(s)
                continue
            for arch in ARCHETYPES:
                out.append({**s, "arch": arch,
                            "evs": archetype_spread(arch, sp, s["nature"], self.dex),
                            "n": s["n"] * _arch_prior_mult(arch, s["nature"])})
        return out

    # -- stats -----------------------------------------------------------
    def _base(self, species, stat):
        """Return one integer base stat from dex, or ``None`` if unavailable."""
        if not self.dex or species not in self.dex["species"]:
            return None
        return self.dex["species"][species]["baseStats"][stat]

    def _particle_speed(self, k, p, ctx=None, sp=None, j=None):
        """sp=None picks the speed SP to evaluate at: for a free-SP particle
        under strict speed the MIDPOINT of its narrowed feasible interval
        (j = particle index), else the particle's own spread if concrete, else
        0 (the conservative floor for 'any'-spread particles)."""
        base = self._base(self.species[k], "spe")
        if base is None:
            return None
        if sp is None:
            if self.strict_spe and j is not None and self._free_sp(p):
                lo, hi = self.sp_bounds[k][j]["spe"]
                sp = (lo + hi) // 2
            else:
                sp = p["evs"][5] if p.get("evs") else 0
        spe = calc_stat(base, "spe", p["nature"], sp)
        if p["item"] == "choicescarf" and not self.constraints[k]["consumed"]:
            spe = int(spe * 1.5)
        if ctx:
            spe *= boost_mult(ctx["spe"])
            if ctx["tw"]:
                spe *= 2
            if ctx["par"]:
                spe *= 0.5
        return spe

    def _my_speed(self, idx, ctx, sp=None):
        """Return effective speed for a known own-team mon and context."""
        s = self.my_team[idx]
        base = self._base(_sid(s["species"]), "spe")
        if base is None:
            return None
        spe = calc_stat(base, "spe", s["nature"], s["evs"][5] if sp is None else sp)
        if s["item"] == "choicescarf":
            spe = int(spe * 1.5)
        spe *= boost_mult(ctx["spe"])
        if ctx["tw"]:
            spe *= 2
        if ctx["par"]:
            spe *= 0.5
        return spe

    # -- constraint machinery ---------------------------------------------
    def _apply(self, k, keep, cause="?"):
        """keep(particle) -> bool | float multiplier. A pure-Python pass over
        <= n_particles (200) floats — microseconds. The filter's real cost is
        calc_batch in _damage_evidence; `--audit` prints both to confirm."""
        self._apply_list(k, [float(keep(p)) for p in self.particles[k]], cause)

    def _oracle_mass(self, k):
        """Total weight on particles sharing the true set's identity (audit
        only; self.oracle_keys must be set). Grouped by _set_key because
        archetype expansion splits one set across several particles."""
        key = self.oracle_keys[k]
        return sum(w for w, p in zip(self.weights[k], self.particles[k])
                   if self.oracle_keyfn(p) == key)

    def _apply_list(self, k, mults, cause="?"):
        """Apply multipliers, normalize, and track any depletion cause."""
        before = self._oracle_mass(k) if self.oracle_keys else None
        new = [wi * m for wi, m in zip(self.weights[k], mults)]
        if sum(new) <= 0:
            self.soft_depletions[k] += 1
            self.soft_by_cause[cause] += 1
            new = [pr * self._hard_ok(k, p) for pr, p in
                   zip(self.priors[k], self.particles[k])]
        if sum(new) <= 0:
            self.hard_depletions[k] += 1
            self.hard_by_cause[cause] += 1
            new = self._hard_depletion_fallback(k)
        total = sum(new)
        self.weights[k] = [x / total for x in new]
        if before is not None and before > 1e-9:
            after = self._oracle_mass(k)
            if after < before * 0.5:
                self.oracle_loss_by_cause[cause] += (before - after) / before
            if after <= 1e-9:
                self.oracle_kills_by_cause[cause] += 1

    def _hard_ok(self, k, p):
        """Return whether a particle satisfies all hard public reveals."""
        c = self.constraints[k]
        return float(c["moves"] <= set(p["moves"])
                     and (c["item"] is None or p["item"] == c["item"])
                     and (c["ability"] is None or p["ability"] == c["ability"]))

    def _hard_depletion_fallback(self, k):
        """No train set satisfies the hard reveals. Instead of collapsing to
        the raw joint prior (which still cannot contain a never-seen
        COMBINATION), stitch particles by crossing the two marginal buckets --
        (moveset,nature) x (item,ability) -- keeping only combinations
        consistent with the reveals, so a moveset seen with one item/ability
        can pair with an item/ability seen on a different set. The stitched
        particles are APPENDED to this mon's arrays (arch='any' -> free SP +
        the slack safety net, and open to further evidence). Returns the new
        weight vector over the extended list; falls back to the raw prior when
        factoring is off or yields nothing (e.g. a genuinely novel move)."""
        if not self.factored_fallback:
            return list(self.priors[k])
        c = self.constraints[k]
        mn = [(v, n) for v, n in self._bucket_mn[k] if c["moves"] <= set(v[0])]
        ia = [(v, n) for v, n in self._bucket_ia[k]
              if (c["item"] is None or v[0] == c["item"])
              and (c["ability"] is None or v[1] == c["ability"])]
        if not mn or not ia:
            return list(self.priors[k])
        existing = {_set_key(p) for p in self.particles[k]}
        cand = []
        for (moves, nature), n1 in mn:
            for (item, ability), n2 in ia:
                key = (tuple(sorted(moves)), item, ability, nature)
                if key in existing:
                    continue
                existing.add(key)
                cand.append((n1 * n2, {"moves": moves, "item": item,
                                       "ability": ability, "nature": nature,
                                       "evs": None, "arch": "any"}))
        if not cand:
            return list(self.priors[k])
        cand.sort(key=lambda wp: -wp[0])
        cand = cand[:self.cfg.n_particles]
        tot = sum(w for w, _ in cand)
        new = [0.0] * len(self.particles[k])   # every old particle is inconsistent
        for w, part in cand:
            pr = w / tot
            self.particles[k].append(part)
            self.priors[k].append(pr)
            self.sp_bounds[k].append({"atk": [0, MAX_SP], "spa": [0, MAX_SP],
                                      "spe": [0, MAX_SP]})
            new.append(pr)
        return new

    def _resample_check(self, k):
        """Mix prior mass when the alive-particle fraction falls too low."""
        alive = sum(1 for w in self.weights[k] if w > 1e-9)
        if alive / len(self.weights[k]) < self.cfg.resample_floor:
            mixed = [0.9 * w + 0.1 * pr * self._hard_ok(k, p) for w, pr, p in
                     zip(self.weights[k], self.priors[k], self.particles[k])]
            total = sum(mixed) or 1.0
            self.weights[k] = [x / total for x in mixed]

    # -- update from one turn's events -------------------------------------
    def update(self, events, viewer):
        """Consume ``list[BeliefEvent]`` and mutate posterior weights/bands."""
        opp = "p2" if viewer == "p1" else "p1"
        for ev in events:
            if ev[0] == "reveal" and ev[1] == opp:
                _, _, k, kind, name = ev
                c = self.constraints[k]
                if kind == "move":
                    c["moves"].add(name)
                    self._apply(k, lambda p: name in p["moves"], "reveal_move")
                elif kind == "item":
                    c["item"] = name
                    self._apply(k, lambda p: p["item"] == name, "reveal_item")
                elif kind == "ability":
                    c["ability"] = name
                    self._apply(k, lambda p: p["ability"] == name, "reveal_ability")
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
        """Apply one same-priority move-order constraint; return ``None``."""
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
                # our own SP may be hidden too (prep replays: both sheets
                # redact it), so OUR side of the inequality stays one-sided.
                # Their side is exact for archetype particles (concrete
                # spread) and one-sided only for the 'any' catch-all.
                mine_hi = self._my_speed(m_idx, m_ctx, sp=MAX_SP)

                if self.strict_spe:
                    self._apply_list(
                        k, self._strict_speed_mults(k, o_ctx, mine, mine_hi, first),
                        "speed")
                    continue

                def ok(p, mine=mine, mine_hi=mine_hi, first=first, k=k, ctx=o_ctx):
                    theirs = self._particle_speed(k, p, ctx)
                    if theirs is None:
                        return True
                    if first:                  # they acted before my mon
                        if not p.get("evs"):
                            theirs = self._particle_speed(k, p, ctx, sp=MAX_SP)
                        return theirs >= mine
                    return mine_hi >= theirs

                self._apply(k, ok, "speed")

    def _strict_speed_mults(self, k, ctx, mine, mine_hi, first):
        """Interval inversion of one same-priority move-order fact. `first` =
        the opponent's mon acted before my mon (already flipped for Trick
        Room), so their on-field speed >= mine; else <= mine_hi. For each
        free-SP particle we find the speed-SP sub-range consistent with the
        inequality and INTERSECT it into the particle's feasible band, killing
        (multiplier 0) only when the band empties -- i.e. only when no speed
        investment at all could reproduce the observed order. Known-SP
        particles keep the exact single-point test. Our side stays one-sided
        (mine = our slowest, mine_hi = our fastest) because our SP is hidden in
        prep replays. The context (Choice Scarf, Tailwind, paralysis, boosts)
        is baked into _particle_speed, so a mistracked modifier here would show
        up as a false kill -- exactly what the speed-context tests check."""
        mults = []
        for j, p in enumerate(self.particles[k]):
            theirs0 = self._particle_speed(k, p, ctx, sp=0)
            if theirs0 is None:                 # no dex speed -> no evidence
                mults.append(1.0)
                continue
            if not self._free_sp(p):            # concrete spread: exact test
                theirs = self._particle_speed(k, p, ctx)
                mults.append(float(theirs >= mine if first else mine_hi >= theirs))
                continue
            band = self.sp_bounds[k][j]["spe"]
            if first:                           # need theirs(sp) >= mine
                new_lo = next((sp for sp in range(MAX_SP + 1)
                               if self._particle_speed(k, p, ctx, sp=sp) >= mine), None)
                if new_lo is None:              # too slow even fully invested
                    mults.append(0.0)
                    continue
                band[0] = max(band[0], new_lo)
            else:                               # need theirs(sp) <= mine_hi
                new_hi = next((sp for sp in range(MAX_SP, -1, -1)
                               if self._particle_speed(k, p, ctx, sp=sp) <= mine_hi), None)
                if new_hi is None:              # too fast even at 0 SP
                    mults.append(0.0)
                    continue
                band[1] = min(band[1], new_hi)
            mults.append(1.0 if band[0] <= band[1] else 0.0)
        return mults

    def _damage_evidence(self, ev, viewer, opp):
        """Apply one eligible observed-damage constraint; return ``None``."""
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
        their_attack = atk_side == opp    # hidden attack SP raises damage,
        #                                   hidden bulk SP lowers it
        field = {"weather": ctx["weather"], "terrain": ctx["terrain"],
                 "screens": ctx["screens"]}

        if atk_side == opp and self.my_team:      # their attack, our known defender
            k = atk_idx
            d = self.my_team[def_idx]
            dfd = {"species": _sid(d["species"]), "level": d["level"],
                   "item": d["item"], "ability": d["ability"], "nature": d["nature"],
                   "evs": d["evs"], "boosts": ctx["def_boosts"]}
            if self.strict_atk:                   # invert to feasible attack SP
                self._apply_list(
                    k, self._strict_attack_mults(k, move, dfd, field, ctx,
                                                 frac, tol, truncated), "damage_atk")
                return
            reqs = [request(self._as_attacker(k, p, ctx), dfd, move, field)
                    for p in self.particles[k]]
            slacks = [self._atk_slack(k, p, move) for p in self.particles[k]]
        elif atk_side == viewer and self.my_team:  # our attack, their unknown defender
            k = def_idx
            a = self.my_team[atk_idx]
            atk = {"species": _sid(a["species"]), "level": a["level"],
                   "item": a["item"], "ability": a["ability"], "nature": a["nature"],
                   "evs": a["evs"], "boosts": ctx["atk_boosts"],
                   "status": "brn" if ctx["burn"] else ""}
            reqs = [request(atk, self._as_defender(k, p, ctx), move, field)
                    for p in self.particles[k]]
            slacks = [self._bulk_slack(k, p, move) for p in self.particles[k]]
        else:
            return
        res = self.bridge.calc_batch(reqs)

        def ok(r, s):
            if r is None:
                return 1.0
            lo, hi = r
            if their_attack:      # their unknown atk SP can only add
                hi = hi * s
            else:                 # their unknown bulk SP can only subtract
                lo = lo * s
            if truncated:
                return float(hi >= frac - tol)
            return float(lo - tol <= frac <= hi + tol)

        self._apply_list(k, [ok(r, s) for r, s in zip(res, slacks)],
                         "damage_atk" if their_attack else "damage_def")

    def _strict_attack_mults(self, k, move, dfd, field, ctx, frac, tol, truncated):
        """Damage WE took -> per-particle survival multiplier by inverting the
        calc for the opponent's attack SP. Free-SP particles are grouped by
        attacker hypothesis (nature, item, ability); each group is calc'd once
        across an SP grid (the calc is the exact forward oracle, so all its
        truncation is respected), inverted to the feasible attack-SP interval
        consistent with the observed fraction, and that interval is intersected
        into every member's band. A particle dies only when its band empties --
        i.e. when no attack investment at all, for its nature/item/ability,
        could deal the observed damage -- which is the strict kill on attack
        EVs/nature we want, without the coarse-archetype over-kill. Known-SP
        particles keep the exact single-point test. Defensive inference is NOT
        done here (that branch stays on the old slack)."""
        cat = (self.dex or {}).get("moves", {}).get(move, {}).get("category")
        key = "atk" if cat == "Physical" else "spa" if cat == "Special" else None
        particles = self.particles[k]
        mults = [1.0] * len(particles)
        if key is None:                       # status / unknown move: no info
            return mults
        consumed = self.constraints[k]["consumed"]
        grid = list(range(0, MAX_SP + 1, max(1, self.cfg.strict_sp_step)))
        if grid[-1] != MAX_SP:
            grid.append(MAX_SP)

        # concrete redacted-spread archetypes: strict interval inversion.
        groups = {}
        for j, p in enumerate(particles):
            if p.get("arch") not in (None, "any"):
                gkey = (p["nature"], None if consumed else p["item"], p["ability"])
                groups.setdefault(gkey, []).append(j)
        if groups:
            reqs, tags = [], []
            for gkey, js in groups.items():
                p0 = particles[js[0]]
                for sp in grid:
                    reqs.append(request(self._atk_hypo(k, p0, ctx, key, sp),
                                        dfd, move, field))
                    tags.append((gkey, sp))
            res = self.bridge.calc_batch(reqs)
            per_group = {}
            for (gkey, sp), r in zip(tags, res):
                per_group.setdefault(gkey, {})[sp] = r
            for gkey, js in groups.items():
                lo, hi, valid = _feasible_sp_range(per_group[gkey], frac, tol, truncated)
                if not valid:                 # calc gave nothing -> no evidence
                    continue
                for j in js:
                    if lo is None:            # no SP explains it -> kill
                        mults[j] = 0.0
                        continue
                    band = self.sp_bounds[k][j][key]
                    band[0], band[1] = max(band[0], lo), min(band[1], hi)
                    mults[j] = 1.0 if band[0] <= band[1] else 0.0

        # the 'any' catch-all + authored/determinized sets keep the OLD test.
        # 'any' KEEPS its upward investment slack (_atk_slack), which absorbs
        # un-modeled attacker damage boosts -- Supreme Overlord, Analytic, etc.
        # -- so a boosted hit that exceeds the calc's max at every SP still
        # can't drive the true SET to zero (its 'any' particle survives). That
        # cushion is exactly why removing it made strict WORSE than the old path
        # in the audit. Authored sets get slack 1.0 (exact), as before.
        other = [(j, p) for j, p in enumerate(particles)
                 if p.get("arch") in (None, "any")]
        if other:
            # only the attack stat affects OUTGOING damage, so a concrete build
            # (evs set -- the Pikalytics-spread particles) is calc'd with all SP
            # isolated onto `key` at its own attack SP. Builds sharing
            # (nature,item,ability,attack-SP) then collapse to ONE cached calc
            # (covered mons run ~180 concrete builds, most maxing attack), which
            # is the covered-path speedup. The 'any' cushion (evs None) keeps the
            # slack request unchanged.
            reqs = []
            for _, p in other:
                if p.get("evs"):
                    atk = self._atk_hypo(k, p, ctx, key, p["evs"][STAT_KEYS.index(key)])
                else:
                    atk = self._as_attacker(k, p, ctx)
                reqs.append(request(atk, dfd, move, field))
            res = self.bridge.calc_batch(reqs)
            for (j, p), r in zip(other, res):
                if r is None:
                    continue
                hi = r[1] * self._atk_slack(k, p, move)   # slack 1.0 for concrete
                mults[j] = (float(hi >= frac - tol) if truncated
                            else float(r[0] - tol <= frac <= hi + tol))
        return mults

    def _atk_hypo(self, k, p, ctx, stat_key, sp):
        """Attacker hypothesis with all stat points isolated onto the attack
        stat being inverted (the attacker's other stats never affect its
        outgoing damage), so the calc's damage is a clean monotone function of
        `sp` and the inversion is exact."""
        evs = [0] * 6
        evs[STAT_KEYS.index(stat_key)] = sp
        return {"species": self._species_cur(k), "level": 50,
                "item": None if self.constraints[k]["consumed"] else p["item"],
                "ability": p["ability"], "nature": p["nature"], "evs": evs,
                "boosts": ctx["atk_boosts"], "status": "brn" if ctx["burn"] else "",
                "alliesFainted": ctx.get("allies_fainted")}

    def _atk_slack(self, k, p, move):
        """Largest multiplier hidden attack SP (0..MAX_SP, redacted on team
        sheets) can put on this particle's damage: the calc ran at SP 0, and
        damage is linear in the attack stat modulo rounding. Archetype
        particles carry a concrete spread, so the calc is exact: no slack."""
        if p.get("evs"):
            return 1.0
        cat = (self.dex or {}).get("moves", {}).get(move, {}).get("category")
        key = "atk" if cat == "Physical" else "spa"
        base = self._base(self._species_cur(k), key)
        if base is None or not cat:
            return self.cfg.investment_slack
        return calc_stat(base, key, p["nature"], MAX_SP) / calc_stat(base, key, p["nature"])

    def _bulk_slack(self, k, p, move):
        """Smallest factor hidden bulk SP can shrink the observed damage
        fraction to: frac = dmg/maxhp scales as 1/(defense * HP). Exact for
        archetype particles: no slack."""
        if p.get("evs"):
            return 1.0
        cat = (self.dex or {}).get("moves", {}).get(move, {}).get("category")
        key = "def" if cat == "Physical" else "spd"
        sid = self._species_cur(k)
        bd, bh = self._base(sid, key), self._base(sid, "hp")
        if bd is None or bh is None or not cat:
            return 1 / self.cfg.investment_slack
        n = p["nature"]
        return (calc_stat(bd, key, n) * calc_stat(bh, "hp", n)) / (
            calc_stat(bd, key, n, MAX_SP) * calc_stat(bh, "hp", n, MAX_SP))

    def _as_attacker(self, k, p, ctx):
        """Return canonical calc attacker mapping for one particle."""
        return {"species": self._species_cur(k), "level": 50,
                "item": None if self.constraints[k]["consumed"] else p["item"],
                "ability": p["ability"], "nature": p["nature"],
                "evs": p.get("evs"),
                "boosts": ctx["atk_boosts"], "status": "brn" if ctx["burn"] else "",
                "alliesFainted": ctx.get("allies_fainted")}

    def _as_defender(self, k, p, ctx):
        """Return canonical calc defender mapping for one particle."""
        return {"species": self._species_cur(k), "level": 50,
                "item": None if self.constraints[k]["consumed"] else p["item"],
                "ability": p["ability"], "nature": p["nature"],
                "evs": p.get("evs"),
                "boosts": ctx["def_boosts"]}

    def _species_cur(self, k):
        """Return current inferred species id, applying a revealed mega."""
        sp = self.species[k]
        c = self.constraints[k]
        if not c["mega"]:
            return sp
        stone = (self.dex or {}).get("items", {}).get(c["item"] or "", {}).get("megaStone")
        return (stone or {}).get(sp) or sp + "mega"

    # -- outputs ------------------------------------------------------------
    def top_particle(self, k):
        """Return the highest-weight ``ParticleSet`` at team index ``k``."""
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
            speeds = [(self._particle_speed(k, p, j=j) or 0, w)
                      for j, (p, w) in enumerate(zip(ps, ws))]
            speeds.sort()
            lo = _quantile(speeds, 0.05)
            hi = _quantile(speeds, 0.95)
            bulk = 0.0
            hp_b, d_b, sd_b = (self._base(sp, s) for s in ("hp", "def", "spd"))
            if hp_b is not None:
                for p, w in zip(ps, ws):
                    evs = p.get("evs") or [0] * 6
                    hp = calc_stat(hp_b, "hp", p["nature"], evs[0])
                    df = calc_stat(d_b, "def", p["nature"], evs[2])
                    sd = calc_stat(sd_b, "spd", p["nature"], evs[4])
                    bulk += w * hp * (df + sd) / 2
            arch, p_arch = self.arch_posterior(k)[0]
            nat, p_nat = self.nature_posterior(k)[0]
            out[k] = {"item": item, "p_item": p_item,
                      "spe_lo": lo, "spe_hi": hi, "bulk": bulk,
                      "arch": arch, "p_arch": p_arch,
                      "nature": nat, "p_nature": p_nat}
        return out

    def arch_posterior(self, k):
        """[(archetype, prob)] sorted by prob. Particles with an authored /
        sampled concrete spread count under 'any' (their spread is known, an
        archetype label adds nothing)."""
        acc = {}
        for w, p in zip(self.weights[k], self.particles[k]):
            acc[p.get("arch") or "any"] = acc.get(p.get("arch") or "any", 0.0) + w
        return sorted(acc.items(), key=lambda kv: -kv[1])

    def nature_posterior(self, k):
        """[(nature, prob)] sorted by prob. The inferred nature sub-dimension:
        for a covered mon this is the real posterior (which Pikalytics build
        survived the speed/damage facts); for an uncovered mon every particle is
        the redacted 'serious', so it collapses to serious (honestly unknown)."""
        acc = {}
        for w, p in zip(self.weights[k], self.particles[k]):
            acc[p["nature"]] = acc.get(p["nature"], 0.0) + w
        return sorted(acc.items(), key=lambda kv: -kv[1])

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
                j = rng.choices(range(len(self.particles[k])),
                                weights=self.weights[k])[0]
                p = self.particles[k][j]
                s = {"species": sp, "moves": list(p["moves"]),
                     "item": p["item"], "ability": p["ability"],
                     "nature": p["nature"]}
                evs = self._sampled_evs(k, j, p)
                if evs is not None:
                    s["evs"] = evs
                team.append(s)
            teams.append(team)
        return teams

    def _sampled_evs(self, k, j, p):
        """Concrete spread for one determinized sample. A free-SP particle's
        inferred stats (attack + speed) are drawn from the midpoints of their
        narrowed feasible bands so the sampled opponent is consistent with the
        evidence; defensive stats come from the archetype. Non-free particles
        pass their known spread through unchanged."""
        base = list(p["evs"]) if p.get("evs") else None
        if not self._free_sp(p) or not (self.strict_spe or self.strict_atk):
            return base
        evs = base if base is not None else [0] * 6
        b = self.sp_bounds[k][j]
        if self.strict_spe:
            evs[5] = sum(b["spe"]) // 2
        if self.strict_atk:                    # only the stat this set invests
            att = _att_key(self.species[k], p["nature"], self.dex)
            evs[STAT_KEYS.index(att)] = sum(b[att]) // 2
        return evs


def determinized(sets, cfg=CFG):
    """A collapsed belief: one particle per mon, weight 1. The search uses it
    inside a determinization, where the opponent's sets are fixed, so
    summary()/top_particle() feed the tokenizer the sampled 'truth' through
    the same interface the real filter uses."""
    usage = {_sid(s["species"]): [(1, list(s["moves"]), s["item"],
                                   s["ability"], s["nature"], s.get("evs"))]
             for s in sets}
    return OpponentBelief([_sid(s["species"]) for s in sets], usage, cfg)


def _feasible_sp_range(results, frac, tol, truncated):
    """results: {sp: (min_frac, max_frac) | None} from the calc across an SP
    grid. Returns (lo, hi, any_valid): the smallest and largest grid SP whose
    damage roll-range is consistent with the observed `frac` (within `tol`).
    Damage is monotone in SP, so the consistent set is a contiguous run; any
    holes left by roll truncation are bridged (first..last consistent SP),
    the conservative choice that never kills a particle a hole would spare.
    lo is None with any_valid True means no SP is consistent -> kill; any_valid
    False means the calc returned nothing usable -> treat as no evidence."""
    lo = hi = None
    any_valid = False
    for sp in sorted(results):
        r = results[sp]
        if r is None:
            continue
        any_valid = True
        mn, mx = r
        ok = (mx >= frac - tol) if truncated else (mx >= frac - tol and mn <= frac + tol)
        if ok:
            if lo is None:
                lo = sp
            hi = sp
    return lo, hi, any_valid


def _quantile(sorted_pairs, q):
    """Return a weighted quantile from ascending ``(value, weight)`` pairs."""
    acc = 0.0
    for v, w in sorted_pairs:
        acc += w
        if acc >= q:
            return v
    return sorted_pairs[-1][0] if sorted_pairs else 0


def _sid(name):
    """Return lowercase alphanumeric Showdown id."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------------------
# `python beliefs.py --audit [max_battles]` — replay held-out battles through
# the filter and measure how often it breaks down (the module-docstring
# question). Run after `data.py parse`, on the training box.
# ---------------------------------------------------------------------------

def _set_key(s):
    """Return hashable full hidden-set identity including nature."""
    # team sheets redact stat points, so identity is (moves, item, ability, nature)
    return (tuple(sorted(s["moves"])), s["item"], s["ability"], s["nature"])


def _id_key(s):
    """Nature-free identity (moves, item, ability). The dataset ALSO redacts
    nature (every oracle set is 'serious'), and the spreads prior now hypothesizes
    real natures per particle, so oracle grading must drop nature -- otherwise a
    correct discrete set with an inferred Jolly nature would count as 'not in
    prior' against the serious placeholder. Used only for --audit oracle tracking."""
    return (tuple(sorted(s["moves"])), s["item"], s["ability"])


def audit(max_battles, cfg=CFG):
    """Print held-out posterior, depletion, and latency metrics."""
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

    # stream the test-split records and stop once we have max_battles, so the
    # 1.15GB bo3 pickle is never fully loaded (that pickle.load was its own OOM)
    from data import iter_battles
    paths = [cfg.parsed_dir / f"{fn[len('logs_'):-len('.json')]}.pkl"
             for fn in cfg.dataset_files]
    battles = []
    for rec in iter_battles(*paths):
        if rec["split"] == "test":
            battles.append(rec)
            if len(battles) >= max_battles:
                break

    mons = soft = hard = in_prior = top1 = depleted_battles = 0
    oracle_mass = 0.0
    by_species = Counter()
    soft_by_cause, hard_by_cause = Counter(), Counter()
    oracle_loss_by_cause, oracle_kills_by_cause = Counter(), Counter()
    t0 = time.perf_counter()
    for rec in battles:
        battle_depleted = False
        for p in ("p1", "p2"):
            opp = "p2" if p == "p1" else "p1"
            oracle = rec["teams"][opp]
            bel = OpponentBelief([_sid(s["species"]) for s in oracle], usage,
                                 cfg, bridge, my_team=rec["teams"][p])
            # attribute what collapses each mon's TRUE set mass (see _apply_list).
            # nature-free identity: the oracle's nature/SP are redacted, and the
            # spreads prior hypothesizes real natures per particle.
            bel.oracle_keyfn = _id_key
            bel.oracle_keys = {k: _id_key(s) for k, s in enumerate(oracle)}
            for turn in rec["turns"]:
                bel.update(turn["events"], viewer=p)
            soft_by_cause.update(bel.soft_by_cause)
            hard_by_cause.update(bel.hard_by_cause)
            oracle_loss_by_cause.update(bel.oracle_loss_by_cause)
            oracle_kills_by_cause.update(bel.oracle_kills_by_cause)
            for k, s in enumerate(oracle):
                mons += 1
                # archetype expansion: several particles share one set
                # identity, so grade the set's TOTAL mass across its variants
                mass_by_key = {}
                for pt, w in zip(bel.particles[k], bel.weights[k]):
                    key = _id_key(pt)
                    mass_by_key[key] = mass_by_key.get(key, 0.0) + w
                ok = _id_key(s) in mass_by_key
                in_prior += ok
                if ok:
                    m = mass_by_key[_id_key(s)]
                    oracle_mass += m
                    top1 += m == max(mass_by_key.values())
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

    # --- which lever dominates -------------------------------------------
    # soft/hard depletions attributed to the evidence channel that caused
    # them. reveal_* => prior-coverage lever (widen/factor the prior);
    # speed/damage => archetype-sharpness lever (soften evidence).
    def _fmt(counter):
        tot = sum(counter.values()) or 1
        return "  ".join(f"{c}={n} ({n / tot:.0%})"
                         for c, n in counter.most_common())
    print("\nsoft depletions by cause: " + (_fmt(soft_by_cause) or "none"))
    print("hard depletions by cause: " + (_fmt(hard_by_cause) or "none"))
    # what collapses the TRUE set's mass when it IS present. Any weight on
    # reveal_* here is a BUG (the oracle set contains every revealed move):
    # suspect name normalization / mega form / consumed-item / ability
    # attribution. speed/damage weight is the archetype-sharpness lever.
    print("\noracle-set mass loss by cause (relative mass lost, when present):")
    print("  " + (_fmt(oracle_loss_by_cause) or "none"))
    print("oracle set driven to ~0 by cause (should have NO reveal_*):")
    print("  " + (_fmt(oracle_kills_by_cause) or "none"))
    if bridge:
        bridge.close()


if __name__ == "__main__":
    if "--audit" in sys.argv:
        i = sys.argv.index("--audit")
        audit(int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 500)
