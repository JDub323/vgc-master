"""Tests for phase-3.1 strict SP inversion (beliefs.py):

  * speed:  move-order fact -> feasible speed-SP interval (pure Python, no node)
  * attack: damage WE take -> feasible attack-SP interval (needs the node calc)

The speed suite is the "context audit" the plan called for: it exercises
Choice Scarf, Tailwind, paralysis, boosts and Trick Room and checks that every
particle the filter keeps/kills is EXACTLY the set that is consistent/inconsistent
with the move order, computed independently. A mistracked speed modifier would
show up as a keep/kill that disagrees with the brute-force truth.

Run:  python tests/test_strict_inversion.py           (speed + range: no node)
      python tests/test_strict_inversion.py --bridge   (also the attack calc)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataclasses

from config import CFG as _CFG
from beliefs import (OpponentBelief, calc_stat, boost_mult, load_dex,
                     _feasible_sp_range, _set_key, MAX_SP)

# These suites validate the ARCHETYPE + neutral-nature SP-interval machinery,
# which is now the fallback path for species NOT covered by spreads.json. Some
# use covered species (kingambit, garchomp), so pin spreads_prior off here to
# exercise the interval path deterministically; the covered-mon (spreads/nature)
# path has its own suite in test_spreads_nature.py.
CFG = dataclasses.replace(_CFG, spreads_prior=False)

DEX = load_dex(CFG)
NEUTRAL = {"spe": 0, "tw": False, "par": False}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def base_spe(species):
    return DEX["species"][species]["baseStats"]["spe"]


def indep_speed(base, nature, sp, scarf, ctx):
    """Independent re-derivation of on-field speed, mirroring the documented
    Champions formula + modifier order. Used as the ground truth the filter's
    keep/kill decisions are checked against."""
    spe = calc_stat(base, "spe", nature, sp)
    if scarf:
        spe = int(spe * 1.5)
    spe *= boost_mult(ctx["spe"])
    if ctx["tw"]:
        spe *= 2
    if ctx["par"]:
        spe *= 0.5
    return spe


def make_belief(species, opp_sets, my_team, bridge=None):
    """species: the ONE opponent mon (sid, must be in dex).
    opp_sets: list of (moves, item, ability, nature).
    my_team: list of dicts (species sid, nature, item, evs)."""
    usage = {species: [[10, list(mv), it, ab, na]
                       for (mv, it, ab, na) in opp_sets]}
    return OpponentBelief([species], usage, CFG, bridge, my_team=my_team)


def order_event(opp_move, my_move, first, my_ctx=NEUTRAL, opp_ctx=NEUTRAL,
                tr=False):
    """One same-priority move-order event. `first`=opponent's mon moved before
    mine. viewer is p1, so my mon is p1 and the opponent is p2."""
    opp = ("p2", 0, opp_move, dict(opp_ctx))
    me = ("p1", 0, my_move, dict(my_ctx))
    order = [opp, me] if first else [me, opp]
    return ("move_order", order, {"tr": tr})


def alive(bel, j):
    return bel.weights[0][j] > 1e-9


# a real same-priority pair of damaging moves that exist in the dex
OPP_MOVE, MY_MOVE = "ironhead", "earthquake"


def _priority(m):
    return DEX["moves"].get(m, {}).get("priority", 0)


assert _priority(OPP_MOVE) == _priority(MY_MOVE) == 0, "test moves must share priority"


# --------------------------------------------------------------------------- #
# _feasible_sp_range unit tests (pure logic, no calc)
# --------------------------------------------------------------------------- #

def test_feasible_range_contiguous():
    # min/max_frac rise with sp; observed 0.50 sits in sp 2..4's ranges
    res = {0: (0.30, 0.36), 2: (0.44, 0.52), 4: (0.48, 0.57), 6: (0.60, 0.70)}
    lo, hi, valid = _feasible_sp_range(res, 0.50, 0.0, truncated=False)
    assert valid and (lo, hi) == (2, 4), (lo, hi)
    print("ok  _feasible_sp_range contiguous interval")


def test_feasible_range_truncated_is_lower_bound():
    # a KO caps the observed fraction: only max_frac >= obs matters -> [lo, top]
    res = {0: (0.30, 0.36), 2: (0.44, 0.52), 4: (0.48, 0.60)}
    lo, hi, valid = _feasible_sp_range(res, 0.50, 0.0, truncated=True)
    assert valid and (lo, hi) == (2, 4), (lo, hi)
    print("ok  _feasible_sp_range truncated -> lower-bound only")


def test_feasible_range_none_and_empty():
    lo, hi, valid = _feasible_sp_range({0: None, 2: None}, 0.5, 0.0, False)
    assert not valid and lo is None, "all-None must be 'no evidence'"
    lo, hi, valid = _feasible_sp_range({0: (0.1, 0.2), 2: (0.2, 0.3)}, 0.9, 0.0, False)
    assert valid and lo is None, "no consistent sp must report kill"
    print("ok  _feasible_sp_range none/empty handling")


def test_feasible_range_bridges_holes():
    # sp 2 inconsistent (a truncation hole) between two consistent points:
    # we bridge first..last consistent -> conservative, never over-kills
    res = {0: (0.48, 0.52), 2: (0.60, 0.61), 4: (0.49, 0.53)}
    lo, hi, valid = _feasible_sp_range(res, 0.50, 0.0, truncated=False)
    assert (lo, hi) == (0, 4), (lo, hi)
    print("ok  _feasible_sp_range bridges truncation holes")


# --------------------------------------------------------------------------- #
# speed inversion: soundness/completeness sweep (the context audit)
# --------------------------------------------------------------------------- #

def check_speed_consistency(opp_species, my_species, opp_sets, my_nat,
                            opp_ctx, my_ctx, first, tr):
    """Test the inversion FUNCTION directly (before soft-depletion / resample
    can rebuild prior mass and hide a kill): every free-SP particle gets a
    survival multiplier IFF some speed SP in [0,MAX] reproduces the observed
    order under the SAME context, computed independently. This is the core
    correctness + context audit."""
    my_team = [{"species": my_species, "nature": my_nat, "item": "",
                "evs": [0] * 6, "level": 50}]
    bel = make_belief(opp_species, opp_sets, my_team)
    obase = base_spe(opp_species)
    mbase = base_spe(my_species)
    # our side is one-sided: mine=our slowest (sp 0), mine_hi=our fastest (MAX)
    mine = indep_speed(mbase, my_nat, 0, False, my_ctx)
    mine_hi = indep_speed(mbase, my_nat, MAX_SP, False, my_ctx)
    eff_first = (not first) if tr else first
    mults = bel._strict_speed_mults(0, dict(opp_ctx), mine, mine_hi, eff_first)
    for j, p in enumerate(bel.particles[0]):
        scarf = p["item"] == "choicescarf"
        feasible = any(
            (indep_speed(obase, p["nature"], sp, scarf, opp_ctx) >= mine)
            if eff_first else
            (mine_hi >= indep_speed(obase, p["nature"], sp, scarf, opp_ctx))
            for sp in range(MAX_SP + 1))
        assert (mults[j] > 0) == feasible, (
            f"particle {j} nat={p['nature']} scarf={scarf} eff_first={eff_first} "
            f"ctx={opp_ctx}: mult={mults[j]} but feasible={feasible}")


def test_speed_context_sweep():
    """Sweep scarf/tailwind/paralysis/boosts/TR and both order directions
    across a slow and a fast opponent; every keep/kill must match truth."""
    opp_sets = [
        (["ironhead", "suckerpunch"], "leftovers", "supremeoverlord", "adamant"),
        (["ironhead", "suckerpunch"], "choicescarf", "supremeoverlord", "jolly"),
        (["ironhead", "suckerpunch"], "leftovers", "supremeoverlord", "brave"),  # -spe
    ]
    combos = 0
    for opp_species in ("kingambit", "dragapult"):
        for my_species in ("incineroar", "dragonite", "blissey"):
            for first in (True, False):
                for tr in (False, True):
                    for opp_ctx in (NEUTRAL, {"spe": 1, "tw": False, "par": False},
                                    {"spe": 0, "tw": True, "par": False},
                                    {"spe": 0, "tw": False, "par": True}):
                        check_speed_consistency(
                            opp_species, my_species, opp_sets, "serious",
                            opp_ctx, NEUTRAL, first, tr)
                        combos += 1
    print(f"ok  speed context sweep: {combos} scenarios, keep/kill all match truth")


def test_speed_scarf_differential():
    """Concrete, hand-checked: a mon fast enough that a non-scarf variant can
    NEVER outspeed (dead) but a scarf variant can (alive), with the expected
    narrowed band. Proves scarf context is actually applied."""
    opp_species, my_species = "kingambit", "garchomp"    # 50 vs 102 base spe
    obase, mbase = base_spe(opp_species), base_spe(my_species)
    mine = calc_stat(mbase, "spe", "serious", 0)          # our slowest
    # precondition: non-scarf maxed < mine <= scarf can reach
    assert calc_stat(obase, "spe", "serious", MAX_SP) < mine, "pick a faster my_species"
    assert int(calc_stat(obase, "spe", "serious", MAX_SP) * 1.5) >= mine
    opp_sets = [
        (["ironhead"], "leftovers", "supremeoverlord", "serious"),
        (["ironhead"], "choicescarf", "supremeoverlord", "serious"),
    ]
    my_team = [{"species": my_species, "nature": "serious", "item": "",
                "evs": [0] * 6, "level": 50}]
    bel = make_belief(opp_species, opp_sets, my_team)
    bel.update([order_event(OPP_MOVE, MY_MOVE, first=True)], viewer="p1")
    for j, p in enumerate(bel.particles[0]):
        scarf = p["item"] == "choicescarf"
        assert alive(bel, j) == scarf, f"particle {j} scarf={scarf} alive={alive(bel,j)}"
        if scarf:                                   # band lo = min sp to outspeed
            need = next(sp for sp in range(MAX_SP + 1)
                        if int(calc_stat(obase, "spe", "serious", sp) * 1.5) >= mine)
            assert bel.sp_bounds[0][j]["spe"][0] == need, bel.sp_bounds[0][j]["spe"]
    print("ok  speed scarf differential (non-scarf dies, scarf survives, band narrowed)")


def test_speed_tailwind_revives():
    """Directional invariant: if opponent acted first, adding THEIR tailwind
    (doubles their speed) can only make more particles feasible, never fewer.
    Checked on the raw multipliers so depletion/resample can't confound it."""
    opp_sets = [(["ironhead"], "leftovers", "supremeoverlord", n)
                for n in ("adamant", "modest", "brave", "timid")]
    mbase = base_spe("dragonite")
    mine = calc_stat(mbase, "spe", "serious", 0)
    mine_hi = calc_stat(mbase, "spe", "serious", MAX_SP)
    my_team = [{"species": "dragonite", "nature": "serious", "item": "",
                "evs": [0] * 6, "level": 50}]
    m_no = make_belief("kingambit", opp_sets, my_team)._strict_speed_mults(
        0, NEUTRAL, mine, mine_hi, first=True)
    m_tw = make_belief("kingambit", opp_sets, my_team)._strict_speed_mults(
        0, {"spe": 0, "tw": True, "par": False}, mine, mine_hi, first=True)
    for j in range(len(m_no)):
        assert m_tw[j] > 0 or m_no[j] == 0, \
            "tailwind (faster) must not kill a particle that survived without it"
    assert sum(m > 0 for m in m_tw) >= sum(m > 0 for m in m_no)
    print("ok  speed tailwind directional invariant (faster -> not fewer survivors)")


def test_speed_tr_flip_matches():
    """The Trick Room flip lives in _speed_evidence: updating with (order-first
    = False, tr=True) must land the SAME feasible speed bands as (order-first =
    True, tr=False). sp_bounds are set by the inversion and untouched by
    resample, so this isolates the flip."""
    opp_sets = [(["ironhead"], "leftovers", "supremeoverlord", "serious"),
                (["ironhead"], "choicescarf", "supremeoverlord", "timid")]
    my_team = [{"species": "incineroar", "nature": "serious", "item": "",
                "evs": [0] * 6, "level": 50}]
    b1 = make_belief("kingambit", opp_sets, my_team)
    b1.update([order_event(OPP_MOVE, MY_MOVE, first=False, tr=True)], viewer="p1")
    b2 = make_belief("kingambit", opp_sets, my_team)
    b2.update([order_event(OPP_MOVE, MY_MOVE, first=True, tr=False)], viewer="p1")
    assert [b["spe"] for b in b1.sp_bounds[0]] == [b["spe"] for b in b2.sp_bounds[0]], \
        "Trick Room flip must equal the un-flipped opposite order"
    print("ok  speed Trick Room flip matches un-flipped opposite order")


def test_speed_impossible_kills_all():
    """Opponent that is slower than my mon even fully invested, but acted
    first: no SP explains it -> every free particle dies (soft depletion)."""
    opp_sets = [(["ironhead"], "leftovers", "supremeoverlord", "serious")]
    my_team = [{"species": "dragapult", "nature": "serious", "item": "",
                "evs": [0] * 6, "level": 50}]           # 142 base, very fast
    assert calc_stat(base_spe("kingambit"), "spe", "serious", MAX_SP) < \
        calc_stat(base_spe("dragapult"), "spe", "serious", 0)
    bel = make_belief("kingambit", opp_sets, my_team)
    bel.update([order_event(OPP_MOVE, MY_MOVE, first=True)], viewer="p1")
    assert bel.soft_depletions[0] > 0, "impossible order should deplete"
    assert bel.soft_by_cause["speed"] > 0, bel.soft_by_cause
    print("ok  speed impossible order -> soft depletion tagged 'speed'")


def test_speed_off_recovers_old_path():
    """strict_speed_ev False must still run (the old archetype binary path)."""
    import dataclasses
    cfg = dataclasses.replace(CFG, strict_speed_ev=False)
    opp_sets = [(["ironhead"], "leftovers", "supremeoverlord", "serious")]
    my_team = [{"species": "incineroar", "nature": "serious", "item": "",
                "evs": [0] * 6, "level": 50}]
    bel = OpponentBelief(["kingambit"], {"kingambit": [[10, ["ironhead"],
                         "leftovers", "supremeoverlord", "serious"]]},
                         cfg, None, my_team=my_team)
    bel.update([order_event(OPP_MOVE, MY_MOVE, first=True)], viewer="p1")
    assert sum(bel.weights[0]) > 0
    print("ok  strict_speed_ev=False falls back to old path cleanly")


# --------------------------------------------------------------------------- #
# attack inversion round-trip (needs the node calc bridge)
# --------------------------------------------------------------------------- #

def _dmg_event(move, frac, def_hp_before=1.0):
    ctx = {"crit": False, "spread": False, "multi": False, "burn": False,
           "weather": None, "terrain": None, "atk_boosts": {}, "def_boosts": {},
           "screens": [], "def_hp_before": def_hp_before, "def_transformed": False}
    # ("dmg", atk_side, atk_idx, move, def_side, def_idx, frac, ctx)
    return ("dmg", "p2", 0, move, "p1", 0, frac, ctx)


def _one_frac(bridge, atk_species, nature, item, ability, sp, key,
              defender, move):
    from beliefs import STAT_KEYS
    from damage import request
    evs = [0] * 6
    evs[STAT_KEYS.index(key)] = sp
    atk = {"species": atk_species, "level": 50, "item": item, "ability": ability,
           "nature": nature, "evs": evs}
    return bridge.calc_batch([request(atk, defender, move, {})])[0]


def test_attack_roundtrip(bridge):
    """Pick a TRUE attack SP, produce the damage the calc says it deals, feed
    it, and assert (a) the true set survives with the true SP inside its band,
    (b) the band actually narrowed, (c) every surviving free particle is
    genuinely consistent and every killed one is genuinely inconsistent."""
    atk_species, move, key = "kingambit", "ironhead", "atk"
    nature, item, ability = "adamant", "leftovers", "supremeoverlord"
    # a physically bulky, steel-neutral wall: survives the hit and its damage
    # is clearly SP-sensitive, so the band must narrow above 0.
    defender = {"species": "hippowdon", "level": 50, "item": "", "ability": "",
                "nature": "impish", "evs": [0] * 6, "boosts": {}}
    true_sp = MAX_SP
    rng = _one_frac(bridge, atk_species, nature, item, ability, true_sp, key,
                    defender, move)
    rng0 = _one_frac(bridge, atk_species, nature, item, ability, 0, key,
                     defender, move)
    assert rng is not None and rng0 is not None, "calc failed"
    frac = rng[1]                                   # observed = a max roll
    assert frac < 1.0, f"defender must survive (non-truncated); got {frac}"
    assert rng0[1] < frac - CFG.damage_tolerance, \
        f"pick a bulkier defender so SP matters: sp0 max {rng0[1]} vs obs {frac}"

    opp_sets = [(["ironhead", "suckerpunch"], item, ability, nature),
                (["ironhead", "suckerpunch"], item, ability, "modest")]  # -atk
    my_team = [{"species": "hippowdon", "nature": "impish", "item": "",
                "evs": [0] * 6, "ability": "", "level": 50}]
    bel = OpponentBelief([atk_species], {atk_species: [[10, mv, it, ab, na]
                         for (mv, it, ab, na) in opp_sets]}, CFG, bridge,
                         my_team=my_team)
    assert bel.strict_atk, "bridge present -> strict_atk on"
    bel.update([_dmg_event(move, frac)], viewer="p1")

    for j, p in enumerate(bel.particles[0]):
        if p.get("arch") in (None, "any"):
            continue                       # 'any'/authored use the slack path
        band = bel.sp_bounds[0][j]["atk"]
        # independent feasibility: does any sp reproduce frac for this hypo?
        feas_sp = [sp for sp in range(0, MAX_SP + 1, CFG.strict_sp_step)
                   if (lambda r: r and r[0] - CFG.damage_tolerance <= frac
                       <= r[1] + CFG.damage_tolerance)(
                       _one_frac(bridge, atk_species, p["nature"], item, ability,
                                 sp, key, defender, move))]
        assert alive(bel, j) == bool(feas_sp), (
            f"j={j} nat={p['nature']} alive={alive(bel,j)} feas={feas_sp}")
        if alive(bel, j):
            assert band[0] <= min(feas_sp) and band[1] >= max(feas_sp) - CFG.strict_sp_step, \
                (band, feas_sp)

    adamant_alive = any(alive(bel, j) for j, p in enumerate(bel.particles[0])
                        if p["nature"] == "adamant")
    modest_alive = any(alive(bel, j) for j, p in enumerate(bel.particles[0])
                       if p["nature"] == "modest")
    assert adamant_alive, "the TRUE (adamant) set must survive its own damage"
    band_true = next(bel.sp_bounds[0][j]["atk"] for j, p in enumerate(bel.particles[0])
                     if p["nature"] == "adamant" and p.get("arch") not in (None, "any"))
    assert band_true[0] > 0, f"a big hit should raise the low bound: {band_true}"
    print(f"ok  attack round-trip: true set survives (band {band_true}), "
          f"modest {'survives' if modest_alive else 'killed'}")


def test_attack_defender_side_untouched(bridge):
    """Our attack on the opponent's mon must NOT invoke strict inversion (the
    defensive-EV branch stays on the old slack): sp_bounds unchanged."""
    my_team = [{"species": "kingambit", "nature": "adamant", "item": "",
                "ability": "supremeoverlord", "evs": [0] * 6, "level": 50}]
    bel = OpponentBelief(["blissey"], {"blissey": [[10, ["seismictoss"],
                         "leftovers", "naturalcure", "bold"]]}, CFG, bridge,
                         my_team=my_team)
    before = [dict((s, list(v)) for s, v in b.items()) for b in bel.sp_bounds[0]]
    # ("dmg", atk_side=p1 (us), atk_idx, move, def_side=p2 (them), def_idx, frac, ctx)
    ctx = {"crit": False, "spread": False, "multi": False, "burn": False,
           "weather": None, "terrain": None, "atk_boosts": {}, "def_boosts": {},
           "screens": [], "def_hp_before": 1.0, "def_transformed": False}
    bel.update([("dmg", "p1", 0, "ironhead", "p2", 0, 0.25, ctx)], viewer="p1")
    after = [dict((s, list(v)) for s, v in b.items()) for b in bel.sp_bounds[0]]
    assert before == after, "defender-side damage must not touch attack-SP bands"
    print("ok  attack: defender-side (our attack) leaves SP bands untouched")


# --------------------------------------------------------------------------- #
# integration: a normal update still produces the stable summary schema
# --------------------------------------------------------------------------- #

def test_summary_schema_stable():
    opp_sets = [(["ironhead", "suckerpunch"], "leftovers", "supremeoverlord", "adamant")]
    my_team = [{"species": "dragonite", "nature": "serious", "item": "",
                "evs": [0] * 6, "level": 50}]
    bel = make_belief("kingambit", opp_sets, my_team)
    bel.update([order_event(OPP_MOVE, MY_MOVE, first=True)], viewer="p1")
    s = bel.summary()[0]
    assert set(s) == {"item", "p_item", "spe_lo", "spe_hi", "bulk", "arch",
                      "p_arch", "nature", "p_nature"}, s
    assert s["spe_lo"] <= s["spe_hi"]
    teams = bel.sample_sets(3)
    assert len(teams) == 3 and all(len(t) == 1 for t in teams)
    print("ok  summary schema unchanged + sample_sets works under strict")


# --------------------------------------------------------------------------- #
# factored / stitched depletion fallback
# --------------------------------------------------------------------------- #

# two train sets whose moveset and item never co-occur:
_SET_A = (["ironhead", "suckerpunch", "kowtowcleave", "protect"],
          "leftovers", "supremeoverlord", "adamant")
_SET_B = (["swordsdance", "irondefense", "stealthrock", "taunt"],
          "airballoon", "supremeoverlord", "jolly")
# the TRUE set = A's moves stitched onto B's item (in NO joint train set):
_ORACLE_KEY = (tuple(sorted(_SET_A[0])), "airballoon", "supremeoverlord", "adamant")


def _stitch_belief(cfg):
    usage = {"kingambit": [[10, list(_SET_A[0]), _SET_A[1], _SET_A[2], _SET_A[3]],
                           [8, list(_SET_B[0]), _SET_B[1], _SET_B[2], _SET_B[3]]]}
    bel = OpponentBelief(["kingambit"], usage, cfg, None, my_team=[])
    # reveal one distinguishing move of A + B's item -> no joint set survives
    bel.update([("reveal", "p2", 0, "move", "ironhead"),
                ("reveal", "p2", 0, "item", "airballoon")], viewer="p1")
    return bel


def _oracle_mass(bel):
    return sum(w for w, p in zip(bel.weights[0], bel.particles[0])
              if _set_key(p) == _ORACLE_KEY)


def test_factored_fallback_stitches_oracle():
    """A hard depletion whose true set is a never-seen COMBINATION of two train
    sets is representable after stitching; the old raw-prior fallback can't."""
    import dataclasses
    on = _stitch_belief(dataclasses.replace(CFG, factored_fallback=True))
    assert on.hard_depletions[0] > 0, "reveals should force hard depletion"
    assert _oracle_mass(on) > 0, "stitched fallback must contain the true combo"
    assert abs(sum(on.weights[0]) - 1.0) < 1e-9, "weights stay normalized"
    off = _stitch_belief(dataclasses.replace(CFG, factored_fallback=False))
    assert off.hard_depletions[0] > 0
    assert _oracle_mass(off) == 0, "raw-prior fallback cannot hold the combo"
    print("ok  factored fallback stitches the true combo (raw prior cannot)")


def test_factored_fallback_novel_move_is_graceful():
    """A genuinely novel move exists in no bucket -> no stitch possible -> fall
    back to the raw prior without crashing, weights still valid."""
    import dataclasses
    cfg = dataclasses.replace(CFG, factored_fallback=True)
    usage = {"kingambit": [[10, list(_SET_A[0]), _SET_A[1], _SET_A[2], _SET_A[3]]]}
    bel = OpponentBelief(["kingambit"], usage, cfg, None, my_team=[])
    bel.update([("reveal", "p2", 0, "move", "boomburst")], viewer="p1")  # never seen
    assert bel.hard_depletions[0] > 0
    assert abs(sum(bel.weights[0]) - 1.0) < 1e-9
    print("ok  factored fallback degrades to raw prior on a genuinely novel move")


# --------------------------------------------------------------------------- #

PURE = [test_feasible_range_contiguous, test_feasible_range_truncated_is_lower_bound,
        test_feasible_range_none_and_empty, test_feasible_range_bridges_holes,
        test_speed_context_sweep, test_speed_scarf_differential,
        test_speed_tailwind_revives, test_speed_tr_flip_matches,
        test_speed_impossible_kills_all,
        test_speed_off_recovers_old_path, test_summary_schema_stable,
        test_factored_fallback_stitches_oracle,
        test_factored_fallback_novel_move_is_graceful]

BRIDGE = [test_attack_roundtrip, test_attack_defender_side_untouched]


if __name__ == "__main__":
    assert DEX, "needs artifacts/dex.json (python env.py --dump-dex)"
    for t in PURE:
        t()
    if "--bridge" in sys.argv:
        from damage import DamageBridge
        bridge = DamageBridge(CFG)
        try:
            for t in BRIDGE:
                t(bridge)
        finally:
            bridge.close()
    else:
        print("(skipping attack-calc tests; pass --bridge to run them)")
    print("all strict-inversion tests passed")
