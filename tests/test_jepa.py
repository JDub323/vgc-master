"""Standalone smoke/contract tests for the JEPA world-model experiment.

Runs without Node or trained weights: it builds the vocab from artifacts (or a
tiny synthetic dex), exercises feature extraction, a full model forward, the
matrix solver, an end-to-end ``choose`` against a mocked tracker/belief, and one
training step on a synthetic shard. Run: ``python tests/test_jepa.py``.
"""

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actions import N_SLOT_ACTIONS
from agents.jepa_world_model.v1 import JEPAWorldModelChooser
from config import CFG
from jepa.config import JEPAConfig
from jepa.features import FeatureExtractor, N_MON_SCALAR, action_arrays
from jepa.solver import solve_matrix
from jepa.vocab import JEPAVocab, STATUSES, TERRAINS, WEATHERS
from models.jepa_wm import JEPAWorldModel

JCFG = JEPAConfig(d_model=96, d_ff=192, n_enc_layers=2, n_pred_layers=1,
                  n_heads=4, n_determinizations=2, use_damage_features=False)


def _vocab():
    """Build a vocab from artifacts, or a tiny synthetic dex if none exist."""
    v = JEPAVocab.build(CFG)
    if len(v.species) > 4:
        return v
    dex = {"species": {f"mon{i}": {"baseStats": {"hp": 80, "atk": 100,
            "def": 80, "spa": 100, "spd": 80, "spe": 90}, "types": ["Water"]}
            for i in range(12)},
           "moves": {f"mv{i}": {"priority": 0, "category": "Physical",
            "basePower": 80, "type": "Water", "target": "normal"}
            for i in range(12)},
           "items": {}}
    return JEPAVocab([f"mon{i}" for i in range(12)],
                     [f"mv{i}" for i in range(12)], ["static"], ["static"], dex)


def _mon_set(name, species, moves):
    """Return a minimal own-team ``PokemonSet``."""
    return {"name": name, "species": species, "item": "", "ability": "static",
            "moves": moves, "nature": "serious", "evs": [0] * 6, "gender": "N",
            "level": 50}


def _own_view(k, species, moves, active_slot):
    """Build a ``MonOwnView`` dict for team index ``k``."""
    return {"team_idx": k, "species_cur": species, "hp": 1.0, "status": "",
            "boosts": dict.fromkeys(("atk", "def", "spa", "spd", "spe",
                                     "accuracy", "evasion"), 0),
            "fainted": False, "active_slot": active_slot,
            "appeared": active_slot is not None, "mega_done": False,
            "turns_active": 1 if active_slot is not None else 0, "protect_ct": 0,
            "item_consumed": False, "set": _mon_set(f"Mon{k}", species, moves)}


def _opp_view(k, species, active_slot):
    """Build a ``MonOpponentView`` dict for team index ``k``."""
    return {"team_idx": k, "species_cur": species, "level": 50, "gender": "N",
            "hp": 1.0, "status": "", "boosts": dict.fromkeys(
                ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"), 0),
            "fainted": False, "active_slot": active_slot,
            "appeared": active_slot is not None, "mega_done": False,
            "turns_active": 1 if active_slot is not None else 0, "protect_ct": 0,
            "revealed_moves": [], "revealed_item": None, "item_consumed": False,
            "revealed_ability": None}


def _mock_battle(vocab):
    """Build a mock (tracker, belief, request, view) with two active mons/side."""
    sp = [k for k in vocab.species if k not in ("__pad__", "__unk__")][:6]
    mv = [k for k in vocab.moves if k not in ("__pad__", "__unk__")][:4]
    my_team = [_own_view(k, sp[k], mv, {0: 0, 1: 1}.get(k)) for k in range(6)]
    opp_team = [_opp_view(k, sp[k], {0: 0, 1: 1}.get(k)) for k in range(6)]
    conds = {"tailwind": False, "reflect": False, "lightscreen": False,
             "auroraveil": False}
    view = {"turn": 3, "weather": "", "terrain": "", "trickroom": False,
            "my": {"team": my_team, "mega_available": True, "conditions": dict(conds)},
            "opp": {"team": opp_team, "mega_available": True, "conditions": dict(conds)}}

    mons = [SimpleNamespace(team_idx=k, set=my_team[k]["set"]) for k in range(6)]
    tracker = SimpleNamespace(_view=lambda side: view,
                              sides={"p1": SimpleNamespace(mons=mons)})

    def summary():
        return {k: {"item": "", "p_item": 0.5, "spe_lo": 80.0, "spe_hi": 130.0,
                    "bulk": 12000.0, "nature": "serious", "p_nature": 0.4}
                for k in range(6)}

    belief = SimpleNamespace(
        summary=summary,
        top_particle=lambda k: {"moves": mv, "item": "", "ability": "static",
                                "nature": "serious"},
        sample_sets=lambda n, rng: [[{"species": sp[k], "moves": mv, "item": "",
                                       "ability": "static", "nature": "serious"}
                                     for k in range(6)] for _ in range(n)])

    def moves_req():
        return [{"move": m, "id": m, "pp": 10, "disabled": False,
                 "target": "normal"} for m in mv[:2]]
    pokemon = [{"ident": f"p1: Mon{k}", "details": sp[k],
                "condition": "100/100", "active": k in (0, 1)} for k in range(4)]
    request = {"active": [{"moves": moves_req()}, {"moves": moves_req()}],
               "side": {"id": "p1", "pokemon": pokemon}}
    return tracker, belief, request, view


def test_features_and_actions():
    """Feature extraction yields the fixed layout and resolvable actions."""
    vocab = _vocab()
    _, belief, _, view = _mock_battle(vocab)
    pos = FeatureExtractor(vocab).extract(view, belief.summary(),
                                          brought=[0, 1, 2, 3])
    assert pos.mon_cat.shape == (12, 10)
    assert pos.mon_scalar.shape == (12, N_MON_SCALAR)
    assert pos.my_active == {0: 0, 1: 1} and pos.opp_active == {0: 0, 1: 1}
    act = action_arrays(pos, (1, 1), (1, 1), vocab)
    assert act.shape == (12, 7)
    assert act[0, 0] == 1  # my slot-0 mon takes a move


def test_model_forward():
    """Encode/predict/value/grounded/policy heads return correct shapes."""
    vocab = _vocab()
    model = JEPAWorldModel(vocab.sizes(), JCFG, vocab.state())
    _, belief, _, view = _mock_battle(vocab)
    pos = FeatureExtractor(vocab).extract(view, belief.summary())
    b = _batch([pos], model)
    z = model.encode(b)
    assert z.shape == (1, 16, JCFG.d_model)
    act = torch.as_tensor(action_arrays(pos, (1, 1), (1, 1), vocab))[None]
    zp = model.predict(z, act, b["dmg"])
    assert zp.shape == z.shape
    assert model.value(zp).shape == (1,)
    g = model.grounded(zp)
    assert g["hp"].shape == (1, 12) and g["status"].shape == (1, 12, len(STATUSES))
    my, opp = model.policies(z)
    assert my.shape == (1, 2, N_SLOT_ACTIONS) and opp.shape == (1, 2, N_SLOT_ACTIONS)


def test_solver_finds_mixed_strategy():
    """A matching-pennies payoff matrix yields a ~uniform mixed strategy."""
    m = np.array([[1.0, -1.0], [-1.0, 1.0]])
    p, q, val = solve_matrix(m, 500)
    assert abs(p[0] - 0.5) < 0.05 and abs(q[0] - 0.5) < 0.05
    assert abs(val) < 0.05
    # a dominated row collapses to the dominant one
    p2, _, _ = solve_matrix(np.array([[1.0, 1.0], [0.0, 0.0]]), 500)
    assert p2[0] > 0.9


def test_chooser_end_to_end():
    """A random-init chooser returns a legal joint action and full ChoiceInfo."""
    vocab = _vocab()
    model = JEPAWorldModel(vocab.sizes(), JCFG, vocab.state())
    chooser = JEPAWorldModelChooser(model, vocab, CFG, JCFG, seed=0, bridge=None)
    tracker, belief, request, _ = _mock_battle(vocab)
    joint, info = chooser.choose(tracker, belief, "p1", request, [0, 1, 2, 3],
                                 temperature=0.0)
    assert isinstance(joint, tuple) and len(joint) == 2
    assert joint[0].kind in ("move", "switch", "pass")
    for key in ("value", "solve", "strategy", "q", "opp_pred", "health"):
        assert key in info
    assert -1.0 <= info["value"] <= 1.0
    # the real planner ran (not the first-legal fallback)
    assert info["health"].get("determinizations") == JCFG.n_determinizations
    assert info["strategy"] and info["strategy"][0][0] != "fallback:first-legal"
    # the chosen joint converts to a valid Showdown choice string
    from actions import joint_choice
    name_to_idx = {f"Mon{k}": k for k in range(6)}
    choice = joint_choice(request, joint, name_to_idx)
    assert isinstance(choice, str) and choice
    # temperature > 0 still returns a legal action
    joint2, _ = chooser.choose(tracker, belief, "p1", request, [0, 1, 2, 3],
                               temperature=1.0)
    assert isinstance(joint2, tuple)


def test_training_step_runs():
    """One synthetic minibatch backpropagates through every loss term."""
    from train_jepa import losses
    vocab = _vocab()
    model = JEPAWorldModel(vocab.sizes(), JCFG, vocab.state())
    sizes = vocab.sizes()
    b = 8
    rng = np.random.default_rng(0)
    cat = lambda n, hi: torch.as_tensor(rng.integers(0, hi, size=(b, n)))

    def pos(pref):
        return {pref + "gcat": torch.stack([
                    torch.as_tensor(rng.integers(0, len(WEATHERS), b)),
                    torch.as_tensor(rng.integers(0, len(TERRAINS), b))], 1),
                pref + "gscal": torch.rand(b, 18),
                pref + "mcat": torch.stack([
                    torch.as_tensor(rng.integers(0, min(sizes["species"], 12), (b, 12)))
                    for _ in range(10)], -1),
                pref + "mscal": torch.rand(b, 12, N_MON_SCALAR),
                pref + "dmg": torch.rand(b, 6, 6)}
    sh = {**pos("cur_"), **pos("nxt_"),
          "act": torch.as_tensor(rng.integers(0, 3, (b, 12, 7))),
          "value": (rng.integers(0, 2, b) * 2 - 1).astype("float32"),
          "weight": np.ones(b, "float32"),
          "a_slot": rng.integers(0, N_SLOT_ACTIONS, (b, 2)),
          "b_slot": rng.integers(0, N_SLOT_ACTIONS, (b, 2))}
    sh["value"] = torch.as_tensor(sh["value"])
    sh["weight"] = torch.as_tensor(sh["weight"])
    sh["a_slot"] = torch.as_tensor(sh["a_slot"])
    sh["b_slot"] = torch.as_tensor(sh["b_slot"])
    # cur/nxt status column must be a valid status class
    for pref in ("cur_", "nxt_"):
        sh[pref + "mcat"][..., 7] = torch.as_tensor(
            rng.integers(0, len(STATUSES), (b, 12)))
    loss, metrics = losses(model, sh, JCFG)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["total"] > 0


def _batch(positions, model):
    """Stack positions into a device tensor dict (test-local helper)."""
    dev = next(model.parameters()).device
    t = lambda arr, dt: torch.as_tensor(np.stack(arr), dtype=dt, device=dev)
    return {"gcat": t([p.global_cat for p in positions], torch.long),
            "gscal": t([p.global_scalar for p in positions], torch.float32),
            "mcat": t([p.mon_cat for p in positions], torch.long),
            "mscal": t([p.mon_scalar for p in positions], torch.float32),
            "dmg": t([p.dmg_edge for p in positions], torch.float32)}


def test_legal_my_joints_from_view():
    """Own legal-joint enumeration from a view yields move+switch joints."""
    from jepa.features import legal_my_joints
    vocab = _vocab()
    _, _, _, view = _mock_battle(vocab)
    joints = legal_my_joints(view, vocab, 256)
    assert joints and all(len(j) == 2 for j in joints)
    kinds = {a.kind for j in joints for a in j}
    assert "move" in kinds


def test_consequence_model_forward():
    """Consequence predictor + policy/value heads return correct shapes."""
    from models.jepa_consequence import JEPAConsequenceModel
    from jepa.features import my_action_arrays
    vocab = _vocab()
    model = JEPAConsequenceModel(vocab.sizes(), JCFG, vocab.state())
    _, belief, _, view = _mock_battle(vocab)
    pos = FeatureExtractor(vocab).extract(view, belief.summary())
    b = _batch([pos], model)
    z = model.encode(b)
    act = torch.as_tensor(my_action_arrays(pos, (1, 1), vocab))[None]
    c = model.consequence(z, act, b["dmg"], None)
    assert c.shape == (1, JCFG.d_model)
    assert model.score(c).shape == (1,) and model.value(c).shape == (1,)


def test_consequence_chooser_end_to_end():
    """A random-init consequence chooser returns a legal move + ChoiceInfo."""
    from agents.jepa_world_model.v2 import JEPAConsequenceChooser
    from models.jepa_consequence import JEPAConsequenceModel
    vocab = _vocab()
    model = JEPAConsequenceModel(vocab.sizes(), JCFG, vocab.state())
    chooser = JEPAConsequenceChooser(model, vocab, CFG, JCFG, seed=0, bridge=None)
    tracker, belief, request, _ = _mock_battle(vocab)
    joint, info = chooser.choose(tracker, belief, "p1", request, [0, 1, 2, 3],
                                 temperature=0.0)
    assert isinstance(joint, tuple) and joint[0].kind in ("move", "switch", "pass")
    assert info["strategy"] and info["strategy"][0][0] != "fallback:first-legal"
    assert info["health"].get("determinizations") == JCFG.cons_determinizations
    from actions import joint_choice
    assert joint_choice(request, joint, {f"Mon{k}": k for k in range(6)})


def test_consequence_training_step():
    """One synthetic consequence minibatch backpropagates through every loss."""
    from train_consequence import losses
    from models.jepa_consequence import JEPAConsequenceModel
    vocab = _vocab()
    model = JEPAConsequenceModel(vocab.sizes(), JCFG, vocab.state())
    sizes = vocab.sizes()
    b, nc = 8, 6
    rng = np.random.default_rng(0)

    def pos(pref):
        return {pref + "gcat": torch.stack([
                    torch.as_tensor(rng.integers(0, len(WEATHERS), b)),
                    torch.as_tensor(rng.integers(0, len(TERRAINS), b))], 1),
                pref + "gscal": torch.rand(b, 18),
                pref + "mcat": torch.stack([torch.as_tensor(
                    rng.integers(0, min(sizes["species"], 12), (b, 12)))
                    for _ in range(10)], -1),
                pref + "mscal": torch.rand(b, 12, N_MON_SCALAR),
                pref + "dmg": torch.rand(b, 6, 6)}
    sh = {**pos("cur_"), **pos("nxt_"),
          "value": torch.as_tensor((rng.integers(0, 2, b) * 2 - 1).astype("float32")),
          "weight": torch.ones(b),
          "my_act": torch.as_tensor(rng.integers(0, 3, (b, 12, 7))),
          "cand_acts": torch.as_tensor(rng.integers(0, 3, (b, nc, 12, 7))),
          "cand_mask": torch.ones(b, nc, dtype=torch.bool),
          "a_index": torch.zeros(b, dtype=torch.long)}
    for pref in ("cur_", "nxt_"):
        sh[pref + "mcat"][..., 7] = torch.as_tensor(
            rng.integers(0, len(STATUSES), (b, 12)))
    loss, metrics = losses(model, sh, JCFG)
    loss.backward()
    assert torch.isfinite(loss) and metrics["total"] > 0


def test_scaled_config_builds_and_runs():
    """The ~6x scaled consequence config forwards and lands in 30M-90M params."""
    from jepa.config import scaled_consequence
    from models.jepa_consequence import JEPAConsequenceModel
    from jepa.features import my_action_arrays
    vocab = _vocab()
    sj = scaled_consequence()
    model = JEPAConsequenceModel(vocab.sizes(), sj, vocab.state())
    n = sum(p.numel() for p in model.parameters())
    assert 30e6 < n < 90e6, f"scaled params {n/1e6:.1f}M outside 30-90M"
    _, belief, _, view = _mock_battle(vocab)
    pos = FeatureExtractor(vocab).extract(view, belief.summary())
    b = _batch([pos], model)
    z = model.encode(b)
    act = torch.as_tensor(my_action_arrays(pos, (1, 1), vocab))[None]
    c = model.consequence(z, act, b["dmg"], None)
    assert c.shape == (1, sj.d_model)
    assert model.score(c).shape == (1,)


def test_selfplay_losses_backprop():
    """sp_losses backpropagates; advantage weights are clipped and positive."""
    from selfplay_jepa import sp_losses
    from models.jepa_consequence import JEPAConsequenceModel
    vocab = _vocab()
    model = JEPAConsequenceModel(vocab.sizes(), JCFG, vocab.state())
    sizes = vocab.sizes()
    b, nc = 6, 5
    rng = np.random.default_rng(1)

    def pos(pref):
        return {pref + "gcat": torch.stack([
                    torch.as_tensor(rng.integers(0, len(WEATHERS), b)),
                    torch.as_tensor(rng.integers(0, len(TERRAINS), b))], 1),
                pref + "gscal": torch.rand(b, 18),
                pref + "mcat": torch.stack([torch.as_tensor(
                    rng.integers(0, min(sizes["species"], 12), (b, 12)))
                    for _ in range(10)], -1),
                pref + "mscal": torch.rand(b, 12, N_MON_SCALAR),
                pref + "dmg": torch.rand(b, 6, 6)}
    sh = {**pos("cur_"), **pos("nxt_"),
          "my_act": torch.as_tensor(rng.integers(0, 3, (b, 12, 7))),
          "cand_acts": torch.as_tensor(rng.integers(0, 3, (b, nc, 12, 7))),
          "cand_mask": torch.ones(b, nc, dtype=torch.bool),
          "a_index": torch.zeros(b, dtype=torch.long),
          "value": torch.as_tensor(
              (rng.integers(0, 2, b) * 2 - 1).astype("float32")),
          "weight": torch.ones(b),
          "has_nxt": torch.tensor([True] * (b - 1) + [False])}
    for pref in ("cur_", "nxt_"):        # status column must be a valid class
        sh[pref + "mcat"][..., 7] = torch.as_tensor(
            rng.integers(0, len(STATUSES), (b, 12)))
    loss, m = sp_losses(model, sh, JCFG)
    loss.backward()
    assert torch.isfinite(loss)
    assert 0 < m["adv_w"] <= JCFG.spj_w_max


def test_recorder_sample_from_plan():
    """RecorderConsequenceBot._sample keeps the chosen action at index 0."""
    import random as _random
    from selfplay_jepa import RecorderConsequenceBot
    vocab = _vocab()
    _, belief, _, view = _mock_battle(vocab)
    pos = FeatureExtractor(vocab).extract(view, belief.summary())
    acts = np.stack([np.full((12, 7), i, dtype=np.int64) for i in range(20)])
    plan = {"pos": pos, "cands": list(range(20)), "cand_acts": acts,
            "chosen": 7, "scores": np.zeros(20), "values": np.zeros(20)}
    bot = object.__new__(RecorderConsequenceBot)   # skip Bot.__init__ (no sim)
    bot.eps, bot.exp_rng, bot.n_cand = 0.0, _random.Random(0), 12
    s = bot._sample(plan, 7)
    assert s["cand_acts"].shape == (12, 12, 7)
    assert (s["cand_acts"][0] == acts[7]).all()      # chosen at index 0
    assert s["a_index"] == 0 and s["cand_mask"].all()
    rows = {int(s["cand_acts"][j, 0, 0]) for j in range(12)}
    assert len(rows) == 12                            # negatives are distinct


TESTS = [test_features_and_actions, test_model_forward,
         test_solver_finds_mixed_strategy, test_chooser_end_to_end,
         test_training_step_runs, test_legal_my_joints_from_view,
         test_consequence_model_forward, test_consequence_chooser_end_to_end,
         test_consequence_training_step, test_scaled_config_builds_and_runs,
         test_selfplay_losses_backprop, test_recorder_sample_from_plan]


if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"ok  {t.__name__}")
    print("all jepa tests passed")
