"""Unit tests for the exp/lead-switch selectors (expert / value / LeadNet).

Kept light on purpose: the type-chart fallback path needs no Node process,
the value selectors run against a stub net, and LeadNet trains one step on a
synthetic example — so the module gates in a couple of seconds. The real
bridge/checkpoint behavior is exercised by the smoke scripts and the round
robin itself.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from agents.lead_switch.expert import (ExpertLeadSelector,
                                       ExpertSwitchSelector,
                                       enumerate_previews, preview_order,
                                       team_choice_string)
from agents.lead_switch.lscfg import LSCFG
from agents.lead_switch.matchup import MatchupModel, type_multiplier
from beliefs import OpponentBelief
from config import CFG
from data import LogParser, Side, parse_packed_team, sid
from env import TEAM_A, TEAM_B
from observe_game import cts_placeholder
from tokenizer import PositionTokenizer


def _fixture():
    """Tracker + belief for TEAM_A (me, p1) vs TEAM_B preview (them, p2)."""
    usage = json.loads((CFG.artifacts_dir / "usage_stats.json").read_text())
    teams = {"p1": parse_packed_team(TEAM_A), "p2": parse_packed_team(TEAM_B)}
    tracker = LogParser("t", 0, "", CFG.format_id)
    tracker.sides = {"p1": Side(teams["p1"]),
                     "p2": Side([cts_placeholder(s) for s in teams["p2"]])}
    belief = OpponentBelief([sid(s["species"]) for s in teams["p2"]],
                            usage, CFG, None, my_team=teams["p1"])
    return tracker, belief, teams


def _force_switch_fixture():
    """Fixture advanced to a p1 slot-0 faint with a forceSwitch request."""
    tracker, belief, teams = _fixture()
    mons = tracker.sides["p1"].mons
    mons[0].fainted, mons[0].hp, mons[0].active_slot = True, 0.0, 0
    mons[1].active_slot, mons[1].appeared = 1, True
    for slot, k in enumerate((3, 5)):
        m = tracker.sides["p2"].mons[k]
        m.active_slot, m.appeared = slot, True
    req = {"forceSwitch": [True, False],
           "side": {"pokemon": [
               {"ident": f"p1: {s['name']}", "details": s["species"],
                "condition": "0 fnt" if i == 0 else "100/100",
                "active": i in (0, 1)}
               for i, s in enumerate(teams["p1"])]}}
    return tracker, belief, teams, req


def test_enumerate_previews_is_the_90_grid():
    """6 choose 4 brings x lead pairs = 90 combos, all distinct and legal."""
    combos = list(enumerate_previews(6, 4))
    assert len(combos) == 90
    assert len(set(combos)) == 90
    for lead, back in combos:
        order = preview_order(lead, back)
        assert len(order) == 4 and len(set(order)) == 4
        assert team_choice_string(order).startswith("team ")


def test_type_chart_sanity():
    """A few well-known multipliers hold, including immunities."""
    assert type_multiplier("Water", ["Fire"]) == 2
    assert type_multiplier("Electric", ["Ground"]) == 0
    assert type_multiplier("Ice", ["Dragon", "Flying"]) == 4
    assert type_multiplier("Fighting", ["Ghost"]) == 0


def test_matchup_tables_shape_and_signal():
    """Chart-fallback tables are full-size, bounded, and not constant."""
    tracker, belief, _ = _fixture()
    my_sets = [m.set for m in tracker.sides["p1"].mons]
    off, dfn, spd = MatchupModel(CFG, None).tables(my_sets, belief)
    assert len(off) == 6 and all(len(r) == 6 for r in off)
    flat = [v for r in off for v in r]
    assert min(flat) >= 0 and max(flat) <= 1.5
    assert max(flat) > min(flat), "matchup table carries no signal"
    assert all(0.0 <= v <= 1.0 for r in spd for v in r)


def test_expert_lead_selector_returns_legal_order():
    """Order is 4 distinct preview indices; scores are strictly ranked."""
    tracker, belief, _ = _fixture()
    order, info = ExpertLeadSelector(CFG, None).choose(tracker, belief, "p1")
    assert len(order) == 4 and len(set(order)) == 4
    assert all(0 <= i < 6 for i in order)
    scores = [s for _, s in info["top"]]
    assert scores == sorted(scores, reverse=True)


def test_expert_lead_selector_is_deterministic():
    """Two runs on the same position agree exactly."""
    tracker, belief, _ = _fixture()
    sel = ExpertLeadSelector(CFG, None)
    assert sel.choose(tracker, belief, "p1")[0] == \
        sel.choose(tracker, belief, "p1")[0]


def test_expert_switch_selector_answers_forced_slots():
    """One switch per forced slot, pass elsewhere, no fainted picks."""
    tracker, belief, teams, req = _force_switch_fixture()
    choice = ExpertSwitchSelector(CFG, None).choose(req, tracker, belief, "p1")
    parts = [c.strip() for c in choice.split(",")]
    assert len(parts) == 2 and parts[1] == "pass"
    assert parts[0].startswith("switch ")
    pos = int(parts[0].split()[1])
    assert req["side"]["pokemon"][pos - 1]["condition"] != "0 fnt"
    assert not req["side"]["pokemon"][pos - 1]["active"]


class _StubNet:
    """predict_batch stub: deterministic pseudo-values from token sums."""

    def predict_batch(self, toks):
        """Return (None, values, None) with values a function of the tokens."""
        vals = np.array([(t.astype(np.int64).sum() % 97) / 97.0 - 0.5
                         for t in toks])
        return None, vals, None


def test_value_selectors_run_on_hypothetical_states():
    """Value lead + switch selectors produce legal answers via a stub net."""
    from agents.lead_switch.value import (ValueLeadSelector,
                                          ValueSwitchSelector)
    tok = PositionTokenizer.load(CFG)
    tracker, belief, _ = _fixture()
    order, info = ValueLeadSelector(_StubNet(), tok, CFG, None).choose(
        tracker, belief, "p1")
    assert len(order) == 4 and len(set(order)) == 4
    assert 0 <= info["expert_rank"] < LSCFG.v_my_top

    tracker, belief, teams, req = _force_switch_fixture()
    choice = ValueSwitchSelector(_StubNet(), tok, CFG, None).choose(
        req, tracker, belief, "p1")
    assert choice.split(",")[0].strip().startswith("switch ")


def test_leadnet_train_step_and_inference():
    """LeadNet fits one synthetic example and selects a legal preview."""
    import torch

    from agents.lead_switch.leadnet import (LeadNet, NNLeadSelector, PAIRS,
                                            batches, loss_terms,
                                            team_features)
    tok = PositionTokenizer.load(CFG)
    tracker, belief, teams = _fixture()
    my_f, opp_s = team_features(tok, teams["p1"],
                                [sid(s["species"]) for s in teams["p2"]])
    ex = [{"my": my_f, "opp": opp_s, "pair": PAIRS.index((2, 3)),
           "bring_t": np.array([0, 0, 1, 1, 1, 1], dtype=np.float32),
           "bring_m": np.ones(6, dtype=np.float32),
           "w": np.float32(1.0)}] * 8
    torch.manual_seed(0)
    net = LeadNet(tok.vocab_size())
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    for _ in range(30):
        for b in batches(ex, 8, shuffle=False):
            ce, bce = loss_terms(net, b)
            (ce + 0.5 * bce).backward()
            opt.step()
            opt.zero_grad()
    with torch.no_grad():
        _, pair = net(torch.from_numpy(my_f).unsqueeze(0),
                      torch.from_numpy(opp_s).unsqueeze(0))
    assert int(pair[0].argmax()) == PAIRS.index((2, 3)), \
        "LeadNet failed to fit a single repeated example"

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "leadnet.pt"
        net.save(p)
        order, _ = NNLeadSelector(p, tok).choose(tracker, belief, "p1")
    assert len(order) == 4 and len(set(order)) == 4
    assert order[0] == 2 and order[1] == 3, \
        "selector should lead with the overfit pair"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
