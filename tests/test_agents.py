"""Focused contracts for versioned agents and modular search bricks."""

import dataclasses
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actions import N_JOINT_ACTIONS, SlotAction, joint_index
from agents.determinized_duct.v1 import DeterminizedDUCTChooser
from agents.encoding.v1 import TokenPositionEncoder
from agents.evaluation import EvaluationStore, evaluate_policy_prior
from agents.evaluators.v1 import PolicyValueLeafEvaluator
from agents.interfaces import BeliefModel, MoveChooser
from agents.priors.v1 import PolicyValuePrior
from agents.registry import (REGISTRY, implementation_source_hashes,
                             verify_implementation_sources)
from agents.search.v1 import DecoupledUCTSearcher
from agents.spec import AgentSpec, default_duct_spec
from beliefs import OpponentBelief
from config import CFG
from search.mcts import Searcher, _joint_priors
from search.node import Node


def actions(n):
    return [(SlotAction("move", move_slot=i), SlotAction("pass"))
            for i in range(n)]


def test_move_chooser_contract_and_named_v1_equivalence():
    fixture = Path(__file__).parent / "fixtures/tiny_random_agent/agent.json"
    chooser = REGISTRY.build(AgentSpec.load(fixture))
    assert isinstance(chooser, MoveChooser)
    assert DeterminizedDUCTChooser.choose is Searcher.choose


def test_registry_applies_archived_brick_config():
    archived_cfg = dataclasses.replace(
        CFG, use_damage_features=False, sims_per_move=17)
    runtime_cfg = dataclasses.replace(archived_cfg, sims_per_move=1)
    chooser = REGISTRY.build(
        default_duct_spec(archived_cfg), model=None, tokenizer=None,
        cfg=runtime_cfg, sidecar=SimpleNamespace())
    assert chooser.cfg.sims_per_move == 17
    assert chooser.bridge is None
    chooser.close()


def test_tiny_archived_random_agent_runs_deterministic_scenario():
    fixture = Path(__file__).parent / "fixtures/tiny_random_agent/agent.json"
    spec = AgentSpec.load(fixture)
    chooser = REGISTRY.build(spec)
    mon = SimpleNamespace(set={"name": "A"}, team_idx=0)
    tracker = SimpleNamespace(sides={"p1": SimpleNamespace(mons=[mon])})
    request = {
        "wait": True,
        "side": {"pokemon": [{"ident": "p1: A", "details": "A",
                                "condition": "1/1", "active": True}]},
    }
    action, info = chooser.choose(
        tracker, None, "p1", request, [0], temperature=0.0)
    assert action == (SlotAction("pass"), SlotAction("pass"))
    assert info["strategy"] == [("uniform random", 1.0)]


def test_policy_prior_masks_normalizes_prunes_and_falls_back():
    legal = actions(3)
    dist = np.zeros(N_JOINT_ACTIONS)
    dist[joint_index(*legal[0])] = 0.1
    dist[joint_index(*legal[1])] = 0.6
    dist[joint_index(*legal[2])] = 0.3
    prior = PolicyValuePrior()
    p, kept = prior.legal_priors(dist, legal, top_k=2)
    assert kept == [legal[1], legal[2]]
    np.testing.assert_allclose(p, [2 / 3, 1 / 3])
    old_p, old_kept = _joint_priors(dist, legal, 2)
    np.testing.assert_array_equal(p, old_p)
    assert kept == old_kept

    p, kept = prior.legal_priors(np.zeros_like(dist), legal, top_k=None)
    assert kept == legal
    np.testing.assert_allclose(p, np.full(3, 1 / 3))


def test_position_encoder_is_exact_tokenizer_seam_without_damage():
    state = {"turn": 7}

    class Tracker:
        def _view(self, side):
            assert side == "p1"
            return state

    class Belief:
        def summary(self):
            return {0: {"item": "sitrusberry"}}

    class Tokenizer:
        def encode(self, got_state, got_summary, got_damage):
            assert got_state is state
            assert got_damage == {}
            return np.array([got_state["turn"], len(got_summary)])

    tok, belief = Tokenizer(), Belief()
    encoder = TokenPositionEncoder(tok)
    expected = tok.encode(state, belief.summary(), {})
    np.testing.assert_array_equal(
        encoder.encode(Tracker(), "p1", belief), expected)


def test_leaf_evaluator_preserves_batch_and_value_orientation():
    expected = (np.zeros((2, N_JOINT_ACTIONS)), np.array([-0.25, 0.5]), {})

    class Model:
        def predict_batch(self, tokens):
            assert tokens.shape == (2, 3)
            return expected

    evaluator = PolicyValueLeafEvaluator(Model())
    assert evaluator.predict_batch(np.zeros((2, 3))) is expected
    assert evaluator.value(expected[1], 0) == -0.25
    assert evaluator.terminal_value("p1", "p1", "p2") == 1.0
    assert evaluator.terminal_value("p2", "p1", "p2") == -1.0

    search = DecoupledUCTSearcher()
    chooser = SimpleNamespace(
        cfg=SimpleNamespace(rollout_depth=1), health=Counter(),
        leaf_evaluator=evaluator)
    det = SimpleNamespace(my="p1", opp="p2")
    win = SimpleNamespace(ended=True, winner="p1")
    loss = SimpleNamespace(ended=True, winner="p2")
    leaf = SimpleNamespace(value=0.75)
    assert search.leaf_value(chooser, det, win, None, leaf) == 1.0
    assert search.leaf_value(chooser, det, loss, None, leaf) == -1.0


def test_belief_v1_satisfies_external_lifecycle_contract():
    cfg = dataclasses.replace(
        CFG, spreads_prior=False, spread_archetypes=False, n_particles=2)
    usage = {"pikachu": [[1, ["thunderbolt"], "lightball",
                            "static", "timid"]]}
    belief = OpponentBelief(["pikachu"], usage, cfg, bridge=None, my_team=[])
    assert isinstance(belief, BeliefModel)
    belief.update([], viewer="p1")
    assert belief.top_particle(0)["item"] == "lightball"
    assert len(belief.sample_sets(2, random.Random(3))) == 2
    assert belief.summary()[0]["item"] == "lightball"


def test_searcher_backup_and_root_aggregation_with_fake_nodes():
    mine, opp = actions(2), actions(1)
    node = Node(mine, opp, [0.7, 0.3], [1.0])
    node.update(1, 0, 0.5)
    assert node.n == 1 and node.my_w[1] == 0.5 and node.opp_w[0] == -0.5

    other = Node(mine, opp, [0.7, 0.3], [1.0])
    other.update(1, 0, -0.25)
    rows = DecoupledUCTSearcher().aggregate_root([
        SimpleNamespace(root=node), SimpleNamespace(root=other)])
    assert rows[0][0] == mine[1]
    assert rows[0][1:] == [2.0, 0.25]


def test_agent_spec_round_trip_paths_and_registry(tmp_path):
    spec = default_duct_spec(CFG, runtime={"python": "test"})
    path = tmp_path / "agent.json"
    spec.dump(path)
    loaded = AgentSpec.load(path)
    assert loaded == spec
    assert REGISTRY.validate(loaded) == loaded
    assert loaded.resolve(tmp_path, "ckpt.pt") == tmp_path / "ckpt.pt"
    try:
        loaded.resolve(tmp_path, "../escape")
    except ValueError:
        pass
    else:
        raise AssertionError("archive path traversal was accepted")

    data = json.loads(path.read_text())
    data["agent_impl"] = "agents.missing.v99.Chooser"
    try:
        REGISTRY.validate(AgentSpec.from_dict(data))
    except KeyError:
        pass
    else:
        raise AssertionError("unknown archived implementation was accepted")


def test_agent_spec_source_identity_fails_closed():
    spec = default_duct_spec(CFG)
    hashes = implementation_source_hashes(spec)
    assert "agents/determinized_duct/v1.py" in hashes
    assert "search/mcts.py" in hashes
    tampered = dataclasses.replace(
        spec, source={"files": dict(hashes) | {"search/mcts.py": "0" * 64}})
    try:
        verify_implementation_sources(tampered)
    except RuntimeError:
        pass
    else:
        raise AssertionError("changed implementation source was accepted")


def test_brick_evaluation_results_are_append_only(tmp_path):
    legal = actions(2)
    distribution = np.zeros(N_JOINT_ACTIONS)
    target = joint_index(*legal[1])
    distribution[target] = 1.0
    store = EvaluationStore(tmp_path / "results.jsonl")
    evaluate_policy_prior(
        "agents.priors.v1.PolicyValuePrior", PolicyValuePrior(),
        [(distribution, legal, target)], top_k=1, store=store)
    evaluate_policy_prior(
        "agents.priors.v1.PolicyValuePrior", PolicyValuePrior(),
        [(distribution, legal, target)], top_k=2, store=store)
    rows = store.load(suite="policy_prior")
    assert len(rows) == 2
    assert all(row["metrics"]["recall_at_k"] == 1.0 for row in rows)


if __name__ == "__main__":
    no_tmp = [
        test_move_chooser_contract_and_named_v1_equivalence,
        test_registry_applies_archived_brick_config,
        test_tiny_archived_random_agent_runs_deterministic_scenario,
        test_policy_prior_masks_normalizes_prunes_and_falls_back,
        test_position_encoder_is_exact_tokenizer_seam_without_damage,
        test_leaf_evaluator_preserves_batch_and_value_orientation,
        test_belief_v1_satisfies_external_lifecycle_contract,
        test_searcher_backup_and_root_aggregation_with_fake_nodes,
        test_agent_spec_source_identity_fails_closed,
    ]
    for test in no_tmp:
        test()
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        test_agent_spec_round_trip_paths_and_registry(tmp_path)
        test_brick_evaluation_results_are_append_only(tmp_path)
    print("all modular agent tests passed")
