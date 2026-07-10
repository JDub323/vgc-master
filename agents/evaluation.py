"""Comparable, append-only quality and latency evaluations for agent bricks.

Each evaluator accepts explicit cases so tests, replay datasets, and scenario
miners can share the same result format without coupling bricks to one data
loader. Calling an evaluator always appends a timestamped JSON record; previous
results remain available for implementation-to-implementation comparisons.
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from actions import joint_index
from config import CFG, config_snapshot


@dataclass
class BrickEvaluation:
    """One JSON-serializable, timestamped brick benchmark record."""
    brick_impl: str
    suite: str
    metrics: dict
    cases: int
    created: str = field(default_factory=lambda: datetime.now(
        timezone.utc).isoformat())
    config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class EvaluationStore:
    """Append-only JSONL result history."""

    def __init__(self, path=None, cfg=CFG):
        """Target ``path`` or ``artifacts/brick_evaluations/results.jsonl``."""
        self.path = Path(path or (
            cfg.artifacts_dir / "brick_evaluations" / "results.jsonl"))

    def append(self, result):
        """Append one ``BrickEvaluation`` JSON line and return ``result``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(asdict(result), sort_keys=True) + "\n")
        return result

    def load(self, *, brick_impl=None, suite=None):
        """Return decoded rows, optionally filtered by implementation/suite."""
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text().splitlines()
                if line.strip()]
        if brick_impl:
            rows = [row for row in rows if row["brick_impl"] == brick_impl]
        if suite:
            rows = [row for row in rows if row["suite"] == suite]
        return rows


def _save(impl, suite, metrics, n, store, cfg, metadata):
    """Build, append, and return one ``BrickEvaluation``."""
    result = BrickEvaluation(
        brick_impl=impl, suite=suite, metrics=metrics, cases=n,
        config=config_snapshot(cfg), metadata=dict(metadata or {}))
    return (store or EvaluationStore(cfg=cfg)).append(result)


def evaluate_policy_prior(impl, prior, cases, top_k, *, store=None, cfg=CFG,
                          metadata=None):
    """Measure legal normalization, recall@k, calibration, NLL, and latency.

    A case is ``(joint_distribution, legal_joints, target_joint_index)``.
    """
    cases = list(cases)
    hits, normalized, confidences, correct, nll = 0, 0, [], [], []
    started = time.perf_counter()
    for distribution, legal, target in cases:
        probabilities, retained = prior.legal_priors(
            distribution, legal, top_k)
        indices = [joint_index(*action) for action in retained]
        normalized += int(np.isfinite(probabilities).all()
                          and abs(float(probabilities.sum()) - 1.0) < 1e-9)
        hits += int(target in indices)
        pick = int(np.argmax(probabilities))
        confidences.append(float(probabilities[pick]))
        correct.append(float(indices[pick] == target))
        if target in indices:
            nll.append(-math.log(max(float(probabilities[indices.index(target)]),
                                      1e-12)))
    elapsed = time.perf_counter() - started
    n = len(cases)
    # Expected calibration error over five fixed confidence bins.
    ece = 0.0
    for bin_index, lo in enumerate(np.linspace(0.0, 0.8, 5)):
        hi = lo + 0.2
        selected = [i for i, confidence in enumerate(confidences)
                    if lo <= confidence < hi or
                    (bin_index == 4 and confidence == hi)]
        if selected:
            avg_conf = np.mean([confidences[i] for i in selected])
            avg_acc = np.mean([correct[i] for i in selected])
            ece += len(selected) / max(1, n) * abs(avg_conf - avg_acc)
    return _save(impl, "policy_prior", {
        "recall_at_k": hits / max(1, n),
        "normalization_rate": normalized / max(1, n),
        "nll_on_retained": float(np.mean(nll)) if nll else None,
        "calibration_ece": float(ece),
        "latency_ms_per_case": elapsed * 1000 / max(1, n),
    }, n, store, cfg, metadata)


def evaluate_position_encoder(impl, encoder, cases, *, store=None, cfg=CFG,
                              metadata=None):
    """Measure token equivalence and latency.

    A case is ``(tracker, side_id, belief, expected_tokens)``.
    """
    cases = list(cases)
    exact = 0
    started = time.perf_counter()
    for tracker, side_id, belief, expected in cases:
        actual = encoder.encode(tracker, side_id, belief)
        exact += int(np.array_equal(actual, expected))
    elapsed = time.perf_counter() - started
    n = len(cases)
    return _save(impl, "position_encoder", {
        "token_exact_rate": exact / max(1, n),
        "latency_ms_per_case": elapsed * 1000 / max(1, n),
    }, n, store, cfg, metadata)


def evaluate_leaf_evaluator(impl, evaluator, batches, *, store=None, cfg=CFG,
                            metadata=None):
    """Measure value calibration, terminal-sign accuracy, and latency.

    A case is ``(tokens, target_values)``; targets at exactly -1/0/1 are also
    counted as terminal cases.
    """
    batches = list(batches)
    errors, terminal_correct, terminal_n = [], 0, 0
    positions = 0
    started = time.perf_counter()
    for tokens, targets in batches:
        _, values, _ = evaluator.predict_batch(tokens)
        targets, values = np.asarray(targets), np.asarray(values)
        errors.extend(np.abs(values - targets).tolist())
        positions += len(targets)
        terminal = np.isin(targets, (-1.0, 0.0, 1.0))
        terminal_n += int(terminal.sum())
        terminal_correct += int((np.sign(values[terminal]) ==
                                 np.sign(targets[terminal])).sum())
    elapsed = time.perf_counter() - started
    return _save(impl, "leaf_evaluator", {
        "value_mae": float(np.mean(errors)) if errors else None,
        "terminal_sign_accuracy": terminal_correct / max(1, terminal_n),
        "latency_ms_per_position": elapsed * 1000 / max(1, positions),
    }, positions, store, cfg, metadata)


def evaluate_belief_model(impl, belief, cases, *, samples=100, store=None,
                          cfg=CFG, metadata=None):
    """Measure deductions, oracle top/sample rate, depletion, and latency.

    Each case is ``(events, viewer, mon_index, oracle_subset)``. The subset is
    a mapping of set fields that must match (for example item + ability).
    """
    cases = list(cases)
    top_hits, sample_hits = 0, 0

    def matches(candidate, oracle):
        return all(candidate.get(key) == value for key, value in oracle.items())

    started = time.perf_counter()
    for events, viewer, mon_index, oracle in cases:
        belief.update(events, viewer)
        top_hits += int(matches(belief.top_particle(mon_index), oracle))
        draws = belief.sample_sets(samples)
        sample_hits += sum(matches(team[mon_index], oracle) for team in draws)
    elapsed = time.perf_counter() - started
    n = len(cases)
    soft = sum(getattr(belief, "soft_depletions", []))
    hard = sum(getattr(belief, "hard_depletions", []))
    return _save(impl, "belief_model", {
        "oracle_top_rate": top_hits / max(1, n),
        "oracle_sample_mass": sample_hits / max(1, n * samples),
        "soft_depletions": soft,
        "hard_depletions": hard,
        "latency_ms_per_case": elapsed * 1000 / max(1, n),
    }, n, store, cfg, metadata)


def evaluate_searcher(impl, searcher, cases, *, store=None, cfg=CFG,
                      metadata=None):
    """Measure fake/scenario pass rate, simulations/sec, and aggregation.

    Each case is ``(chooser, determinizations, budget, predicate)``. Predicate
    receives the aggregated root rows after search.
    """
    cases = list(cases)
    passed, sims = 0, 0
    started = time.perf_counter()
    for chooser, determinizations, budget, predicate in cases:
        before = chooser.health.get("sims", 0)
        searcher.run(chooser, determinizations, budget)
        rows = searcher.aggregate_root(determinizations)
        passed += int(predicate(rows))
        sims += chooser.health.get("sims", 0) - before
    elapsed = time.perf_counter() - started
    n = len(cases)
    return _save(impl, "searcher", {
        "scenario_pass_rate": passed / max(1, n),
        "simulations_per_second": sims / max(elapsed, 1e-12),
        "latency_ms_per_case": elapsed * 1000 / max(1, n),
    }, n, store, cfg, metadata)
