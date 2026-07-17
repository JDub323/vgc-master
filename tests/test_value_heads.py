"""exp/value-head contract tests: heads, combined model, and label alignment.

Everything runs on a tiny random-init PolicyValueNet — no checkpoint, data, or
Node processes — so this file stays green in a bare checkout."""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.evaluators.v1 import PolicyValueLeafEvaluator
from agents.interfaces import LeafEvaluator
from config import Config
from models.policy_value import PolicyValueNet
from models.value_heads import (AttnPoolHead, CLSMLPHead, ValueAugmentedNet,
                                ValueNet, load_value_agent, save_combined)

N_TOKENS, VOCAB = 24, 64
OPP_POS = [10, 11, 12, 13, 14, 15]


def tiny_base():
    """Deterministic small baseline net (d=32, 1 layer, no dropout)."""
    torch.manual_seed(0)
    cfg = replace(Config(), d_model=32, n_layers=1, n_heads=4, d_ff=64,
                  dropout=0.0)
    m = PolicyValueNet(VOCAB, N_TOKENS, OPP_POS, n_moves=8, n_items=5,
                       n_abilities=5, cfg=cfg, policy_head="joint")
    m.eval()
    return m


def tokens(n=3):
    """Deterministic random token batch."""
    return np.random.default_rng(7).integers(0, VOCAB, size=(n, N_TOKENS),
                                             dtype=np.int64)


def test_head_shapes_and_zero_init():
    """Heads emit [B, 1+n_aux] and start at exactly v=0 (zero-init out)."""
    for head in (CLSMLPHead(32), AttnPoolHead(32, n_heads=4)):
        head.eval()
        h = torch.randn(3, N_TOKENS, 32)
        out = head(h)
        assert out.shape == (3, 3)
        assert torch.allclose(out[:, 0], torch.zeros(3))


def test_value_net_copies_trunk():
    """ValueNet.from_base clones the baseline trunk weights exactly."""
    base = tiny_base()
    net = ValueNet.from_base(base, head="attnpool")
    assert torch.equal(net.emb.weight, base.emb.weight)
    assert torch.equal(net.pos, base.pos)
    out = net(torch.as_tensor(tokens()))
    assert out.shape == (3, 3)


def test_combined_policy_is_bit_identical_and_value_swapped():
    """ValueAugmentedNet: baseline dists/aux unchanged, value from the head."""
    base = tiny_base()
    head = AttnPoolHead(32, n_heads=4)
    combined = ValueAugmentedNet(base, head, "head", output="bce")
    t = tokens()
    dists_b, values_b, aux_b = base.predict_batch(t)
    dists_c, values_c, aux_c = combined.predict_batch(t)
    assert np.array_equal(dists_b, dists_c)
    for k in aux_b:
        assert np.array_equal(aux_b[k], aux_c[k])
    assert values_c.shape == values_b.shape
    assert np.all(np.abs(values_c) <= 1.0)
    # zero-init head -> sigmoid(0) -> v exactly 0, unlike the baseline head
    assert np.allclose(values_c, 0.0)


def test_temperature_is_monotone_shrink():
    """Higher calibration temperature moves values toward 0, same signs."""
    base = tiny_base()
    net = ValueNet.from_base(base)
    torch.manual_seed(1)
    with torch.no_grad():                     # un-zero the head output
        net.head.mlp[-1].weight.normal_(0, 0.5)
        net.head.mlp[-1].bias.normal_(0, 0.5)
    t = tokens(5)
    sharp = ValueAugmentedNet(base, net, "net", "bce", temperature=1.0)
    soft = ValueAugmentedNet(base, net, "net", "bce", temperature=4.0)
    _, v1, _ = sharp.predict_batch(t)
    _, v4, _ = soft.predict_batch(t)
    assert np.all(np.abs(v4) <= np.abs(v1) + 1e-9)
    nz = np.abs(v1) > 1e-6
    assert np.all(np.sign(v1[nz]) == np.sign(v4[nz]))


def test_combined_checkpoint_roundtrip():
    """save_combined -> load_value_agent reproduces predictions exactly."""
    base = tiny_base()
    head = CLSMLPHead(32)
    torch.manual_seed(2)
    with torch.no_grad():
        head.mlp[-1].weight.normal_(0, 0.5)
    combined = ValueAugmentedNet(base, head, "head", "bce", temperature=1.7)
    base_raw = {"hp": base.hp, "state": base.state_dict(),
                "cfg": base.cfg_snapshot}
    t = tokens()
    dists, values, _ = combined.predict_batch(t)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "combined.pt"
        save_combined(path, base_raw, head, "head", "bce", 1.7,
                      meta={"candidate": "test"})
        loaded = load_value_agent(path, Config(), "cpu")
    dists2, values2, _ = loaded.predict_batch(t)
    assert np.array_equal(dists, dists2)
    assert np.allclose(values, values2, atol=1e-6)
    assert loaded.temperature == 1.7


def test_evaluator_contract():
    """The combined model satisfies the v1 leaf-evaluator brick contract."""
    base = tiny_base()
    combined = ValueAugmentedNet(base, CLSMLPHead(32), "head")
    ev = PolicyValueLeafEvaluator(combined)
    assert isinstance(ev, LeafEvaluator)
    dists, values, _ = ev.predict_batch(tokens())
    assert ev.value(values, 0) == float(values[0])
    assert ev.terminal_value("p1", "p1", "p2") == 1.0
    assert ev.terminal_value("p2", "p1", "p2") == -1.0
    assert ev.terminal_value(None, "p1", "p2") == 0.0


def test_battle_margins():
    """Margins read the last observed state from the right perspective."""
    from value_labels import battle_margins
    mon = lambda hp, fainted: {"hp": hp, "fainted": fainted}
    state = {
        "p1": {"my": {"team": [mon(1.0, False), mon(0.0, True)]},
               "opp": {"team": [mon(0.5, False), mon(0.0, True)]}},
        "p2": {"my": {"team": [mon(0.5, False), mon(0.0, True)]},
               "opp": {"team": [mon(1.0, False), mon(0.0, True)]}},
    }
    rec = {"turns": [{"states": None}, {"states": state}]}
    mf1, mh1 = battle_margins(rec, "p1")
    mf2, mh2 = battle_margins(rec, "p2")
    assert mf1 == 0.0 and mf2 == 0.0            # one faint each
    assert abs(mh1 - (1.0 - 0.5) / 6.0) < 1e-9
    assert abs(mh1 + mh2) < 1e-9                # perspectives are mirrored


def test_fit_temperature_improves_overconfident_logits():
    """Grid calibration finds T>1 for overconfident logits and lowers NLL."""
    from value_lab import fit_temperature
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.5).astype(np.float64)
    logits = (2 * y - 1) * 6.0 + rng.normal(0, 6.0, size=2000)
    t = fit_temperature(logits, y, output="bce")
    assert t > 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} passed")
    print("all value-head tests passed")
