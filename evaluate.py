"""Predictor benchmarks on held-out test battles.

Headline metric: pruned-set recall@k — the fraction of positions where the
human's actual joint action is inside the model's top-k joint actions. That is
the direct measure of whether pruning the search to the model's top-k is safe.

Also: top-1/3/5 joint accuracy, perplexity (log-loss), and calibration (ECE),
since the probabilities weight the search.

Joint distributions come from the per-slot factorization (outer product),
masked to game-legal combos (no double mega, no double switch to the same mon)
and renormalized — the same recombination the search uses.

CLI: python evaluate.py [checkpoint]
     --worst N    decode the N test positions the model gets most wrong
                  (readable state + human action vs model top-3) — the
                  fastest way to see WHERE the model should improve
     --aux        auxiliary-head accuracy vs the oracle team sheets, the
                  learned counterpart to `beliefs.py --audit`
"""

import sys

import numpy as np
import torch

from actions import N_SLOT_ACTIONS, from_index, joint_ok
from config import CFG
from models.baselines import MaxDamagePolicy, RandomPolicy
from models.policy_value import PolicyValueNet
from tokenizer import MON_BLOCK, N_MONS, PositionTokenizer
from train import Shards

KS = (1, 3, 5, CFG.top_k_actions, 16)
TGT = {0: "", 1: ">1", 2: ">2", 3: ">ally"}


def joint_mask():
    m = np.ones((N_SLOT_ACTIONS, N_SLOT_ACTIONS), dtype=bool)
    for a in range(N_SLOT_ACTIONS):
        for b in range(N_SLOT_ACTIONS):
            m[a, b] = joint_ok(from_index(a), from_index(b))
    return m


def score(slot_dists, acts, mask):
    """slot_dists [B,2,A], acts [B,2] -> metrics dict."""
    joint = slot_dists[:, 0, :, None] * slot_dists[:, 1, None, :]   # [B,A,A]
    joint *= mask
    joint /= joint.sum(axis=(1, 2), keepdims=True)
    flat = joint.reshape(len(joint), -1)
    label = acts[:, 0] * N_SLOT_ACTIONS + acts[:, 1]
    p_label = flat[np.arange(len(flat)), label]
    order = np.argsort(-flat, axis=1)
    rank = np.argmax(order == label[:, None], axis=1)   # 0-based rank of label

    out = {f"top{k}" if k <= 5 else f"recall@{k}": float((rank < k).mean())
           for k in KS}
    out["perplexity"] = float(np.exp(-np.log(np.clip(p_label, 1e-12, 1)).mean()))
    conf = flat.max(axis=1)
    hit = rank == 0
    ece, edges = 0.0, np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(hit[in_bin].mean() - conf[in_bin].mean())
    out["ece"] = float(ece)
    return out


# ---------------------------------------------------------------------------
# model debug: decode the positions the model gets most wrong
# ---------------------------------------------------------------------------

def _blocks(names, base):
    return [names[base + k * MON_BLOCK: base + (k + 1) * MON_BLOCK]
            for k in range(N_MONS)]


def _short(tok_name):
    return tok_name.split(":")[-1].replace("ST_", "")


def describe_state(tok, ids) -> str:
    """Decoded fixed layout -> a compact readable position."""
    names = tok.decode(ids)
    lines = ["  " + " ".join(names[1:5])
             + "  my[" + " ".join(names[5:10]) + "]"
             + " opp[" + " ".join(names[10:15]) + "]"]
    for side, base in (("my ", tok.my_base), ("opp", tok.opp_base)):
        for b in _blocks(names, base):
            if b[0] in ("PAD", "UNSEEN"):
                continue
            moves = ",".join(_short(m) for m in b[6:10]
                             if m not in ("NO_MOVE", "UNK_MOVE"))
            boosts = " ".join(f"{s}{v[6:]}" for s, v in
                              zip(("atk", "def", "spa", "spd", "spe", "acc", "eva"),
                                  b[10:17]) if v != "BOOST_0")
            lines.append(f"  {side} {b[0]:7s} {_short(b[1]):16s} "
                         f"{_short(b[2]):12s} {b[4]:5s} {_short(b[5]):4s} "
                         f"[{moves}]" + (f" {{{boosts}}}" if boosts else ""))
    return "\n".join(lines)


def describe_action(tok, ids, pair) -> str:
    """(slot_a_idx, slot_b_idx) -> readable, resolving move/species names
    from the token blocks."""
    names = tok.decode(ids)
    blocks = _blocks(names, tok.my_base)
    out = []
    for slot, ai in enumerate(pair):
        a = from_index(int(ai))
        if a.kind == "pass":
            out.append("pass")
        elif a.kind == "switch":
            out.append("sw " + _short(blocks[a.switch_to][1]))
        else:
            blk = next((b for b in blocks
                        if b[0] == ("SLOT_A" if slot == 0 else "SLOT_B")), None)
            mv = _short(blk[6 + a.move_slot]) if blk else f"m{a.move_slot + 1}"
            out.append(mv + TGT[a.target] + ("+mega" if a.mega else ""))
    return ", ".join(out)


def worst(flat, acts, ds, tok, n):
    label = acts[:, 0] * N_SLOT_ACTIONS + acts[:, 1]
    p = flat[np.arange(len(flat)), label]
    print(f"\n=== {n} highest-loss test positions (model vs human) ===")
    for i in np.argsort(p)[:n]:
        print(f"\n--- sample {i}, p(human action) = {p[i]:.4f} ---")
        print(describe_state(tok, ds.tokens[i]))
        print(f"  human: {describe_action(tok, ds.tokens[i], acts[i])}")
        for t in np.argsort(-flat[i])[:3]:
            print(f"  model {flat[i][t]:5.1%}: "
                  + describe_action(tok, ds.tokens[i], divmod(int(t), N_SLOT_ACTIONS)))


def aux_report(model, ds, cfg):
    """Aux set-prediction head vs oracle sheets. Compare with
    `beliefs.py --audit`: if the net beats the filter on items, the filter
    prior is too narrow; if the filter wins, the net underuses evidence."""
    it_hit = it_n = ab_hit = ab_n = mv_hit = mv_n = 0
    for i in range(0, len(ds), cfg.batch_size):
        _, _, aux = model.predict_batch(ds.tokens[i:i + cfg.batch_size])
        for pred, true, hn in (
                (aux["items"], ds.opp_items[i:i + cfg.batch_size], "it"),
                (aux["abilities"], ds.opp_abils[i:i + cfg.batch_size], "ab")):
            top1 = pred[:, :, 1:].argmax(-1) + 1     # 0 = unknown, excluded
            m = true != 0
            hit, n = int((top1[m] == true[m]).sum()), int(m.sum())
            if hn == "it":
                it_hit, it_n = it_hit + hit, it_n + n
            else:
                ab_hit, ab_n = ab_hit + hit, ab_n + n
        top4 = np.argsort(-aux["moves"][:, :, 1:], axis=-1)[..., :4] + 1
        true_mv = ds.opp_moves[i:i + cfg.batch_size]
        m = true_mv != 0
        hits = (true_mv[..., :, None] == top4[..., None, :]).any(-1)
        mv_hit += int(hits[m].sum())
        mv_n += int(m.sum())
    print("\naux head vs oracle sheets (per opponent mon per position):")
    print(f"  item    top-1: {it_hit / it_n:.1%}   ({it_n} labels)")
    print(f"  ability top-1: {ab_hit / ab_n:.1%}   ({ab_n} labels)")
    print(f"  moves  hit@4:  {mv_hit / mv_n:.1%}   ({mv_n} labels)")


def main(cfg=CFG):
    ckpt = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else cfg.checkpoint_dir / "ckpt_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Shards("test", cfg)
    files = sorted((cfg.prepped_dir).glob("test_*.npz"))
    dmg_active = np.concatenate([np.load(f)["dmg_active"] for f in files])
    acts = ds.acts.astype(np.int64)
    print(f"test transitions: {len(ds)}")
    mask = joint_mask()

    model = PolicyValueNet.load(ckpt, cfg, device)
    dists = []
    for i in range(0, len(ds), cfg.batch_size):
        d, _, _ = model.predict_batch(ds.tokens[i:i + cfg.batch_size])
        dists.append(d)
    net = np.concatenate(dists).astype(np.float64)
    rows = {"policy net": score(net, acts, mask),
            "max damage": score(MaxDamagePolicy().predict_batch(dmg_active), acts, mask),
            "random": score(RandomPolicy().predict_batch(dmg_active), acts, mask)}

    cols = list(next(iter(rows.values())))
    print(f"\n{'':14s}" + "".join(f"{c:>11s}" for c in cols))
    for name, r in rows.items():
        print(f"{name:14s}" + "".join(f"{r[c]:11.3f}" for c in cols))
    print("\n(recall@k = pruned-set recall: human joint action inside model top-k; "
          "baseline perplexities use eps-smoothed one-hot/uniform dists)")

    if "--worst" in sys.argv or "--aux" in sys.argv:
        tok = PositionTokenizer.load(cfg)
        if "--aux" in sys.argv:
            aux_report(model, ds, cfg)
        if "--worst" in sys.argv:
            joint = net[:, 0, :, None] * net[:, 1, None, :] * mask
            joint /= joint.sum(axis=(1, 2), keepdims=True)
            i = sys.argv.index("--worst")
            worst(joint.reshape(len(joint), -1), acts, ds, tok,
                  int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 20)


if __name__ == "__main__":
    main()
