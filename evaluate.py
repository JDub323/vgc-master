"""Predictor benchmarks on held-out test battles.

Headline metric: pruned-set recall@k — the fraction of positions where the
human's actual joint action is inside the model's top-k joint actions. That is
the direct measure of whether pruning the search to the model's top-k is safe.

Also: top-1/3/5 joint accuracy, perplexity (log-loss), and calibration (ECE),
since the probabilities weight the search.

The model emits joint distributions directly (predict_batch, either head
architecture); the per-slot baselines are recombined by outer product, masked
to game-legal combos and renormalized.

Metrics are reported over two action sets, because they answer different
questions and are NOT comparable to each other:

  static mask     every action legal in SOME position (the ~1521-way mask
                  baked into the model). The historical metric; EXPERIMENTS.md
                  rows are recorded against it, so it stays the default table.
  position-legal  only the actions legal in THIS position, renormalized —
                  what the search's prior actually sees (`PolicyValuePrior`
                  filters legality before top-k). Reconstructed from tokens by
                  `evaluation_common.PositionLegality`, which is permissive
                  about what tokens omit (disabled/PP/trapping/brought-four),
                  so its recall@k is a lower bound on the true number.

CLI: python evaluate.py [checkpoint]
     --worst N    decode the N test positions the model gets most wrong
                  (readable state + human action vs model top-3) — the
                  fastest way to see WHERE the model should improve
     --aux        auxiliary-head accuracy vs the oracle team sheets, the
                  learned counterpart to `beliefs.py --audit`
     --switches   switch-calibration report: does the model systematically
                  underweight switching relative to the humans it cloned?
                  (This measures the PRIOR; scenarios.py's diagnostic
                  positions measure the same bias after search.)
     --value      value-head quality vs the final outcome: MSE/Brier/sign
                  accuracy, calibration, confidence by game phase, and the
                  constant and HP-differential floors it must beat. At the v1
                  search budget the value head decides nearly every leaf, so
                  this is the counterpart to recall@k for the other head.
     --no-legal   skip the position-legal table (it costs a Python pass over
                  the test split)
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("evaluate.py"):
        raise SystemExit(0)

import sys

import numpy as np

from actions import N_SLOT_ACTIONS, from_index, static_joint_mask
from config import CFG
from evaluation_common import (PositionLegality, apply_legal_mask,
                               load_test_predictions)
from models.baselines import (DamageStatusSwitchCandidates, MaxDamagePolicy,
                              RandomPolicy)
from tokenizer import N_MONS, PositionTokenizer
from train import Shards

KS = (1, 3, 5, CFG.top_k_actions, 16)
TGT = {0: "", 1: ">1", 2: ">2", 3: ">ally"}


def factorized_to_joint(slot_dists):
    """Per-slot dists [B,2,A] (the baselines) -> flat joint [B, A*A],
    statically masked and renormalized — the v1 recombination."""
    joint = slot_dists[:, 0, :, None] * slot_dists[:, 1, None, :]   # [B,A,A]
    joint *= static_joint_mask()
    joint /= joint.sum(axis=(1, 2), keepdims=True)
    return joint.reshape(len(joint), -1)


def score_legal(flat, label_mask, legal):
    """Metrics over the position-legal action set, with label sets.

    Restricts and renormalizes the distribution over ``legal`` exactly as
    ``PolicyValuePrior`` does, then scores against ``label_mask`` — the set of
    joint actions consistent with the human's recorded label (a singleton
    unless the label's target needed projecting; see
    ``PositionLegality.label_mask``). p(label) is the summed mass of the set,
    i.e. the likelihood of the observed behavior, and a hit@k means any member
    of the set landed in the top k.

    Ranking goes through ``argsort`` rather than counting strictly-better
    actions, so ties break by action index — the same way the prior's own
    ``np.argsort(-p)[:k]`` cut does. It matters: a uniform prior ties every
    legal action, and counting strictly-better mass would score all of them
    rank 0 and report recall@k = 100% for the random floor."""
    flat = apply_legal_mask(flat, legal)
    p_label = (flat * label_mask).sum(axis=1)
    order = np.argsort(-flat, axis=1)
    in_set = np.take_along_axis(label_mask, order, axis=1)
    rank = np.where(in_set.any(axis=1), in_set.argmax(axis=1), flat.shape[1])
    out = {f"top{k}" if k <= 5 else f"recall@{k}": float((rank < k).mean())
           for k in KS}
    out["perplexity"] = float(np.exp(-np.log(np.clip(p_label, 1e-12, 1)).mean()))
    conf, hit = flat.max(axis=1), rank == 0
    ece, edges = 0.0, np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(hit[in_bin].mean() - conf[in_bin].mean())
    out["ece"] = float(ece)
    return out


def score(flat, acts):
    """flat joint dists [B, A*A], acts [B,2] -> metrics dict."""
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

def _blocks(tok, names, base):
    return [names[base + k * tok.mon_block: base + (k + 1) * tok.mon_block]
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
        for b in _blocks(tok, names, base):
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
    blocks = _blocks(tok, names, tok.my_base)
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


def switch_report(net, acts, k=CFG.top_k_actions):
    """Prior-level switch calibration. If the model's average switch mass
    tracks the human switch rate AND recall@k on human-switch positions
    matches non-switch positions, the prior is fine and any in-game
    underswitching lives in the search (reconstruction gaps, determinization
    paranoia). If these numbers are skewed, it is a data/architecture issue
    and no amount of search tuning fixes it."""
    sw = np.zeros(N_SLOT_ACTIONS * N_SLOT_ACTIONS, dtype=bool)
    for a in range(N_SLOT_ACTIONS):
        for b in range(N_SLOT_ACTIONS):
            sw[a * N_SLOT_ACTIONS + b] = (from_index(a).kind == "switch"
                                          or from_index(b).kind == "switch")
    label = acts[:, 0] * N_SLOT_ACTIONS + acts[:, 1]
    human_sw = sw[label]
    p_sw = net[:, sw].sum(1)
    order = np.argsort(-net, axis=1)
    rank = np.argmax(order == label[:, None], axis=1)
    print("\nswitch calibration (test split):")
    print(f"  humans chose a switch:        {human_sw.mean():.1%} of positions")
    print(f"  model mean P(any switch):     {p_sw.mean():.1%}"
          f"   (ratio {p_sw.mean() / max(1e-9, human_sw.mean()):.2f} — "
          "<1 = underweights switching)")
    print(f"  mean P(switch) when human did:    {p_sw[human_sw].mean():.1%}")
    print(f"  mean P(switch) when human didn't: {p_sw[~human_sw].mean():.1%}")
    for name, m in (("human switched", human_sw), ("human stayed", ~human_sw)):
        r = rank[m]
        print(f"  {name:16s}: top-1 {(r == 0).mean():.1%}   "
              f"recall@{k} {(r < k).mean():.1%}   ({m.sum()} positions)")
    print("  (recall gap between the two rows = how often search pruning "
          "throws away exactly the switches humans found)")


def _token_lookup(tok, prefix, cast=float):
    """Return an id-indexed array of the numeric suffix of ``prefix`` tokens.

    Non-matching ids map to NaN, so a fancy-indexed block of tokens can be
    reduced with ``np.nansum`` without decoding anything."""
    out = np.full(max(tok.vocab.values()) + 1, np.nan)
    for name, i in tok.vocab.items():
        if name.startswith(prefix):
            out[i] = cast(name[len(prefix):])
    return out


def value_report(values, ds, tok, cfg):
    """Report value-head quality against the recorded final outcome.

    The target is the same one training used: z = +1 if the encoded player won
    that game, else -1 (`data.prep`), so every position in a won game is a +1
    example regardless of how even it was. That makes the honest reading
    phase-dependent, which is why the breakdown by turn bucket matters more
    than the headline: a value head should be near-chance and diffident early,
    and accurate and confident late.

    Three floors, because the headline flatters a weak head without them. The
    labels are NOT 50/50 — `data.prep` drops transitions whose action is
    unobservable (flinch, sleep, KO'd before acting) and that correlates with
    losing, so the winner's side contributes more rows and "always predict
    win" already scores the base rate. Predicting 0 everywhere gives MSE
    1.000. And the HP differential is what a value head must beat to have
    learned anything beyond counting HP — if it doesn't, search built on it
    inherits exactly the myopia `rollout_depth` exists to paper over."""
    z = ds.value.astype(np.float64)
    v = np.clip(values, -1.0, 1.0)
    p, y = (v + 1) / 2, (z + 1) / 2          # win probability / outcome
    n = len(z)
    sign_acc = float(np.mean(np.sign(v) == np.sign(z)))
    base = float(y.mean())
    print(f"\nvalue head vs final outcome ({n} test transitions, "
          f"unweighted; z = +1 win / -1 loss):")
    print(f"  MSE          {np.mean((v - z) ** 2):.3f}   "
          f"(predicting 0 everywhere = 1.000)")
    print(f"  MAE          {np.mean(np.abs(v - z)):.3f}")
    print(f"  Brier        {np.mean((p - y) ** 2):.3f}   "
          f"(predicting 0.5 everywhere = 0.250)")
    print(f"  sign acc     {sign_acc:.1%}   (does it pick the winner?)")
    print(f"  mean |v|     {np.mean(np.abs(v)):.3f}   (mean confidence)")

    hp = _token_lookup(tok, "HP_")
    hp_of = lambda base_i: np.nansum(
        hp[ds.tokens[:, [base_i + k * tok.mon_block + 4
                         for k in range(N_MONS)]]], axis=1)
    diff = hp_of(tok.my_base) - hp_of(tok.opp_base)
    edge = diff != 0
    hp_acc = float(np.mean(np.sign(diff[edge]) == np.sign(z[edge]))) \
        if edge.any() else float("nan")
    print("\n  floors it has to beat:")
    print(f"    base rate      {max(base, 1 - base):.1%}   "
          f'("always predict win"; labels are {base:.1%} wins, not 50/50 — '
          f"prep drops unobservable-action rows and that tracks losing)")
    print(f"    HP differential{hp_acc:>7.1%}   (sign of summed HP%, on the "
          f"{edge.mean():.0%} of rows with an HP edge; the rest are ties, "
          f"mostly turn 1)")
    print(f"    value head     {sign_acc:>7.1%}   "
          f"({sign_acc - max(base, 1 - base):+.1%} vs base rate)")

    edges = np.linspace(0, 1, 11)
    ece = 0.0
    print("\n  calibration (predicted win% vs actual):")
    print(f"    {'bin':>12s} {'n':>7s} {'predicted':>10s} {'actual':>8s} "
          f"{'gap':>7s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (p > lo) & (p <= hi) if lo > 0 else (p >= 0) & (p <= hi)
        if not in_bin.any():
            continue
        pred, actual = p[in_bin].mean(), y[in_bin].mean()
        ece += in_bin.mean() * abs(actual - pred)
        print(f"    {lo:.1f}-{hi:.1f}".ljust(17)
              + f"{int(in_bin.sum()):7d} {pred:10.1%} {actual:8.1%} "
                f"{actual - pred:+7.1%}")
    print(f"  ECE {ece:.3f}")

    turn = _token_lookup(tok, "TURN_", cast=int)
    buckets = turn[ds.tokens[:, 1]]
    print("\n  by game phase (a value head should start unsure and end sure):")
    print(f"    {'turn bucket':>12s} {'n':>7s} {'sign acc':>9s} {'MSE':>7s} "
          f"{'mean |v|':>9s}")
    for b in sorted(set(buckets[~np.isnan(buckets)])):
        m = buckets == b
        print(f"    {int(b):>12d} {int(m.sum()):7d} "
              f"{np.mean(np.sign(v[m]) == np.sign(z[m])):9.1%} "
              f"{np.mean((v[m] - z[m]) ** 2):7.3f} {np.mean(np.abs(v[m])):9.3f}")


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


def _table(rows):
    """Print one name -> metrics-dict table."""
    cols = list(next(iter(rows.values())))
    print(f"\n{'':14s}" + "".join(f"{c:>11s}" for c in cols))
    for name, r in rows.items():
        print(f"{name:14s}" + "".join(f"{r[c]:11.3f}" for c in cols))


def main(cfg=CFG):
    ckpt = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else cfg.checkpoint_dir / "ckpt_best.pt"
    ds, dmg_active, model, net, values = load_test_predictions(ckpt, cfg)
    acts = ds.acts.astype(np.int64)
    print(f"test transitions: {len(ds)}")
    max_dmg = factorized_to_joint(MaxDamagePolicy().predict_batch(dmg_active))
    rnd = factorized_to_joint(RandomPolicy().predict_batch(dmg_active))

    print("\n=== static mask: every action legal in SOME position (~1521) ===")
    print("the historical metric — EXPERIMENTS.md rows are recorded here")
    _table({"policy net": score(net, acts),
            "max damage": score(max_dmg, acts),
            "random": score(rnd, acts)})
    print("\n(recall@k = pruned-set recall: human joint action inside model top-k; "
          "baseline perplexities use eps-smoothed one-hot/uniform dists)")

    tok = PositionTokenizer.load(cfg)
    if "--no-legal" not in sys.argv:
        pl = PositionLegality(tok, cfg.artifacts_dir / "dex.json")
        legal = pl.mask(ds.tokens)
        label_mask, projected = pl.label_mask(ds.tokens, acts)
        covered = (label_mask & legal).any(axis=1)
        print("\n=== position-legal: only actions legal HERE, renormalized ===")
        print("what the search's prior actually ranks (PolicyValuePrior masks "
              "legality before top-k)")
        _table({"policy net": score_legal(net, label_mask, legal),
                "max damage": score_legal(max_dmg, label_mask, legal),
                "random": score_legal(rnd, label_mask, legal)})
        print(f"\nlegal actions/position: mean {legal.sum(1).mean():.1f}, "
              f"median {int(np.median(legal.sum(1)))} (of {legal.shape[1]})")
        print(f"self-check — label set intersects the legal set: "
              f"{covered.mean():.2%} (should be ~100%; the human played it. "
              f"A shortfall means the reconstruction is wrong, not the model.)")
        print(f"labels needing target projection: {projected / len(ds):.2%} "
              f"— data.py records a (move, target) the search cannot propose "
              f"(spread move without its [spread] tag; single-target move with "
              f"no target ref defaulting to AUTO). Projected onto the move's "
              f"real target codes and scored as a set, so this is a metric "
              f"fix, not a data fix: those rows still train the model toward "
              f"an index search never reads.")
        print("NOT comparable to the static-mask table above. Permissive about "
              "what tokens omit (disabled/PP/trapping/brought-four), so this "
              "recall@k is a LOWER bound on the true number.")

    sanity = DamageStatusSwitchCandidates(tok, cfg.artifacts_dir / "dex.json")
    ranked = sanity.ranked(ds.tokens, dmg_active, 16)
    labels = acts[:, 0] * N_SLOT_ACTIONS + acts[:, 1]
    print("\nsanity candidate-set recall (4 max-damage target combos first; "
          "then seeded-random status/damage/switch combos):")
    for k in (cfg.top_k_actions, 16):
        hit = np.mean([int(label in row[:k]) for label, row in zip(labels, ranked)])
        print(f"  recall@{k}: {hit:.3f}")

    if "--value" in sys.argv:
        value_report(values, ds, tok, cfg)
    if "--switches" in sys.argv:
        switch_report(net, acts)
    if "--worst" in sys.argv or "--aux" in sys.argv:
        if "--aux" in sys.argv:
            aux_report(model, ds, cfg)
        if "--worst" in sys.argv:
            i = sys.argv.index("--worst")
            worst(net, acts, ds, tok,
                  int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 20)


if __name__ == "__main__":
    main()
