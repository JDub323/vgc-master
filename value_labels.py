"""Sidecar end-of-game margin labels for value training (exp/value-head).

The prepped shards carry one value target per transition: z = +-1 final
outcome. That is maximally high-variance (every position of a won game is a +1
example, however even it was). This pass derives two *shaped* auxiliary
targets per row — constant across a game like z, but graded:

  margin_faints  (opponent faints - own faints) / 4 at the last observed
                 state, clipped to [-1, 1]  (faints are public both ways)
  margin_hp      (own HP-fraction sum - opponent HP-fraction sum) / 6 at the
                 last observed state, clipped to [-1, 1]

Both are measured at the last decision point the parser recorded, from the
CTS-observable view (unrevealed opponent mons count as healthy), so they are
slight *underestimates* of the true final margin — acceptable for an
auxiliary shaping loss.

Alignment without touching the shared shards: `data.prep` writes rows in a
fully deterministic order (battle stream order x p1/p2 x turns where
``states`` and that side's action are observable). This pass replays exactly
that enumeration — skipping the expensive belief/damage/tokenize work — and
emits one margin row per shard row. Alignment is **proven, not assumed**:
each row's sample weight and outcome are recomputed and asserted equal to the
shard columns before anything is written.

CLI: python value_labels.py [--check-only]
Writes artifacts/value_labels/{train,val,test}.npz with ``margins`` float32
[N, 2] aligned 1:1 with the concatenated ``<split>_*.npz`` shard rows.
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("value_labels.py"):
        raise SystemExit(0)

import sys
from glob import glob

import numpy as np

from config import CFG
from data import battle_weight, iter_battles

SPLITS = ("train", "val", "test")


def battle_margins(rec, p):
    """Return ``(margin_faints, margin_hp)`` for perspective ``p``'s game."""
    last = None
    for turn in rec["turns"]:
        if turn["states"] is not None:
            last = turn["states"][p]
    if last is None:
        return 0.0, 0.0
    my, opp = last["my"]["team"], last["opp"]["team"]
    my_f = sum(bool(m["fainted"]) for m in my)
    opp_f = sum(bool(m["fainted"]) for m in opp)
    my_hp = sum(0.0 if m["fainted"] else float(m["hp"]) for m in my)
    opp_hp = sum(0.0 if m["fainted"] else float(m["hp"]) for m in opp)
    clip = lambda x: max(-1.0, min(1.0, x))
    return clip((opp_f - my_f) / 4.0), clip((my_hp - opp_hp) / 6.0)


def enumerate_rows(cfg=CFG):
    """Replay ``data.prep``'s exact row enumeration without the heavy work.

    Yields nothing; returns ``{split: dict(margins, weight, value)}`` with one
    entry per shard row, in shard row order."""
    files = [cfg.parsed_dir / f"{fn[len('logs_'):-len('.json')]}.pkl"
             for fn in cfg.dataset_files]
    max_ts = 0
    for rec in iter_battles(*files):
        max_ts = max(max_ts, rec["ts"])
    rows = {s: {"margins": [], "weight": [], "value": []} for s in SPLITS}
    for rec in iter_battles(*files):
        out = rows[rec["split"]]
        for p in ("p1", "p2"):
            n = sum(1 for turn in rec["turns"]
                    if turn["states"] is not None
                    and turn["actions"][p] is not None)
            if not n:
                continue
            mf, mh = battle_margins(rec, p)
            w = battle_weight(rec, p, max_ts, cfg)
            z = 1 if rec["winner"] == p else -1
            out["margins"].extend([(mf, mh)] * n)
            out["weight"].extend([w] * n)
            out["value"].extend([z] * n)
    return rows


def shard_columns(split, cfg=CFG):
    """Concatenate one split's shard ``weight``/``value`` columns in order."""
    files = sorted(glob(str(cfg.prepped_dir / f"{split}_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"no {split} shards under {cfg.prepped_dir} — run data.py prep "
            "on this machine first")
    parts = [np.load(f) for f in files]
    return (np.concatenate([p["weight"] for p in parts]),
            np.concatenate([p["value"] for p in parts]))


def build(cfg=CFG, check_only=False):
    """Derive, verify, and write the sidecar margin labels for every split."""
    rows = enumerate_rows(cfg)
    out_dir = cfg.artifacts_dir / "value_labels"
    for split in SPLITS:
        got = rows[split]
        weight, value = shard_columns(split, cfg)
        n_ours, n_shards = len(got["weight"]), len(weight)
        assert n_ours == n_shards, (
            f"{split}: enumerated {n_ours} rows but shards hold {n_shards} — "
            "parsed pickles and prepped shards disagree; re-check that both "
            "came from the same parse/prep run")
        assert np.array_equal(np.asarray(got["value"], dtype=np.int8), value), \
            f"{split}: recomputed outcomes do not match shard 'value' column"
        assert np.allclose(np.asarray(got["weight"], dtype=np.float32), weight,
                           rtol=1e-5, atol=1e-7), \
            f"{split}: recomputed weights do not match shard 'weight' column"
        margins = np.asarray(got["margins"], dtype=np.float32)
        print(f"{split}: {n_ours} rows aligned; margin_faints mean "
              f"{margins[:, 0].mean():+.3f}, margin_hp mean "
              f"{margins[:, 1].mean():+.3f}")
        if not check_only:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / f"{split}.npz", margins=margins)
            print(f"wrote {out_dir / (split + '.npz')}")


def load_margins(split, cfg=CFG):
    """Return one split's ``[N, 2]`` margins, or ``None`` if not built."""
    path = cfg.artifacts_dir / "value_labels" / f"{split}.npz"
    if not path.exists():
        return None
    return np.load(path)["margins"]


if __name__ == "__main__":
    build(check_only="--check-only" in sys.argv[1:])
