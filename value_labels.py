"""Sidecar end-of-game margin and game-progression labels (exp/value-head).

The prepped shards carry one value target per transition: z = +-1 final
outcome. That is maximally high-variance — every position of a won game is a
+1 example, however even it was, and however much of the eventual result
depended on play many turns after this one (comebacks, opponent blunders).
This pass derives per-row auxiliary quantities that let training discount
that noise instead of fitting it:

  margin_faints  (opponent faints - own faints) / 4 at the last observed
                 state, clipped to [-1, 1]  (faints are public both ways)
  margin_hp      (own HP-fraction sum - opponent HP-fraction sum) / 6 at the
                 last observed state, clipped to [-1, 1]
  progression    this row's turn number / the battle's final turn number,
                 clipped to [0, 1] — 0 near team preview, 1 at the last turn
                 played. A proxy for "how much of the eventual outcome this
                 specific position could plausibly explain" — early turns are
                 the ones most likely to get an outcome unrelated to the
                 position (the opponent can still blunder, get read, or claw
                 back), so a value_lab.py candidate can down-weight them
                 instead of trying to fit that noise.
  abandoned      1 when this row's game ended with at most one total faint
                 across both sides — rage-quits, disconnects, and misclick
                 concessions whose +-1 outcome label is close to pure noise
                 (measured at ~6% of games, almost all over by turn 3; a
                 decisive KO ending shows the loser at 3+ faints at the last
                 observed state because the 4th lands during the final
                 turn's resolution). value_lab.py drops these rows from the
                 value loss by default.

margin_faints/margin_hp are measured at the last decision point the parser
recorded, from the CTS-observable view (unrevealed opponent mons count as
healthy), so they are slight *underestimates* of the true final margin —
acceptable for an auxiliary shaping loss.

Alignment without touching the shared shards: `data.prep` writes rows in a
fully deterministic order (battle stream order x p1/p2 x turns where
``states`` and that side's action are observable). This pass replays exactly
that enumeration — skipping the expensive belief/damage/tokenize work — and
emits one row per shard row. Alignment is **proven, not assumed**: each row's
sample weight and outcome are recomputed and asserted equal to the shard
columns, and progression is cross-checked against the shard's own decoded
TURN_ bucket token (the same bucketing tokenizer.py already applies), before
anything is written.

CLI: python value_labels.py [--check-only]
Writes artifacts/value_labels/{train,val,test}.npz with ``margins`` float32
[N, 2], ``progression`` float32 [N], ``abandoned`` uint8 [N], and
``final_turn`` int16 [N], aligned 1:1 with the concatenated ``<split>_*.npz``
shard rows.
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
from tokenizer import TURN_EDGES, PositionTokenizer

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


def battle_final_turn(rec):
    """Return the highest recorded turn number in this battle (>= 1)."""
    return max((t["n"] for t in rec["turns"]), default=0)


def battle_total_faints(rec):
    """Total faints across both sides at the last observed state.

    Perspective-independent (faints are public), so p1's view suffices: its
    ``my`` team is p1's mons, its ``opp`` team is p2's. A normally-decisive
    game ends with the loser at 3+ faints (the 4th lands during the final
    turn's resolution, after the last decision), so <= 1 total faint marks a
    game abandoned early — rage-quit, disconnect, or misclick concession —
    whose +-1 outcome label carries little information about the positions."""
    last = None
    for turn in rec["turns"]:
        if turn["states"] is not None:
            last = turn["states"]["p1"]
    if last is None:
        return 0
    return (sum(bool(m["fainted"]) for m in last["my"]["team"])
            + sum(bool(m["fainted"]) for m in last["opp"]["team"]))


def turn_bucket(turn_n):
    """Bucket a raw turn number the same way ``tokenizer.py``'s TURN_ token
    does, so progression can be cross-checked against the encoded token."""
    return sum(int(turn_n > e) for e in TURN_EDGES)


def enumerate_rows(cfg=CFG):
    """Replay ``data.prep``'s exact row enumeration without the heavy work.

    Returns ``{split: dict(margins, weight, value, progression, turn_bucket,
    abandoned, final_turn)}`` with one entry per shard row, in shard row
    order. ``abandoned`` and ``final_turn`` are per-game and broadcast to
    every row of that game."""
    files = [cfg.parsed_dir / f"{fn[len('logs_'):-len('.json')]}.pkl"
             for fn in cfg.dataset_files]
    max_ts = 0
    for rec in iter_battles(*files):
        max_ts = max(max_ts, rec["ts"])
    rows = {s: {"margins": [], "weight": [], "value": [], "progression": [],
               "turn_bucket": [], "abandoned": [], "final_turn": []}
            for s in SPLITS}
    for rec in iter_battles(*files):
        out = rows[rec["split"]]
        final_turn = max(1, battle_final_turn(rec))
        abandoned = int(battle_total_faints(rec) <= 1)
        for p in ("p1", "p2"):
            mf, mh = battle_margins(rec, p)
            w = battle_weight(rec, p, max_ts, cfg)
            z = 1 if rec["winner"] == p else -1
            for turn in rec["turns"]:
                if turn["states"] is None or turn["actions"][p] is None:
                    continue
                out["margins"].append((mf, mh))
                out["weight"].append(w)
                out["value"].append(z)
                out["progression"].append(min(1.0, turn["n"] / final_turn))
                out["turn_bucket"].append(turn_bucket(turn["n"]))
                out["abandoned"].append(abandoned)
                out["final_turn"].append(final_turn)
    return rows


def shard_columns(split, cfg=CFG):
    """Concatenate one split's ``weight``/``value`` columns and the TURN_
    token column (``tokens[:, 1]``) in shard order.

    Only column 1 of the token matrix is pulled, not the whole ~561-wide
    array — loading full ``tokens`` for the train split would cost ~14GB and
    is unnecessary for the alignment cross-check."""
    files = sorted(glob(str(cfg.prepped_dir / f"{split}_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"no {split} shards under {cfg.prepped_dir} — run data.py prep "
            "on this machine first")
    weight, value, turn_tok = [], [], []
    for f in files:
        part = np.load(f)
        weight.append(part["weight"])
        value.append(part["value"])
        turn_tok.append(part["tokens"][:, 1])
    return (np.concatenate(weight), np.concatenate(value),
            np.concatenate(turn_tok))


def build(cfg=CFG, check_only=False):
    """Derive, verify, and write the sidecar labels for every split."""
    rows = enumerate_rows(cfg)
    tok = PositionTokenizer.load(cfg)
    turn_token_id = {i: name for name, i in tok.vocab.items()
                     if name.startswith("TURN_")}
    out_dir = cfg.artifacts_dir / "value_labels"
    for split in SPLITS:
        got = rows[split]
        weight, value, turn_tok = shard_columns(split, cfg)
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
        shard_bucket = np.array(
            [int(turn_token_id[tid][len("TURN_"):]) for tid in turn_tok])
        assert np.array_equal(np.asarray(got["turn_bucket"]), shard_bucket), \
            (f"{split}: recomputed turn buckets do not match the shard's own "
             "TURN_ token — progression would be misaligned")
        margins = np.asarray(got["margins"], dtype=np.float32)
        progression = np.asarray(got["progression"], dtype=np.float32)
        abandoned = np.asarray(got["abandoned"], dtype=np.uint8)
        final_turn = np.asarray(got["final_turn"], dtype=np.int16)
        print(f"{split}: {n_ours} rows aligned (incl. turn-bucket "
              f"cross-check); margin_faints mean {margins[:, 0].mean():+.3f}, "
              f"margin_hp mean {margins[:, 1].mean():+.3f}, "
              f"progression mean {progression.mean():.3f}; "
              f"abandoned rows {abandoned.mean():.1%}, "
              f"rows from games >14 turns "
              f"{float(np.mean(final_turn > 14)):.1%} (value_lab drops both "
              f"from the value loss by default)")
        if not check_only:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / f"{split}.npz", margins=margins,
                                progression=progression, abandoned=abandoned,
                                final_turn=final_turn)
            print(f"wrote {out_dir / (split + '.npz')}")


def load_sidecar(split, cfg=CFG):
    """Return one split's sidecar arrays as a dict, or ``None`` if not built.

    Keys: ``margins`` [N,2], ``progression`` [N], ``abandoned`` [N] (uint8),
    ``final_turn`` [N] (int16)."""
    path = cfg.artifacts_dir / "value_labels" / f"{split}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in ("margins", "progression", "abandoned",
                                 "final_turn") if k in data}


def load_margins(split, cfg=CFG):
    """Return one split's ``[N, 2]`` margins, or ``None`` if not built."""
    side = load_sidecar(split, cfg)
    return None if side is None else side["margins"]


def load_progression(split, cfg=CFG):
    """Return one split's ``[N]`` progression fractions, or ``None``."""
    side = load_sidecar(split, cfg)
    return None if side is None else side["progression"]


if __name__ == "__main__":
    build(check_only="--check-only" in sys.argv[1:])
