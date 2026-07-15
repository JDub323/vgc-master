"""One-time legality/label sidecar builder for the seq2seq pointer model.

Training the pointer model needs, per transition row: the per-slot legal
action masks (reconstructed from tokens — legality is not stored in the
shards) and the per-slot *projected* label sets (the recorded label, or its
projection onto the move's real target codes for the 8.7% of labels whose
target the legal set never contains, KNOWN_ISSUES.md #3). Reconstructing
those runs per-row Python (``PositionLegality`` decodes every row), far too
slow to sit inside a training loop — so this script runs it once per split
and saves the result as a bit-packed sidecar aligned with the shard rows:

  artifacts/prepped/seq2seq_<split>.npz
    legal_a, legal_b   uint8 [N, 5]   packed bool[39] legal masks per slot
    label_a, label_b   uint8 [N, 5]   packed bool[39] projected label sets
    n_rows             int            alignment check against the shards

Row order matches ``train.Shards`` exactly (both concatenate the same sorted
``<split>_*.npz`` glob). The existing shards are never touched. Masks come
from ``models.seq2seq.slot_legal_masks`` / ``slot_label_sets``, the same
functions ``predict_batch`` uses at play time, so train and play cannot
drift.

CLI: python seq2seq_prep.py [split ...]     (default: train val test)
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("seq2seq_prep.py"):
        raise SystemExit(0)

import sys
import time
from glob import glob

import numpy as np

from actions import N_SLOT_ACTIONS
from config import CFG
from evaluation_common import PositionLegality
from models.seq2seq import slot_label_sets, slot_legal_masks
from tokenizer import PositionTokenizer


def pack(mask):
    """Bit-pack a bool[N, 39] mask to uint8[N, 5] (np.packbits, big-endian)."""
    return np.packbits(mask, axis=1)


def unpack(packed):
    """Invert ``pack``: uint8[N, 5] -> bool[N, 39]."""
    return np.unpackbits(packed, axis=1)[:, :N_SLOT_ACTIONS].astype(bool)


def build_split(split, legality, cfg=CFG, chunk=2000):
    """Build and save one split's sidecar; return its per-row statistics."""
    files = sorted(glob(str(cfg.prepped_dir / f"{split}_*.npz")))
    if not files:
        raise FileNotFoundError(f"no {split} shards under {cfg.prepped_dir}")
    tokens = np.concatenate([np.load(f)["tokens"] for f in files])
    acts = np.concatenate([np.load(f)["acts"] for f in files])
    n = len(tokens)
    la = np.zeros((n, N_SLOT_ACTIONS), dtype=bool)
    lb = np.zeros_like(la)
    sa = np.zeros_like(la)
    sb = np.zeros_like(la)
    projected = 0
    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        la[lo:hi], lb[lo:hi] = slot_legal_masks(legality, tokens[lo:hi])
        sa[lo:hi], sb[lo:hi], p = slot_label_sets(legality, tokens[lo:hi],
                                                  acts[lo:hi])
        projected += p
        done, dt = hi, time.time() - t0
        print(f"\r{split}: {done}/{n} rows "
              f"({done / dt:.0f} rows/s, {dt:.0f}s)", end="", flush=True)
    print()
    # a label set that misses the legal grid entirely would train on -inf;
    # count it here so train_seq2seq can zero those rows knowingly
    label_legal = ((sa & la).any(1)) & ((sb & lb).any(1))
    out = cfg.prepped_dir / f"seq2seq_{split}.npz"
    np.savez_compressed(out, legal_a=pack(la), legal_b=pack(lb),
                        label_a=pack(sa), label_b=pack(sb), n_rows=n)
    stats = {"rows": n, "projected": projected,
             "label_outside_legal": int((~label_legal).sum()),
             "mean_legal_a": float(la.sum(1).mean()),
             "mean_legal_b": float(lb.sum(1).mean())}
    print(f"{split}: wrote {out} | projected labels {projected} "
          f"({projected / n:.1%}) | label outside legal superset "
          f"{stats['label_outside_legal']} "
          f"({stats['label_outside_legal'] / n:.2%}) | mean legal "
          f"A {stats['mean_legal_a']:.1f} B {stats['mean_legal_b']:.1f}")
    return stats


def main(cfg=CFG):
    """Build the sidecars for the splits named on the CLI (default: all)."""
    splits = [a for a in sys.argv[1:] if not a.startswith("-")] \
        or ["train", "val", "test"]
    legality = PositionLegality(PositionTokenizer.load(cfg),
                                cfg.artifacts_dir / "dex.json")
    for split in splits:
        build_split(split, legality, cfg)


if __name__ == "__main__":
    main()
