"""Behavioral probes for a trained joint-action policy.

CLI: python probe_policy.py [checkpoint]
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("probe_policy.py"):
        raise SystemExit(0)

import json
import sys

import numpy as np

from actions import N_SLOT_ACTIONS, from_index
from config import CFG
from evaluation_common import load_test_predictions
from tokenizer import PositionTokenizer


def event_mask(kind, tok=None, tokens=None, moves=None, quantifier="both"):
    """Mask joints where ``both`` or ``any`` slots perform an event.

    Move events are position-dependent, so their move slots are decoded from
    each row.  ``status`` deliberately excludes the move named Protect.
    """
    combine = np.logical_and if quantifier == "both" else np.logical_or
    if kind == "switch":
        one = np.array([from_index(i).kind == "switch" for i in range(N_SLOT_ACTIONS)])
        return np.broadcast_to(combine(one[:, None], one[None, :]).reshape(1, -1),
                               (len(tokens), N_SLOT_ACTIONS ** 2))
    masks = np.zeros((len(tokens), N_SLOT_ACTIONS ** 2), dtype=bool)
    for r, ids in enumerate(tokens):
        names = tok.decode(ids)
        per_slot = []
        for slot, slot_name in enumerate(("SLOT_A", "SLOT_B")):
            blocks = [names[tok.my_base + k * tok.mon_block:
                            tok.my_base + (k + 1) * tok.mon_block] for k in range(6)]
            blk = next((b for b in blocks if b[0] == slot_name), None)
            move_slots = set()
            if blk is not None:
                for m, token in enumerate(blk[6:10]):
                    move = token[5:] if token.startswith("move:") else None
                    if kind == "protect" and move == "protect":
                        move_slots.add(m)
                    elif (kind == "status" and move != "protect" and move and
                          moves.get(move, {}).get("category") == "Status"):
                        move_slots.add(m)
            per_slot.append(np.array([from_index(i).kind == "move" and
                                      from_index(i).move_slot in move_slots
                                      for i in range(N_SLOT_ACTIONS)]))
        masks[r] = combine(per_slot[0][:, None], per_slot[1][None, :]).reshape(-1)
    return masks


def report(name, mask, net, labels, ks=(6, 16)):
    actual = mask[np.arange(len(labels)), labels]
    mass = (net * mask).sum(1)
    order = np.argsort(-net, axis=1)
    print(f"\n{name}:")
    print(f"  human frequency:       {actual.mean():.3%} ({actual.sum()}/{len(actual)})")
    print(f"  model mean probability:{mass.mean():.3%}")
    print(f"  model top-1 frequency: {mask[np.arange(len(mask)), order[:, 0]].mean():.3%}")
    for k in ks:
        present = np.take_along_axis(mask, order[:, :k], axis=1).any(1)
        print(f"  present in top-{k:<2}:     {present.mean():.3%}")


def main(cfg=CFG):
    ckpt = sys.argv[1] if len(sys.argv) > 1 else cfg.checkpoint_dir / "ckpt_best.pt"
    ds, _, _, net, _ = load_test_predictions(ckpt, cfg)
    labels = ds.acts[:, 0].astype(int) * N_SLOT_ACTIONS + ds.acts[:, 1].astype(int)
    tok = PositionTokenizer.load(cfg)
    with open(cfg.artifacts_dir / "dex.json") as f:
        moves = json.load(f)["moves"]
    report("both Pokémon switch",
           event_mask("switch", tokens=ds.tokens), net, labels)
    report("both Pokémon use Protect",
           event_mask("protect", tok, ds.tokens, moves), net, labels)
    report("at least one Pokémon switches",
           event_mask("switch", tokens=ds.tokens, quantifier="any"), net, labels)
    report("at least one Pokémon uses Protect",
           event_mask("protect", tok, ds.tokens, moves, "any"), net, labels)
    report("at least one Pokémon uses a non-Protect status move",
           event_mask("status", tok, ds.tokens, moves, "any"), net, labels)


if __name__ == "__main__":
    main()
