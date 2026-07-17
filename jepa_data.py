"""Build paired-transition shards for the JEPA world model.

Reads the shared parsed battles (``data.iter_battles`` over ``artifacts/parsed``)
and emits, for both perspectives of every game, transitions
``(features(s_t), a_true, b_true) -> features(s_{t+1})`` plus the game outcome
and a sample weight. The belief filter is run turn-by-turn exactly as in
``data.prep`` so the opponent tokens carry the same CTS posterior the live
agent sees; the opponent's *moveset* is taken from the oracle sheet (the true
determinization), which is also what makes the recorded action's move-slot
indices resolve to the right move identities.

Writes to a NEW directory (default ``artifacts/jepa_prepped``); it never touches
the layout-3 shards under ``artifacts/prepped``.

CLI: python jepa_data.py [--out DIR] [--limit N] [--damage]
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import CFG
from data import battle_weight, iter_battles, sid
from jepa.config import JCFG
from jepa.features import action_arrays
from jepa.vocab import JEPAVocab


def _parsed_files(cfg):
    """Resolve the parsed-battle pickle paths from the configured dataset."""
    return [cfg.parsed_dir / f"{fn[len('logs_'):-len('.json')]}.pkl"
            for fn in cfg.dataset_files]


def prep(out_dir, limit=None, use_damage=False, cfg=CFG, jcfg=JCFG):
    """Generate transition shards; return the total transition count written."""
    import json

    from beliefs import OpponentBelief
    vocab = JEPAVocab.build(cfg)
    from jepa.features import FeatureExtractor
    extractor = FeatureExtractor(vocab)
    bridge = None
    if use_damage:
        from damage import DamageBridge
        bridge = DamageBridge(cfg)
    usage = json.loads((cfg.artifacts_dir / "usage_stats.json").read_text())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab_state.json").write_text(json.dumps(vocab.state()))

    files = _parsed_files(cfg)
    max_ts = n_battles = 0
    for i, rec in enumerate(iter_battles(*files)):
        if limit is not None and i >= limit:   # keep smokes off the 1.9GB file
            break
        max_ts = max(max_ts, rec["ts"])
        n_battles += 1

    buf = defaultdict(list)
    shard_by_split = {"train": 0, "val": 0, "test": 0}
    total = 0

    def flush(split, force=False):
        """Write one shard for ``split`` when the buffer is full enough."""
        rows = buf.get(("n", split), 0)
        if not rows or (not force and rows < jcfg.shard_size):
            return
        arrs = {k[1]: np.stack(v) for k, v in buf.items()
                if isinstance(k, tuple) and k[0] == "d" and k[2] == split}
        path = out_dir / f"{split}_{shard_by_split[split]:03d}.npz"
        np.savez_compressed(path, **arrs)
        print(f"wrote {path.name} ({rows} transitions)")
        shard_by_split[split] += 1
        for key in [k for k in buf if isinstance(k, tuple) and k[-1] == split]:
            buf[key] = []
        buf[("n", split)] = 0

    def add(split, sample):
        """Append one transition sample's arrays into the split's buffer."""
        for name, arr in sample.items():
            buf[("d", name, split)].append(arr)
        buf[("n", split)] = buf.get(("n", split), 0) + 1

    for bi, rec in enumerate(iter_battles(*files)):
        if limit is not None and bi >= limit:
            break
        split = rec["split"]
        for p in ("p1", "p2"):
            opp = "p2" if p == "p1" else "p1"
            belief = OpponentBelief([sid(s["species"]) for s in rec["teams"][opp]],
                                    usage, cfg,
                                    bridge if cfg.use_belief_damage_updates else None,
                                    my_team=rec["teams"][p])
            w = battle_weight(rec, p, max_ts, cfg)
            outcome = 1 if rec["winner"] == p else -1
            opp_ms = [[sid(m) for m in s["moves"]][:4]
                      for s in rec["teams"][opp]]
            for sample in _battle_steps(rec, p, opp, belief, extractor, vocab,
                                        opp_ms, outcome, w, bridge if use_damage
                                        else None):
                add(split, sample)
                total += 1
            for s in shard_by_split:
                flush(s)
        if (bi + 1) % 500 == 0:
            print(f"{bi + 1}/{n_battles} battles -> {total} transitions")
    for s in shard_by_split:
        flush(s, force=True)
    if bridge:
        bridge.close()
    print(f"done: {total} transitions across {sum(shard_by_split.values())} shards")
    return total


def _battle_steps(rec, p, opp, belief, extractor, vocab, opp_ms, outcome, w,
                  bridge):
    """Yield one transition dict per consecutive labelled turn pair."""
    from damage import damage_features
    steps = []
    for turn in rec["turns"]:
        if turn["states"] is not None:
            state = turn["states"][p]
            summ = belief.summary()
            dmg = damage_features(state, belief, bridge) if bridge else None
            pos = extractor.extract(state, summ, opp_movesets=opp_ms, dmg=dmg)
            acts = turn["actions"]
            steps.append((pos, acts[p] if acts else None,
                          acts[opp] if acts else None))
        belief.update(turn["events"], viewer=p)
    for i in range(len(steps) - 1):
        pos, a, b = steps[i]
        if a is None or b is None:
            continue
        nxt = steps[i + 1][0]
        act = action_arrays(pos, a, b, vocab)
        yield {
            "cur_gcat": pos.global_cat.astype(np.int16),
            "cur_gscal": pos.global_scalar,
            "cur_mcat": pos.mon_cat.astype(np.int16),
            "cur_mscal": pos.mon_scalar,
            "cur_dmg": pos.dmg_edge,
            "act": act.astype(np.int16),
            "nxt_gcat": nxt.global_cat.astype(np.int16),
            "nxt_gscal": nxt.global_scalar,
            "nxt_mcat": nxt.mon_cat.astype(np.int16),
            "nxt_mscal": nxt.mon_scalar,
            "nxt_dmg": nxt.dmg_edge,
            "value": np.int8(outcome),
            "weight": np.float32(w),
            "a_slot": np.array(a, dtype=np.int16),
            "b_slot": np.array(b, dtype=np.int16),
        }


def main():
    """CLI entry: build transition shards from parsed battles."""
    args = sys.argv[1:]

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    out = opt("--out", str(CFG.artifacts_dir / "jepa_prepped"))
    limit = opt("--limit")
    prep(out, limit=int(limit) if limit else None, use_damage="--damage" in args)


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
    else:
        main()
