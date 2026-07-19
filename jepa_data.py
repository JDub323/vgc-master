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

The default (world-model) payload feeds ``train_jepa.py``; ``--consequence``
emits the own-move candidate/future payload for ``train_consequence.py``;
``--seq`` emits padded consecutive-transition windows (positions + both-sides
actions) for the v3 multi-step dynamics trainer ``train_strategy.py``.

CLI: python jepa_data.py [--out DIR] [--limit N] [--damage]
                         [--consequence | --seq]
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_RNG = random.Random(0)          # deterministic negative-candidate shuffling

from actions import from_index
from config import CFG
from data import battle_weight, iter_battles, sid
from jepa.config import JCFG
from jepa.features import (PASS_JOINT, action_arrays, legal_my_joints,
                           my_action_arrays)
from jepa.vocab import JEPAVocab


def _parsed_files(cfg):
    """Resolve the parsed-battle pickle paths from the configured dataset."""
    return [cfg.parsed_dir / f"{fn[len('logs_'):-len('.json')]}.pkl"
            for fn in cfg.dataset_files]


def prep(out_dir, limit=None, use_damage=False, mode="wm", cfg=CFG, jcfg=JCFG):
    """Generate transition shards; return the total transition count written.

    ``mode='wm'`` emits the v1 world-model payload (joint (a,b) + next state);
    ``mode='consequence'`` emits the v2 payload (own-move candidate sets + the
    future position that the consequence vector is trained to predict)."""
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
                                        else None, mode, jcfg.n_cand):
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
                  bridge, mode="wm", n_cand=12):
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
            steps.append((pos, state, acts[p] if acts else None,
                          acts[opp] if acts else None))
        belief.update(turn["events"], viewer=p)
    if mode == "seq":
        yield from _seq_windows(steps, outcome, w, vocab)
        return
    for i in range(len(steps) - 1):
        pos, state, a, b = steps[i]
        nxt = steps[i + 1][0]
        if mode == "consequence":
            if a is None:                      # only my action is required here
                continue
            yield _consequence_sample(pos, state, nxt, a, outcome, w, vocab, n_cand)
        else:
            if a is None or b is None:
                continue
            yield _wm_sample(pos, nxt, a, b, outcome, w, vocab)


def _seq_windows(steps, outcome, w, vocab, jcfg=JCFG):
    """Yield v3 sequence windows: K actions + K+1 positions, padded + masked.

    Multi-step JEPA unrolls need consecutive chains of fully-labelled turns.
    Runs are maximal sublists where both sides' actions are known; each run
    emits windows every ``seq_stride`` steps, padded to ``seq_len`` with
    ``n_steps`` recording the valid transition count."""
    K = jcfg.seq_len
    runs, cur = [], []
    for pos, _state, a, b in steps:
        if a is None or b is None:             # unlabelled turn breaks the chain
            if len(cur) > 1:
                runs.append(cur)
            cur = []
        else:
            cur.append((pos, a, b))
    if len(cur) > 1:
        runs.append(cur)
    for run in runs:
        for i in range(0, len(run) - 1, jcfg.seq_stride):
            m = min(K, len(run) - 1 - i)
            if m < 1:
                break
            yield _seq_sample(run[i:i + m + 1], m, K, outcome, w, vocab)


def _seq_sample(chunk, m, K, outcome, w, vocab):
    """Build one padded window sample from ``m`` transitions (m <= K)."""
    p0 = chunk[0][0]
    gcat = np.zeros((K + 1, 2), dtype=np.int16)
    gscal = np.zeros((K + 1,) + p0.global_scalar.shape, dtype=np.float32)
    mcat = np.zeros((K + 1,) + p0.mon_cat.shape, dtype=np.int16)
    mscal = np.zeros((K + 1,) + p0.mon_scalar.shape, dtype=np.float32)
    dmg = np.zeros((K + 1, 6, 6), dtype=np.float32)
    act = np.zeros((K, 12, 7), dtype=np.int16)
    a_slot = np.zeros((K, 2), dtype=np.int16)
    b_slot = np.zeros((K, 2), dtype=np.int16)
    for k in range(m + 1):
        pos = chunk[k][0]
        gcat[k] = pos.global_cat
        gscal[k] = pos.global_scalar
        mcat[k] = pos.mon_cat
        mscal[k] = pos.mon_scalar
        dmg[k] = pos.dmg_edge
    for k in range(m):
        pos, a, b = chunk[k]
        act[k] = action_arrays(pos, a, b, vocab)
        a_slot[k] = a
        b_slot[k] = b
    return {"pos_gcat": gcat, "pos_gscal": gscal, "pos_mcat": mcat,
            "pos_mscal": mscal, "pos_dmg": dmg, "act": act,
            "a_slot": a_slot, "b_slot": b_slot,
            "n_steps": np.int16(m), "value": np.int8(outcome),
            "weight": np.float32(w)}


def _cur_nxt(pos, nxt, outcome, w):
    """Shared cur/next/value/weight arrays for either shard mode."""
    return {
        "cur_gcat": pos.global_cat.astype(np.int16), "cur_gscal": pos.global_scalar,
        "cur_mcat": pos.mon_cat.astype(np.int16), "cur_mscal": pos.mon_scalar,
        "cur_dmg": pos.dmg_edge,
        "nxt_gcat": nxt.global_cat.astype(np.int16), "nxt_gscal": nxt.global_scalar,
        "nxt_mcat": nxt.mon_cat.astype(np.int16), "nxt_mscal": nxt.mon_scalar,
        "nxt_dmg": nxt.dmg_edge,
        "value": np.int8(outcome), "weight": np.float32(w)}


def _wm_sample(pos, nxt, a, b, outcome, w, vocab):
    """One world-model (v1) transition: joint (a,b) + explicit next state."""
    return {**_cur_nxt(pos, nxt, outcome, w),
            "act": action_arrays(pos, a, b, vocab).astype(np.int16),
            "a_slot": np.array(a, dtype=np.int16),
            "b_slot": np.array(b, dtype=np.int16)}


def _consequence_sample(pos, state, nxt, a, outcome, w, vocab, n_cand):
    """One consequence (v2) sample: own-move candidate set + future target.

    The taken action is at candidate index 0; the other candidates are a
    *shuffled* sample of the full legal own-joint set, so the negatives are a
    diverse mix of attacks/targets/switches (not switches-first). This forces
    the policy head to rank the human's move against genuine alternatives rather
    than a trivial attack-vs-switch split, and the JEPA loss matches candidate
    0's consequence vector to the future ``nxt``."""
    a_true = (from_index(int(a[0])), from_index(int(a[1])))
    others = [c for c in legal_my_joints(state, vocab, 256) if c != a_true]
    _RNG.shuffle(others)
    cands = [a_true] + others[:n_cand - 1]
    cand_acts = np.zeros((n_cand, 12, 7), dtype=np.int16)
    cand_mask = np.zeros(n_cand, dtype=bool)
    for j, cd in enumerate(cands):
        cand_acts[j] = action_arrays(pos, cd, PASS_JOINT, vocab)
        cand_mask[j] = True
    return {**_cur_nxt(pos, nxt, outcome, w),
            "my_act": my_action_arrays(pos, a_true, vocab).astype(np.int16),
            "cand_acts": cand_acts, "cand_mask": cand_mask,
            "a_index": np.int16(0)}


def main():
    """CLI entry: build transition shards from parsed battles."""
    args = sys.argv[1:]

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    mode = ("consequence" if "--consequence" in args
            else "seq" if "--seq" in args else "wm")
    default_out = {"consequence": "jepa_cons_prepped",
                   "seq": "jepa_seq_prepped"}.get(mode, "jepa_prepped")
    out = opt("--out", str(CFG.artifacts_dir / default_out))
    limit = opt("--limit")
    prep(out, limit=int(limit) if limit else None,
         use_damage="--damage" in args, mode=mode)


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
    else:
        main()
