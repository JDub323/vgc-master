"""Sanity probes for a trained v3 Strategy-JEPA checkpoint.

Answers, on held-out sequence windows, the questions that decide how the
chooser should be configured — before any games are played:

  counterfactual value ranking   Does ``value(T(z, a_true, b))`` exceed
      ``value(T(z, a_other, b))`` for alternative own actions (drawn from
      other positions in the batch)? ~0.5 means the value head carries no
      off-policy signal and ``solver_eta`` should stay near 0 (play the
      prior); well above 0.5 means the payoff matrix is trustworthy and eta
      can rise. This is THE number behind the 0-for-17 failure mode.
  value calibration              Mean predicted value grouped by the actual
      game outcome (won/lost positions should separate cleanly).
  policy quality                 Per-slot top-1/top-3 of the prior head on
      observed actions (the anchor of the anchored solve).

CLI: python probe_strategy.py [--ckpt PATH] [--data DIR] [--shards N]
                              [--batches N]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from config import CFG
from train_strategy import _batches, _pos_at


@torch.no_grad()
def probe(ckpt, data_dir, n_shards=1, n_batches=40, bs=64):
    """Run all probes; print a report and return the metrics dict."""
    from models.jepa_strategy import JEPAStrategyModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = JEPAStrategyModel.load(ckpt, device)
    model.eval()
    paths = sorted(Path(data_dir).glob("val_*.npz")) or \
        sorted(Path(data_dir).glob("*.npz"))
    cf_hits = cf_n = 0
    top1 = top3 = pol_n = 0
    v_by_outcome = {1: [], -1: [], 0: []}
    for bi, sh in enumerate(_batches(paths, bs, device, shuffle=True,
                                     limit_shards=n_shards)):
        if bi >= n_batches:
            break
        b = sh["value"].shape[0]
        if b < 4:
            continue
        z0 = model.encode(_pos_at(sh, 0))
        act = sh["act"][:, 0]
        zp = model.step(z0, act, sh["pos_dmg"][:, 0])
        v_true = model.value(zp)

        # counterfactual: swap the OWN half of the action arrays (mon tokens
        # 0..5) with another row's, keeping the opponent half fixed
        perm = torch.roll(torch.arange(b, device=device), 1)
        act_cf = act.clone()
        act_cf[:, :6] = act[perm][:, :6]
        differs = (act_cf != act).any(-1).any(-1)
        v_cf = model.value(model.step(z0, act_cf, sh["pos_dmg"][:, 0]))
        usable = differs & sh["a_mask"][:, 0]
        cf_hits += int(((v_true > v_cf) & usable).sum())
        cf_n += int(usable.sum())

        for z, out in zip(v_true.cpu().numpy(), sh["value"].cpu().numpy()):
            v_by_outcome[int(out)].append(float(z))

        my_logits, _ = model.policies(z0)
        am = sh["a_mask"][:, 0]
        if am.any():
            tgt = sh["a_slot"][:, 0][am]
            lg = my_logits[am]
            top1 += int((lg.argmax(-1) == tgt).sum())
            top3 += int((lg.topk(3, -1).indices == tgt[..., None]).any(-1).sum())
            pol_n += int(am.sum()) * 2

    cf = cf_hits / max(1, cf_n)
    report = {
        "counterfactual_value_ranking": round(cf, 4),
        "value_mean_when_won": round(float(np.mean(v_by_outcome[1] or [0])), 4),
        "value_mean_when_lost": round(float(np.mean(v_by_outcome[-1] or [0])), 4),
        "policy_top1_per_slot": round(top1 / max(1, pol_n), 4),
        "policy_top3_per_slot": round(top3 / max(1, pol_n), 4),
        "n_counterfactuals": cf_n,
    }
    print(json.dumps(report, indent=1))
    if cf < 0.55:
        print("VERDICT: value head has ~no counterfactual signal -- set "
              "solver_eta near 0 (play the prior) until what-if grounding "
              "lands.")
    else:
        print(f"VERDICT: value ranks the human move {cf:.0%} of the time -- "
              "eta in spread units is usable.")
    return report


def main():
    """CLI entry: probe one checkpoint against held-out window shards."""
    args = sys.argv[1:]

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    probe(opt("--ckpt", str(CFG.checkpoint_dir / "jepa" / "jepa_strategy_l2.pt")),
          opt("--data", str(CFG.artifacts_dir / "jepa_seq_prepped")),
          n_shards=int(opt("--shards", 1)),
          n_batches=int(opt("--batches", 40)))


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
    else:
        main()
