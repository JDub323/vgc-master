"""Train LeadNet (bring-4/lead-2 imitation) from the parsed replay dataset.

Streams the parsed battle pickles (data.py parse output), extracts one
example per game per perspective (leads are public in turn 1; brought mons
are the ones that appeared), and fits the small transformer in
agents/lead_switch/leadnet.py. Uses the SAME vocab.json as the baseline
tokenizer and the same match-id splits, so val/test stay honest. Reports
lead-pair top-1/top-3 against the human choice plus the 'team 1234' floor
(how often humans actually led with the first two — what the baseline
adapter plays every game).

Runs fine on a laptop CPU: the net is ~1M params and the dataset is one
row per game, not per turn.

CLI: python train_leads.py [--epochs N] [--max-battles N] [--batch N]
                           [--out PATH] [--seed N]
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("train_leads.py"):
        raise SystemExit(0)

import sys
import time
from pathlib import Path

import torch

from agents.lead_switch.leadnet import (LeadNet, batches, evaluate_leadnet,
                                        extract_examples, loss_terms)
from agents.lead_switch.lscfg import LSCFG
from config import CFG
from tokenizer import PositionTokenizer


def main(cfg=CFG, ls=LSCFG):
    """CLI entry: extract examples, train, evaluate, save the checkpoint."""
    args = sys.argv[1:]

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    epochs = int(opt("--epochs", ls.nn_epochs))
    batch = int(opt("--batch", ls.nn_batch))
    max_battles = int(opt("--max-battles", 0)) or None
    seed = int(opt("--seed", 0))
    out = Path(opt("--out", cfg.checkpoint_dir / "leadnet.pt"))

    torch.manual_seed(seed)
    tok = PositionTokenizer.load(cfg)
    files = [cfg.parsed_dir / f"{fn[len('logs_'):-len('.json')]}.pkl"
             for fn in cfg.dataset_files]
    files = [f for f in files if f.exists()]
    assert files, "no parsed battles — run `python data.py parse` first"
    print(f"extracting preview examples from {len(files)} parsed files ...")
    ex = extract_examples(files, tok, cfg, ls, max_battles=max_battles)
    print(f"examples: train {len(ex['train'])}, val {len(ex['val'])}, "
          f"test {len(ex['test'])}")
    assert ex["train"], "no training examples extracted"

    model = LeadNet(tok.vocab_size(), ls)
    opt_ = torch.optim.AdamW(model.parameters(), lr=ls.nn_lr,
                             weight_decay=0.01)
    best_top1, t0 = 0.0, time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0.0
        for b in batches(ex["train"], batch, shuffle=True, seed=seed + ep):
            ce, bce = loss_terms(model, b, ls)
            loss = ce + ls.nn_bring_weight * bce
            opt_.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_.step()
            tot += float(loss) * len(b["pair"])
            n += len(b["pair"])
        val = evaluate_leadnet(model, ex["val"], ls) if ex["val"] else \
            {"pair_top1": 0.0, "pair_top3": 0.0, "n": 0,
             "first_pair_rate": 0.0}
        print(f"epoch {ep}: loss {tot / max(1, n):.4f}  "
              f"val top-1 {val['pair_top1']:.3f} top-3 {val['pair_top3']:.3f} "
              f"({time.time() - t0:.0f}s)")
        if val["pair_top1"] >= best_top1:
            best_top1 = val["pair_top1"]
            model.save(out)
            print(f"  saved {out}")
    if ex["test"]:
        best = LeadNet.load(out, ls)
        t = evaluate_leadnet(best, ex["test"], ls)
        print(f"test: n={t['n']}  lead-pair top-1 {t['pair_top1']:.3f}  "
              f"top-3 {t['pair_top3']:.3f}  "
              f"(humans led with the first two {t['first_pair_rate']:.1%} "
              f"of games — the 'team 1234' adapter floor)")


if __name__ == "__main__":
    main()
