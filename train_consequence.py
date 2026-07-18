"""Train the JEPA-Consequence model on own-move / future shards.

End-to-end objective (see ``JEPA_DESIGN.md`` "Consequence variant"):

  * JEPA latent loss — the taken move's predicted consequence vector is matched
    (smooth-L1) to an EMA target-encoder's embedding of the realized future
    position. No explicit next state is reconstructed; because one (position,
    move) maps to many futures, the predictor learns the expected future
    embedding and thereby the engine/opponent/luck dynamics.
  * Policy behavior-cloning — the policy head ranks each legal move's predicted
    consequence vector; cross-entropy toward the human's actual move makes the
    consequence vectors sufficient for comparing moves.
  * Value — win probability read off the taken move's consequence (real game
    outcome, not a handcrafted target).
  * VICReg on the encoder latents (anti-collapse), with the EMA target's
    stop-grad asymmetry.

CLI: python train_consequence.py [--data DIR] [--out PATH] [--epochs N]
                                 [--bs N] [--limit-shards N]
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from jepa.config import JCFG
from jepa.vocab import JEPAVocab
from models.jepa_consequence import JEPAConsequenceModel
from train_jepa import _vicreg


def _load_shard(path, device):
    """Load one consequence npz shard into a dict of device tensors."""
    z = np.load(path)
    long = lambda k: torch.as_tensor(z[k].astype(np.int64), device=device)
    flt = lambda k: torch.as_tensor(z[k].astype(np.float32), device=device)
    return {
        "cur_gcat": long("cur_gcat"), "cur_gscal": flt("cur_gscal"),
        "cur_mcat": long("cur_mcat"), "cur_mscal": flt("cur_mscal"),
        "cur_dmg": flt("cur_dmg"),
        "nxt_gcat": long("nxt_gcat"), "nxt_gscal": flt("nxt_gscal"),
        "nxt_mcat": long("nxt_mcat"), "nxt_mscal": flt("nxt_mscal"),
        "nxt_dmg": flt("nxt_dmg"),
        "value": flt("value"), "weight": flt("weight"),
        "my_act": long("my_act"), "cand_acts": long("cand_acts"),
        "cand_mask": torch.as_tensor(z["cand_mask"], device=device),
        "a_index": long("a_index"),
    }


def _pos(sh, prefix):
    """Slice a shard dict into an encoder-ready position tensor dict."""
    return {"gcat": sh[prefix + "gcat"], "gscal": sh[prefix + "gscal"],
            "mcat": sh[prefix + "mcat"], "mscal": sh[prefix + "mscal"],
            "dmg": sh[prefix + "dmg"]}


def losses(model, sh, jcfg):
    """Total loss + metrics for one consequence minibatch."""
    cur, nxt = _pos(sh, "cur_"), _pos(sh, "nxt_")
    z = model.encode(cur)                                   # [B,16,d]
    b, _, d = z.shape

    xi = model.sample_noise(b, z.device)
    c_true = model.consequence(z, sh["my_act"], cur["dmg"], xi)     # [B,d]
    target = model.target_context(nxt)                             # [B,d] sg
    jepa_l = F.smooth_l1_loss(c_true, target)

    w = sh["weight"]
    wn = w / w.mean().clamp_min(1e-6)
    v = model.value(c_true)
    value_l = (wn * (v - sh["value"]) ** 2).mean()

    # policy behavior-cloning over the candidate own moves
    nc = sh["cand_acts"].shape[1]
    z_rep = z.unsqueeze(1).expand(b, nc, z.shape[1], d).reshape(b * nc, z.shape[1], d)
    dmg_rep = cur["dmg"].unsqueeze(1).expand(b, nc, 6, 6).reshape(b * nc, 6, 6)
    act = sh["cand_acts"].reshape(b * nc, 12, 7)
    xi2 = model.sample_noise(b * nc, z.device)
    c_cand = model.consequence(z_rep, act, dmg_rep, xi2)           # [B*nc,d]
    logits = model.score(c_cand).reshape(b, nc)
    logits = logits.masked_fill(~sh["cand_mask"], float("-inf"))
    bc_l = F.cross_entropy(logits, sh["a_index"])

    var_l, cov_l = _vicreg(z, jcfg.vicreg_gamma)
    total = (jcfg.w_jepa_c * jepa_l + jcfg.w_value_c * value_l + jcfg.w_bc * bc_l
             + jcfg.w_vicreg_var * var_l + jcfg.w_vicreg_cov * cov_l)
    acc = (logits.argmax(-1) == sh["a_index"]).float().mean()
    return total, {"total": total.item(), "jepa": jepa_l.item(),
                   "value": value_l.item(), "bc": bc_l.item(),
                   "bc_acc": acc.item(), "var": var_l.item()}


def _batches(paths, bs, device, shuffle=True, limit_shards=None):
    """Yield minibatch shard-dicts, loading one npz shard at a time."""
    paths = list(paths)[:limit_shards] if limit_shards else list(paths)
    for path in paths:
        sh = _load_shard(path, device)
        n = sh["value"].shape[0]
        order = torch.randperm(n, device=device) if shuffle \
            else torch.arange(n, device=device)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            yield {k: v[idx] for k, v in sh.items()}


def train(data_dir, out_path, cfg=CFG, jcfg=JCFG, epochs=None, limit_shards=None):
    """Fit the consequence model; write best/last checkpoints; return best loss."""
    data_dir = Path(data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from beliefs import load_dex
    vstate_p = data_dir / "vocab_state.json"
    vocab = (JEPAVocab.from_state(json.loads(vstate_p.read_text()), load_dex(cfg))
             if vstate_p.exists() else JEPAVocab.build(cfg))

    train_paths = sorted(data_dir.glob("train_*.npz")) or sorted(data_dir.glob("*.npz"))
    # Record whether the shards actually carry damage edges, so the chooser
    # only builds a damage bridge at play time when the model trained on one.
    jcfg.use_damage_features = bool(train_paths) and \
        float(np.abs(np.load(train_paths[0])["cur_dmg"]).max()) > 0
    model = JEPAConsequenceModel(vocab.sizes(), jcfg, vocab.state()).to(device)
    val_paths = sorted(data_dir.glob("val_*.npz"))
    epochs = epochs or jcfg.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=jcfg.lr,
                            weight_decay=jcfg.weight_decay)
    per = [int(np.load(p)["value"].shape[0]) for p in
           (train_paths[:limit_shards] if limit_shards else train_paths)]
    steps_total = max(1, epochs * sum(per) // jcfg.batch_size)
    step = 0

    def lr_at(s):
        """Cosine schedule with linear warmup."""
        if s < jcfg.warmup_steps:
            return jcfg.lr * (s + 1) / jcfg.warmup_steps
        prog = (s - jcfg.warmup_steps) / max(1, steps_total - jcfg.warmup_steps)
        return 0.5 * jcfg.lr * (1 + math.cos(math.pi * min(1.0, prog)))

    best = float("inf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        agg = {}
        for sh in _batches(train_paths, jcfg.batch_size, device,
                           limit_shards=limit_shards):
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step)
            loss, m = losses(model, sh, jcfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), jcfg.grad_clip)
            opt.step()
            model.update_ema()
            step += 1
            for k, val in m.items():
                agg[k] = agg.get(k, 0.0) + val
            agg["_n"] = agg.get("_n", 0) + 1
        tr = {k: agg[k] / agg["_n"] for k in agg if k != "_n"}
        vl = _evaluate(model, val_paths, jcfg, device) if val_paths else tr
        print(f"epoch {ep}: train total={tr['total']:.4f} jepa={tr['jepa']:.4f} "
              f"bc={tr['bc']:.4f} bc_acc={tr['bc_acc']:.3f} value={tr['value']:.4f} "
              f"| val total={vl['total']:.4f} bc_acc={vl['bc_acc']:.3f}")
        model.save(out_path.with_name(out_path.stem + "_last.pt"))
        if vl["total"] < best:
            best = vl["total"]
            model.save(out_path)
            print(f"  saved best -> {out_path} (val total {best:.4f})")
    return best


@torch.no_grad()
def _evaluate(model, paths, jcfg, device):
    """Average the loss metrics over the validation shards."""
    model.eval()
    agg, n = {}, 0
    for sh in _batches(paths, jcfg.batch_size, device, shuffle=False):
        _, m = losses(model, sh, jcfg)
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in agg.items()}


def main():
    """CLI entry: train the consequence model from shards per argv flags."""
    args = sys.argv[1:]

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    jcfg = JCFG
    data = opt("--data", str(CFG.artifacts_dir / "jepa_cons_prepped"))
    out = opt("--out", str(CFG.checkpoint_dir / "jepa" / "jepa_consequence.pt"))
    epochs = opt("--epochs")
    if opt("--bs"):
        jcfg.batch_size = int(opt("--bs"))
    ls = opt("--limit-shards")
    train(data, out, jcfg=jcfg, epochs=int(epochs) if epochs else None,
          limit_shards=int(ls) if ls else None)


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
    else:
        main()
