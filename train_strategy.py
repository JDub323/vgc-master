"""Train the v3 Strategy-JEPA dynamics on sequence-window shards.

Objective (``JEPA_V3_DESIGN.md`` §4, stage 1): unroll the latent dynamics
``T(Z, a, b)`` along real trajectories and match every step against the EMA
target encoder's embedding of the realized position — multi-step JEPA, with a
per-step discount so the far horizon shapes without dominating. The same
unrolled latents also train the value head (so payoff matrices read off ``T``
outputs are grounded), and the window-start latents train the own-prior and
opponent-policy heads (candidate generators for the matrix search). VICReg on
the online latents guards collapse alongside the EMA stop-grad.

CLI: python train_strategy.py [--data DIR] [--out PATH] [--epochs N] [--bs N]
                              [--limit-shards N] [--large]
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
from models.jepa_strategy import JEPAStrategyModel
from train_jepa import _vicreg


def _load_shard(path, device):
    """Load one sequence-window npz shard into a dict of device tensors."""
    z = np.load(path)
    long = lambda k: torch.as_tensor(z[k].astype(np.int64), device=device)
    flt = lambda k: torch.as_tensor(z[k].astype(np.float32), device=device)
    return {"pos_gcat": long("pos_gcat"), "pos_gscal": flt("pos_gscal"),
            "pos_mcat": long("pos_mcat"), "pos_mscal": flt("pos_mscal"),
            "pos_dmg": flt("pos_dmg"), "act": long("act"),
            "a_slot": long("a_slot"), "b_slot": long("b_slot"),
            "n_steps": long("n_steps"), "value": flt("value"),
            "weight": flt("weight")}


def _pos_at(sh, k):
    """Slice the position tensors at window step ``k`` into an encoder dict."""
    return {"gcat": sh["pos_gcat"][:, k], "gscal": sh["pos_gscal"][:, k],
            "mcat": sh["pos_mcat"][:, k], "mscal": sh["pos_mscal"][:, k],
            "dmg": sh["pos_dmg"][:, k]}


def losses(model, sh, jcfg):
    """Multi-step JEPA + policy + value losses for one window minibatch."""
    K = sh["act"].shape[1]
    z0 = model.encode(_pos_at(sh, 0))
    w = sh["weight"]
    wn = w / w.mean().clamp_min(1e-6)

    # policy heads at the window start (windows stride over every offset, so
    # step-0 supervision covers the whole trajectory distribution)
    my_logits, opp_logits = model.policies(z0)
    pol_l = (F.cross_entropy(my_logits[:, 0], sh["a_slot"][:, 0, 0])
             + F.cross_entropy(my_logits[:, 1], sh["a_slot"][:, 0, 1])
             + F.cross_entropy(opp_logits[:, 0], sh["b_slot"][:, 0, 0])
             + F.cross_entropy(opp_logits[:, 1], sh["b_slot"][:, 0, 1]))

    # value off the encoded present
    v0 = model.value(z0)
    value_l = (wn * (v0 - sh["value"]) ** 2).mean()

    # multi-step latent unroll vs EMA targets of the realized positions
    zhat = z0
    jepa_l = z0.new_zeros(())
    jepa_norm = 0.0
    for k in range(1, K + 1):
        dmg = sh["pos_dmg"][:, 0] if k == 1 else None
        zhat = model.step(zhat, sh["act"][:, k - 1], dmg)
        mask = (sh["n_steps"] >= k).float()
        if mask.sum() == 0:
            break
        target = model.target_encode(_pos_at(sh, k))
        per = F.smooth_l1_loss(zhat, target, reduction="none").mean((1, 2))
        g = jcfg.unroll_gamma ** (k - 1)
        jepa_l = jepa_l + g * (per * mask).sum() / mask.sum()
        jepa_norm += g
        vk = model.value(zhat)
        value_l = value_l + g * ((vk - sh["value"]) ** 2 * mask).sum() \
            / mask.sum().clamp_min(1.0)
    jepa_l = jepa_l / max(jepa_norm, 1e-6)
    value_l = value_l / (1.0 + jepa_norm)

    var_l, cov_l = _vicreg(z0, jcfg.vicreg_gamma)
    total = (jcfg.w_jepa_s * jepa_l + jcfg.w_value_s * value_l
             + jcfg.w_policy_s * pol_l
             + jcfg.w_vicreg_var * var_l + jcfg.w_vicreg_cov * cov_l)
    my_acc = (my_logits.argmax(-1) == sh["a_slot"][:, 0]).float().mean()
    return total, {"total": total.item(), "jepa": jepa_l.item(),
                   "value": value_l.item(), "policy": pol_l.item(),
                   "my_acc": my_acc.item(), "var": var_l.item()}


def _batches(paths, bs, device, shuffle=True, limit_shards=None):
    """Yield minibatch tensor dicts, one loaded npz shard at a time."""
    paths = list(paths)[:limit_shards] if limit_shards else list(paths)
    for path in paths:
        sh = _load_shard(path, device)
        n = sh["value"].shape[0]
        order = torch.randperm(n, device=device) if shuffle \
            else torch.arange(n, device=device)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            yield {k: v[idx] for k, v in sh.items()}


def train(data_dir, out_path, cfg=CFG, jcfg=JCFG, epochs=None,
          limit_shards=None):
    """Fit the strategy model; write best/last checkpoints; return best loss."""
    data_dir = Path(data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from beliefs import load_dex
    vstate_p = data_dir / "vocab_state.json"
    vocab = (JEPAVocab.from_state(json.loads(vstate_p.read_text()),
                                  load_dex(cfg))
             if vstate_p.exists() else JEPAVocab.build(cfg))

    train_paths = sorted(data_dir.glob("train_*.npz")) or sorted(
        data_dir.glob("*.npz"))
    val_paths = sorted(data_dir.glob("val_*.npz"))
    jcfg.use_damage_features = bool(train_paths) and \
        float(np.abs(np.load(train_paths[0])["pos_dmg"]).max()) > 0
    model = JEPAStrategyModel(vocab.sizes(), jcfg, vocab.state()).to(device)

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
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), jcfg.grad_clip)
            opt.step()
            model.update_ema()
            step += 1
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            agg["_n"] = agg.get("_n", 0) + 1
        tr = {k: agg[k] / agg["_n"] for k in agg if k != "_n"}
        vl = _evaluate(model, val_paths, jcfg, device) if val_paths else tr
        print(f"epoch {ep}: train total={tr['total']:.4f} jepa={tr['jepa']:.4f}"
              f" value={tr['value']:.4f} policy={tr['policy']:.4f}"
              f" my_acc={tr['my_acc']:.3f} | val total={vl['total']:.4f}"
              f" my_acc={vl['my_acc']:.3f}")
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
    """CLI entry: train the strategy model from window shards per argv."""
    args = sys.argv[1:]

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    jcfg = JCFG
    if "--large" in args:
        from jepa.config import scaled_consequence
        jcfg = scaled_consequence(jcfg)
    data = opt("--data", str(CFG.artifacts_dir / "jepa_seq_prepped"))
    out = opt("--out", str(CFG.checkpoint_dir / "jepa" / "jepa_strategy.pt"))
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
