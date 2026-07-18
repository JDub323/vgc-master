"""Train the JEPA world model on paired-transition shards.

Objective (see ``JEPA_DESIGN.md``): predict the next-state *latent* of the true
transition (matched against an EMA target encoder), read the game outcome off
that predicted latent, and ground the latent with next-state decoders
(HP/faint/status/field). Opponent- and my-policy heads are trained as candidate
generators; VICReg keeps the encoder from collapsing.

CLI: python train_jepa.py [--data DIR] [--out PATH] [--epochs N] [--bs N]
                          [--limit-shards N]
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
from jepa.features import (GS_MY_AV, GS_MY_LS, GS_MY_REF, GS_MY_TW, GS_OPP_AV,
                           GS_OPP_LS, GS_OPP_REF, GS_OPP_TW, GS_TR, MC_STATUS,
                           MS_FAINTED, MS_HP)
from jepa.vocab import JEPAVocab
from models.jepa_wm import JEPAWorldModel

SCREEN_COLS = [GS_MY_TW, GS_MY_REF, GS_MY_LS, GS_MY_AV,
               GS_OPP_TW, GS_OPP_REF, GS_OPP_LS, GS_OPP_AV]


def _load_shard(path, device):
    """Load one npz transition shard into a dict of device tensors."""
    z = np.load(path)
    long = lambda k: torch.as_tensor(z[k].astype(np.int64), device=device)
    flt = lambda k: torch.as_tensor(z[k].astype(np.float32), device=device)
    return {
        "cur_gcat": long("cur_gcat"), "cur_gscal": flt("cur_gscal"),
        "cur_mcat": long("cur_mcat"), "cur_mscal": flt("cur_mscal"),
        "cur_dmg": flt("cur_dmg"), "act": long("act"),
        "nxt_gcat": long("nxt_gcat"), "nxt_gscal": flt("nxt_gscal"),
        "nxt_mcat": long("nxt_mcat"), "nxt_mscal": flt("nxt_mscal"),
        "nxt_dmg": flt("nxt_dmg"), "value": flt("value"), "weight": flt("weight"),
        "a_slot": long("a_slot"), "b_slot": long("b_slot"),
    }


def _pos(sh, prefix):
    """Slice a shard dict into an encoder-ready position tensor dict."""
    return {"gcat": sh[prefix + "gcat"], "gscal": sh[prefix + "gscal"],
            "mcat": sh[prefix + "mcat"], "mscal": sh[prefix + "mscal"],
            "dmg": sh[prefix + "dmg"]}


def _index(d, idx):
    """Return a new dict with every tensor value indexed by ``idx``."""
    return {k: v[idx] for k, v in d.items()}


def losses(model, sh, jcfg):
    """Compute the total loss and a metrics dict for one minibatch shard."""
    cur, nxt = _pos(sh, "cur_"), _pos(sh, "nxt_")
    z = model.encode(cur)
    zp = model.predict(z, sh["act"], cur["dmg"])
    v = model.value(zp)
    w = sh["weight"]
    wn = w / w.mean().clamp_min(1e-6)

    value_l = (wn * (v - sh["value"]) ** 2).mean()
    target = model.target_encode(nxt)
    jepa_l = F.smooth_l1_loss(zp, target)

    g = model.grounded(zp)
    hp_l = F.mse_loss(g["hp"], nxt["mscal"][..., MS_HP])
    faint_l = F.binary_cross_entropy_with_logits(g["faint"],
                                                 nxt["mscal"][..., MS_FAINTED])
    status_l = F.cross_entropy(g["status"].reshape(-1, g["status"].shape[-1]),
                               nxt["mcat"][..., MC_STATUS].reshape(-1))
    field_l = (F.cross_entropy(g["weather"], nxt["gcat"][:, 0])
               + F.cross_entropy(g["terrain"], nxt["gcat"][:, 1])
               + F.binary_cross_entropy_with_logits(g["tr"], nxt["gscal"][:, GS_TR])
               + F.binary_cross_entropy_with_logits(
                   g["screens"], nxt["gscal"][:, SCREEN_COLS]))

    my_logits, opp_logits = model.policies(z)
    my_l = (F.cross_entropy(my_logits[:, 0], sh["a_slot"][:, 0])
            + F.cross_entropy(my_logits[:, 1], sh["a_slot"][:, 1]))
    opp_l = (F.cross_entropy(opp_logits[:, 0], sh["b_slot"][:, 0])
             + F.cross_entropy(opp_logits[:, 1], sh["b_slot"][:, 1]))

    var_l, cov_l = _vicreg(z, jcfg.vicreg_gamma)

    total = (jcfg.w_jepa * jepa_l + jcfg.w_value * value_l
             + jcfg.w_ground_hp * hp_l + jcfg.w_ground_faint * faint_l
             + jcfg.w_ground_status * status_l + jcfg.w_ground_field * field_l
             + jcfg.w_my_prior * my_l + jcfg.w_opp_policy * opp_l
             + jcfg.w_vicreg_var * var_l + jcfg.w_vicreg_cov * cov_l)
    my_acc = (my_logits.argmax(-1) == sh["a_slot"]).float().mean()
    return total, {"total": total.item(), "jepa": jepa_l.item(),
                   "value": value_l.item(), "hp": hp_l.item(),
                   "my_acc": my_acc.item(), "var": var_l.item()}


def _vicreg(z, gamma):
    """VICReg variance (hinge) + covariance losses over entity latents."""
    x = z.reshape(-1, z.shape[-1])
    x = x - x.mean(0, keepdim=True)
    std = torch.sqrt(x.var(0) + 1e-4)
    var_l = F.relu(gamma - std).mean()
    n, d = x.shape
    cov = (x.T @ x) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_l = off.pow(2).sum() / d
    return var_l, cov_l


def _batches(paths, bs, device, shuffle=True, limit_shards=None):
    """Yield minibatch shard-dicts, loading one npz shard at a time."""
    paths = list(paths)[:limit_shards] if limit_shards else list(paths)
    for path in paths:
        sh = _load_shard(path, device)
        n = sh["value"].shape[0]
        order = torch.randperm(n, device=device) if shuffle \
            else torch.arange(n, device=device)
        for i in range(0, n, bs):
            yield _index(sh, order[i:i + bs])


def train(data_dir, out_path, cfg=CFG, jcfg=JCFG, epochs=None,
          limit_shards=None):
    """Fit the world model and write best/last checkpoints; return best loss."""
    data_dir = Path(data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from beliefs import load_dex
    vstate_p = data_dir / "vocab_state.json"
    if vstate_p.exists():
        vocab = JEPAVocab.from_state(json.loads(vstate_p.read_text()), load_dex(cfg))
    else:
        vocab = JEPAVocab.build(cfg)
    train_paths = sorted(data_dir.glob("train_*.npz")) or sorted(
        data_dir.glob("*.npz"))
    val_paths = sorted(data_dir.glob("val_*.npz"))
    # Record whether the shards carry damage edges so the chooser only builds a
    # damage bridge at play when the model trained on one (train/play parity).
    jcfg.use_damage_features = bool(train_paths) and \
        float(np.abs(np.load(train_paths[0])["cur_dmg"]).max()) > 0
    model = JEPAWorldModel(vocab.sizes(), jcfg, vocab.state()).to(device)
    epochs = epochs or jcfg.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=jcfg.lr,
                            weight_decay=jcfg.weight_decay)
    steps_total = max(1, epochs * sum(
        int(np.load(p)["value"].shape[0]) for p in
        (train_paths[:limit_shards] if limit_shards else train_paths)
    ) // jcfg.batch_size)
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
              f"value={tr['value']:.4f} my_acc={tr['my_acc']:.3f} | "
              f"val total={vl['total']:.4f} value={vl['value']:.4f}")
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
    """CLI entry: train from transition shards per argv flags."""
    args = sys.argv[1:]

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    jcfg = JCFG          # use_damage_features is auto-detected from the shards
    data = opt("--data", str(CFG.artifacts_dir / "jepa_prepped"))
    out = opt("--out", str(CFG.checkpoint_dir / "jepa" / "jepa_wm.pt"))
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
