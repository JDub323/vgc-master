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
from models.jepa_strategy import JEPAStrategyModel, _scale_grad
from train_jepa import _vicreg


def _load_shard(path, device, builder=None):
    """Load one sequence-window npz shard into a dict of device tensors.

    With a :class:`jepa.candidates.CandidateBuilder`, also attach sampled
    negative candidates for the BC scoring heads (both sides), synthesized
    from the stored step-0 position arrays and disk-cached next to the shard
    (``negcache/``) so the per-row Python enumeration runs once, not once per
    epoch."""
    z = np.load(path)
    long = lambda k: torch.as_tensor(z[k].astype(np.int64), device=device)
    flt = lambda k: torch.as_tensor(z[k].astype(np.float32), device=device)
    sh = {"pos_gcat": long("pos_gcat"), "pos_gscal": flt("pos_gscal"),
          "pos_mcat": long("pos_mcat"), "pos_mscal": flt("pos_mscal"),
          "pos_dmg": flt("pos_dmg"), "act": long("act"),
          "a_slot": long("a_slot"), "b_slot": long("b_slot"),
          "a_mask": torch.as_tensor(z["a_mask"], device=device),
          "b_mask": torch.as_tensor(z["b_mask"], device=device),
          "n_steps": long("n_steps"), "value": flt("value"),
          "margin": long("margin"), "weight": flt("weight")}
    if builder is not None:
        ng = _negatives_cached(Path(path), z, builder)
        for k, arr in ng.items():
            sh[k] = torch.as_tensor(
                arr.astype(np.int64) if arr.dtype != bool else arr,
                device=device)
    return sh


def _negatives_cached(path, z, builder):
    """Per-shard negative candidates, computed once and cached on disk."""
    cache = path.parent / "negcache" / f"{path.stem}.n{builder.n_neg}.npz"
    if cache.exists():
        c = np.load(cache)
        return {k: c[k] for k in ("negs_my", "negs_my_mask",
                                  "negs_opp", "negs_opp_mask")}
    mcat, mscal = z["pos_mcat"][:, 0], z["pos_mscal"][:, 0]
    negs_my, m_my = builder.negatives(mcat, mscal, z["a_slot"][:, 0],
                                      z["a_mask"][:, 0], side="my")
    negs_opp, m_opp = builder.negatives(mcat, mscal, z["b_slot"][:, 0],
                                        z["b_mask"][:, 0], side="opp")
    out = {"negs_my": negs_my, "negs_my_mask": m_my,
           "negs_opp": negs_opp, "negs_opp_mask": m_opp}
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def _pos_at(sh, k):
    """Slice the position tensors at window step ``k`` into an encoder dict."""
    return {"gcat": sh["pos_gcat"][:, k], "gscal": sh["pos_gscal"][:, k],
            "mcat": sh["pos_mcat"][:, k], "mscal": sh["pos_mscal"][:, k],
            "dmg": sh["pos_dmg"][:, k]}


def _masked_ce(logits, target, mask):
    """Cross-entropy over the rows where ``mask`` is set (0 when none)."""
    if mask.sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits[mask], target[mask])


def losses(model, sh, jcfg):
    """Multi-step JEPA + policy + margin-value losses for one window batch.

    Encoder-trunk ownership: the policy heads receive only
    ``policy_grad_scale`` of their gradient into the trunk — 1.0 let the
    policy CEs conquer the encoder in stage 1 (JEPA loss rose all run), 0.0
    starved the heads in stage 2 (my_acc collapsed to 0.14). Policy CE is a
    mean per head (not a sum) and masked to observed actions (AK_UNK steps
    train the dynamics but never the policy)."""
    K = sh["act"].shape[1]
    z0 = model.encode(_pos_at(sh, 0))

    # policy heads at the window start (stride-1 windows cover every offset)
    my_logits, opp_logits = model.policies(z0,
                                           grad_scale=jcfg.policy_grad_scale)
    am, bm = sh["a_mask"][:, 0], sh["b_mask"][:, 0]
    pol_l = (_masked_ce(my_logits[:, 0], sh["a_slot"][:, 0, 0], am)
             + _masked_ce(my_logits[:, 1], sh["a_slot"][:, 0, 1], am)
             + _masked_ce(opp_logits[:, 0], sh["b_slot"][:, 0, 0], bm)
             + _masked_ce(opp_logits[:, 1], sh["b_slot"][:, 0, 1], bm)) / 4.0

    # distributional margin value off the encoded present
    margin_bin = (sh["margin"] + model.margin_half).clamp(
        0, jcfg.n_margin_bins - 1)
    value_l = F.cross_entropy(model.margin_logits(z0), margin_bin)
    value_norm = 1.0

    # multi-step latent unroll vs EMA targets of the realized positions
    zhat = z0
    jepa_l = z0.new_zeros(())
    jepa_norm = 0.0
    for k in range(1, K + 1):
        dmg = sh["pos_dmg"][:, 0] if k == 1 else None
        zhat = model.step(zhat, sh["act"][:, k - 1], dmg)
        mask = sh["n_steps"] >= k
        if mask.sum() == 0:
            break
        target = model.target_encode(_pos_at(sh, k))
        per = F.smooth_l1_loss(zhat, target, reduction="none").mean((1, 2))
        g = jcfg.unroll_gamma ** (k - 1)
        jepa_l = jepa_l + g * (per * mask.float()).sum() / mask.sum()
        jepa_norm += g
        value_l = value_l + g * _masked_ce(model.margin_logits(zhat),
                                           margin_bin, mask)
        value_norm += g
    jepa_l = jepa_l / max(jepa_norm, 1e-6)
    value_l = value_l / value_norm

    # BC candidate CE through T: the human's joint action (candidate 0, the
    # stored positive with the other side marked AK_UNK) vs sampled legal
    # negatives, each scored by the score heads off T's output — v2's
    # candidate CE, re-expressed through v3's dynamics
    score_l = z0.new_zeros(())
    score_acc = torch.zeros(())
    if "negs_my" in sh and jcfg.w_score_s > 0:
        z_s = _scale_grad(z0, jcfg.score_grad_scale)
        dmg0, mscal0 = sh["pos_dmg"][:, 0], sh["pos_mscal"][:, 0]
        s_my = _cand_logits(model, model.score, z_s, dmg0,
                            _half_unk(sh["act"][:, 0], mscal0, 6),
                            sh["negs_my"], sh["negs_my_mask"])
        s_opp = _cand_logits(model, model.opp_score, z_s, dmg0,
                             _half_unk(sh["act"][:, 0], mscal0, 0),
                             sh["negs_opp"], sh["negs_opp_mask"])
        sm = am & sh["negs_my_mask"].any(-1)
        so = bm & sh["negs_opp_mask"].any(-1)
        tgt0 = torch.zeros(s_my.shape[0], dtype=torch.long, device=s_my.device)
        score_l = (_masked_ce(s_my, tgt0, sm) + _masked_ce(s_opp, tgt0, so)) / 2
        if sm.any():
            score_acc = (s_my.argmax(-1) == 0)[sm].float().mean()

    var_l, cov_l = _vicreg(z0, jcfg.vicreg_gamma)
    total = (jcfg.w_jepa_s * jepa_l + jcfg.w_value_s * value_l
             + jcfg.w_policy_s * pol_l + jcfg.w_score_s * score_l
             + jcfg.w_vicreg_var * var_l + jcfg.w_vicreg_cov * cov_l)
    my_acc = (my_logits.argmax(-1) == sh["a_slot"][:, 0])[am].float().mean() \
        if am.any() else torch.zeros(())
    return total, {"total": total.item(), "jepa": jepa_l.item(),
                   "value": value_l.item(), "policy": pol_l.item(),
                   "score": score_l.item(), "score_acc": score_acc.item(),
                   "my_acc": my_acc.item(), "var": var_l.item()}


def _half_unk(act, mscal0, lo):
    """Blank one side of the joint action arrays, marking actives ``AK_UNK``.

    ``lo=6`` keeps the own action and marginalizes the opponent (own-score
    positive); ``lo=0`` is the symmetric opponent-score positive."""
    from jepa.features import AK_UNK, MS_ACTIVE
    out = act.clone()
    out[:, lo:lo + 6] = 0
    active = mscal0[:, lo:lo + 6, MS_ACTIVE] > 0.5
    out[:, lo:lo + 6, 0] = active.long() * AK_UNK
    return out


def _cand_logits(model, head, z, dmg, pos_act, negs, neg_mask):
    """Score [positive] + negatives through one batched ``T`` application.

    Returns ``[B, 1+n_neg]`` logits with padded negatives masked to -inf."""
    b, n_neg = negs.shape[0], negs.shape[1]
    c = 1 + n_neg
    cand = torch.cat([pos_act.unsqueeze(1), negs], 1)          # [B,C,12,7]
    zp = model.step(
        z.unsqueeze(1).expand(-1, c, -1, -1).reshape(b * c, *z.shape[1:]),
        cand.reshape(b * c, *cand.shape[2:]),
        dmg.unsqueeze(1).expand(-1, c, -1, -1).reshape(b * c, 6, 6))
    logits = head(zp).view(b, c)
    valid = torch.cat([torch.ones_like(neg_mask[:, :1]), neg_mask], 1)
    return logits.masked_fill(~valid, float("-inf"))


def _batches(paths, bs, device, shuffle=True, limit_shards=None, builder=None):
    """Yield minibatch tensor dicts, one loaded npz shard at a time."""
    paths = list(paths)[:limit_shards] if limit_shards else list(paths)
    for path in paths:
        sh = _load_shard(path, device, builder)
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
    builder = None
    if jcfg.w_score_s > 0:
        from jepa.candidates import CandidateBuilder
        builder = CandidateBuilder(vocab, jcfg.n_score_neg)

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
                           limit_shards=limit_shards, builder=builder):
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
        vl = _evaluate(model, val_paths, jcfg, device, builder) \
            if val_paths else tr
        print(f"epoch {ep}: train total={tr['total']:.4f} jepa={tr['jepa']:.4f}"
              f" value={tr['value']:.4f} policy={tr['policy']:.4f}"
              f" score={tr['score']:.4f} score_acc={tr['score_acc']:.3f}"
              f" my_acc={tr['my_acc']:.3f} | val total={vl['total']:.4f}"
              f" score_acc={vl['score_acc']:.3f} my_acc={vl['my_acc']:.3f}")
        model.save(out_path.with_name(out_path.stem + "_last.pt"))
        if vl["total"] < best:
            best = vl["total"]
            model.save(out_path)
            print(f"  saved best -> {out_path} (val total {best:.4f})")
    return best


@torch.no_grad()
def _evaluate(model, paths, jcfg, device, builder=None):
    """Average the loss metrics over the validation shards."""
    model.eval()
    agg, n = {}, 0
    for sh in _batches(paths, jcfg.batch_size, device, shuffle=False,
                       builder=builder):
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
