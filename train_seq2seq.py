"""Behavior cloning for the seq2seq pointer model (exp/seq2seq-pointer).

Same data, splits, sample weights, loss weights, optimizer, schedule, and
epoch count as train.py — the architecture (models/seq2seq.py) and the
policy loss shape differ. The pointer model scores only position-legal
actions, so its policy loss is a *set* cross-entropy over the chain-rule
joint: -log sum_{(a,b) in label set} P(a) P(b|a), where the label set is the
recorded action projected onto its move's real target codes
(KNOWN_ISSUES.md #3; sets built once by seq2seq_prep.py). Value MSE and the
aux set losses are train.py's verbatim. Rows whose projected label set
misses the legal superset entirely (counted by seq2seq_prep) contribute no
policy gradient and are reported as invalid_frac.

Requires the seq2seq_<split>.npz sidecars from ``python seq2seq_prep.py``.
Checkpoints go to cfg.checkpoint_dir as seq2seq_ckpt_last.pt /
seq2seq_ckpt_best.pt — the baseline's ckpt_*.pt files are never touched.

CLI: python train_seq2seq.py [epochs] [--smoke]
     --smoke   no shards needed: synthetic batch through build/forward/
               backward/save/load/predict_batch, plus the PositionLegality
               overhead measurement at B=2 (the search-expansion shape).
               Run this before shipping the worktree to the training box.
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("train_seq2seq.py"):
        raise SystemExit(0)

import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from actions import N_SLOT_ACTIONS
from config import CFG, config_snapshot
from models.policy_value import clean_state_dict
from models.seq2seq import Seq2SeqPointerNet
from tokenizer import PositionTokenizer
from train import Shards, make_loader

NEG = -1e9   # finite -inf stand-in: keeps logsumexp/backward NaN-free


class Seq2SeqShards(Shards):
    """Shards plus the per-row legality/label sidecar from seq2seq_prep.py."""

    def __init__(self, split, cfg=CFG):
        """Load the split and its aligned ``seq2seq_<split>.npz`` sidecar."""
        super().__init__(split, cfg)
        p = cfg.prepped_dir / f"seq2seq_{split}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run `python seq2seq_prep.py {split}` first")
        side = np.load(p)
        if int(side["n_rows"]) != len(self.tokens):
            raise ValueError(
                f"{p} has {int(side['n_rows'])} rows but the {split} shards "
                f"have {len(self.tokens)} — rebuild the sidecar")
        self.legal_a, self.legal_b = side["legal_a"], side["legal_b"]
        self.label_a, self.label_b = side["label_a"], side["label_b"]

    def _unpack(self, packed, idxs):
        """Unpack uint8[B, 5] sidecar rows to a bool[B, 39] tensor."""
        bits = np.unpackbits(packed[idxs], axis=1)[:, :N_SLOT_ACTIONS]
        return torch.from_numpy(bits.astype(bool))

    def __getitem__(self, idxs):
        """Return train.Shards' batch tuple + the four mask tensors."""
        return super().__getitem__(idxs) + (
            self._unpack(self.legal_a, idxs), self._unpack(self.legal_b, idxs),
            self._unpack(self.label_a, idxs), self._unpack(self.label_b, idxs))


def compute_loss_seq2seq(model, batch, cfg=CFG):
    """Return ``(scalar_loss, detached_metric_tensors)`` for one batch.

    Set-CE on the chain-rule joint over the projected label set; value MSE
    and aux losses are copied from train.compute_loss. top1_legal scores the
    legal-masked joint argmax against the label set (position-legal metric —
    not comparable to the static-mask top1 in EXPERIMENTS.md)."""
    (tokens, acts, value_t, w, items_t, abils_t, moves_t,
     legal_a, legal_b, label_a, label_b) = batch
    log_pa, log_pb, value, (item_lg, abil_lg, move_lg) = model(
        tokens, legal_a, legal_b)

    joint_logp = log_pa.unsqueeze(-1) + log_pb              # [B, 39, 39]
    label_set = (label_a.unsqueeze(2) & label_b.unsqueeze(1)
                 & model.joint_ok_mask)
    legal_grid = (legal_a.unsqueeze(2) & legal_b.unsqueeze(1)
                  & model.joint_ok_mask)
    valid = (label_set & legal_grid).flatten(1).any(1)
    ll = torch.logsumexp(
        joint_logp.clamp_min(NEG).masked_fill(~label_set, NEG).flatten(1), 1)
    ce = -torch.where(valid, ll, torch.zeros_like(ll))
    policy_loss = (ce * w).mean()
    top1 = label_set.flatten(1).gather(
        1, joint_logp.flatten(1).argmax(1, keepdim=True)).squeeze(1)
    value_loss = (F.mse_loss(value, value_t, reduction="none") * w).mean()

    item_loss = F.cross_entropy(item_lg.flatten(0, 1), items_t.flatten(),
                                ignore_index=0)
    abil_loss = F.cross_entropy(abil_lg.flatten(0, 1), abils_t.flatten(),
                                ignore_index=0)
    moves_hot = torch.zeros_like(move_lg).scatter_(-1, moves_t, 1.0)
    moves_hot[..., 0] = 0
    move_loss = F.binary_cross_entropy_with_logits(move_lg, moves_hot)
    aux_loss = item_loss + abil_loss + move_loss

    loss = (policy_loss + cfg.value_loss_weight * value_loss
            + cfg.aux_set_loss_weight * aux_loss)
    return loss, {"loss": loss.detach(), "policy": policy_loss.detach(),
                  "value": value_loss.detach(), "aux": aux_loss.detach(),
                  "top1_legal": top1.detach().float().mean(),
                  "invalid_frac": (~valid).detach().float().mean()}


def run_epoch(model, loader, device, opt=None, sched=None, cfg=CFG):
    """train.run_epoch with the seq2seq loss (mirrored, not modified)."""
    model.train(opt is not None)
    agg, n = {}, 0
    autocast = torch.autocast(device, dtype=torch.bfloat16,
                              enabled=device == "cuda")
    for batch in loader:
        batch = [t.to(device, non_blocking=True) for t in batch]
        for t in batch:
            torch._dynamo.mark_dynamic(t, 0)
        with torch.set_grad_enabled(opt is not None), autocast:
            loss, stats = compute_loss_seq2seq(model, batch, cfg)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
        bs = len(batch[0])
        n += bs
        for k, v in stats.items():
            agg[k] = agg.get(k, 0.0) + v * bs
    return {k: float(v / n) for k, v in agg.items()}


def build_model(tok, cfg, device):
    """Construct a fresh Seq2SeqPointerNet from the loaded tokenizer."""
    return Seq2SeqPointerNet(
        tok.vocab_size(), tok.n_tokens, tok.opp_species_positions(),
        len(tok.move_list), len(tok.item_list), len(tok.ability_list),
        cfg, policy_head="joint",
        slot_token_ids=(tok.vocab["SLOT_A"], tok.vocab["SLOT_B"])).to(device)


def smoke(cfg=CFG):
    """Sharded-data-free self-test: forward/backward/save/load/predict_batch
    round trip on synthetic masks, plus the B=2 legality-overhead timing."""
    tok = PositionTokenizer.load(cfg)
    model = build_model(tok, cfg, "cpu")
    n = sum(p.numel() for p in model.parameters())
    print(f"seq2seq-pointer: {n / 1e6:.2f}M params")
    B = 4
    rng = np.random.default_rng(0)
    tokens = torch.from_numpy(
        rng.integers(0, tok.vocab_size(), (B, tok.n_tokens)).astype(np.int64))
    legal = torch.zeros(B, N_SLOT_ACTIONS, dtype=torch.bool)
    legal[:, [1, 3, 9, 33, 34]] = True    # a few moves + two switches
    label_a = torch.zeros_like(legal)
    label_a[:, 1] = True
    label_b = torch.zeros_like(legal)
    label_b[:, [3, 33]] = True            # a projected two-action set
    batch = (tokens, torch.ones((B, 2), dtype=torch.long),
             torch.zeros(B), torch.ones(B),
             torch.ones((B, 6), dtype=torch.long),
             torch.ones((B, 6), dtype=torch.long),
             torch.ones((B, 6, 4), dtype=torch.long),
             legal, legal.clone(), label_a, label_b)
    loss, stats = compute_loss_seq2seq(model, batch, cfg)
    loss.backward()
    grads = sum(p.grad.abs().sum().item() for p in model.parameters()
                if p.grad is not None)
    assert math.isfinite(float(loss)) and math.isfinite(grads)
    print(f"forward/backward ok, loss {float(loss):.3f} "
          f"top1_legal {float(stats['top1_legal']):.2f}")

    dists, values, aux = model.predict_batch(tokens.numpy())
    assert dists.shape == (B, 1521) and values.shape == (B,)
    assert np.allclose(dists.sum(1), 1.0, atol=1e-4)
    assert np.isfinite(dists).all()
    t0 = time.time()
    reps = 20
    for _ in range(reps):
        model.predict_batch(tokens.numpy()[:2])
    per_call = (time.time() - t0) / reps * 1000
    print(f"predict_batch ok; B=2 full call {per_call:.1f} ms "
          f"(includes per-row PositionLegality reconstruction)")

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ck.pt"
        model.save(p)
        m2 = Seq2SeqPointerNet.load(p, cfg)
        d2, v2, _ = m2.predict_batch(tokens.numpy())
        assert np.allclose(dists, d2, atol=1e-5)
        assert np.allclose(values, v2, atol=1e-5)
    from models.seq2seq import load_any_policy_model
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ck.pt"
        model.save(p)
        assert isinstance(load_any_policy_model(p, cfg), Seq2SeqPointerNet)
    print("save/load + dispatch smoke ok")


def main(cfg=CFG):
    """Mirror train.main with the pointer model, sidecar shards, and loss."""
    if "--smoke" in sys.argv:
        smoke(cfg)
        return
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 \
        and not sys.argv[1].startswith("--") else cfg.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    tok = PositionTokenizer.load(cfg)
    train_ds, val_ds = Seq2SeqShards("train", cfg), Seq2SeqShards("val", cfg)
    print(f"train {len(train_ds)} / val {len(val_ds)} transitions, "
          f"device={device}")
    train_dl = make_loader(train_ds, cfg.batch_size, True, device, cfg)
    val_dl = make_loader(val_ds, cfg.batch_size, False, device, cfg)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last = cfg.checkpoint_dir / "seq2seq_ckpt_last.pt"
    if last.exists():
        model = Seq2SeqPointerNet.load(last, cfg, device)
        start_epoch = torch.load(last, map_location="cpu",
                                 weights_only=False)["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")
    else:
        model = build_model(tok, cfg, device)
        start_epoch = 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params")
    if cfg.compile_model and device == "cuda":
        model = torch.compile(model)
        print("torch.compile on")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay,
                            fused=device == "cuda")
    total_steps = max(1, len(train_dl) * epochs)

    def lr_lambda(step):
        """Warmup then cosine decay, identical to train.py."""
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        t = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(cfg.checkpoint_dir / "tb_seq2seq")

    best_val = float("inf")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr = run_epoch(model, train_dl, device, opt, sched, cfg)
        with torch.no_grad():
            va = run_epoch(model, val_dl, device, cfg=cfg)
        for k in tr:
            tb.add_scalar(f"train/{k}", tr[k], epoch)
            tb.add_scalar(f"val/{k}", va[k], epoch)
        print(f"epoch {epoch:3d} | "
              f"train loss {tr['loss']:.4f} pol {tr['policy']:.4f} "
              f"val {tr['value']:.4f} aux {tr['aux']:.4f} "
              f"top1L {tr['top1_legal']:.3f} | "
              f"val loss {va['loss']:.4f} top1L {va['top1_legal']:.3f} "
              f"inv {va['invalid_frac']:.4f} | "
              f"{time.time() - t0:.0f}s")
        ck = {"hp": getattr(model, "_orig_mod", model).hp,
              "state": clean_state_dict(model), "epoch": epoch,
              "cfg": config_snapshot(cfg)}
        torch.save(ck, last)
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(ck, cfg.checkpoint_dir / "seq2seq_ckpt_best.pt")
    tb.close()


if __name__ == "__main__":
    main()
