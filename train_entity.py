"""Behavior cloning for the entity-hybrid model (exp/entity-hybrid).

Same data, splits, loss, optimizer, schedule, and epoch count as train.py —
only the architecture differs (models/entity_hybrid.py) plus an optional
permutation augmentation (entity_augment.py). Reuses train.py's Shards,
make_loader, compute_loss, and run_epoch directly so the recipes cannot
drift apart silently.

Checkpoints are written to cfg.checkpoint_dir as entity_ckpt_last.pt /
entity_ckpt_best.pt — the baseline's ckpt_*.pt files are never touched.

CLI: python train_entity.py [epochs] [--augment] [--smoke]
     --augment    per-row team-order + move-slot permutation with action and
                  aux labels remapped (see entity_augment.py). Run A of the
                  experiment trains WITHOUT this for a clean protocol match
                  against the baseline; run B turns it on.
     --smoke      no shards needed: random tokens through build/forward/
                  backward/save/load, then exit. Run this before shipping the
                  worktree to the training box.
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("train_entity.py"):
        raise SystemExit(0)

import math
import sys
import time

import numpy as np
import torch

from config import CFG, config_snapshot
from models.entity_hybrid import EntityHybridNet
from models.policy_value import clean_state_dict
from tokenizer import PositionTokenizer
from train import Shards, compute_loss, make_loader, run_epoch


class AugmentedShards(Shards):
    """Shards whose batches pass through entity_augment.augment_batch."""

    def __init__(self, split, tok, seed=0, cfg=CFG):
        """Load the split and keep the vocab slot ids the remap needs."""
        super().__init__(split, cfg)
        self.rng = np.random.default_rng(seed)
        self.slot_a = tok.vocab["SLOT_A"]
        self.slot_b = tok.vocab["SLOT_B"]

    def __getitem__(self, idxs):
        """Augment the numpy batch, then tensorize exactly like Shards."""
        from entity_augment import augment_batch
        tokens, acts, items, abils, moves = augment_batch(
            self.tokens[idxs], self.acts[idxs], self.opp_items[idxs],
            self.opp_abils[idxs], self.opp_moves[idxs], self.rng,
            self.slot_a, self.slot_b)
        return (torch.from_numpy(tokens.astype(np.int64)),
                torch.from_numpy(acts.astype(np.int64)),
                torch.from_numpy(self.value[idxs].astype(np.float32)),
                torch.from_numpy(self.weight[idxs].astype(np.float32)),
                torch.from_numpy(items.astype(np.int64)),
                torch.from_numpy(abils.astype(np.int64)),
                torch.from_numpy(moves.astype(np.int64)))


def build_model(tok, cfg, device):
    """Construct a fresh EntityHybridNet from the loaded tokenizer."""
    return EntityHybridNet(
        tok.vocab_size(), tok.n_tokens, tok.opp_species_positions(),
        len(tok.move_list), len(tok.item_list), len(tok.ability_list),
        cfg, policy_head="joint").to(device)


def smoke(cfg=CFG):
    """Sharded-data-free self-test: forward/backward/save/load round trip."""
    tok = PositionTokenizer.load(cfg)
    model = build_model(tok, cfg, "cpu")
    n = sum(p.numel() for p in model.parameters())
    print(f"entity-hybrid: {n / 1e6:.2f}M params")
    B = 4
    rng = np.random.default_rng(0)
    tokens = torch.from_numpy(
        rng.integers(0, tok.vocab_size(), (B, tok.n_tokens)).astype(np.int64))
    batch = (tokens,
             # a fixed statically-legal joint action (random pairs can hit the
             # -inf mask and NaN the smoke loss)
             torch.ones((B, 2), dtype=torch.long),
             # aux labels must not be all-ignore_index(0) or the CE is NaN
             torch.zeros(B), torch.ones(B),
             torch.ones((B, 6), dtype=torch.long),
             torch.ones((B, 6), dtype=torch.long),
             torch.ones((B, 6, 4), dtype=torch.long))
    loss, stats = compute_loss(model, batch, cfg)
    loss.backward()
    print(f"forward/backward ok, loss {float(loss):.3f}")
    dists, values, aux = model.predict_batch(tokens.numpy())
    assert dists.shape == (B, 1521) and values.shape == (B,)
    assert np.allclose(dists.sum(1), 1.0, atol=1e-4)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ck.pt"
        model.save(p)
        m2 = EntityHybridNet.load(p, cfg)
        d2, v2, _ = m2.predict_batch(tokens.numpy())
        assert np.allclose(dists, d2, atol=1e-5) and np.allclose(values, v2, atol=1e-5)
    from entity_augment import augment_batch
    t2, a2, *_ = augment_batch(
        tokens.numpy().astype(np.uint16), batch[1].numpy().astype(np.int8),
        np.zeros((B, 6), np.int16), np.zeros((B, 6), np.int16),
        np.zeros((B, 6, 4), np.int16), rng, tok.vocab["SLOT_A"],
        tok.vocab["SLOT_B"])
    assert t2.shape == tokens.shape and a2.shape == (B, 2)
    print("save/load + augment smoke ok")


def main(cfg=CFG):
    """Mirror train.main with the entity model and optional augmentation."""
    if "--smoke" in sys.argv:
        smoke(cfg)
        return
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 \
        and not sys.argv[1].startswith("--") else cfg.epochs
    augment = "--augment" in sys.argv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    tok = PositionTokenizer.load(cfg)
    train_ds = AugmentedShards("train", tok, cfg=cfg) if augment \
        else Shards("train", cfg)
    val_ds = Shards("val", cfg)      # validation is never augmented
    print(f"train {len(train_ds)} / val {len(val_ds)} transitions, "
          f"device={device}, augment={augment}")
    train_dl = make_loader(train_ds, cfg.batch_size, True, device, cfg)
    val_dl = make_loader(val_ds, cfg.batch_size, False, device, cfg)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last = cfg.checkpoint_dir / "entity_ckpt_last.pt"
    if last.exists():
        model = EntityHybridNet.load(last, cfg, device)
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
    tb = SummaryWriter(cfg.checkpoint_dir / "tb_entity")

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
              f"top1 {tr['top1_joint']:.3f} | "
              f"val loss {va['loss']:.4f} top1 {va['top1_joint']:.3f} | "
              f"{time.time() - t0:.0f}s")
        ck = {"hp": getattr(model, "_orig_mod", model).hp,
              "state": clean_state_dict(model), "epoch": epoch,
              "cfg": config_snapshot(cfg)}
        torch.save(ck, last)
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(ck, cfg.checkpoint_dir / "entity_ckpt_best.pt")
    tb.close()


if __name__ == "__main__":
    main()
