"""Behavior cloning on human transitions (both perspectives of every game).

Loss = weighted CE on the joint action (masked 39x39 softmax)
     + value_loss_weight * MSE(value, final outcome)
     + aux_set_loss_weight * (CE item + CE ability + BCE moves) on oracle sets.

AMP (bf16 on cuda), AdamW, cosine LR with warmup, gradient clipping.
Checkpoints go to cfg.checkpoint_dir — point that at Google Drive on Colab.
TensorBoard scalars plus a plain terminal table per epoch.

CLI: python train.py [epochs]
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("train.py"):
        raise SystemExit(0)

import math
import sys
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (BatchSampler, DataLoader, Dataset,
                              RandomSampler, SequentialSampler)

from actions import N_SLOT_ACTIONS
from config import CFG, config_snapshot
from models.policy_value import PolicyValueNet, clean_state_dict
from tokenizer import PositionTokenizer


class Shards(Dataset):
    """In-memory concatenation of one BC split's NPZ arrays."""

    def __init__(self, split, cfg=CFG):
        """Load and concatenate ``<split>_*.npz`` from ``cfg.prepped_dir``."""
        files = sorted(glob(str(cfg.prepped_dir / f"{split}_*.npz")))
        parts = [np.load(f) for f in files]
        self.tokens = np.concatenate([p["tokens"] for p in parts])
        self.acts = np.concatenate([p["acts"] for p in parts])
        self.value = np.concatenate([p["value"] for p in parts])
        self.weight = np.concatenate([p["weight"] for p in parts])
        self.opp_items = np.concatenate([p["opp_items"] for p in parts])
        self.opp_abils = np.concatenate([p["opp_abils"] for p in parts])
        self.opp_moves = np.concatenate([p["opp_moves"] for p in parts])
        self.weight = self.weight / self.weight.mean()

    def __len__(self):
        """Return the number of transition rows."""
        return len(self.tokens)

    def __getitem__(self, idxs):
        """idxs is a whole batch of indices (see make_loader): numpy fancy
        indexing materializes each field once per batch instead of running
        Python per sample — the loader was the bottleneck, not the GPU."""
        return (torch.from_numpy(self.tokens[idxs].astype(np.int64)),
                torch.from_numpy(self.acts[idxs].astype(np.int64)),
                torch.from_numpy(self.value[idxs].astype(np.float32)),
                torch.from_numpy(self.weight[idxs].astype(np.float32)),
                torch.from_numpy(self.opp_items[idxs].astype(np.int64)),
                torch.from_numpy(self.opp_abils[idxs].astype(np.int64)),
                torch.from_numpy(self.opp_moves[idxs].astype(np.int64)))


def make_loader(ds, batch_size, shuffle, device, cfg=CFG):
    """Batched-index loading: the sampler yields index lists, __getitem__
    returns ready tensors, automatic collation is off (batch_size=None)."""
    base = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    return DataLoader(ds, batch_size=None,
                      sampler=BatchSampler(base, batch_size, drop_last=False),
                      num_workers=cfg.num_workers,
                      pin_memory=device == "cuda",
                      persistent_workers=cfg.num_workers > 0)


def compute_loss(model, batch, cfg=CFG):
    """Return ``(scalar_loss, detached_metric_tensors)`` for one BC batch."""
    tokens, acts, value_t, w, items_t, abils_t, moves_t = batch
    pol, value, (item_lg, abil_lg, move_lg) = model(tokens)
    label = acts[:, 0] * N_SLOT_ACTIONS + acts[:, 1]
    logits = pol.masked_fill(~model.joint_mask, float("-inf"))
    ce = F.cross_entropy(logits, label, reduction="none")
    top1 = logits.argmax(-1) == label
    policy_loss = (ce * w).mean()
    value_loss = (F.mse_loss(value, value_t, reduction="none") * w).mean()

    item_loss = F.cross_entropy(item_lg.flatten(0, 1), items_t.flatten(),
                                ignore_index=0)
    abil_loss = F.cross_entropy(abil_lg.flatten(0, 1), abils_t.flatten(),
                                ignore_index=0)
    moves_hot = torch.zeros_like(move_lg).scatter_(-1, moves_t, 1.0)
    moves_hot[..., 0] = 0                       # index 0 = unknown/pad
    move_loss = F.binary_cross_entropy_with_logits(move_lg, moves_hot)
    aux_loss = item_loss + abil_loss + move_loss

    loss = (policy_loss + cfg.value_loss_weight * value_loss
            + cfg.aux_set_loss_weight * aux_loss)
    # stats stay 0-dim GPU tensors; .item() every step would sync the device
    return loss, {"loss": loss.detach(), "policy": policy_loss.detach(),
                  "value": value_loss.detach(), "aux": aux_loss.detach(),
                  "top1_joint": top1.detach().float().mean()}


def run_epoch(model, loader, device, opt=None, sched=None, cfg=CFG):
    """Run one train/eval epoch and return mean Python-float metrics."""
    model.train(opt is not None)
    agg, n = {}, 0
    autocast = torch.autocast(device, dtype=torch.bfloat16,
                              enabled=device == "cuda")
    for batch in loader:
        batch = [t.to(device, non_blocking=True) for t in batch]
        # batch dim is dynamic (drop_last=False remainders, val loader):
        # without this, every new shape recompiles the whole graph
        for t in batch:
            torch._dynamo.mark_dynamic(t, 0)
        with torch.set_grad_enabled(opt is not None), autocast:
            loss, stats = compute_loss(model, batch, cfg)
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
    return {k: float(v / n) for k, v in agg.items()}   # one sync per epoch


def main(cfg=CFG):
    """Build/resume, train, validate, and checkpoint from CLI/config inputs."""
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else cfg.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")     # TF32 outside autocast
    tok = PositionTokenizer.load(cfg)
    train_ds, val_ds = Shards("train", cfg), Shards("val", cfg)
    print(f"train {len(train_ds)} / val {len(val_ds)} transitions, device={device}")
    train_dl = make_loader(train_ds, cfg.batch_size, True, device, cfg)
    # same batch as training: a larger no-grad batch barely helps (val is 5%
    # of the data) but its activation peak sets the GPU memory high-water mark
    val_dl = make_loader(val_ds, cfg.batch_size, False, device, cfg)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last = cfg.checkpoint_dir / "ckpt_last.pt"
    if last.exists():
        model = PolicyValueNet.load(last, cfg, device)
        start_epoch = torch.load(last, map_location="cpu", weights_only=False)["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")
    else:
        model = PolicyValueNet(tok.vocab_size(), tok.n_tokens,
                               tok.opp_species_positions(), len(tok.move_list),
                               len(tok.item_list), len(tok.ability_list), cfg,
                               policy_head="joint").to(device)
        start_epoch = 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params")
    if cfg.compile_model and device == "cuda":
        model = torch.compile(model)   # shares params; checkpoints unaffected
        print("torch.compile on (first train + first val step build the graphs)")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay,
                            fused=device == "cuda")
    total_steps = max(1, len(train_dl) * epochs)

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        t = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(cfg.checkpoint_dir / "tb")

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
              f"train loss {tr['loss']:.4f} pol {tr['policy']:.4f} val {tr['value']:.4f} "
              f"aux {tr['aux']:.4f} top1 {tr['top1_joint']:.3f} | "
              f"val loss {va['loss']:.4f} top1 {va['top1_joint']:.3f} | "
              f"{time.time() - t0:.0f}s")
        ck = {"hp": model.hp, "state": clean_state_dict(model), "epoch": epoch,
              "cfg": config_snapshot(cfg)}
        torch.save(ck, last)
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(ck, cfg.checkpoint_dir / "ckpt_best.pt")
    tb.close()


if __name__ == "__main__":
    main()
