"""Behavior cloning on human transitions (both perspectives of every game).

Loss = weighted CE on the two slot actions
     + value_loss_weight * MSE(value, final outcome)
     + aux_set_loss_weight * (CE item + CE ability + BCE moves) on oracle sets.

AMP (bf16 on cuda), AdamW, cosine LR with warmup, gradient clipping.
Checkpoints go to cfg.checkpoint_dir — point that at Google Drive on Colab.
TensorBoard scalars plus a plain terminal table per epoch.

CLI: python train.py [epochs]
"""

import math
import sys
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import CFG
from models.policy_value import PolicyValueNet
from tokenizer import PositionTokenizer


class Shards(Dataset):
    def __init__(self, split, cfg=CFG):
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
        return len(self.tokens)

    def __getitem__(self, i):
        return (self.tokens[i].astype(np.int64), self.acts[i].astype(np.int64),
                np.float32(self.value[i]), np.float32(self.weight[i]),
                self.opp_items[i].astype(np.int64),
                self.opp_abils[i].astype(np.int64),
                self.opp_moves[i].astype(np.int64))


def compute_loss(model, batch, cfg=CFG):
    tokens, acts, value_t, w, items_t, abils_t, moves_t = batch
    slots, value, (item_lg, abil_lg, move_lg) = model(tokens)
    ce = (F.cross_entropy(slots[:, 0], acts[:, 0], reduction="none")
          + F.cross_entropy(slots[:, 1], acts[:, 1], reduction="none"))
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
    with torch.no_grad():
        joint_hit = ((slots[:, 0].argmax(-1) == acts[:, 0])
                     & (slots[:, 1].argmax(-1) == acts[:, 1])).float().mean()
    return loss, {"loss": loss.item(), "policy": policy_loss.item(),
                  "value": value_loss.item(), "aux": aux_loss.item(),
                  "top1_joint": joint_hit.item()}


def run_epoch(model, loader, device, opt=None, sched=None, cfg=CFG):
    model.train(opt is not None)
    agg, n = {}, 0
    autocast = torch.autocast(device, dtype=torch.bfloat16,
                              enabled=device == "cuda")
    for batch in loader:
        batch = [t.to(device, non_blocking=True) for t in batch]
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
    return {k: v / n for k, v in agg.items()}


def main(cfg=CFG):
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else cfg.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = PositionTokenizer.load(cfg)
    train_ds, val_ds = Shards("train", cfg), Shards("val", cfg)
    print(f"train {len(train_ds)} / val {len(val_ds)} transitions, device={device}")
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=device == "cuda")
    val_dl = DataLoader(val_ds, cfg.batch_size, num_workers=cfg.num_workers)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last = cfg.checkpoint_dir / "ckpt_last.pt"
    if last.exists():
        model = PolicyValueNet.load(last, cfg, device)
        start_epoch = torch.load(last, map_location="cpu", weights_only=False)["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")
    else:
        model = PolicyValueNet(tok.vocab_size(), tok.n_tokens,
                               tok.opp_species_positions(), len(tok.move_list),
                               len(tok.item_list), len(tok.ability_list), cfg).to(device)
        start_epoch = 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
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
        ck = {"hp": model.hp, "state": model.state_dict(), "epoch": epoch}
        torch.save(ck, last)
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(ck, cfg.checkpoint_dir / "ckpt_best.pt")
    tb.close()


if __name__ == "__main__":
    main()
