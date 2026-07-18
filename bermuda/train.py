"""The LSMC fit: regress realized terminal payoff on φ, per phase.

`--target outcome` (default) is Longstaff–Schwartz's regress-realized-payoffs
estimator: every state on a path carries the path's final ±1. `--target
boot` is the Tsitsiklis–Van Roy regress-later ablation: targets mix the
outcome with the current model's value at the next decision of the same
(game, side) filtration, recomputed each epoch (fitted value iteration).

Diagnostics after training (held-out games, grouped split):
  - MSE and sign-accuracy (decided games only)
  - per-phase calibration: mean predicted V vs realized outcome by turn bucket
  - martingale residuals: mean/std of V(s_{t+1}) − V(s_t) per phase — drift
    localizes where the value function is systematically wrong (the
    Andersen–Broadie "hedging error" lens, plan.md §2.6).

CLI:
  python bermuda/train.py --shards DIR[,DIR...] --out CKPT.pt
      [--epochs E] [--batch B] [--lr LR] [--target outcome|boot]
      [--val-frac F] [--device cpu|cuda] [--seed S]
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import json
import time
from pathlib import Path

import numpy as np
import torch

from bermuda.config import BCFG
from bermuda.model import ValueMLP


def load_shards(dirs):
    """Concatenate every shard under the given directories."""
    arrays = {k: [] for k in ("feats", "turns", "z", "gid", "side", "step")}
    files = []
    for d in dirs:
        files += sorted(Path(d).glob("shard_*.npz"))
    assert files, f"no shards under {list(map(str, dirs))} — run paths.py"
    for f in files:
        blob = np.load(f)
        for k in arrays:
            arrays[k].append(blob[k])
    out = {k: np.concatenate(v) for k, v in arrays.items()}
    print(f"loaded {len(files)} shards: {len(out['z'])} rows, "
          f"{len(np.unique(out['gid']))} games")
    return out


def group_split(gid, val_frac, seed=13):
    """Boolean val mask, split by game so no battle leaks across the split."""
    h = (gid * np.int64(2654435761) + seed) % 1000
    return h < int(val_frac * 1000)


def order_index(data):
    """Row order sorted by (gid, side, step) plus next-row-in-path map."""
    order = np.lexsort((data["step"], data["side"], data["gid"]))
    nxt = np.full(len(order), -1, dtype=np.int64)
    same = ((data["gid"][order][1:] == data["gid"][order][:-1])
            & (data["side"][order][1:] == data["side"][order][:-1]))
    nxt[np.where(same)[0]] = order[np.where(same)[0] + 1]
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    return order, nxt, back


def fit(shard_dirs, out_path, epochs=None, batch=None, lr=None,
        target="outcome", val_frac=None, device="cpu", seed=0, bcfg=BCFG):
    torch.manual_seed(seed)
    data = load_shards(shard_dirs)
    n = len(data["z"])
    val = group_split(data["gid"], val_frac or bcfg.val_frac)
    feats = torch.from_numpy(np.ascontiguousarray(data["feats"]))
    turns = torch.from_numpy(np.minimum(data["turns"], 25)
                             .astype(np.int64))
    z = torch.from_numpy(data["z"].astype(np.float32))
    tr_idx = np.where(~val)[0]
    va_idx = np.where(val)[0]
    print(f"split: {len(tr_idx)} train rows / {len(va_idx)} val rows")

    model = ValueMLP(feats.shape[1], bcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr or bcfg.lr,
                            weight_decay=bcfg.weight_decay)
    batch = batch or bcfg.batch_size
    epochs = epochs or bcfg.epochs
    order, nxt, back = order_index(data)

    for ep in range(epochs):
        y = z.clone()
        if target == "boot":       # TvR regress-later ablation
            with torch.no_grad():
                preds = model.predict_np(data["feats"], turns.numpy())
            boot = np.where(nxt[back] >= 0, preds[np.maximum(nxt[back], 0)],
                            data["z"].astype(np.float32))
            y = torch.from_numpy(
                (0.5 * data["z"] + 0.5 * boot).astype(np.float32))
        model.train()
        perm = tr_idx[np.random.default_rng(seed + ep).permutation(
            len(tr_idx))]
        t0, tot, nb = time.time(), 0.0, 0
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(
                model(feats[j].to(device), turns[j].to(device)),
                y[j].to(device))
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        model.eval()
        with torch.no_grad():
            vp = model.predict_np(data["feats"][va_idx], turns[va_idx].numpy())
        vz = data["z"][va_idx].astype(np.float32)
        mse = float(np.mean((vp - vz) ** 2))
        dec = vz != 0
        sign = float(np.mean(np.sign(vp[dec]) == vz[dec])) if dec.any() else 0
        print(f"epoch {ep + 1}/{epochs}: train {tot / max(1, nb):.4f}  "
              f"val mse {mse:.4f}  sign-acc {sign:.3f}  "
              f"({time.time() - t0:.0f}s)")

    diagnostics(model, data, va_idx, order, nxt, back)
    out_path = Path(out_path)
    model.save(out_path, meta={
        "rows": n, "games": int(len(np.unique(data["gid"]))),
        "target": target, "epochs": epochs, "val_mse": mse,
        "sign_acc": sign, "shards": list(map(str, shard_dirs))})
    print(f"saved {out_path}")
    return out_path


def diagnostics(model, data, va_idx, order, nxt, back):
    """Calibration + martingale residual drift per phase on held-out games."""
    preds = model.predict_np(data["feats"], data["turns"].astype(np.int64))
    va = np.zeros(len(preds), dtype=bool)
    va[va_idx] = True
    print("phase   n(val)   mean V̂   mean z   |resid drift|  resid std")
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15), (15, 26)):
        m = va & (data["turns"] >= lo) & (data["turns"] < hi)
        if not m.any():
            continue
        has_next = m & (nxt[back] >= 0)
        resid = (preds[np.maximum(nxt[back], 0)] - preds)[has_next]
        print(f"{lo:2d}-{hi:<3d} {int(m.sum()):8d}  "
              f"{preds[m].mean():+.3f}  {data['z'][m].mean():+.3f}  "
              f"{abs(resid.mean()) if len(resid) else 0:12.4f}  "
              f"{resid.std() if len(resid) else 0:.4f}")


def main():
    import sys
    args = sys.argv[1:]

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    if not args or "--help" in args:
        print(__doc__)
        return
    fit(shard_dirs=opt("--shards", str(BCFG.shards_dir / "gen0")).split(","),
        out_path=opt("--out", str(BCFG.default_ckpt)),
        epochs=int(opt("--epochs")) if opt("--epochs") else None,
        batch=int(opt("--batch")) if opt("--batch") else None,
        lr=float(opt("--lr")) if opt("--lr") else None,
        target=opt("--target", "outcome"),
        val_frac=float(opt("--val-frac")) if opt("--val-frac") else None,
        device=opt("--device", "cpu"),
        seed=int(opt("--seed", 0)))


if __name__ == "__main__":
    main()
