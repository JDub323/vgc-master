"""Shared loading/inference helpers for offline policy evaluation tools."""

from pathlib import Path

import numpy as np
import torch

from config import CFG
from models.policy_value import PolicyValueNet
from train import Shards


def load_test_predictions(checkpoint, cfg=CFG):
    """Load the test split, damage grids, model, and joint predictions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Shards("test", cfg)
    files = sorted(Path(cfg.prepped_dir).glob("test_*.npz"))
    if not files:
        raise FileNotFoundError(f"no test shards under {cfg.prepped_dir}")
    dmg_active = np.concatenate([np.load(f)["dmg_active"] for f in files])
    model = PolicyValueNet.load(checkpoint, cfg, device)
    dists = []
    for i in range(0, len(ds), cfg.batch_size):
        dist, _, _ = model.predict_batch(ds.tokens[i:i + cfg.batch_size])
        dists.append(dist)
    return ds, dmg_active, model, np.concatenate(dists).astype(np.float64)
