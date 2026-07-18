"""The regression basis: a phase-conditioned value MLP.

Classic LSMC fits one polynomial regression per exercise date; sharing one
net across dates with a learned phase (turn-bucket) embedding is the same
estimator with statistical strength shared across dates. Output is tanh —
the expected terminal payoff in [-1, 1] from the viewer's side.
"""

import numpy as np
import torch
from torch import nn

from bermuda.config import BCFG
from bermuda.features import FEAT_DIM


class ValueMLP(nn.Module):
    """φ ⊕ phase-embedding -> tanh(E[outcome])."""

    def __init__(self, feat_dim=FEAT_DIM, bcfg=BCFG):
        super().__init__()
        self.feat_dim = feat_dim
        self.phase_buckets = bcfg.phase_buckets
        self.phase = nn.Embedding(bcfg.phase_buckets, bcfg.phase_dim)
        layers, d = [], feat_dim + bcfg.phase_dim
        for _ in range(bcfg.depth):
            layers += [nn.Linear(d, bcfg.hidden), nn.GELU(),
                       nn.Dropout(bcfg.dropout)]
            d = bcfg.hidden
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, turns):
        ph = self.phase(turns.clamp(0, self.phase_buckets - 1))
        return torch.tanh(self.mlp(torch.cat([x, ph], dim=-1))).squeeze(-1)

    # ---- persistence -----------------------------------------------------
    def save(self, path, meta=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.state_dict(),
                    "meta": {"feat_dim": self.feat_dim,
                             "feat_version": BCFG.feat_version,
                             **(meta or {})}}, path)

    @classmethod
    def load(cls, path, device="cpu", bcfg=BCFG):
        blob = torch.load(path, map_location=device, weights_only=False)
        meta = blob.get("meta", {})
        assert meta.get("feat_version") == bcfg.feat_version, \
            f"checkpoint feat_version {meta.get('feat_version')} != " \
            f"code {bcfg.feat_version} — retrain or pin the code"
        model = cls(meta.get("feat_dim", FEAT_DIM), bcfg)
        model.load_state_dict(blob["state"])
        model.to(device).eval()
        model.meta = meta
        return model

    @torch.no_grad()
    def predict_np(self, feats, turns):
        """np[N,D] float32, np[N] int -> np[N] values (batched, eval mode)."""
        was_training = self.training
        self.eval()
        out = []
        for i in range(0, len(feats), 4096):
            x = torch.from_numpy(np.ascontiguousarray(feats[i:i + 4096]))
            t = torch.from_numpy(np.ascontiguousarray(
                turns[i:i + 4096])).long()
            out.append(self(x, t).cpu().numpy())
        if was_training:
            self.train()
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
