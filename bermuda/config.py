"""Every BERMUDA knob in one dataclass, mirroring the repo's config.py style."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BermudaConfig:
    """Budgets, risk, net size, and paths for the LSMC agent."""

    feat_version: int = 1

    # ---- paths ----
    shards_dir: Path = Path("artifacts/bermuda/paths")
    ckpt_dir: Path = Path("artifacts/checkpoints")
    default_ckpt: Path = Path("artifacts/checkpoints/bermuda.pt")

    # ---- play-time exercise policy ----
    n_scenarios: int = 8        # K frozen scenarios per decision
    n_candidates: int = 12      # A joint actions kept by the heuristic screen
    risk_lambda0: float = 0.8   # entropic CE scale; lambda = lambda0 * (-V̄)
    opp_temp: float = 0.5       # opponent-model softmax temperature
    opp_uniform_mix: float = 0.10  # opponent-model mass spread uniformly
    max_recon_tries: int = 2    # extra attempts when a scenario fails to build

    # ---- path collection ----
    heuristic_temp: float = 0.7    # gen-0 behavior temperature (diffuse measure)
    explore_temp: float = 0.5      # gen>=1 behavior softmax-over-CE temperature
    max_turns: int = 300

    # ---- regression ----
    hidden: int = 256
    depth: int = 3
    phase_buckets: int = 26        # turn 0..24 plus 25+ bucket
    phase_dim: int = 16
    dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 6
    val_frac: float = 0.10         # group split by game id


BCFG = BermudaConfig()


def apply_runtime_env(cfg):
    """Honor $VGC_NODE_DIR the way agent_server does, for worktree runs that
    share the main checkout's Node install. Returns the (mutated) repo cfg."""
    node_dir = os.environ.get("VGC_NODE_DIR")
    if node_dir:
        cfg.node_dir = Path(node_dir)
    return cfg
