"""The one place every knob lives. Change model size or format here, nowhere else."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---- format ----
    # Play format. Switch to "gen9championsvgc2026regma" for Reg M-A.
    format_id: str = "gen9championsvgc2026regmb"
    # Dataset files to train on (HF repo cameronangliss/vgc-battle-logs).
    dataset_name: str = "cameronangliss/vgc-battle-logs"
    dataset_files: tuple = (
        "logs_gen9championsvgc2026regma.json",
        "logs_gen9championsvgc2026regmabo3.json",
        "logs_gen9championsvgc2026regmb.json",
        "logs_gen9championsvgc2026regmbbo3.json",
    )

    # ---- paths ----
    data_dir: Path = Path("artifacts/data")            # raw downloaded logs
    parsed_dir: Path = Path("artifacts/parsed")        # parsed battle pickles
    prepped_dir: Path = Path("artifacts/prepped")      # tokenized npz shards
    artifacts_dir: Path = Path("artifacts")            # vocab.json, usage_stats.json, dex.json
    checkpoint_dir: Path = Path("artifacts/checkpoints")  # point at Drive on Colab
    node_dir: Path = Path("artifacts/node")            # where npm installed @smogon/calc / pokemon-showdown
    node_bin: str = "node"

    # ---- data prep ----
    val_frac: float = 0.05
    test_frac: float = 0.05
    split_seed: int = 7
    # sample weight = format_weight * rating_weight * recency_weight
    format_weights: dict = field(default_factory=lambda: {
        "gen9championsvgc2026regma": 0.4,
        "gen9championsvgc2026regmabo3": 0.4,
        "gen9championsvgc2026regmb": 1.0,
        "gen9championsvgc2026regmbbo3": 1.0,
    })
    unrated_weight: float = 0.6            # rating weight for games with no ladder rating
    rating_pivot: float = 1200.0           # rating_weight = clip(rating / pivot, 0.5, 1.5)
    recency_halflife_days: float = 90.0
    use_damage_features: bool = True       # needs node + @smogon/calc
    use_belief_damage_updates: bool = True # damage-likelihood particle killing (needs node)
    belief_damage_hits_per_pair: int = 2   # only the first N hits per (attacker, defender) constrain
    shard_size: int = 50_000               # transitions per npz shard

    # ---- tokenizer ----
    n_dmg_buckets: int = 20
    n_hp_buckets: int = 20
    n_speed_buckets: int = 12              # belief speed-range buckets, 25 speed each
    speed_bucket_width: float = 25.0
    n_prob_buckets: int = 5                # belief P(scarf) buckets
    n_bulk_buckets: int = 8                # belief bulk buckets over hp*(def+spd)/2
    bulk_bucket_width: float = 3500.0      # L50 bulk products run ~8k (frail) to ~25k+

    # ---- beliefs ----
    n_particles: int = 200
    resample_floor: float = 0.25           # resample when alive mass fraction drops below
    damage_tolerance: float = 0.03         # slack on observed damage fraction (replay HP is /100)
    # OTS sheets redact stat training (verified: showteam SP fields are empty
    # and sim damage at 0 SP undershoots replay damage). Real stats sit between
    # 0 SP and the 32-SP cap; beliefs.py derives exact one-sided bounds from
    # base stats + nature. This constant is only the fallback multiplier when
    # the species/move is missing from dex.json:
    investment_slack: float = 1.35

    # ---- model ----
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1

    # ---- training ----
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 8
    warmup_steps: int = 250
    grad_clip: float = 1.0
    value_loss_weight: float = 0.5
    aux_set_loss_weight: float = 0.2
    num_workers: int = 2

    # ---- search (phase 2) ----
    top_k_actions: int = 6                 # per-player pruning width; also eval recall@k
    n_determinizations: int = 4
    sims_per_move: int = 400
    play_temperature: float = 1.0
    solve_endgame_at: int = 2              # solve to terminal when <=N mons per side
    c_puct: float = 1.5                    # exploration constant in decoupled PUCT

    # ---- human-vs-bot play ----
    showdown_port: int = 8000              # local pokemon-showdown server
    dashboard_port: int = 8010             # bot-thoughts dashboard (play.py)


CFG = Config()
