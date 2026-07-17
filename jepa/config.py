"""Knobs for the JEPA world-model experiment.

Kept out of the repo-wide ``config.Config`` on purpose: that module is one of
the ten hashed ``BEHAVIOR_SOURCE_FILES``, so adding a field there would
invalidate every existing archive. These knobs are experiment-local; the global
``CFG`` is still used for paths/format/belief settings.
"""

from dataclasses import dataclass


@dataclass
class JEPAConfig:
    """Typed model, training, and planner settings for the JEPA world model."""

    # ---- model dims ----
    d_model: int = 192
    d_embed: int = 64          # per-categorical embedding width
    n_heads: int = 6
    n_enc_layers: int = 3      # encoder (context) transformer depth
    n_pred_layers: int = 2     # predictor (dynamics) transformer depth
    d_ff: int = 512
    dropout: float = 0.1

    # ---- training ----
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 6
    warmup_steps: int = 200
    grad_clip: float = 1.0
    ema_decay: float = 0.996           # target-encoder EMA
    # loss weights
    w_jepa: float = 1.0                # latent prediction (representation space)
    w_value: float = 1.0               # outcome MSE on the taken transition
    w_ground_hp: float = 1.0           # next-HP regression per mon
    w_ground_faint: float = 0.5        # next-faint BCE per mon
    w_ground_status: float = 0.25      # next-status CE per mon
    w_ground_field: float = 0.25       # next field-condition heads
    w_ground_order: float = 0.25       # who-moved-first CE
    w_opp_policy: float = 0.5          # opponent action CE (candidate generator)
    w_my_prior: float = 0.5            # my action CE (candidate generator)
    w_vicreg_var: float = 1.0          # VICReg variance (anti-collapse)
    w_vicreg_cov: float = 0.04         # VICReg covariance (decorrelate dims)
    vicreg_gamma: float = 1.0          # target per-dim std floor

    # ---- data prep ----
    shard_size: int = 40_000
    unroll: int = 1                    # latent-consistency horizon (>1 = lever)

    # ---- planner (play time) ----
    n_determinizations: int = 2        # belief samples to average V over
    top_k_mine: int = 6                # my candidate joint actions
    top_k_opp: int = 6                 # opponent candidate joint actions
    solver_iters: int = 256            # regret-matching iterations
    use_damage_features: bool = True   # ally->foe damage edges (needs bridge)
    play_temperature: float = 1.0


JCFG = JEPAConfig()
