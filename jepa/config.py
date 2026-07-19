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

    # ---- consequence variant (v2: pure latent move-consequence embedding) ----
    # A JEPA model that predicts, per legal MY move, a single latent
    # "consequence" vector that already integrates the opponent's response and
    # chance (no explicit next state, no (a,b) matrix). A policy head ranks
    # those vectors. See JEPA_DESIGN.md "Consequence variant".
    horizon: int = 1                   # plies ahead the target future encodes
    n_cpred_layers: int = 2            # consequence-predictor transformer depth
    n_cand: int = 12                   # BC candidate cap per position
    # encoded-luck latent: wired as a predictor input (jepa/features + the
    # ConsequencePredictor accept it). Default 0 = deterministic, so the
    # consequence vector IS the distribution summary (its mean), which is what
    # the value/policy heads need. A stochastic (CVAE-style) latent that samples
    # the consequence distribution is the documented lever: set noise_dim>0.
    noise_dim: int = 0                 # encoded-luck latent width (0 = deterministic)
    ensemble_m: int = 4                # decision-time luck samples averaged (noise_dim>0)
    cons_determinizations: int = 2     # belief samples averaged at play time
    w_jepa_c: float = 1.0              # consequence-latent prediction loss
    w_bc: float = 1.0                  # policy-head behavior-cloning loss
    w_value_c: float = 0.5             # outcome value off the consequence vector
    w_spread: float = 0.05             # luck-latent usage (spread) regularizer

    # ---- v3 Strategy-JEPA stage 1 (models/jepa_strategy.py) ----
    # Recursive latent dynamics T(Z, a, b) trained with multi-step JEPA
    # unrolls on sequence windows; play is a depth-`plan_depth` latent
    # matrix-tree search (depth 1 = v1-style one-ply matrix at v2 speed).
    seq_len: int = 5                   # actions per training window (positions = +1)
    seq_stride: int = 1                # window stride (dense; chains no longer
    #                                    break on unobservable turns, so windows
    #                                    are cheap and the data is 4x richer)
    unroll_gamma: float = 0.8          # per-step discount on multi-step JEPA loss
    w_jepa_s: float = 1.0              # latent dynamics prediction loss
    w_policy_s: float = 0.5            # policy CE (mean per head, masked)
    # Fraction of the policy gradient allowed into the encoder trunk. 1.0
    # reproduces stage 1 (policy CEs conquered the encoder, JEPA loss rose all
    # run); 0.0 reproduces stage 2's full detach (policy collapsed to my_acc
    # 0.14 — a linear head on frozen features is too weak, and the mushy prior
    # let the value head's off-policy bias run the anchored solve into
    # 0-for-17 degenerate play). 0.1 lets the heads shape features without
    # owning them; the heads themselves are MLPs so they can also learn on
    # mostly-frozen features.
    policy_grad_scale: float = 0.1
    w_value_s: float = 0.5             # margin-distribution CE off encoded + unrolled
    n_margin_bins: int = 9             # final mon differential -4..+4
    plan_depth: int = 1                # play-time latent search depth (1 or 2)
    top_k_mine_s: int = 6              # root candidate width, own side
    top_k_opp_s: int = 6               # root candidate width, opponent side
    child_k: int = 4                   # per-side candidate width at depth >= 2
    # Prior-anchored equilibrium: p ∝ prior·exp(eta·Mq). eta -> 0 plays the
    # policy prior; eta -> inf approaches Nash on the value matrix. The payoff
    # matrix is STANDARDIZED (per-decision mean/std) inside the solve, so eta
    # is in units of that decision's own value spread — a miscalibrated value
    # head cannot buy influence just by being confidently spread out.
    solver_eta: float = 1.5

    # ---- consequence self-play (selfplay_jepa.py) ----
    # jepa-c decides ~300x faster than DUCT because it never touches the sim
    # at decision time, so self-play is sim-bound: throughput scales with
    # procs x workers almost linearly until the box runs out of Node processes.
    spj_games_per_iter: int = 400      # games per generate->train iteration
    spj_procs: int = 4                 # generator subprocesses (beat the GIL)
    spj_workers: int = 6               # game threads per process (sim-bound)
    spj_max_turns: int = 300           # stall cap -> tie (z=0)
    spj_temp_turns: int = 8            # tau=1 for the first N turns...
    spj_final_temp: float = 0.35       # ...then this (generation only)
    spj_eps: float = 0.03              # eps-uniform candidate exploration
    spj_beta: float = 1.0              # advantage temperature in exp(A/beta)
    spj_w_max: float = 5.0             # advantage-weight clip (stability)
    spj_human_mix: float = 0.25        # fraction of human-BC rows mixed per iter
    spj_buffer_iters: int = 4          # train on the last N iterations' shards
    spj_epochs: int = 2                # epochs per iteration
    spj_lr: float = 5e-5               # fine-tune LR (BC used 3e-4 fresh)
    spj_gate_games: int = 40           # argmax gate current-vs-best per iter
    spj_gate_keep: float = 0.55        # promote to best at/above this winrate
    spj_league_every: int = 3          # add current to the league every N iters
    spj_p_mirror: float = 0.5          # opponent sampling: current model
    spj_p_league: float = 0.3          # ...a random league checkpoint
    spj_p_anchor: float = 0.2          # ...the frozen starting anchor


JCFG = JEPAConfig()


def scaled_consequence(base=None):
    """A ~6x-parameter consequence config (~50M): richer latents, still fast.

    16 tokens x ~60 candidates per decision keeps inference in single-digit
    milliseconds on GPU at this size; the self-play bottleneck stays the sim."""
    import dataclasses
    return dataclasses.replace(base or JEPAConfig(), d_model=448, d_embed=96,
                               n_heads=8, n_enc_layers=5, n_cpred_layers=3,
                               d_ff=1792)
