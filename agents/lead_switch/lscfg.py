"""Every knob of the exp/lead-switch experiment in one dataclass.

Deliberately NOT fields on config.Config: config.py is one of the ten hashed
behavior-source files, and this experiment must leave the frozen baseline's
source identity untouched. The server exposes the interesting knobs as CLI
flags; everything defaults to the values the exported bundles were run with.
"""

from dataclasses import dataclass


@dataclass
class LSConfig:
    """Typed constants for the expert scorer, value selectors, and lead net."""

    # ---- expert lead selection ----
    n_bring: int = 4               # VGC bring-4 (request may override)
    w_speed: float = 0.25          # weight on lead speed advantage vs. pair
    w_syn_fakeout: float = 0.10    # lead carries Fake Out
    w_syn_redirect: float = 0.05   # lead carries Rage Powder / Follow Me
    w_syn_speedctl: float = 0.08   # slow lead pair backed by its own speed control
    opp_lead_temp: float = 3.0     # softmax sharpness over their 15 lead pairs
    w_coverage: float = 0.5        # bring-4 coverage term vs. lead-pair term
    cov_mean: float = 0.7          # coverage: mean-case share
    cov_worst: float = 0.3         # coverage: worst-case (walled) share

    # ---- expert forced-switch selection ----
    sw_in_weight: float = 1.2      # incoming-damage weight (danger aversion)
    sw_spd_weight: float = 0.15    # speed tiebreak weight
    sw_hp_floor: float = 0.25      # hp fraction floor when scaling danger

    # ---- value-head selectors (frozen baseline net, no retraining) ----
    v_my_top: int = 12             # my expert-pruned preview combos evaluated
    v_opp_top: int = 6             # their expert-ranked lead pairs evaluated
    v_maximin_alpha: float = 0.5   # score = a*mean + (1-a)*min over their leads
    v_opp_switch_cap: int = 4      # opponent replacement scenarios averaged

    # ---- trained lead network ----
    nn_d_model: int = 64
    nn_layers: int = 2
    nn_heads: int = 4
    nn_dropout: float = 0.1
    nn_lr: float = 3e-4
    nn_epochs: int = 6
    nn_batch: int = 512
    nn_loser_weight: float = 0.3   # imitation weight for the losing side's picks
    nn_bring_weight: float = 0.5   # bring-head BCE weight vs. lead-pair CE
    nn_bring_lambda: float = 1.0   # inference: bring logits vs. lead-pair logit


LSCFG = LSConfig()
