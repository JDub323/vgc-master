"""JEPA-Consequence chooser: rank predicted latent move-consequences.

Implements ``agents.interfaces.MoveChooser``. For each legal own joint move it
predicts a single latent consequence vector (integrating the opponent's
response and chance), scores it with the policy head, and plays the highest
(temperature-sampled). Values/scores are averaged over belief determinizations
and, when a luck latent is enabled, over a small noise ensemble. There is no
opponent-action enumeration and no matrix game — the opponent and randomness are
compressed inside the consequence vector. See ``JEPA_DESIGN.md``.
"""

import sys

import numpy as np
import torch

from actions import SlotAction, legal_joint_actions, _pos_maps
from config import CFG
from jepa.config import JCFG
from jepa.features import FeatureExtractor, my_action_arrays
from agents.jepa_world_model.v1 import _decoded, _info


class JEPAConsequenceChooser:
    """Pick the own move whose predicted latent consequence scores highest."""

    def __init__(self, model, vocab, cfg=CFG, jcfg=JCFG, seed=0, bridge=None):
        """Bind a loaded consequence model + vocab and optional damage bridge."""
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.vocab = vocab
        self.cfg = cfg
        self.jcfg = jcfg
        self.extractor = FeatureExtractor(vocab)
        import random
        self.rng = random.Random(seed)
        self.bridge = bridge
        self.device = next(model.parameters()).device
        # Self-play recording hook: when True, _plan stashes the exact
        # position/candidates/scores it acted on in self.last_plan, so
        # training samples are literally the play-time distribution.
        self.record = False
        self.last_plan = None

    def choose(self, tracker, belief, my_id, request, brought,
               opp_brought=None, temperature=None, root_noise=None):
        """Return one legal ``JointAction`` and display ``ChoiceInfo``."""
        temp = self.jcfg.play_temperature if temperature is None else temperature
        view = tracker._view(my_id)
        name_to_idx = {m.set["name"]: m.team_idx
                       for m in tracker.sides[my_id].mons}
        idx_of_pos = _pos_maps(request, name_to_idx)[0]
        legal = legal_joint_actions(request, idx_of_pos)
        if not legal:
            return (SlotAction("pass"), SlotAction("pass")), _info("no legal move")
        if len(legal) == 1:
            return legal[0], _info("forced", 0.0)
        try:
            return self._plan(view, legal, belief, brought, opp_brought, temp)
        except Exception as exc:
            print(f"[jepa-c] planning fell back: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return legal[0], _info("fallback:first-legal")

    def close(self):
        """Close the owned damage bridge, if any."""
        if self.bridge is not None:
            self.bridge.close()

    def _plan(self, view, legal, belief, brought, opp_brought, temp):
        """Score every candidate move by its predicted latent consequence."""
        cands = legal[:64]
        summary = belief.summary() if belief is not None else {}
        dmg = self._damage(view, belief)
        dets = (belief.sample_sets(self.jcfg.cons_determinizations, self.rng)
                if belief is not None else [None])
        c = len(cands)
        scores = np.zeros(c, dtype=np.float64)
        values = np.zeros(c, dtype=np.float64)
        passes = 0
        first_pos, acts_np, act = None, None, None
        for det in dets:
            opp_ms = _movesets(det)
            # brought/opp_brought left as None to match prep (jepa_data passes
            # neither), so the brought feature has the same value in train/play.
            pos = self.extractor.extract(view, summary, opp_movesets=opp_ms,
                                         dmg=dmg)
            if acts_np is None:
                # candidate action arrays only fill the OWN side, which does not
                # vary across determinizations -- build once, reuse every det
                first_pos = pos
                acts_np = np.stack(
                    [my_action_arrays(pos, a, self.vocab) for a in cands])
                act = torch.as_tensor(acts_np, dtype=torch.long,
                                      device=self.device)
            batch = self._batch(pos, c)
            z = self.model.encode(batch)
            m = max(1, self.jcfg.ensemble_m if self.jcfg.noise_dim else 1)
            for _ in range(m):
                xi = self.model.sample_noise(c, self.device)
                cons = self.model.consequence(z, act, batch["dmg"], xi)
                scores += self.model.score(cons).cpu().numpy()
                values += self.model.value(cons).cpu().numpy()
                passes += 1
        scores /= passes
        values /= passes

        idx = self._pick(scores, temp)
        if self.record:
            # exact play-time distribution for self-play training samples
            self.last_plan = {"pos": first_pos, "cands": cands,
                              "cand_acts": acts_np, "chosen": idx,
                              "scores": scores, "values": values}
        p = _softmax(scores)
        info = {"value": float(values[idx]), "solve": False,
                "strategy": _decoded(cands, p),
                "q": _decoded(cands, values),
                "opp_pred": [],
                "health": {"candidates": float(c),
                           "determinizations": float(len(dets)),
                           "ensemble": float(passes // len(dets))}}
        return cands[idx], info

    def _damage(self, view, belief):
        """Compute ally->foe damage edges when a bridge and belief exist."""
        if self.bridge is None or belief is None:
            return None
        from damage import damage_features
        try:
            return damage_features(view, belief, self.bridge)
        except Exception:
            return None

    def _batch(self, pos, n):
        """Replicate one position into an ``n``-row device tensor dict."""
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.device
                                          ).unsqueeze(0).expand(n, *a.shape)
        return {"gcat": t(pos.global_cat, torch.long),
                "gscal": t(pos.global_scalar, torch.float32),
                "mcat": t(pos.mon_cat, torch.long),
                "mscal": t(pos.mon_scalar, torch.float32),
                "dmg": t(pos.dmg_edge, torch.float32)}

    def _pick(self, scores, temp):
        """Argmax at ``temp==0``, else a temperature-softmax sample."""
        if temp <= 1e-6:
            return int(np.argmax(scores))
        p = _softmax(scores / temp)
        return int(self.rng.choices(range(len(p)), weights=p)[0])


def _movesets(team):
    """Sampled team -> 6 x up-to-4 move sids in preview order (``None`` -> [])."""
    if team is None:
        return None
    return [[mv for mv in s.get("moves", [])][:4] for s in team]


def _softmax(x):
    """Numerically stable softmax of a 1-D array."""
    e = np.exp(x - x.max())
    return e / e.sum()


def build_jepa_consequence_chooser(ckpt=None, cfg=CFG, jcfg=JCFG, seed=0):
    """Build the consequence chooser from a checkpoint, or random-init if absent."""
    from pathlib import Path

    from beliefs import load_dex
    from jepa.vocab import JEPAVocab
    from models.jepa_consequence import JEPAConsequenceModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(ckpt) if ckpt else None
    model = None
    if ckpt and ckpt.exists():
        try:
            model, vstate = JEPAConsequenceModel.load(ckpt, device)
            vocab = (JEPAVocab.from_state(vstate, load_dex(cfg)) if vstate
                     else JEPAVocab.build(cfg))
        except Exception as exc:
            print(f"[jepa-c] '{ckpt}' is not a consequence checkpoint ({exc}); "
                  "using a random-init net", file=sys.stderr)
            model = None
    if model is None:
        vocab = JEPAVocab.build(cfg)
        model = JEPAConsequenceModel(vocab.sizes(), jcfg, vocab.state()).to(device)
    # Use the model's own (checkpoint-stored) planner/training knobs so play
    # matches training -- crucially use_damage_features, which train_consequence
    # sets from whether the shards actually carried damage edges. Feeding damage
    # edges the model never trained on would push it off-distribution.
    mjcfg = model.jcfg
    bridge = None
    if getattr(mjcfg, "use_damage_features", False):
        try:
            from damage import DamageBridge
            bridge = DamageBridge(cfg)
        except Exception as exc:
            print(f"[jepa-c] damage bridge unavailable ({exc}); no edges",
                  file=sys.stderr)
    return JEPAConsequenceChooser(model, vocab, cfg, mjcfg, seed, bridge)
