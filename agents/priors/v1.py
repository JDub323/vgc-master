"""Frozen v1 legal-action masking and top-k policy prior."""

import numpy as np

from actions import joint_index


class PolicyValuePrior:
    """Map a model joint distribution onto the retained legal joint actions.

    The order of operations intentionally matches the original search v1:
    normalize over all position-legal actions, select top-k from that result,
    then renormalize the retained actions. A zero-mass legal slice falls back
    to uniform before pruning.
    """

    def legal_priors(self, joint_dist, legal_joints, top_k=None):
        """Return ``(float64[K] probabilities, K retained JointAction pairs)``."""
        joints = list(legal_joints)
        if not joints:
            raise ValueError("legal_joints must not be empty")
        p = np.array([joint_dist[joint_index(a, b)] for a, b in joints],
                     dtype=np.float64)
        total = p.sum()
        p = p / total if total > 0 else np.full(len(joints), 1.0 / len(joints))
        if top_k and len(joints) > top_k:
            keep = np.argsort(-p)[:top_k]
            joints = [joints[i] for i in keep]
            p = p[keep] / p[keep].sum()
        return p, joints
