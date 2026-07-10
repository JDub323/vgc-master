"""Frozen v1 uniform-random legal joint-action chooser."""

import random

from actions import legal_joint_actions
from agents.common import single_action_info


class RandomChooser:
    """Uniformly sample one position-legal joint action."""

    bridge = None

    def __init__(self, rng=None):
        """Use the supplied ``random.Random``-like object or a fresh RNG."""
        self.rng = rng or random.Random()

    def choose(self, tracker, belief, my_id, request, brought,
               opp_brought=None, temperature=None, root_noise=None):
        """Return ``(JointAction, ChoiceInfo)``; belief/search knobs are ignored."""
        from search.mcts import _pos_maps

        name_to_idx = {
            mon.set["name"]: mon.team_idx
            for mon in tracker.sides[my_id].mons
        }
        joints = legal_joint_actions(
            request, _pos_maps(request, name_to_idx)[0])
        return self.rng.choice(joints), single_action_info("uniform random")

    def close(self):
        """No-op: the random chooser owns no subprocess resources."""
        pass
