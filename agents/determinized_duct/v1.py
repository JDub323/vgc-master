"""Frozen v1 determinized decoupled-UCT architecture."""

from search.mcts import Searcher, joint_choice
from agents.beliefs.v1 import OpponentBelief


class DeterminizedDUCTChooser(Searcher):
    """Algorithmic public name for the original full-search agent."""

    def __init__(self, *args, **kwargs):
        """Initialize the legacy-compatible orchestrator and pin belief v1."""
        super().__init__(*args, **kwargs)
        self.belief_model_cls = OpponentBelief
