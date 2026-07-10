"""Versioned, composable battle agents.

Gameplay code should depend on :class:`MoveChooser`; archived agents should be
constructed through :mod:`agents.registry` so implementation IDs, rather than
today's imports, determine the code path that is run.
"""

from agents.interfaces import (BeliefModel, LeafEvaluator, MoveChooser,
                               PolicyPrior, PositionEncoder, Searcher)
from agents.spec import AgentSpec, BrickSpec

__all__ = [
    "AgentSpec", "BeliefModel", "BrickSpec", "LeafEvaluator", "MoveChooser",
    "PolicyPrior", "PositionEncoder", "Searcher",
]
