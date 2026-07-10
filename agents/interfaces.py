"""Small runtime contracts shared by agent architectures and bricks.

Concrete dictionary/array layouts referenced below are defined in
``contracts.py`` and explained with examples in ``DATA_CONTRACTS.md``.
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from numpy.typing import NDArray

from contracts import (BeliefEvent, BeliefSummary, ChoiceInfo, JointAction,
                       ModelPrediction, PositionState, RootAggregateRow,
                       ShowdownRequest, SideID)


@runtime_checkable
class MoveChooser(Protocol):
    """The only interface required to play games and earn Elo."""

    bridge: object | None

    def choose(self, tracker: Any, belief: "BeliefModel", my_id: SideID,
               request: ShowdownRequest, brought: Sequence[int],
               opp_brought: Sequence[int] | None = None,
               temperature: float | None = None,
               root_noise: tuple[float, float] | None = None
               ) -> tuple[JointAction, ChoiceInfo]:
        """Choose one legal joint action and return display/training diagnostics."""
        ...


@runtime_checkable
class BeliefModel(Protocol):
    """Externally-owned hidden-information state for one battle."""

    def update(self, events: Sequence[BeliefEvent], viewer: SideID) -> None:
        """Consume newly public events; mutate this battle's posterior in place."""
        ...

    def summary(self) -> BeliefSummary:
        """Return tokenizer-facing posterior summaries keyed by team index."""
        ...

    def sample_sets(self, n: int, rng: Any) -> list[list[dict[str, Any]]]:
        """Draw ``n`` full opponent teams in team-preview order."""
        ...

    def top_particle(self, k: int) -> dict:
        """Return the highest-posterior hidden set for opponent index ``k``."""
        ...


@runtime_checkable
class PositionEncoder(Protocol):
    """Convert a tracker view plus external belief state into model tokens."""

    def encode(self, tracker: Any, side_id: SideID, belief: BeliefModel,
               belief_summary: BeliefSummary | None = None) -> NDArray[Any]:
        """Return one fixed-layout ``uint16[n_tokens]`` token vector."""
        ...


@runtime_checkable
class PolicyPrior(Protocol):
    """Mask and prune a model joint distribution for one position."""

    def legal_priors(self, joint_dist: NDArray[Any],
                     legal_joints: Sequence[JointAction],
                     top_k: int | None = None
                     ) -> tuple[NDArray[Any], list[JointAction]]:
        """Return normalized retained probabilities and matching actions."""
        ...


@runtime_checkable
class LeafEvaluator(Protocol):
    """Batched policy/value inference and searching-side value orientation."""

    def predict_batch(self, tokens: NDArray[Any]) -> ModelPrediction:
        """Map ``[B,n_tokens]`` tokens to joint priors, values, and aux arrays."""
        ...

    def value(self, values: NDArray[Any], index: int = 0) -> float:
        """Extract one searching-player-oriented scalar from a value batch."""
        ...

    def terminal_value(self, winner: SideID | None, searching_side: SideID,
                       opponent_side: SideID) -> float:
        """Return ``+1/-1/0`` for win/loss/tie from ``searching_side``."""
        ...


@runtime_checkable
class Searcher(Protocol):
    """Search mechanics used by a top-level chooser orchestrator."""

    def run(self, chooser: MoveChooser, determinizations: list,
            simulations_per_determinization: int) -> None:
        """Mutate each determinization's tree by running its simulation budget."""
        ...

    def aggregate_root(self, determinizations: list,
                       policy_only: bool = False) -> list[RootAggregateRow]:
        """Return ``[joint_action, count, value_sum]`` rows sorted by count."""
        ...
