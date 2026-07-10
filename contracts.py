"""Shared data-shape vocabulary for manual review and type documentation.

The project deliberately passes plain dictionaries between the parser,
beliefs, tokenizer, simulator, agents, and archive loader.  These ``TypedDict``
definitions document those wire shapes without changing their runtime
representation.  Optional keys reflect legacy archives, partial Showdown
requests, and belief-sampled sets before :func:`env.full_set` normalization.

For exhaustive per-function input/output contracts, see ``DATA_CONTRACTS.md``.
"""

from typing import Any, Literal, TypeAlias, TypedDict

from numpy.typing import NDArray

SideID: TypeAlias = Literal["p1", "p2"]
DamageCell: TypeAlias = tuple[float, float]
DamageKey: TypeAlias = tuple[int, int, int]
DamageFeatures: TypeAlias = dict[DamageKey, DamageCell]
JointIndex: TypeAlias = int
JointAction: TypeAlias = tuple[Any, Any]  # tuple[SlotAction, SlotAction]
RootAggregateRow: TypeAlias = list[Any]


class PokemonSet(TypedDict, total=False):
    """Normalized team-preview set; ``evs`` are Champions stat points."""

    name: str
    species: str
    item: str
    ability: str
    moves: list[str]
    nature: str
    evs: list[int]
    gender: str
    level: int


class ParticleSet(TypedDict, total=False):
    """One hidden-set hypothesis stored by ``OpponentBelief``."""

    moves: tuple[str, ...]
    item: str
    ability: str
    nature: str
    evs: list[int] | None
    arch: str | None
    n: float


class MonOwnView(TypedDict, total=False):
    """One own-team mon in ``LogParser._view`` output."""

    team_idx: int
    species_cur: str
    hp: float
    status: str
    boosts: dict[str, int]
    fainted: bool
    active_slot: int | None
    appeared: bool
    mega_done: bool
    turns_active: int
    protect_ct: int
    item_consumed: bool
    set: PokemonSet


class MonOpponentView(TypedDict, total=False):
    """CTS-visible opponent mon; unrevealed set fields are absent."""

    team_idx: int
    species_cur: str
    level: int
    gender: str
    hp: float
    status: str
    boosts: dict[str, int]
    fainted: bool
    active_slot: int | None
    appeared: bool
    mega_done: bool
    turns_active: int
    protect_ct: int
    revealed_moves: list[str]
    revealed_item: str | None
    item_consumed: bool
    revealed_ability: str | None


class SideView(TypedDict):
    """One side inside a position snapshot."""

    team: list[MonOwnView] | list[MonOpponentView]
    mega_available: bool
    conditions: dict[str, bool]


class PositionState(TypedDict):
    """CTS-observable position returned by ``LogParser._view(side_id)``."""

    turn: int
    weather: str
    terrain: str
    trickroom: bool
    my: SideView
    opp: SideView


class BeliefSummaryEntry(TypedDict, total=False):
    """Posterior summary for one opponent team-preview index."""

    item: str
    p_item: float
    spe_lo: float
    spe_hi: float
    bulk: float
    arch: str
    p_arch: float
    nature: str
    p_nature: float


BeliefSummary: TypeAlias = dict[int, BeliefSummaryEntry]


class ShowdownPokemonRequest(TypedDict, total=False):
    """One party entry in ``request['side']['pokemon']``."""

    ident: str
    details: str
    condition: str
    active: bool


class ShowdownMoveRequest(TypedDict, total=False):
    """One legal move description inside an active-slot request."""

    move: str
    id: str
    pp: int
    disabled: bool
    target: str


class ShowdownActiveRequest(TypedDict, total=False):
    """Legality data for one active slot."""

    moves: list[ShowdownMoveRequest]
    trapped: bool
    canMegaEvo: bool


class ShowdownSideRequest(TypedDict, total=False):
    """Identity and current party order embedded in a request."""

    id: SideID
    name: str
    pokemon: list[ShowdownPokemonRequest]


class ShowdownRequest(TypedDict, total=False):
    """Partial Pokémon Showdown request consumed by action/chooser code."""

    wait: bool
    teamPreview: bool
    maxChosenTeamSize: int
    forceSwitch: list[bool]
    active: list[ShowdownActiveRequest | None]
    side: ShowdownSideRequest


class ChoiceInfo(TypedDict, total=False):
    """Diagnostics returned beside a chosen joint action."""

    value: float
    solve: bool
    visits: list[tuple[JointIndex, float]]
    strategy: list[tuple[str, float]]
    q: list[tuple[str, float]]
    opp_pred: list[tuple[str, float]]
    health: dict[str, float]


# Parser/belief event tuples are discriminated by element 0.  Exact variants
# are documented in DATA_CONTRACTS.md because tuple element types vary by tag.
BeliefEvent: TypeAlias = tuple[Any, ...]


class ModelAuxOutput(TypedDict):
    """Numpy auxiliary predictions from ``PolicyValueNet.predict_batch``."""

    items: NDArray[Any]
    abilities: NDArray[Any]
    moves: NDArray[Any]


ModelPrediction: TypeAlias = tuple[NDArray[Any], NDArray[Any], ModelAuxOutput]


class BrickSpecJson(TypedDict, total=False):
    """JSON representation of one versioned brick."""

    impl: str
    cfg: dict[str, Any]
    checkpoint: str
    vocab: str
    assets: dict[str, str]


class AgentSpecJson(TypedDict, total=False):
    """On-disk ``agent.json`` schema version 1."""

    schema_version: int
    agent_impl: str
    architecture: str
    config: str
    bricks: dict[str, BrickSpecJson]
    assets: dict[str, str]
    runtime: dict[str, Any]
    source: dict[str, Any]
    archive: dict[str, Any]


class BenchmarkResult(TypedDict, total=False):
    """One game row appended to ``benchmarks/registry.json``."""

    a: str
    b: str
    team_a: str
    team_b: str
    winner: Literal["a", "b", "tie"]
    turns: int
    sims: int
    sims_a: int
    sims_b: int
    rollout_depth: int
    rollout_depth_a: int
    rollout_depth_b: int
    temp: float
    date: str
    era_a: str
    era_b: str
    era_run: str
    era_run_a: str
    era_run_b: str
    git: str
    search_impl: str
    agent_impl_a: str
    agent_impl_b: str
    architecture_a: str
    architecture_b: str


class ParsedTurn(TypedDict):
    """One parsed protocol turn; turn zero has no state/action labels."""

    n: int
    states: dict[SideID, PositionState] | None
    actions: dict[SideID, tuple[int, int] | None] | None
    events: list[BeliefEvent]


class ParsedBattle(TypedDict, total=False):
    """CTS reconstruction and oracle labels emitted by ``LogParser.parse``."""

    tag: str
    format: str
    ts: int
    match_id: str
    players: dict[SideID, str]
    ratings: dict[SideID, int | None]
    teams: dict[SideID, list[PokemonSet]]
    winner: SideID
    turns: list[ParsedTurn]
    split: Literal["train", "val", "test"]


class BCShard(TypedDict):
    """Arrays stored in one behavior-cloning ``<split>_NNN.npz`` shard."""

    tokens: NDArray[Any]
    acts: NDArray[Any]
    value: NDArray[Any]
    weight: NDArray[Any]
    opp_items: NDArray[Any]
    opp_abils: NDArray[Any]
    opp_moves: NDArray[Any]
    dmg_active: NDArray[Any]


class SelfPlayShard(TypedDict):
    """Arrays returned by generation and stored in one self-play NPZ shard."""

    tokens: NDArray[Any]
    pol_idx: NDArray[Any]
    pol_p: NDArray[Any]
    value: NDArray[Any]
    weight: NDArray[Any]
    acts: NDArray[Any]
    opp_items: NDArray[Any]
    opp_abils: NDArray[Any]
    opp_moves: NDArray[Any]


class SelfPlaySample(TypedDict):
    """One in-memory decision recorded before conversion to NPZ arrays."""

    tokens: NDArray[Any]
    visits: list[tuple[JointIndex, float]]
    act: tuple[int, int]
