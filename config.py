"""The one place every knob lives. Change model size or format here, nowhere else."""

import ast
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Typed source of every path, data, model, belief, search, and play knob."""
    # ---- format ----
    # Play format. Switch to "gen9championsvgc2026regma" for Reg M-A.
    format_id: str = "gen9championsvgc2026regmb"
    # Dataset files to train on (HF repo cameronangliss/vgc-battle-logs).
    dataset_name: str = "cameronangliss/vgc-battle-logs"
    dataset_files: tuple = (
        "logs_gen9championsvgc2026regma.json",
        "logs_gen9championsvgc2026regmabo3.json",
        "logs_gen9championsvgc2026regmb.json",
        "logs_gen9championsvgc2026regmbbo3.json",
    )

    # ---- paths ----
    data_dir: Path = Path("artifacts/data")            # raw downloaded logs
    parsed_dir: Path = Path("artifacts/parsed")        # parsed battle pickles
    prepped_dir: Path = Path("artifacts/prepped")      # tokenized npz shards
    artifacts_dir: Path = Path("artifacts")            # vocab.json, usage_stats.json, dex.json
    checkpoint_dir: Path = Path("artifacts/checkpoints")  # point at Drive on Colab
    node_dir: Path = Path("artifacts/node")            # where npm installed @smogon/calc / pokemon-showdown
    node_bin: str = "node"

    # ---- data prep ----
    val_frac: float = 0.05
    test_frac: float = 0.05
    split_seed: int = 7
    # sample weight = format_weight * rating_weight * recency_weight
    format_weights: dict = field(default_factory=lambda: {
        "gen9championsvgc2026regma": 0.4,
        "gen9championsvgc2026regmabo3": 0.4,
        "gen9championsvgc2026regmb": 1.0,
        "gen9championsvgc2026regmbbo3": 1.0,
    })
    unrated_weight: float = 0.6            # rating weight for games with no ladder rating
    rating_pivot: float = 1200.0           # rating_weight = clip(rating / pivot, 0.5, 1.5)
    recency_halflife_days: float = 90.0
    use_damage_features: bool = True       # needs node + @smogon/calc
    use_belief_damage_updates: bool = True # damage-likelihood particle killing (needs node)
    belief_damage_hits_per_pair: int = 2   # only the first N hits per (attacker, defender) constrain
    shard_size: int = 50_000               # transitions per npz shard

    # ---- tokenizer ----
    n_dmg_buckets: int = 20
    n_hp_buckets: int = 20
    n_speed_buckets: int = 12              # belief speed-range buckets, 25 speed each
    speed_bucket_width: float = 25.0
    n_prob_buckets: int = 5                # belief P(scarf) buckets
    n_bulk_buckets: int = 8                # belief bulk buckets over hp*(def+spd)/2
    bulk_bucket_width: float = 3500.0      # L50 bulk products run ~8k (frail) to ~25k+

    # ---- beliefs ----
    n_particles: int = 200
    # expand each train-split set into SP-spread archetype particles
    # (beliefs.ARCHETYPES); off = pre-phase-3 behavior (one-sided bounds only)
    spread_archetypes: bool = True
    resample_floor: float = 0.25           # resample when alive mass fraction drops below
    damage_tolerance: float = 0.03         # slack on observed damage fraction (replay HP is /100)
    # OTS sheets redact stat training (verified: showteam SP fields are empty
    # and sim damage at 0 SP undershoots replay damage). Real stats sit between
    # 0 SP and the 32-SP cap; beliefs.py derives exact one-sided bounds from
    # base stats + nature. This constant is only the fallback multiplier when
    # the species/move is missing from dex.json:
    investment_slack: float = 1.35
    # phase-3.1 strict SP inversion. Damage WE take -> feasible attack-SP
    # interval per (nature,item,ability) hypothesis, using the Node calc as an
    # exact forward oracle over an SP grid; particles are killed/narrowed by
    # attack EVs/nature WITHOUT the coarse-archetype over-kill (an archetype
    # whose fixed spread misses the observed damage no longer dies -- SP is a
    # free latent constrained to an interval). Speed does the same via the
    # move-order inequality (pure-Python calc_stat, no bridge). Defensive-EV
    # inference deliberately stays on the old one-sided slack (see
    # investigations.txt). Set either False to recover pre-3.1 (archetype
    # binary) behavior for an A/B against beliefs.py --audit.
    strict_attack_ev: bool = True
    strict_speed_ev: bool = True
    strict_sp_step: int = 2      # attack-SP grid step (SP) for the calc oracle
    # Objective (nature, SP-spread) prior from Pikalytics (artifacts/spreads.json,
    # built by build_spreads.py). The dataset's sheets redact nature+SP (every
    # set is 'serious'/0-EV), so the filter had no prior for that latent and
    # inverted at NEUTRAL nature -- matching ~0% of real sets and killing the
    # true set on any +nature mon. When on and the species is covered, each
    # redacted usage set is expanded into the top-K real (spread,nature) builds
    # (concrete evs + real nature, tested EXACTLY) plus one 'any' slack cushion
    # for off-list spreads; SP is fixed to the Pikalytics values and NATURE is
    # the inferred sub-dimension (which build survives the speed/damage facts).
    # Uncovered species fall back to the hand-built archetype + neutral-nature
    # SP-interval path. Off recovers pre-spreads behavior for an A/B.
    spreads_prior: bool = True
    spreads_top_k: int = 10        # (spread,nature) builds kept per covered set
    spreads_any_weight: float = 0.08  # prior mass on the off-list 'any' cushion
    # On HARD depletion (no train set satisfies the hard reveals), stitch new
    # particles by crossing two independent marginal buckets -- (moves,nature)
    # x (item,ability) -- filtered by the reveals, so a moveset seen with one
    # item/ability can pair with an item/ability seen on a different set. Lets
    # the filter represent novel-but-plausible COMBINATIONS the joint prior
    # never held, raising the "oracle in prior" ceiling. Backoff only (the
    # joint particles stay the default); False = old raw-prior fallback.
    factored_fallback: bool = True

    # ---- model ----
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1

    # ---- training ----
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 8
    warmup_steps: int = 250
    grad_clip: float = 1.0
    value_loss_weight: float = 0.5
    aux_set_loss_weight: float = 0.2
    num_workers: int = 2
    # torch.compile the training forward/backward (CUDA only; needs triton,
    # so Linux big box yes, Windows laptop no). Checkpoints stay eager-keyed
    # either way (models/policy_value.clean_state_dict).
    compile_model: bool = True
    #                                        torch.compile warmup util/memory spike

    # ---- search (phase 2) ----
    top_k_actions: int = 6                 # per-player pruning width; also eval recall@k
    n_determinizations: int = 4
    sims_per_move: int = 400
    play_temperature: float = 1.0
    solve_endgame_at: int = 2              # solve to terminal when <=N mons per side
    c_puct: float = 1.5                    # exploration constant in decoupled PUCT
    rollout_depth: int = 1                 # plies of real sim (greedy) before the
    #                                        value head is trusted at a new leaf.
    #                                        1 = evaluate the leaf immediately (v2
    #                                        default); >1 looks that many turns
    #                                        further so myopic-HP value mistakes
    #                                        (e.g. double-Protect) surface as the
    #                                        opponent's follow-up gets played out.

    # ---- self-play (phase 3) ----
    # Throughput scales with procs x workers (each worker owns 2 node
    # processes: a shared sidecar + the damage bridge); re-run env.py
    # --benchmark on the target box and adjust to taste. Current values are
    # deliberately tiny for a co-tenant GPU: 1 proc x 2 workers keeps the
    # network-eval batches small so generation sits at single-digit % GPU
    # util (the workload is CPU/Node-bound anyway), and small games/iter keeps
    # each generate->train->checkpoint cycle to a couple hours.
    sp_procs: int = 3                      # generator subprocesses (beat the GIL)
    sp_workers: int = 4                    # game threads per generator process
    #                                        procs*workers = concurrent games:
    #                                        3*4=12 (was 1*2=2). Bigger batches to
    #                                        the per-proc GPU evaluator + more sim
    #                                        parallelism across the 64 cores.
    sp_games_per_iter: int = 150
    sp_sims: int = 160                     # search budget while generating
    sp_buffer_iters: int = 5               # train on the last N iterations
    sp_epochs_per_iter: int = 2
    sp_lr: float = 1e-4                    # fine-tune LR (BC used 3e-4 fresh)
    sp_temp_turns: int = 8                 # tau=1 for the first N turns...
    sp_final_temp: float = 0.25            # ...then this (0 = argmax)
    sp_dirichlet_eps: float = 0.25         # AlphaZero root-noise mix-in
    sp_dirichlet_alpha: float = 0.35       # ~10 / typical root width (~30)
    sp_policy_targets_k: int = 32          # sparse policy target width
    sp_gate_games: int = 20                # quick new-vs-old gate per iteration

    # ---- human-vs-bot play ----
    showdown_port: int = 8000              # local pokemon-showdown server
    dashboard_port: int = 8010             # bot-thoughts dashboard (play.py)


CFG = Config()


# If an archived config predates a behavior-changing field, missing means the
# code that created the archive did not have that feature. Prefer legacy-off
# over silently enabling today's behavior for old bundles.
LEGACY_MISSING_DEFAULTS = {
    "strict_attack_ev": False,
    "strict_speed_ev": False,
    "spreads_prior": False,
    "factored_fallback": False,
}


def _jsonable(v):
    """Recursively convert dataclass values and paths to JSON-safe values."""
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    return v


def config_snapshot(cfg=CFG):
    """Typed JSON snapshot of a Config.

    Old benchmark archives wrote every value through str(); loaders below still
    accept those files, but new checkpoints/archives keep booleans, numbers,
    lists and dicts typed so model/search behavior can be reconstructed later.
    """
    return {k: _jsonable(v) for k, v in dataclasses.asdict(cfg).items()}


def _parse_legacy_string(s):
    """Parse old stringified config values, falling back to the input string."""
    if s in ("True", "False"):
        return s == "True"
    if s == "None":
        return None
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return s


def _coerce_like(default, value):
    """Coerce a decoded value to the type represented by ``default``."""
    if isinstance(value, str):
        value = _parse_legacy_string(value)
    if isinstance(default, Path):
        return Path(value)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, tuple):
        return tuple(value)
    if isinstance(default, dict):
        return dict(value)
    return value


def config_from_snapshot(snapshot, base=None):
    """Build a Config from a saved snapshot, tolerating missing/old fields."""
    if snapshot is None:
        return base or Config()
    if "config" in snapshot and isinstance(snapshot["config"], dict):
        snapshot = snapshot["config"]
    base = base or Config()
    vals = dataclasses.asdict(base)
    for f in dataclasses.fields(Config):
        if f.name in snapshot:
            vals[f.name] = _coerce_like(vals[f.name], snapshot[f.name])
        elif f.name in LEGACY_MISSING_DEFAULTS:
            vals[f.name] = LEGACY_MISSING_DEFAULTS[f.name]
    return Config(**vals)


def load_config_snapshot(path, base=None):
    """Read JSON and return ``config_from_snapshot(..., base=base)``."""
    return config_from_snapshot(json.loads(Path(path).read_text()), base=base)


def config_diff(a, b, fields=None):
    """Return [(field, a_value, b_value), ...] for meaningful config diffs."""
    da, db = config_snapshot(a), config_snapshot(b)
    names = fields or sorted(set(da) | set(db))
    out = []
    for name in names:
        va, vb = da.get(name), db.get(name)
        if va != vb:
            out.append((name, va, vb))
    return out
