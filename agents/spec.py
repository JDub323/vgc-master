"""Serializable manifests for complete, static leaderboard agents."""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from agents.ids import (BELIEF_V1, DETERMINIZED_DUCT_V1, ENCODER_V1,
                        EVALUATOR_V1, PRIOR_V1, SEARCHER_V1)

AGENT_SPEC_FILENAME = "agent.json"


@dataclass(frozen=True)
class BrickSpec:
    """Serializable implementation/config/assets for one injected brick."""
    impl: str
    cfg: dict = field(default_factory=dict)
    checkpoint: str | None = None
    vocab: str | None = None
    assets: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        """Coerce a ``BrickSpec`` or JSON-shaped mapping to ``BrickSpec``."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or not value.get("impl"):
            raise ValueError("each brick must be an object with a non-empty impl")
        return cls(
            impl=str(value["impl"]),
            cfg=dict(value.get("cfg") or {}),
            checkpoint=value.get("checkpoint"),
            vocab=value.get("vocab"),
            assets=dict(value.get("assets") or {}),
        )

    def to_dict(self):
        """Return a JSON-serializable mapping, omitting empty optional fields."""
        return {k: v for k, v in dataclasses.asdict(self).items()
                if v not in (None, {}, [])}


@dataclass(frozen=True)
class AgentSpec:
    """Schema-v1 manifest for one complete immutable leaderboard agent."""
    agent_impl: str
    bricks: dict[str, BrickSpec]
    architecture: str
    schema_version: int = 1
    config: str = "config.json"
    assets: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    archive: dict = field(default_factory=dict)

    def __post_init__(self):
        """Reject unsupported schema versions and missing identity labels."""
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported AgentSpec schema_version {self.schema_version}")
        if not self.agent_impl:
            raise ValueError("agent_impl must not be empty")
        if not self.architecture:
            raise ValueError("architecture must not be empty")

    @classmethod
    def from_dict(cls, value):
        """Coerce an ``AgentSpec`` or decoded ``agent.json`` mapping."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("AgentSpec must be a JSON object")
        bricks = {
            str(name): BrickSpec.from_dict(spec)
            for name, spec in (value.get("bricks") or {}).items()
        }
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            agent_impl=str(value.get("agent_impl") or ""),
            architecture=str(value.get("architecture") or ""),
            config=str(value.get("config") or "config.json"),
            bricks=bricks,
            assets=dict(value.get("assets") or {}),
            runtime=dict(value.get("runtime") or {}),
            source=dict(value.get("source") or {}),
            archive=dict(value.get("archive") or {}),
        )

    @classmethod
    def load(cls, path):
        """Read UTF-8 JSON at ``path`` and return a validated ``AgentSpec``."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        """Return the complete JSON-serializable manifest mapping."""
        return {
            "schema_version": self.schema_version,
            "agent_impl": self.agent_impl,
            "architecture": self.architecture,
            "config": self.config,
            "bricks": {name: spec.to_dict()
                       for name, spec in self.bricks.items()},
            "assets": dict(self.assets),
            "runtime": dict(self.runtime),
            "source": dict(self.source),
            "archive": dict(self.archive),
        }

    def dump(self, path):
        """Write indented schema-v1 JSON to ``path``; return ``None``."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def resolve(self, bundle_dir, relative_path):
        """Resolve an archived path without permitting bundle traversal."""
        if not relative_path:
            return None
        bundle = Path(bundle_dir).resolve()
        path = (bundle / relative_path).resolve()
        try:
            path.relative_to(bundle)
        except ValueError as exc:
            raise ValueError(
                f"AgentSpec path escapes archive: {relative_path!r}") from exc
        return path

    def behavior_paths(self):
        """Every relative behavior asset referenced by this manifest."""
        paths = {self.config, *self.assets.values()}
        for brick in self.bricks.values():
            paths.update(brick.assets.values())
            if brick.checkpoint:
                paths.add(brick.checkpoint)
            if brick.vocab:
                paths.add(brick.vocab)
        return {p for p in paths if p}


def default_duct_spec(cfg, *, runtime=None, source=None, archive=None):
    """Manifest for the behavior-equivalent v1 full-search architecture."""
    belief_cfg = {
        name: getattr(cfg, name) for name in (
            "n_particles", "resample_floor", "damage_tolerance",
            "investment_slack", "belief_damage_hits_per_pair",
            "spread_archetypes", "strict_attack_ev", "strict_speed_ev",
            "strict_sp_step", "spreads_prior", "spreads_top_k",
            "spreads_any_weight", "factored_fallback",
        )
    }
    return AgentSpec(
        agent_impl=DETERMINIZED_DUCT_V1,
        architecture="DeterminizedDUCTChooser",
        bricks={
            "belief_model": BrickSpec(BELIEF_V1, belief_cfg,
                                      assets={
                                          "usage": "usage_stats.json",
                                          "dex": "dex.json",
                                          "spreads": "spreads.json",
                                      }),
            "position_encoder": BrickSpec(
                ENCODER_V1,
                {"use_damage_features": cfg.use_damage_features},
                vocab="vocab.json",
                assets={"dex": "dex.json"}),
            "policy_prior": BrickSpec(
                PRIOR_V1, {"top_k_actions": cfg.top_k_actions},
                checkpoint="ckpt.pt"),
            "leaf_evaluator": BrickSpec(
                EVALUATOR_V1, {"rollout_depth": cfg.rollout_depth},
                checkpoint="ckpt.pt"),
            "searcher": BrickSpec(
                SEARCHER_V1,
                {name: getattr(cfg, name) for name in (
                    "n_determinizations", "sims_per_move",
                    "solve_endgame_at", "c_puct", "rollout_depth",
                )}),
        },
        assets={
            "checkpoint": "ckpt.pt", "vocab": "vocab.json",
            "usage": "usage_stats.json", "dex": "dex.json",
            "spreads": "spreads.json",
        },
        runtime=dict(runtime or {}),
        source=dict(source or {}),
        archive=dict(archive or {}),
    )


def config_from_agent_spec(cfg, spec):
    """Apply authoritative per-brick behavior config to a runtime Config."""
    spec = AgentSpec.from_dict(spec)
    valid = {item.name for item in dataclasses.fields(cfg)}
    values = {}
    owners = {}
    for brick_name, brick in spec.bricks.items():
        for key, value in brick.cfg.items():
            if key not in valid:
                raise ValueError(
                    f"{brick_name}: unknown archived config field {key!r}")
            if key in values and values[key] != value:
                raise ValueError(
                    f"conflicting archived {key}: {owners[key]}={values[key]!r}, "
                    f"{brick_name}={value!r}")
            values[key], owners[key] = value, brick_name
    return dataclasses.replace(cfg, **values)
