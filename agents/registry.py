"""Explicit implementation registry and AgentSpec construction."""

import ast
import hashlib
import inspect
from pathlib import Path

from agents.determinized_duct.v1 import DeterminizedDUCTChooser
from agents.beliefs.v1 import OpponentBelief
from agents.encoding.v1 import TokenPositionEncoder
from agents.evaluators.v1 import PolicyValueLeafEvaluator
from agents.ids import (BELIEF_V1, DETERMINIZED_DUCT_V1, ENCODER_V1,
                        EVALUATOR_V1, MAX_DAMAGE_V1,
                        POLICY_ONLY_V1, PRIOR_V1, RANDOM_V1, SEARCHER_V1)
from agents.max_damage.v1 import MaxDamageChooser
from agents.policy_only.v1 import PolicyOnlyChooser
from agents.priors.v1 import PolicyValuePrior
from agents.random.v1 import RandomChooser
from agents.search.v1 import DecoupledUCTSearcher
from agents.spec import AgentSpec, config_from_agent_spec


class AgentRegistry:
    """Safe allow-list for archived implementation IDs.

    There is deliberately no arbitrary ``importlib`` fallback: a manifest is
    runnable only while every recorded ID still has an explicitly retained
    implementation.
    """

    def __init__(self):
        """Create empty ``impl_id -> class`` maps for agents and bricks."""
        self.agents = {}
        self.bricks = {}

    def register_agent(self, impl, cls):
        """Bind a stable agent implementation ID to a constructor class."""
        self.agents[impl] = cls

    def register_brick(self, impl, cls):
        """Bind a stable brick implementation ID to a constructor class."""
        self.bricks[impl] = cls

    def validate(self, spec):
        """Return a parsed ``AgentSpec`` or raise for unknown/missing IDs."""
        spec = AgentSpec.from_dict(spec)
        if spec.agent_impl not in self.agents:
            raise KeyError(f"unregistered agent implementation: {spec.agent_impl}")
        for name, brick in spec.bricks.items():
            if brick.impl not in self.bricks:
                raise KeyError(
                    f"unregistered {name} implementation: {brick.impl}")
        if spec.agent_impl in (DETERMINIZED_DUCT_V1, POLICY_ONLY_V1):
            required = {
                "belief_model", "position_encoder", "policy_prior",
                "leaf_evaluator", "searcher",
            }
            missing = sorted(required - set(spec.bricks))
            if missing:
                raise ValueError(
                    f"{spec.agent_impl} is missing required bricks: {missing}")
        return spec

    def build(self, spec, *, model=None, tokenizer=None, cfg=None, seed=0,
              debug=False, sidecar=None, apply_spec_config=True):
        """Construct the exact chooser and bricks named by ``spec``.

        Belief state remains external. The resolved belief class is attached
        as ``belief_model_cls`` so game layers can verify/construct the expected
        implementation without transferring lifecycle ownership to the agent.
        """
        spec = self.validate(spec)
        if cfg is not None and apply_spec_config:
            cfg = config_from_agent_spec(cfg, spec)
        cls = self.agents[spec.agent_impl]
        if spec.agent_impl == RANDOM_V1:
            return cls()
        if spec.agent_impl == MAX_DAMAGE_V1:
            return cls(cfg)

        wrapped_policy_only = spec.agent_impl == POLICY_ONLY_V1
        full_cls = DeterminizedDUCTChooser if wrapped_policy_only else cls
        chooser = full_cls(model, tokenizer, cfg, seed=seed, debug=debug,
                           sidecar=sidecar)

        def brick(name):
            try:
                return self.bricks[spec.bricks[name].impl]
            except KeyError as exc:
                raise ValueError(
                    f"{spec.agent_impl} requires brick {name!r}") from exc

        chooser.position_encoder = brick("position_encoder")(
            tokenizer, chooser.bridge)
        chooser.policy_prior = brick("policy_prior")()
        chooser.leaf_evaluator = brick("leaf_evaluator")(model)
        chooser.searcher = brick("searcher")()
        chooser.belief_model_cls = brick("belief_model")
        chooser.agent_spec = spec
        return PolicyOnlyChooser(chooser) if wrapped_policy_only else chooser


REGISTRY = AgentRegistry()
REGISTRY.register_agent(DETERMINIZED_DUCT_V1, DeterminizedDUCTChooser)
REGISTRY.register_agent(POLICY_ONLY_V1, PolicyOnlyChooser)
REGISTRY.register_agent(MAX_DAMAGE_V1, MaxDamageChooser)
REGISTRY.register_agent(RANDOM_V1, RandomChooser)
REGISTRY.register_brick(BELIEF_V1, OpponentBelief)
REGISTRY.register_brick(ENCODER_V1, TokenPositionEncoder)
REGISTRY.register_brick(PRIOR_V1, PolicyValuePrior)
REGISTRY.register_brick(EVALUATOR_V1, PolicyValueLeafEvaluator)
REGISTRY.register_brick(SEARCHER_V1, DecoupledUCTSearcher)

AGENTS = REGISTRY.agents
BRICKS = REGISTRY.bricks

# Shared modules called by the versioned v1 classes. They are hashed into new
# archives as well: editing one in place must not silently alter an old agent.
BEHAVIOR_SOURCE_FILES = (
    "actions.py", "beliefs.py", "config.py", "damage.py", "data.py", "env.py",
    "models/policy_value.py", "search/mcts.py", "search/node.py", "tokenizer.py",
)


# Source identity uses docstring-stripped ASTs, so comments/docstrings and
# formatting do not invalidate an archive while logic edits still fail closed.
HASH_SCHEME_AST = "ast-v1"
HASH_SCHEME = HASH_SCHEME_AST


def _strip_docstrings(tree):
    """Remove docstring statements from a parsed AST in place; return it."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            del body[0]
            if not body:            # docstring-only body must stay valid
                body.append(ast.Pass())
    return tree


def _normalized_source_hash(path, scheme=HASH_SCHEME):
    """Return the AST-normalized SHA-256 identity of one source file."""
    data = Path(path).read_bytes()
    if Path(path).suffix != ".py":
        return hashlib.sha256(data).hexdigest()
    if scheme != HASH_SCHEME_AST:
        raise ValueError(f"unknown source hash scheme: {scheme!r}")
    tree = _strip_docstrings(ast.parse(data))
    dump = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(dump.encode()).hexdigest()


def implementation_source_hashes(spec, repo_root=None, scheme=HASH_SCHEME):
    """SHA-256 identity for resolved implementation and shared source files."""
    spec = REGISTRY.validate(spec)
    root = Path(repo_root or Path(__file__).resolve().parent.parent).resolve()
    paths = {root / relative for relative in BEHAVIOR_SOURCE_FILES}
    implementations = [REGISTRY.agents[spec.agent_impl]] + [
        REGISTRY.bricks[brick.impl] for brick in spec.bricks.values()
    ]
    for implementation in implementations:
        for cls in inspect.getmro(implementation):
            try:
                source = inspect.getsourcefile(cls)
            except TypeError:  # built-in bases such as object
                source = None
            if source:
                path = Path(source).resolve()
                if path.is_relative_to(root):
                    paths.add(path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "implementation source files are missing: " + ", ".join(missing))
    return {
        str(path.relative_to(root)): _normalized_source_hash(path, scheme)
        for path in sorted(paths)
    }


def verify_implementation_sources(spec, repo_root=None, allow_drift=False):
    """Compare manifest-recorded source hashes against this checkout.

    Manifests must record the current AST hash scheme. Returns the sorted list
    of drifted paths — empty when identity holds. A non-empty
    result raises unless ``allow_drift`` is true, in which case a loud warning
    is printed and the caller is expected to mark downstream results tainted.
    """
    expected = spec.source.get("files", {})
    if not expected:
        return []
    scheme = spec.source.get("hash_scheme")
    if scheme != HASH_SCHEME:
        raise ValueError(f"unsupported source hash scheme: {scheme!r}")
    current = implementation_source_hashes(spec, repo_root, scheme=scheme)
    differences = [
        path for path in sorted(set(expected) | set(current))
        if expected.get(path) != current.get(path)
    ]
    if not differences:
        return []
    message = (f"archived implementation source identity mismatch "
               f"({scheme}): " + ", ".join(differences))
    if not allow_drift:
        raise RuntimeError(message)
    print(f"  WARNING: {message}\n"
          "  --allow-source-drift is set: running the archive through "
          "CURRENT code; results will be recorded as source-drifted")
    return differences


def build_agent(spec, **kwargs):
    """Convenience wrapper returning ``REGISTRY.build(spec, **kwargs)``."""
    return REGISTRY.build(spec, **kwargs)
