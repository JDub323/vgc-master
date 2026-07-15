"""Stdio agent adapter: one MoveChooser behind a JSON-lines game protocol.

The round-robin coordinator (round_robin.py) plays cross-branch tournaments by
talking to each contestant as a black-box subprocess. This module is that
subprocess: it builds one chooser (the same four families play.py offers),
wraps it in the CTS-honest observe_game.Bot, and answers protocol messages on
stdin with Showdown choice strings on stdout. Whatever an experiment changes
behind build_chooser — tokenizer, action space, no net at all — stays
invisible to the coordinator. The *game protocol* is the compatibility
boundary, not any Python class, which is what lets incompatible branches play
each other without ever merging.

Protocol v1, one JSON object per line (-> coordinator to agent, <- reply):

  -> {"type": "hello", "protocol": 1}
  <- {"type": "ready", "protocol": 1, "name": ..., "agent": ...}
  -> {"type": "game_start", "side": "p1", "team": [PokemonSet, ...],
      "opp_preview": [preview sets], "seed": 123, "temperature": 0.0}
  <- {"type": "game_ready"}
  -> {"type": "lines", "lines": ["|move|...", ...]}          (no reply)
  -> {"type": "request", "rqid": 7, "request": {...}, "deadline_s": null}
  <- {"type": "choice", "rqid": 7, "choice": "move 1 2, move 3",
      "wall_s": 0.41, "error": "..."}                        (error optional)
  -> {"type": "game_end", "winner": "p1"}                    (no reply)
  -> {"type": "quit"}                                        (process exits)

Defaults preserve benchmark.run_game behavior exactly: team preview brings the
first four in order, forced switches answer with a random legal choice, and
move requests go through Bot.decide at the game's temperature. Experiments may
answer any request kind more cleverly (e.g. route forced switches through
search) — the coordinator applies whatever choice string comes back.
``deadline_s`` is advisory: the adapter cannot preempt a running search, so
budget enforcement (playing "default" on late replies) is coordinator-side.

At startup the adapter points sys.stdout at stderr and keeps the real stdout
privately for protocol writes, so stray prints inside model/search code cannot
corrupt the message stream. Set $VGC_NODE_DIR to override Config.node_dir so
exported bundles share one Node/pokemon-showdown install instead of copying it.

CLI: python agent_server.py --agent search|policy|max-damage|random
                            [--ckpt PATH] [--name LABEL] [--seed N]
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("agent_server.py"):
        raise SystemExit(0)

import json
import os
import random
import sys
import time
from pathlib import Path

from config import CFG

PROTOCOL_VERSION = 1


def apply_env_overrides(cfg):
    """Point ``cfg.node_dir`` at $VGC_NODE_DIR (shared Node install), if set."""
    node_dir = os.environ.get("VGC_NODE_DIR")
    if node_dir:
        cfg.node_dir = Path(node_dir)
    return cfg


def build_chooser(kind, ckpt, cfg, seed=0):
    """Construct one registered chooser family (mirrors play.build_chooser).

    Experiment branches extend this function with their own kinds; the
    coordinator only ever sees the entrypoint command recorded in the bundle
    manifest, so a new kind needs no coordinator changes."""
    if kind == "random":
        from agents.random.v1 import RandomChooser
        return RandomChooser(random.Random(seed))
    if kind == "max-damage":
        from agents.max_damage.v1 import MaxDamageChooser
        return MaxDamageChooser(cfg)
    if kind in ("search", "policy"):
        import torch

        from agents.determinized_duct.v1 import DeterminizedDUCTChooser
        from agents.policy_only.v1 import PolicyOnlyChooser
        # exp/entity-hybrid: dispatch on the checkpoint's recorded
        # architecture, so `--ckpt entity_ckpt_best.pt` serves the entity
        # model through the identical DUCT / policy-only stack
        from models.entity_hybrid import load_any_policy_model
        from tokenizer import PositionTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chooser = DeterminizedDUCTChooser(
            load_any_policy_model(ckpt, cfg, device),
            PositionTokenizer.load(cfg), cfg, seed=seed)
        return chooser if kind == "search" else PolicyOnlyChooser(chooser)
    raise SystemExit(
        f"unknown --agent kind {kind!r} — extend agent_server.build_chooser")


class AgentServer:
    """One chooser behind the JSON-lines stdio protocol (module docstring)."""

    def __init__(self, kind, ckpt, name=None, seed=0, cfg=CFG):
        """Load usage stats, build the chooser once; game state comes later."""
        self.cfg = apply_env_overrides(cfg)
        self.kind, self.seed = kind, seed
        self.name = name or kind
        usage_p = self.cfg.artifacts_dir / "usage_stats.json"
        self.usage = json.loads(usage_p.read_text()) if usage_p.exists() else {}
        self.chooser = build_chooser(kind, ckpt, self.cfg, seed)
        self.bot = None
        self.rng = random.Random(seed)     # forced-switch parity rng
        self.temperature = 0.0

    def on_hello(self, msg):
        """Return the ready handshake identifying this adapter."""
        return {"type": "ready", "protocol": PROTOCOL_VERSION,
                "name": self.name, "agent": self.kind}

    def on_game_start(self, msg):
        """Build a fresh CTS-honest Bot for one game; the chooser is reused."""
        from observe_game import Bot
        self.rng = random.Random(msg.get("seed", self.seed))
        self.temperature = float(msg.get("temperature", 0.0))
        self.bot = Bot(msg["side"], msg["team"], msg["opp_preview"],
                       self.chooser, self.usage, self.cfg)
        return {"type": "game_ready"}

    def on_lines(self, msg):
        """Feed protocol lines into the tracker; returns ``None`` (no reply)."""
        if self.bot is not None:
            self.bot.feed(msg["lines"])
        return None

    def on_request(self, msg):
        """Answer one sim request with a choice message (never raises)."""
        req = msg["request"]
        t0 = time.perf_counter()
        error = None
        try:
            if req.get("teamPreview"):
                n = min(req.get("maxChosenTeamSize") or 4,
                        len(self.bot.tracker.sides[self.bot.side].mons))
                self.bot.brought = list(range(n))
                choice = "team " + "".join(str(i + 1) for i in range(n))
            elif req.get("forceSwitch"):
                from env import random_choice
                choice = random_choice(req, self.rng)
            else:
                choice, _ = self.bot.decide(req, self.temperature)
        except Exception as exc:   # keep playing; the coordinator tallies it
            error = f"{type(exc).__name__}: {exc}"
            choice = "default"
        out = {"type": "choice", "rqid": msg.get("rqid"), "choice": choice,
               "wall_s": time.perf_counter() - t0}
        if error:
            out["error"] = error
        return out

    def on_game_end(self, msg):
        """Drop per-game state; returns ``None`` (chooser survives)."""
        self.bot = None
        return None

    def serve(self, stdin, proto_out):
        """Dispatch messages until quit/EOF, then close the chooser."""
        handlers = {"hello": self.on_hello, "game_start": self.on_game_start,
                    "lines": self.on_lines, "request": self.on_request,
                    "game_end": self.on_game_end}
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("type") == "quit":
                break
            reply = handlers[msg["type"]](msg)
            if reply is not None:
                proto_out.write(json.dumps(reply) + "\n")
                proto_out.flush()
        close = getattr(self.chooser, "close", None)
        if close:
            close()


def main():
    """Parse flags, exile prints to stderr, and serve the protocol."""
    args = sys.argv[1:]

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    proto_out = sys.stdout
    sys.stdout = sys.stderr   # stray prints must not corrupt the protocol
    server = AgentServer(opt("--agent", "search"),
                         opt("--ckpt", CFG.checkpoint_dir / "ckpt_best.pt"),
                         name=opt("--name"), seed=int(opt("--seed", 0)))
    server.serve(sys.stdin, proto_out)


if __name__ == "__main__":
    main()
