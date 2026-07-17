"""Protocol server with real team-preview and forced-switch decisions.

The stock agent_server answers team preview with "team 1234" and forced
switches with a random legal pick — the two blind spots this experiment
targets. This server subclasses AgentServer and routes exactly those two
request kinds through pluggable selectors while every move request still goes
through the untouched frozen chooser (the baseline model is never modified):

  --leads  first4 | expert | value | nn
  --switch random | expert | value

  expert  hard-coded damage-calc matchup scoring (agents/lead_switch/expert)
  value   the frozen baseline net's value head over hypothetical post-decision
          positions (agents/lead_switch/value; needs --agent search|policy)
  nn      a trained LeadNet checkpoint (--leadnet, agents/lead_switch/leadnet)

Any selector exception falls back to the stock behavior for that request and
reports the error in the choice message, so a selector bug degrades to the
baseline rather than forfeiting the game.

CLI: python lead_switch_server.py [--agent search] [--ckpt PATH]
         [--leads expert|value|nn|first4] [--switch expert|value|random]
         [--leadnet PATH] [--name LABEL] [--seed N]
"""

if __name__ == "__main__":
    from cli_help import show_help
    if show_help("lead_switch_server.py"):
        raise SystemExit(0)

import sys
import time

from agent_server import AgentServer
from agents.lead_switch.expert import (ExpertLeadSelector,
                                       ExpertSwitchSelector,
                                       team_choice_string)
from agents.lead_switch.lscfg import LSCFG
from config import CFG


class LeadSwitchServer(AgentServer):
    """AgentServer whose preview/forced-switch answers come from selectors."""

    def __init__(self, kind, ckpt, leads="expert", switch="expert",
                 leadnet=None, name=None, seed=0, cfg=CFG, ls=LSCFG):
        """Build the wrapped chooser, then the two selectors from flags."""
        super().__init__(kind, ckpt,
                         name=name or f"{kind}+{leads}-leads/{switch}-switch",
                         seed=seed, cfg=cfg)
        self.ls = ls
        inner = getattr(self.chooser, "chooser", self.chooser)
        bridge = getattr(self.chooser, "bridge", None)
        model = getattr(inner, "model", None)
        tok = getattr(inner, "tok", None)

        def need_net(what):
            """Assert the wrapped chooser exposes the frozen net for ``what``."""
            if model is None or tok is None:
                raise SystemExit(f"--{what} value needs --agent search|policy "
                                 "(the frozen net is the evaluator)")
            return model, tok

        self.lead_sel = None
        if leads == "expert":
            self.lead_sel = ExpertLeadSelector(self.cfg, bridge, ls)
        elif leads == "value":
            from agents.lead_switch.value import ValueLeadSelector
            self.lead_sel = ValueLeadSelector(*need_net("leads"), self.cfg,
                                              bridge, ls)
        elif leads == "nn":
            from agents.lead_switch.leadnet import NNLeadSelector
            from tokenizer import PositionTokenizer
            assert leadnet, "--leads nn needs --leadnet PATH"
            self.lead_sel = NNLeadSelector(leadnet,
                                           PositionTokenizer.load(self.cfg),
                                           ls)
        elif leads != "first4":
            raise SystemExit(f"unknown --leads {leads!r}")

        self.switch_sel = None
        if switch == "expert":
            self.switch_sel = ExpertSwitchSelector(self.cfg, bridge, ls)
        elif switch == "value":
            from agents.lead_switch.value import ValueSwitchSelector
            self.switch_sel = ValueSwitchSelector(*need_net("switch"),
                                                  self.cfg, bridge, ls)
        elif switch != "random":
            raise SystemExit(f"unknown --switch {switch!r}")

    def _stock_preview(self, req):
        """The baseline 'first four in order' preview answer."""
        n = min(req.get("maxChosenTeamSize") or 4,
                len(self.bot.tracker.sides[self.bot.side].mons))
        self.bot.brought = list(range(n))
        return "team " + "".join(str(i + 1) for i in range(n))

    def _preview(self, req):
        """Selector-driven preview answer; sets ``bot.brought``."""
        if self.lead_sel is None:
            return self._stock_preview(req)
        bot = self.bot
        bot.belief.update(bot.tracker.drain_events(), viewer=bot.side)
        n = min(req.get("maxChosenTeamSize") or 4,
                len(bot.tracker.sides[bot.side].mons))
        order, _ = self.lead_sel.choose(bot.tracker, bot.belief, bot.side,
                                        n_bring=n)
        order = order[:n]
        bot.brought = list(order)
        return team_choice_string(order)

    def _force_switch(self, req):
        """Selector-driven forced-switch answer."""
        if self.switch_sel is None:
            from env import random_choice
            return random_choice(req, self.rng)
        bot = self.bot
        bot.belief.update(bot.tracker.drain_events(), viewer=bot.side)
        return self.switch_sel.choose(req, bot.tracker, bot.belief, bot.side)

    def on_request(self, msg):
        """Answer one sim request; selector errors fall back to stock play."""
        req = msg["request"]
        t0 = time.perf_counter()
        error = None
        try:
            if req.get("teamPreview"):
                try:
                    choice = self._preview(req)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    choice = self._stock_preview(req)
            elif req.get("forceSwitch"):
                try:
                    choice = self._force_switch(req)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    from env import random_choice
                    choice = random_choice(req, self.rng)
            else:
                choice, _ = self.bot.decide(req, self.temperature)
        except Exception as exc:    # keep playing; the coordinator tallies it
            error = f"{type(exc).__name__}: {exc}"
            choice = "default"
        out = {"type": "choice", "rqid": msg.get("rqid"), "choice": choice,
               "wall_s": time.perf_counter() - t0}
        if error:
            out["error"] = error
        return out


def main():
    """Parse flags, exile prints to stderr, and serve the protocol."""
    args = sys.argv[1:]

    def opt(flag, default=None):
        """Return the token after ``flag`` in argv, else ``default``."""
        return args[args.index(flag) + 1] if flag in args else default

    proto_out = sys.stdout
    sys.stdout = sys.stderr    # stray prints must not corrupt the protocol
    server = LeadSwitchServer(
        opt("--agent", "search"),
        opt("--ckpt", CFG.checkpoint_dir / "ckpt_best.pt"),
        leads=opt("--leads", "expert"),
        switch=opt("--switch", "expert"),
        leadnet=opt("--leadnet"),
        name=opt("--name"), seed=int(opt("--seed", 0)))
    server.serve(sys.stdin, proto_out)


if __name__ == "__main__":
    main()
