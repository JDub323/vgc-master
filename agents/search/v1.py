"""Frozen v1 decoupled-UCT traversal, simulation, and backup mechanics."""

import copy

import numpy as np

from actions import joint_choice, to_index
from env import SidecarBattle, random_choice


class DecoupledUCTSearcher:
    """The mechanics half of the v1 determinized DUCT architecture.

    Reconstruction, belief sampling, encoding, priors, leaf inference, and
    final temperature sampling remain the top-level chooser's responsibility.
    This brick owns tree traversal, simulation stepping, rollout handling,
    backup, and root visit aggregation. The chooser is an injected hook host;
    fake hosts can therefore exercise search mechanics without a neural net.
    """

    def run(self, chooser, determinizations, simulations_per_determinization):
        """Run an integer budget on every ``DetGame``; mutate nodes in place."""
        for det in determinizations:
            for _ in range(simulations_per_determinization):
                self.simulate(chooser, det)

    def aggregate_root(self, determinizations, policy_only=False):
        """Return sorted ``[JointAction, count, value_sum]`` mutable rows."""
        acc = {}
        for det in determinizations:
            src = det.root.my_p if policy_only else det.root.my_n
            for action, count, value_sum in zip(
                    det.root.my_actions, src, det.root.my_w):
                key = (to_index(action[0]), to_index(action[1]))
                row = acc.setdefault(key, [action, 0.0, 0.0])
                row[1] += count
                row[2] += value_sum
        return sorted(acc.values(), key=lambda row: -row[1])

    def simulate(self, chooser, det):
        """Run one sidecar trajectory and back its scalar result up the path."""
        h = chooser.health
        h["sims"] += 1
        with chooser.dbg("restore"):
            battle = SidecarBattle.restore(chooser.sc, det.root_state)
        with chooser.dbg("copy"):
            tracker = copy.deepcopy(det.seed_tracker)
        node, path, z = det.root, [], None
        for _ in range(120):
            i, j = node.select(chooser.cfg.c_puct)
            path.append((node, i, j))
            h["steps"] += 1
            with chooser.dbg("step"):
                response = battle.step({
                    det.my: joint_choice(
                        battle.requests[det.my], node.my_actions[i],
                        det.name_to_idx[det.my]),
                    det.opp: joint_choice(
                        battle.requests[det.opp], node.opp_actions[j],
                        det.name_to_idx[det.opp]),
                })
            for line in response["log"]:
                tracker.feed(line)
            if response["errors"]:
                h["fallbacks"] += 1
                with chooser.dbg("step"):
                    response = battle.step(
                        {side: "default" for side in response["errors"]})
                for line in response["log"]:
                    tracker.feed(line)
            self.settle(chooser, battle, tracker)
            if battle.ended:
                h["terminals"] += 1
                z = chooser.leaf_evaluator.terminal_value(
                    battle.winner, det.my, det.opp)
                break
            child = node.children.get((i, j))
            if child is None:
                child = chooser._expand(det, battle, tracker)
                node.children[(i, j)] = child
                if not det.solve:
                    h["value_leaves"] += 1
                    z = self.leaf_value(chooser, det, battle, tracker, child)
                    break
            node = child
        h["depth_sum"] += len(path)
        h["depth_max"] = max(h["depth_max"], len(path))
        with chooser.dbg("destroy"):
            battle.destroy()
        for old_node, i, j in path:
            old_node.update(i, j, z if z is not None else 0.0)

    def leaf_value(self, chooser, det, battle, tracker, leaf):
        """Evaluate a leaf from the searching player's value orientation."""
        h = chooser.health
        node, reached = leaf, 1
        for _ in range(max(1, chooser.cfg.rollout_depth) - 1):
            if battle.ended:
                break
            i, j = int(np.argmax(node.my_p)), int(np.argmax(node.opp_p))
            with chooser.dbg("step"):
                response = battle.step({
                    det.my: joint_choice(
                        battle.requests[det.my], node.my_actions[i],
                        det.name_to_idx[det.my]),
                    det.opp: joint_choice(
                        battle.requests[det.opp], node.opp_actions[j],
                        det.name_to_idx[det.opp]),
                })
            for line in response["log"]:
                tracker.feed(line)
            if response["errors"]:
                h["fallbacks"] += 1
                with chooser.dbg("step"):
                    response = battle.step(
                        {side: "default" for side in response["errors"]})
                for line in response["log"]:
                    tracker.feed(line)
            self.settle(chooser, battle, tracker)
            reached += 1
            if battle.ended:
                break
            node = chooser._expand(det, battle, tracker)
        h["leaf_depth_sum"] += reached
        if battle.ended:
            return chooser.leaf_evaluator.terminal_value(
                battle.winner, det.my, det.opp)
        return node.value

    def settle(self, chooser, battle, tracker):
        """Resolve forced switches in place; return ``None`` at a move/terminal."""
        while not battle.ended:
            forced = {
                side: random_choice(battle.requests[side], chooser.rng)
                for side in battle.pending_sides()
                if battle.requests[side].get("forceSwitch")
            }
            if not forced:
                return
            chooser.health["forced_switches"] += len(forced)
            with chooser.dbg("settle"):
                response = battle.step(forced)
            for line in response["log"]:
                tracker.feed(line)
