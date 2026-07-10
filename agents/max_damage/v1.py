"""Frozen v1 greedy expected-damage chooser."""

from actions import (T_FOE_A, T_FOE_B, joint_ok, legal_slot_actions)
from agents.common import single_action_info
from config import CFG
from damage import DamageBridge, damage_features


class MaxDamageChooser:
    """Pick each slot's highest expected immediate damage legal action."""

    def __init__(self, cfg=CFG):
        """Create and own the damage-calculator subprocess for ``cfg``."""
        self.bridge = DamageBridge(cfg)

    def choose(self, tracker, belief, my_id, request, brought,
               opp_brought=None, temperature=None, root_noise=None):
        """Return the greedy legal ``JointAction`` and baseline ``ChoiceInfo``."""
        from search.mcts import _pos_maps

        opp_id = "p2" if my_id == "p1" else "p1"
        view = tracker._view(my_id)
        dmg = damage_features(view, belief, self.bridge)
        name_to_idx = {
            mon.set["name"]: mon.team_idx
            for mon in tracker.sides[my_id].mons
        }
        idx_of_pos = _pos_maps(request, name_to_idx)[0]
        opp_at = {
            mon.active_slot: mon.team_idx
            for mon in tracker.sides[opp_id].mons
            if mon.active_slot is not None and not mon.fainted
        }
        picks = []
        slot_actions = []
        for slot in (0, 1):
            acts = legal_slot_actions(request, slot, idx_of_pos)
            slot_actions.append(acts)
            me = tracker.sides[my_id].active(slot)
            best, best_v = acts[0], -1.0
            for action in acts:
                if action.kind != "move" or me is None:
                    continue
                target = {T_FOE_A: 0, T_FOE_B: 1}.get(action.target)
                if target is None or target not in opp_at:
                    continue
                cell = dmg.get((me.team_idx, action.move_slot, opp_at[target]))
                value = (cell[0] + cell[1]) / 2 if cell else 0.0
                if value > best_v:
                    best, best_v = action, value
            picks.append(best)
        if not joint_ok(*picks):
            picks[1] = next(
                action for action in slot_actions[1]
                if joint_ok(picks[0], action))
        return tuple(picks), single_action_info("max immediate damage")

    def close(self):
        """Close the owned ``DamageBridge`` subprocess."""
        self.bridge.close()
