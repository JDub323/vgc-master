"""Frozen v1 tokenizer/damage-feature position encoder."""

from damage import damage_features


class TokenPositionEncoder:
    """Encode one player's tracker view with the original tokenization path."""

    def __init__(self, tokenizer, damage_bridge=None):
        """Store a tokenizer and optional ``DamageBridge`` (no ownership)."""
        self.tokenizer = tokenizer
        self.bridge = damage_bridge

    def position(self, tracker, side_id, belief):
        """Build the observable state + optional damage-feature pair."""
        state = tracker._view(side_id)
        dmg = damage_features(state, belief, self.bridge) if self.bridge else {}
        return state, dmg

    def encode_position(self, position, belief_summary):
        """Encode ``(PositionState, DamageFeatures)`` to ``uint16[n_tokens]``."""
        state, dmg = position
        return self.tokenizer.encode(state, belief_summary, dmg)

    def encode(self, tracker, side_id, belief, belief_summary=None):
        """Build and encode one side's CTS view; return a 1-D token array."""
        position = self.position(tracker, side_id, belief)
        summary = belief.summary() if belief_summary is None else belief_summary
        return self.encode_position(position, summary)
