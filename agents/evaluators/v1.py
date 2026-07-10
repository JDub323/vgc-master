"""Frozen v1 policy/value model leaf evaluator."""


class PolicyValueLeafEvaluator:
    """Keep the batching seam used by direct and self-play inference."""

    def __init__(self, model):
        """Wrap any object implementing ``predict_batch(tokens)``."""
        self.model = model

    def predict_batch(self, tokens):
        """Return ``(joint_dists[B,1521], values[B], aux_dict)`` as numpy."""
        if self.model is None:
            raise RuntimeError("non-terminal leaf evaluation requires a model")
        return self.model.predict_batch(tokens)

    def value(self, values, index=0):
        """Orient model output from the encoded searching-player view."""
        return float(values[index])

    def terminal_value(self, winner, searching_side, opponent_side):
        """Exact zero-sum terminal result from the searching player's view."""
        return {searching_side: 1.0, opponent_side: -1.0}.get(winner, 0.0)
