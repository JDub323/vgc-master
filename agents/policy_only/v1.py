"""Frozen v1 policy-prior-only chooser."""


class PolicyOnlyChooser:
    """Use a determinized chooser's root priors without simulations."""

    def __init__(self, chooser):
        """Wrap and share ownership of a full determinized chooser."""
        self.chooser = chooser
        # Historical code exposed .searcher; retain it for callers that inspect
        # the wrapped full chooser rather than the Searcher brick.
        self.searcher = chooser
        self.bridge = chooser.bridge
        self.belief_model_cls = getattr(chooser, "belief_model_cls", None)

    def choose(self, tracker, belief, my_id, request, brought,
               opp_brought=None, temperature=None, root_noise=None):
        """Delegate ``choose`` with ``policy_only=True``; return its tuple."""
        return self.chooser.choose(
            tracker, belief, my_id, request, brought,
            opp_brought=opp_brought, temperature=temperature,
            policy_only=True, root_noise=root_noise)

    def close(self):
        """Close the wrapped chooser and its owned subprocesses."""
        self.chooser.close()
