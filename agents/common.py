"""Shared result helpers for simple chooser architectures."""


def single_action_info(description: str, value: float = 0.0) -> dict:
    """Return the ``ChoiceInfo`` shape for a deterministic baseline action."""
    return {
        "value": value,
        "solve": False,
        "strategy": [(description, 1.0)],
        "q": [],
        "opp_pred": [],
        "health": {},
    }
