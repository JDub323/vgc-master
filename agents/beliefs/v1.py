"""Frozen v1 particle-filter belief implementation."""

from beliefs import OpponentBelief as _OpponentBelief


class OpponentBelief(_OpponentBelief):
    """Versioned identity for the original particle-filter behavior."""

__all__ = ["OpponentBelief"]
