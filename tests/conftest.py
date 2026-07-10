"""Shared real-engine fixtures for pytest collection.

The test modules remain directly runnable, but pytest needs explicit fixtures
for the bridge-backed cases that their ``__main__`` paths construct manually.
"""

import pytest

from config import CFG
from damage import DamageBridge
from env import Sidecar


@pytest.fixture(scope="module")
def sc():
    sidecar = Sidecar(CFG)
    try:
        yield sidecar
    finally:
        sidecar.close()


@pytest.fixture(scope="module")
def bridge():
    damage_bridge = DamageBridge(CFG)
    try:
        yield damage_bridge
    finally:
        damage_bridge.close()
