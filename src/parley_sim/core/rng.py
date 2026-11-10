from __future__ import annotations

import hashlib
import random


def seed_for(root_seed: int, name: str) -> int:
    """Independent named substream seed: adding a new consumer of randomness
    never shifts the draws of an existing one."""
    h = hashlib.sha256(f"{root_seed}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def substream(root_seed: int, name: str) -> random.Random:
    return random.Random(seed_for(root_seed, name))
