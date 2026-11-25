from __future__ import annotations

import hashlib
import hmac
import random


def seed_for(root_seed: int, name: str) -> int:
    """Independent named substream seed: adding a new consumer of randomness
    never shifts the draws of an existing one."""
    h = hashlib.sha256(f"{root_seed}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def substream(root_seed: int, name: str) -> random.Random:
    return random.Random(seed_for(root_seed, name))


def secret_seed(secret: str, seed: int) -> int:
    """The scenario seed for a requested `seed` under a server-side `secret`.

    The scenario generator is public and deterministic, so anyone who knows a
    scenario's seed can compute every private value in it, and an agent can
    even recover the seed from its own first observation by search. Keyed with
    a secret, the requested seed still names one scenario (so matched arms on
    servers sharing the secret still pair up), but without the secret the
    scenario can't be computed or looked up. A 63-bit HMAC-SHA256 prefix."""
    if not secret:
        raise ValueError("seed secret must be non-empty")
    mac = hmac.new(secret.encode(), str(int(seed)).encode(), hashlib.sha256).digest()
    return int.from_bytes(mac[:8], "big") >> 1
