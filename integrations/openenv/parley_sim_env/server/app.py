"""FastAPI entrypoint for the parley-sim OpenEnv environment.

Configured from the process environment, so one image serves any arm:

    PARLEY_ENV            environment name (default "support_desk")
    PARLEY_SEATS          JSON seat map; the focal seat is "client", e.g.
                          {"agent": "client", "customer": "policy:hardliner"}
                          or {"agent": "client", "customer": "model:peer"}
    PARLEY_LINEUP         JSON list of model specs for "model:<name>" seats
    PARLEY_ENV_CONFIG     JSON env config, e.g. {"max_turns": 10}
    PARLEY_REWARD_METRIC  reward metric (default "focal_surplus_share")
    PARLEY_MAX_REFUSALS   stop a session after this many provider refusals (default 3)
    MAX_CONCURRENT_ENVS   concurrent sessions (default 8)
    PARLEY_SESSION_TIMEOUT  close sessions idle this many seconds (default 600); must be at
                          least 2 x the sum of the seated model counterparts' timeout_s
    PARLEY_RELEASE_TRACE  "after_done" (default): state() carries the full trace once an
                          episode is done; "never": state() only ever shows what the
                          focal seat saw, plus outcomes and metrics
    PARLEY_SEED_SECRET    optional secret that keys requested seeds into scenario seeds,
                          so a client can't compute a scenario from its seed; use with
                          PARLEY_RELEASE_TRACE=never

Play goes only through WebSocket sessions (/ws); the stateless HTTP /reset, /step
and /state routes are disabled, and OpenEnv's web interface is never mounted
(deploy with `openenv push --no-interface`).

The server has no authentication. Expose it only on a trusted network.

Provider keys for model counterparts (OPENROUTER_API_KEY, ANTHROPIC_API_KEY,
OPENAI_API_KEY) are read from the environment at runtime; never bake them in.
"""

from __future__ import annotations

import json
import os
from typing import Any

from parley_sim.adapters.openenv import RELEASE_TRACE, create_parley_app

DEFAULT_SEATS = {"agent": "client", "customer": "policy:reasonable"}


def _json(name: str, default: Any) -> Any:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} is not valid JSON: {exc}") from exc


def _choice(name: str, choices: tuple[str, ...], default: str) -> str:
    value = (os.getenv(name) or "").strip() or default
    if value not in choices:
        raise SystemExit(f"{name} must be one of {', '.join(choices)}, not {value!r}")
    return value


def build_app():
    return create_parley_app(
        env=os.getenv("PARLEY_ENV", "support_desk"),
        seats=_json("PARLEY_SEATS", DEFAULT_SEATS),
        lineup=_json("PARLEY_LINEUP", []),
        env_config=_json("PARLEY_ENV_CONFIG", {}),
        reward_metric=os.getenv("PARLEY_REWARD_METRIC", "focal_surplus_share"),
        max_refusals=int(os.getenv("PARLEY_MAX_REFUSALS", "3")),
        max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
        session_timeout_s=float(os.getenv("PARLEY_SESSION_TIMEOUT", "600")),
        release_trace=_choice("PARLEY_RELEASE_TRACE", RELEASE_TRACE, "after_done"),
        seed_secret=os.getenv("PARLEY_SEED_SECRET") or None,
    )


app = build_app()


def main() -> None:
    """Run without Docker: `uv run --project . server [--port 8001]`."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    # Local by default: the server has no authentication, and with model
    # counterparts anyone who reaches it spends your provider keys.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
