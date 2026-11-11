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

Provider keys for model counterparts (OPENROUTER_API_KEY, ANTHROPIC_API_KEY,
OPENAI_API_KEY) are read from the environment at runtime; never bake them in.
"""

from __future__ import annotations

import json
import os
from typing import Any

from parley_sim.adapters.openenv import create_parley_app

DEFAULT_SEATS = {"agent": "client", "customer": "policy:reasonable"}


def _json(name: str, default: Any) -> Any:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} is not valid JSON: {exc}") from exc


def build_app():
    return create_parley_app(
        env=os.getenv("PARLEY_ENV", "support_desk"),
        seats=_json("PARLEY_SEATS", DEFAULT_SEATS),
        lineup=_json("PARLEY_LINEUP", []),
        env_config=_json("PARLEY_ENV_CONFIG", {}),
        reward_metric=os.getenv("PARLEY_REWARD_METRIC", "focal_surplus_share"),
        max_refusals=int(os.getenv("PARLEY_MAX_REFUSALS", "3")),
        max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
    )


app = build_app()


def main() -> None:
    """Run without Docker: `uv run --project . server [--port 8001]`."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
