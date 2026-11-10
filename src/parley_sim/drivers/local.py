"""In-process driver: loops an Episode, answering each request with a seat agent."""
from __future__ import annotations

from typing import Mapping, Protocol

from parley_sim.core.protocol import Seat
from parley_sim.envs import EnvEntry
from parley_sim.runtime.episode import (DecisionRequest, DecisionResult, Episode, EpisodeComplete,
                                   EpisodeSpec, Sink)


class SeatAgent(Protocol):
    async def decide(self, req: DecisionRequest) -> DecisionResult: ...


def scripted_policies(entry: EnvEntry, spec: EpisodeSpec) -> dict[Seat, object]:
    return {seat: entry.make_policy(label.split(":", 1)[1], seat)
            for seat, label in spec.seats.items() if label.startswith("policy:")}


async def run_episode(entry: EnvEntry, spec: EpisodeSpec, agents: Mapping[Seat, SeatAgent],
                      sink: Sink | None = None) -> Episode:
    ep = Episode(entry.env(), spec, scripted_policies(entry, spec), sink=sink)
    try:
        event = ep.advance()
        while not isinstance(event, EpisodeComplete):
            event = ep.advance(await agents[event.seat].decide(event))
    except BaseException as exc:
        if ep.result is None:
            ep.abort(repr(exc))
        raise
    return ep
